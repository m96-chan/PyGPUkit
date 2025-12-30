"""Mixture of Experts layer implementation for PyGPUkit LLM.

Provides:
- MoELayer: Mixture of Experts for Mixtral-style models
"""

from __future__ import annotations

import os
import time
from functools import reduce
from typing import TYPE_CHECKING

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import zeros
from pygpukit.ops.basic import (
    concat_axis0,
    mul,
    silu,
)

from .linear import LinearBF16, LinearFP8
from .mlp import MLP

if TYPE_CHECKING:
    from pygpukit.llm.config import TransformerConfig


class MoELayer:
    """Mixture of Experts layer for Mixtral-style models.

    Architecture:
        1. Router: hidden -> [num_experts] logits
        2. Top-K selection with softmax
        3. Expert FFN (SwiGLU) for each selected expert
        4. Weighted combination of expert outputs

    Supports FP8 quantized expert weights via LinearFP8.
    """

    def __init__(
        self,
        config: TransformerConfig,
        gate_weight: GPUArray,  # [num_experts, hidden_size] - router
        expert_weights: list,  # [(gate, up, down), ...] - GPUArray or LinearBF16/LinearFP8
    ):
        self.config = config
        self.num_experts = config.num_experts or len(expert_weights)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size or config.intermediate_size

        # Router (gate) projection
        self.gate = LinearBF16(gate_weight)

        # Expert FFNs
        self.experts: list[MLP] = []
        for gate_proj, up_proj, down_proj in expert_weights:
            expert = MLP(
                config,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
            )
            self.experts.append(expert)

        # Check if all experts use FP8 weights for grouped GEMM optimization
        self._use_grouped_gemm = False
        self._stacked_gate_weight: GPUArray | None = None
        self._stacked_gate_scale: GPUArray | None = None
        self._stacked_up_weight: GPUArray | None = None
        self._stacked_up_scale: GPUArray | None = None
        self._stacked_down_weight: GPUArray | None = None
        self._stacked_down_scale: GPUArray | None = None

        # Check if first expert uses FP8 - use grouped GEMM v2 for optimization
        # TEMP: Disabled for debugging
        if os.environ.get("PYGPUKIT_DISABLE_GROUPED_GEMM") != "1":
            if len(self.experts) > 0 and isinstance(self.experts[0].gate_proj, LinearFP8):
                self._stack_fp8_weights()

    # Profiling flag (set to True to enable timing)
    _profile: bool = True
    _profile_count: int = 0

    def _stack_fp8_weights(self) -> None:
        """Stack FP8 expert weights for grouped GEMM optimization."""
        # Collect weights from all experts
        gate_weights = []
        gate_scales = []
        up_weights = []
        up_scales = []
        down_weights = []
        down_scales = []

        for expert in self.experts:
            if not isinstance(expert.gate_proj, LinearFP8):
                return  # Not all experts are FP8, abort

            gate_weights.append(expert.gate_proj.weight_fp8)
            gate_scales.append(expert.gate_proj.scale_inv)
            up_weights.append(expert.up_proj.weight_fp8)
            up_scales.append(expert.up_proj.scale_inv)
            down_weights.append(expert.down_proj.weight_fp8)
            down_scales.append(expert.down_proj.scale_inv)

        # Stack weights: [num_experts, N, K]
        # gate_proj: [intermediate_size, hidden_size] -> stacked [num_experts, intermediate_size, hidden_size]
        # Each weight is [N, K], stack along new axis 0

        def stack_arrays_fast(arrays: list[GPUArray]) -> GPUArray:
            """Stack arrays along new axis 0 using single allocation + cudaMemcpy."""
            from pygpukit.core.backend import get_native_module

            native = get_native_module()

            # Get shape info from first array
            first = arrays[0]
            num_arrays = len(arrays)
            inner_shape = first.shape  # [N, K] or [N/128, K/128]

            # Calculate strides (nbytes is property, not method)
            bytes_per_array = first._get_native().nbytes

            # Allocate output: [num_arrays, *inner_shape]
            out_shape = [num_arrays] + list(inner_shape)
            out_native = native.empty(out_shape, first._get_native().dtype)
            out = GPUArray._wrap_native(out_native)

            # Copy each array to its slice using cuMemcpy
            for i, arr in enumerate(arrays):
                offset_bytes = i * bytes_per_array
                native.memcpy_device_to_device_offset(
                    arr._get_native(),
                    out._get_native(),
                    0,  # src offset
                    offset_bytes,  # dst offset
                    bytes_per_array,
                )

            return out

        self._stacked_gate_weight = stack_arrays_fast(gate_weights)
        self._stacked_gate_scale = stack_arrays_fast(gate_scales)
        self._stacked_up_weight = stack_arrays_fast(up_weights)
        self._stacked_up_scale = stack_arrays_fast(up_scales)
        self._stacked_down_weight = stack_arrays_fast(down_weights)
        self._stacked_down_scale = stack_arrays_fast(down_scales)

        self._use_grouped_gemm = True
        print(f"[MoE] Stacked {self.num_experts} expert weights for grouped GEMM")

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor [batch, seq, hidden_size] or [seq, hidden_size]

        Returns:
            Output tensor with same shape as input
        """
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        profile = self._profile and MoELayer._profile_count < 3
        if profile:
            native.device_synchronize()
            t0 = time.perf_counter()

        original_shape = x.shape
        # Flatten to [num_tokens, hidden_size]
        if len(original_shape) == 3:
            batch, seq, hidden = original_shape
            num_tokens = batch * seq
            x = x.reshape(num_tokens, hidden)
        else:
            num_tokens, hidden = original_shape

        k = self.num_experts_per_tok

        # Step 1: Compute router logits
        router_logits = self.gate(x)  # [num_tokens, num_experts]
        if profile:
            native.device_synchronize()
            t1 = time.perf_counter()

        # Step 2: Top-K selection
        router_weights = zeros((num_tokens, k), dtype=x.dtype)
        expert_indices = zeros((num_tokens, k), dtype="int32")
        native.moe_topk_with_indices(
            router_logits._get_native(),
            router_weights._get_native(),
            expert_indices._get_native(),
            k,
        )

        # Step 3: Softmax over selected experts
        native.moe_softmax_topk(router_weights._get_native(), k)

        # Step 4: Compute permutation for efficient expert dispatch
        expert_counts = zeros((self.num_experts,), dtype="int32")
        expert_offsets = zeros((self.num_experts + 1,), dtype="int32")
        permute_indices = zeros((num_tokens * k,), dtype="int32")
        reverse_perm = zeros((num_tokens * k,), dtype="int32")
        native.moe_compute_permutation(
            expert_indices._get_native(),
            expert_counts._get_native(),
            expert_offsets._get_native(),
            permute_indices._get_native(),
            reverse_perm._get_native(),
            self.num_experts,
            k,
        )

        # Step 5: Gather hidden states for experts
        gathered = zeros((num_tokens * k, hidden), dtype=x.dtype)
        native.moe_gather(
            x._get_native(),
            permute_indices._get_native(),
            gathered._get_native(),
            k,
        )
        if profile:
            native.device_synchronize()
            t2 = time.perf_counter()

        # Step 6: Run experts
        if self._use_grouped_gemm:
            # Use grouped GEMM for all experts in single kernel launches
            from pygpukit.ops.matmul import grouped_gemm_fp8_bf16

            # Create row_expert_ids from expert_offsets
            M_total = num_tokens * k
            row_expert_ids = zeros((M_total,), dtype="int32")
            native.moe_expand_expert_offsets(
                expert_offsets._get_native(),
                row_expert_ids._get_native(),
                self.num_experts,
            )

            # gate_proj: gathered[M_total, hidden] @ gate_weight[experts, inter, hidden]^T
            gate_out = grouped_gemm_fp8_bf16(
                gathered,
                self._stacked_gate_weight,
                self._stacked_gate_scale,
                row_expert_ids,
            )

            # up_proj: gathered[M_total, hidden] @ up_weight[experts, inter, hidden]^T
            up_out = grouped_gemm_fp8_bf16(
                gathered,
                self._stacked_up_weight,
                self._stacked_up_scale,
                row_expert_ids,
            )

            # SiLU(gate) * up
            intermediate = mul(silu(gate_out), up_out)

            # down_proj: intermediate[M_total, inter] @ down_weight[experts, hidden, inter]^T
            expert_outputs = grouped_gemm_fp8_bf16(
                intermediate,
                self._stacked_down_weight,
                self._stacked_down_scale,
                row_expert_ids,
            )
        else:
            # Fallback: Run experts sequentially
            # Get expert counts on CPU for loop
            expert_counts_cpu = expert_counts.to_numpy()
            expert_offsets_cpu = expert_offsets.to_numpy()

            # Build list of (expert_id, start, count) for non-empty experts
            expert_tasks = []
            for e in range(self.num_experts):
                start = int(expert_offsets_cpu[e])
                count = int(expert_counts_cpu[e])
                if count > 0:
                    expert_tasks.append((e, start, count))

            def run_expert(task: tuple) -> GPUArray:
                e, start, count = task
                expert_input = gathered[start : start + count]
                return self.experts[e](expert_input)

            # Run experts sequentially
            expert_output_list = [run_expert(task) for task in expert_tasks]

            # Concatenate all expert outputs on GPU
            expert_outputs = reduce(concat_axis0, expert_output_list)

        if profile:
            native.device_synchronize()
            t3 = time.perf_counter()

        # Step 7: Scatter and combine outputs
        output = zeros((num_tokens, hidden), dtype=x.dtype)
        native.moe_scatter(
            expert_outputs._get_native(),
            router_weights._get_native(),
            reverse_perm._get_native(),
            output._get_native(),
            k,
        )
        if profile:
            native.device_synchronize()
            t4 = time.perf_counter()
            MoELayer._profile_count += 1
            print(
                f"[MoE Profile] router={t1 - t0:.3f}s, routing={t2 - t1:.3f}s, experts={t3 - t2:.3f}s, scatter={t4 - t3:.3f}s"
            )

        # Reshape back
        if len(original_shape) == 3:
            output = output.reshape(*original_shape)

        return output

    def forward_zero_alloc(
        self,
        x: GPUArray,
        router_logits: GPUArray,
        router_weights: GPUArray,
        expert_indices: GPUArray,
        expert_counts: GPUArray,
        expert_offsets: GPUArray,
        permute_indices: GPUArray,
        reverse_perm: GPUArray,
        row_expert_ids: GPUArray,
        gathered: GPUArray,
        gate_out: GPUArray,
        up_out: GPUArray,
        intermediate: GPUArray,
        expert_outputs: GPUArray,
        output: GPUArray,
    ) -> GPUArray:
        """Zero-allocation forward pass for CUDA Graph support.

        This method uses pre-allocated buffers from DecodeBuffers to avoid
        any memory allocations during forward pass, enabling CUDA Graph capture.

        Args:
            x: Input tensor [1, hidden_size]
            router_logits: Pre-allocated [1, num_experts]
            router_weights: Pre-allocated [1, k]
            expert_indices: Pre-allocated [1, k] int32
            expert_counts: Pre-allocated [num_experts] int32
            expert_offsets: Pre-allocated [num_experts + 1] int32
            permute_indices: Pre-allocated [k] int32
            reverse_perm: Pre-allocated [k] int32
            row_expert_ids: Pre-allocated [k] int32
            gathered: Pre-allocated [k, hidden_size]
            gate_out: Pre-allocated [k, moe_intermediate_size]
            up_out: Pre-allocated [k, moe_intermediate_size]
            intermediate: Pre-allocated [k, moe_intermediate_size]
            expert_outputs: Pre-allocated [k, hidden_size]
            output: Pre-allocated [1, hidden_size]

        Returns:
            The output tensor (same as output parameter)
        """
        from pygpukit.core.backend import get_native_module
        from pygpukit.ops.elementwise import mul
        from pygpukit.ops.matmul import grouped_gemm_fp8_bf16
        from pygpukit.ops.nn import silu

        native = get_native_module()

        k = self.num_experts_per_tok

        # Step 1: Router forward (gate projection)
        self.gate(x, out=router_logits)

        # Step 2: Top-K selection (writes to router_weights and expert_indices)
        native.moe_topk_with_indices(
            router_logits._get_native(),
            router_weights._get_native(),
            expert_indices._get_native(),
            k,
        )

        # Step 3: Softmax over selected experts (in-place)
        native.moe_softmax_topk(router_weights._get_native(), k)

        # Step 4: Compute permutation
        native.moe_compute_permutation(
            expert_indices._get_native(),
            expert_counts._get_native(),
            expert_offsets._get_native(),
            permute_indices._get_native(),
            reverse_perm._get_native(),
            self.num_experts,
            k,
        )

        # Step 5: Gather hidden states
        native.moe_gather(
            x._get_native(),
            permute_indices._get_native(),
            gathered._get_native(),
            k,
        )

        # Step 6: Create row_expert_ids for grouped GEMM
        native.moe_expand_expert_offsets(
            expert_offsets._get_native(),
            row_expert_ids._get_native(),
            self.num_experts,
        )

        # Step 7: Expert computation with grouped GEMM
        # gate_proj: gathered[k, hidden] @ gate_weight[experts, inter, hidden]^T
        grouped_gemm_fp8_bf16(
            gathered,
            self._stacked_gate_weight,
            self._stacked_gate_scale,
            row_expert_ids,
            out=gate_out,
        )

        # up_proj: gathered[k, hidden] @ up_weight[experts, inter, hidden]^T
        grouped_gemm_fp8_bf16(
            gathered,
            self._stacked_up_weight,
            self._stacked_up_scale,
            row_expert_ids,
            out=up_out,
        )

        # SiLU(gate) * up -> intermediate
        silu(gate_out, out=intermediate)
        mul(intermediate, up_out, out=intermediate)

        # down_proj: intermediate[k, inter] @ down_weight[experts, hidden, inter]^T
        grouped_gemm_fp8_bf16(
            intermediate,
            self._stacked_down_weight,
            self._stacked_down_scale,
            row_expert_ids,
            out=expert_outputs,
        )

        # Step 8: Scatter and combine outputs
        native.moe_scatter(
            expert_outputs._get_native(),
            router_weights._get_native(),
            reverse_perm._get_native(),
            output._get_native(),
            k,
        )

        return output


__all__ = [
    "MoELayer",
]
