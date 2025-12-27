"""CUDA Graph-accelerated M=1 decode strategy.

This module provides DecodeM1Graph for single-token decoding with CUDA Graph.

CUDA Graph Architecture:
- Graph captures ONLY stateless operations (projections, norms, RoPE)
- SDPA and KV cache operations run OUTSIDE the graph
- This avoids warmup pollution and ensures correct KV cache handling

Requirements for CUDA Graph usage:
- Fixed shape/dtype/RoPE tables (no dynamic changes)
- Identical kernel path for warmup/capture/replay
- No KV cache pollution during warmup/capture
- H2D copies on capture stream
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.llm.decode.base import DecodeStrategy
from pygpukit.ops.basic import (
    add_inplace,
    bias_add_inplace,
    copy_to,
    embedding_lookup_ptr,
    kv_cache_update_gqa,
    matmul,
    mul_inplace,
    reshape_copy,
    rmsnorm,
    rope_inplace_f32table,
    sdpa_causal_fixed_cache,
    silu,
    transpose_3d_021,
)

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers


class DecodeM1Graph(DecodeStrategy):
    """CUDA Graph-accelerated single-token decode strategy.

    This strategy captures stateless operations in CUDA Graphs and
    executes SDPA/KV cache operations manually outside the graph.

    Graph contains:
    - Embedding lookup (via GPU pointer)
    - Linear projections (QKV, O, MLP gate/up/down)
    - RMSNorm
    - RoPE (via pre-computed GPU tables)
    - Activation functions (SiLU)

    Outside graph (manual execution):
    - KV cache update
    - SDPA attention
    """

    def __init__(self) -> None:
        """Initialize DecodeM1Graph strategy."""
        super().__init__()
        self._graph_ready = False
        self._decode_buffers: DecodeBuffers | None = None

        # Per-phase graphs
        self._embed_graph = None
        self._pre_sdpa_graphs: list = []
        self._post_sdpa_graphs: list = []
        self._final_graph = None

        # Numpy buffers for H2D transfers
        self._pos_np: np.ndarray | None = None
        self._tok_np: np.ndarray | None = None
        self._graph_max_seq_len: int = 0

        # F32 RoPE buffers (for numerical consistency with prefill)
        self._cos_f32: GPUArray | None = None
        self._sin_f32: GPUArray | None = None

    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute decode step (non-graph fallback).

        This method is not used by DecodeM1Graph.
        Use step_graph() instead after calling init_graph().

        Raises:
            NotImplementedError: Always. Use DecodeM1 for non-graph decode.
        """
        raise NotImplementedError(
            "DecodeM1Graph does not support non-graph decode. "
            "Use DecodeM1 for non-graph decode, or call init_graph() and step_graph()."
        )

    def _exec_pre_sdpa(self, block, buffers: DecodeBuffers) -> None:
        """Execute pre-SDPA operations.

        Operations: RMSNorm -> QKV projection -> biases -> reshape -> QK norm -> RoPE
        Output: Q, K, V in buffers (ready for KV cache update and SDPA)
        """
        model = self.model
        attn = block.attn

        # DEBUG: CUDA Graph investigation - pointer tracking (layer 0 only)
        # Kept for future debugging of CUDA Graph capture issues
        # if block is model.blocks[0]:
        #     if not hasattr(self, '_exec_call_count'):
        #         self._exec_call_count = 0
        #     self._exec_call_count += 1
        #     # Print first 5 calls only
        #     if self._exec_call_count <= 5:
        #         print(f"    [EXEC#{self._exec_call_count}] buffers id: {id(buffers)}, norm_out: {hex(buffers.norm_out._get_native().data_ptr())}, qkv_out: {hex(buffers.qkv_proj_out._get_native().data_ptr())}")

        # RMSNorm (attn pre-norm)
        rmsnorm(
            buffers.hidden,
            block.attn_norm.weight,
            block.attn_norm.eps,
            out=buffers.norm_out,
        )

        # Save hidden to residual for later add
        copy_to(buffers.hidden, buffers.residual)

        # QKV projection (fused or separate)
        if attn.qkv_proj is not None:
            # Fused QKV projection
            attn.qkv_proj(buffers.norm_out, out=buffers.qkv_proj_out)

            # Apply biases if present
            if attn.q_proj.bias is not None:
                bias_add_inplace(buffers.q_view, attn.q_proj.bias)
            if attn.k_proj.bias is not None:
                bias_add_inplace(buffers.k_view, attn.k_proj.bias)
            if attn.v_proj.bias is not None:
                bias_add_inplace(buffers.v_view, attn.v_proj.bias)

            # Reshape to 3D: [1, num_heads, head_dim]
            reshape_copy(buffers.q_view, (1, attn.num_heads, attn.head_dim), out=buffers.q)
            reshape_copy(buffers.k_view, (1, attn.num_kv_heads, attn.head_dim), out=buffers.k)
            reshape_copy(buffers.v_view, (1, attn.num_kv_heads, attn.head_dim), out=buffers.v)
        else:
            # Separate Q, K, V projections
            attn.q_proj(buffers.norm_out, out=buffers.q_proj_out)
            attn.k_proj(buffers.norm_out, out=buffers.k_proj_out)
            attn.v_proj(buffers.norm_out, out=buffers.v_proj_out)

            # Apply biases if present
            if attn.q_proj.bias is not None:
                bias_add_inplace(buffers.q_proj_out, attn.q_proj.bias)
            if attn.k_proj.bias is not None:
                bias_add_inplace(buffers.k_proj_out, attn.k_proj.bias)
            if attn.v_proj.bias is not None:
                bias_add_inplace(buffers.v_proj_out, attn.v_proj.bias)

            # Reshape to 3D: [1, num_heads, head_dim]
            reshape_copy(buffers.q_proj_out, (1, attn.num_heads, attn.head_dim), out=buffers.q)
            reshape_copy(buffers.k_proj_out, (1, attn.num_kv_heads, attn.head_dim), out=buffers.k)
            reshape_copy(buffers.v_proj_out, (1, attn.num_kv_heads, attn.head_dim), out=buffers.v)

        # QK Norm (Qwen3) if present
        if attn.q_norm is not None and buffers.q_2d is not None:
            reshape_copy(buffers.q, (attn.num_heads, attn.head_dim), out=buffers.q_flat)
            rmsnorm(buffers.q_flat, attn.q_norm.weight, attn.q_norm.eps, out=buffers.q_2d)
            reshape_copy(buffers.q_2d, (1, attn.num_heads, attn.head_dim), out=buffers.q)
        if attn.k_norm is not None and buffers.k_2d is not None:
            reshape_copy(buffers.k, (attn.num_kv_heads, attn.head_dim), out=buffers.k_flat)
            rmsnorm(buffers.k_flat, attn.k_norm.weight, attn.k_norm.eps, out=buffers.k_2d)
            reshape_copy(buffers.k_2d, (1, attn.num_kv_heads, attn.head_dim), out=buffers.k)

        # Apply RoPE using pre-computed f32 GPU tables
        # Use rope_inplace_f32table for bf16/f16 Q/K with f32 cos/sin tables
        if model.config.use_rope and hasattr(model, "_rope_cos_gpu"):
            embedding_lookup_ptr(model._rope_cos_gpu, self._cos_f32, buffers.position_buf)
            embedding_lookup_ptr(model._rope_sin_gpu, self._sin_f32, buffers.position_buf)
            rope_inplace_f32table(buffers.q, buffers.k, self._cos_f32, self._sin_f32)

        # Transpose Q for SDPA: [1, num_heads, head_dim] -> [num_heads, 1, head_dim]
        transpose_3d_021(buffers.q, out=buffers.q_t)

    def _exec_post_sdpa(self, block, buffers: DecodeBuffers) -> None:
        """Execute post-SDPA operations.

        Operations: transpose -> reshape -> O_proj -> residual add
                   -> MLP norm -> gate_up -> silu -> mul -> down -> residual add
        Input: attn_out in buffers (from SDPA)
        Output: Updated hidden in buffers
        """

        attn = block.attn
        mlp = block.mlp

        # Transpose attention output: [num_heads, 1, head_dim] -> [1, num_heads, head_dim]
        transpose_3d_021(buffers.attn_out, out=buffers.q)

        # Reshape to 2D: [1, hidden_size]
        reshape_copy(buffers.q, (1, attn.num_heads * attn.head_dim), out=buffers.q_proj_out)

        # Output projection -> hidden
        attn.o_proj(buffers.q_proj_out, out=buffers.hidden)

        # Add attention residual
        add_inplace(buffers.hidden, buffers.residual)

        # Save for MLP residual
        copy_to(buffers.hidden, buffers.residual)

        # MLP pre-norm
        rmsnorm(
            buffers.hidden,
            block.mlp_norm.weight,
            block.mlp_norm.eps,
            out=buffers.norm_out,
        )

        # MLP forward (SwiGLU)
        # Note: MoE models are not supported in CUDA Graph mode (checked in init_graph)
        if hasattr(mlp, "gate_up_proj") and mlp.gate_up_proj is not None:
            # Fused gate+up projection (non-MoE)
            mlp.gate_up_proj(buffers.norm_out, out=buffers.gate_up_out)
            silu(buffers.gate_view, out=buffers.gate_view)
            mul_inplace(buffers.gate_view, buffers.up_view)
            mlp.down_proj(buffers.gate_view, out=buffers.mlp_down)
            # MLP output to hidden
            copy_to(buffers.mlp_down, buffers.hidden)
        else:
            # Separate projections (non-MoE)
            mlp.gate_proj(buffers.norm_out, out=buffers.mlp_gate)
            silu(buffers.mlp_gate, out=buffers.mlp_gate)
            mlp.up_proj(buffers.norm_out, out=buffers.mlp_up)
            mul_inplace(buffers.mlp_gate, buffers.mlp_up)
            mlp.down_proj(buffers.mlp_gate, out=buffers.mlp_down)
            # MLP output to hidden
            copy_to(buffers.mlp_down, buffers.hidden)

        # Add MLP residual
        add_inplace(buffers.hidden, buffers.residual)

    def init_graph(self, max_seq_len: int = 512) -> None:
        """Initialize CUDA Graphs for decode.

        Captures multiple graphs:
        - embed_graph: Embedding lookup
        - pre_sdpa_graphs[i]: Layer i pre-SDPA ops (norm, QKV, RoPE)
        - post_sdpa_graphs[i]: Layer i post-SDPA ops (O_proj, MLP)
        - final_graph: Final norm + LM head

        SDPA and KV cache operations are NOT captured.

        Args:
            max_seq_len: Maximum sequence length for RoPE pre-computation.
        """
        import gc

        from pygpukit._native_loader import get_native_module
        from pygpukit.core import default_stream
        from pygpukit.core.factory import from_numpy

        CudaGraph = getattr(get_native_module(), "CudaGraph")  # noqa: B009
        from pygpukit.llm.buffers import DecodeBuffers
        from pygpukit.llm.layers import MoELayer, precompute_freqs_cis

        model = self.model
        dtype = str(model.embed_tokens.dtype)
        use_qk_norm = model.spec is not None and model.spec.use_qk_norm
        lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
        vocab_size = lm_head.shape[0]

        # Detect MoE model - CUDA Graph not yet supported for MoE
        for block in model.blocks:
            if isinstance(block.mlp, MoELayer):
                raise NotImplementedError(
                    "CUDA Graph is not yet supported for MoE models. "
                    "MoE uses grouped GEMM which cannot be captured in CUDA Graph. "
                    "Use non-graph decode mode instead (remove --cuda-graph flag)."
                )

        # MoE config not used for now (CUDA Graph doesn't support MoE)
        moe_config = None

        # Allocate decode buffers (with MoE buffers if needed)
        self._decode_buffers = DecodeBuffers.allocate(
            model.config,
            dtype=dtype,
            use_qk_norm=use_qk_norm,
            vocab_size=vocab_size,
            moe_config=moe_config,
        )
        buffers = self._decode_buffers

        # Pre-compute RoPE tables on GPU (always f32 for numerical consistency)
        # This matches prefill which uses f32 cos/sin tables.
        # bf16/f16 Q/K tensors are promoted to f32 for RoPE computation.
        # Note: We always recreate as f32 because caller may have set different dtype.
        if model.config.use_rope:
            cos_np, sin_np = precompute_freqs_cis(
                model.config.head_dim, max_seq_len, model.config.rope_theta
            )
            model._rope_cos_gpu = from_numpy(cos_np.astype(np.float32))
            model._rope_sin_gpu = from_numpy(sin_np.astype(np.float32))

        # Allocate f32 cos/sin buffers for RoPE lookup (single position)
        from pygpukit.core.factory import zeros

        self._cos_f32 = zeros((1, model.config.head_dim), dtype="float32")
        self._sin_f32 = zeros((1, model.config.head_dim), dtype="float32")

        # Cache transposed lm_head
        if not hasattr(model, "_lm_head_t_cache"):
            lm_head_np = lm_head.to_numpy()
            model._lm_head_t_cache = from_numpy(lm_head_np.T.copy())

        # Numpy buffers for H2D
        self._pos_np = np.array([0], dtype=np.int32)
        self._tok_np = np.array([0], dtype=np.int32)
        self._graph_max_seq_len = max_seq_len

        # Initialize GPU buffers
        self._pos_np[0] = 0
        buffers.position_buf._get_native().copy_from_numpy(self._pos_np)
        self._tok_np[0] = 0
        buffers.token_id_buf._get_native().copy_from_numpy(self._tok_np)

        print("  [CUDA Graph] Capturing graphs (SDPA outside graph)...")
        # DEBUG: CUDA Graph investigation - buffer pointer tracking
        # Kept for future debugging of CUDA Graph capture issues
        # print(f"    [DEBUG A] buffers.norm_out ptr: {hex(buffers.norm_out._get_native().data_ptr())}")
        # print(f"    [DEBUG A] buffers.qkv_proj_out ptr: {hex(buffers.qkv_proj_out._get_native().data_ptr())}")
        # print(f"    [DEBUG A] buffers id: {id(buffers)}")

        # =====================================================================
        # Create dummy KV caches and swap with real ones during warmup/capture
        # This ensures real KV caches are never touched during graph setup
        # =====================================================================
        print("  [CUDA Graph] Creating dummy KV caches...")

        # Save real KV cache references
        real_k_caches = []
        real_v_caches = []
        for block in model.blocks:
            real_k_caches.append(block.attn._k_cache)
            real_v_caches.append(block.attn._v_cache)

        # Create dummy KV caches (same shape/dtype as real ones)
        dummy_k_caches = []
        dummy_v_caches = []
        for block in model.blocks:
            if block.attn._k_cache is not None:
                k_shape = block.attn._k_cache.shape
                v_shape = block.attn._v_cache.shape
                k_dtype = block.attn._k_cache.to_numpy().dtype
                v_dtype = block.attn._v_cache.to_numpy().dtype
                dummy_k = from_numpy(np.zeros(k_shape, dtype=k_dtype))
                dummy_v = from_numpy(np.zeros(v_shape, dtype=v_dtype))
                dummy_k_caches.append(dummy_k)
                dummy_v_caches.append(dummy_v)
            else:
                dummy_k_caches.append(None)
                dummy_v_caches.append(None)

        # Swap to dummy KV caches
        for i, block in enumerate(model.blocks):
            block.attn._k_cache = dummy_k_caches[i]
            block.attn._v_cache = dummy_v_caches[i]

        print("  [CUDA Graph] Warming up kernels (using dummy KV)...")
        # Warmup all kernels (same path as capture)
        for _ in range(3):
            embedding_lookup_ptr(model.embed_tokens, buffers.hidden, buffers.token_id_buf)
            for block in model.blocks:
                self._exec_pre_sdpa(block, buffers)
                # Skip SDPA during warmup - no KV pollution
                self._exec_post_sdpa(block, buffers)
            rmsnorm(
                buffers.hidden, model.final_norm.weight, model.final_norm.eps, out=buffers.norm_out
            )
            copy_to(buffers.norm_out, buffers.hidden)
            matmul(buffers.hidden, model._lm_head_t_cache, out=buffers.logits)
        default_stream().synchronize()

        gc.disable()
        try:
            # Capture embedding graph
            print("  [CUDA Graph] Capturing embedding graph...")
            self._embed_graph = CudaGraph()
            self._embed_graph.begin_capture()
            embedding_lookup_ptr(model.embed_tokens, buffers.hidden, buffers.token_id_buf)
            self._embed_graph.end_capture()

            # Capture per-layer graphs
            self._pre_sdpa_graphs = []
            self._post_sdpa_graphs = []

            for i, block in enumerate(model.blocks):
                # DEBUG: CUDA Graph investigation - weight_t pointer during capture
                # Kept for future debugging of CUDA Graph capture issues
                # if i == 0:
                #     wt = block.attn.qkv_proj._weight_t
                #     if wt is not None:
                #         print(f"    [CAPTURE L0] qkv_proj._weight_t: {hex(wt._get_native().data_ptr())}")
                #     else:
                #         print(f"    [CAPTURE L0] qkv_proj._weight_t: None")

                # Pre-SDPA graph
                pre_graph = CudaGraph()
                pre_graph.begin_capture()
                self._exec_pre_sdpa(block, buffers)
                pre_graph.end_capture()
                self._pre_sdpa_graphs.append(pre_graph)

                # Post-SDPA graph
                post_graph = CudaGraph()
                post_graph.begin_capture()
                self._exec_post_sdpa(block, buffers)
                post_graph.end_capture()
                self._post_sdpa_graphs.append(post_graph)

                if (i + 1) % 10 == 0:
                    print(f"    Captured layer {i + 1}/{len(model.blocks)}")

            # Capture final graph
            print("  [CUDA Graph] Capturing final graph...")
            self._final_graph = CudaGraph()
            self._final_graph.begin_capture()
            rmsnorm(
                buffers.hidden, model.final_norm.weight, model.final_norm.eps, out=buffers.norm_out
            )
            copy_to(buffers.norm_out, buffers.hidden)
            matmul(buffers.hidden, model._lm_head_t_cache, out=buffers.logits)
            self._final_graph.end_capture()

        finally:
            gc.enable()

            # Restore real KV caches after warmup/capture
            print("  [CUDA Graph] Restoring real KV caches...")
            for i, block in enumerate(model.blocks):
                block.attn._k_cache = real_k_caches[i]
                block.attn._v_cache = real_v_caches[i]

            # Free dummy caches
            del dummy_k_caches
            del dummy_v_caches

        self._graph_ready = True
        total = 1 + 2 * len(model.blocks) + 1
        print(f"  [CUDA Graph] Captured {total} graphs")
        print("  [CUDA Graph] SDPA and KV cache ops run outside graph (using real KV)")

    def has_graph(self) -> bool:
        """Check if CUDA Graph is ready."""
        return self._graph_ready

    def step_graph(
        self,
        token_id: int,
        position: int,
        context_len: int,
    ) -> GPUArray:
        """Execute decode step using CUDA Graph with interleaved SDPA.

        Flow:
        1. H2D copy (token_id, position)
        2. embed_graph.replay()
        3. For each layer:
           a. pre_sdpa_graph[i].replay()
           b. KV cache update (manual)
           c. SDPA (manual)
           d. post_sdpa_graph[i].replay()
        4. final_graph.replay()

        Args:
            token_id: Input token ID.
            position: Position in sequence.
            context_len: Total context length.

        Returns:
            Logits buffer [1, vocab_size].
        """
        assert self._graph_ready, "Call init_graph() first"
        assert self._decode_buffers is not None

        model = self.model
        buffers = self._decode_buffers

        # H2D copies - use synchronous copy then device sync to ensure visibility
        # Each CudaGraph has its own cudaStreamNonBlocking stream, which doesn't
        # implicitly sync with other streams. Using device_synchronize ensures
        # all GPU work is complete and data is visible to all streams.
        from pygpukit._native_loader import get_native_module

        device_synchronize = getattr(get_native_module(), "device_synchronize")  # noqa: B009

        self._tok_np[0] = token_id
        self._pos_np[0] = position
        buffers.token_id_buf._get_native().copy_from_numpy(self._tok_np)
        buffers.position_buf._get_native().copy_from_numpy(self._pos_np)

        # Full device sync to ensure H2D visible to all graph streams
        device_synchronize()

        # DEBUG: CUDA Graph investigation - H2D and embedding verification
        # Kept for future debugging of CUDA Graph capture issues
        # import os
        # if os.environ.get("PYGPUKIT_DEBUG_GRAPH") == "1":
        #     import numpy as np
        #     pos_check = buffers.position_buf.to_numpy()
        #     tok_check = buffers.token_id_buf.to_numpy()
        #     print(f"[DEBUG] After H2D: position_buf={pos_check[0]}, token_id_buf={tok_check[0]}")
        #     print(f"[DEBUG] Expected: position={position}, token_id={token_id}")

        # Embedding graph
        self._embed_graph.replay()
        device_synchronize()

        # DEBUG: CUDA Graph investigation - embedding output verification
        # Kept for future debugging of CUDA Graph capture issues
        # if os.environ.get("PYGPUKIT_DEBUG_GRAPH") == "1":
        #     hidden_np = buffers.hidden.to_numpy()
        #     print(f"[DEBUG] After embed: hidden[:5]={hidden_np[0, :5]}, sum={np.sum(hidden_np):.4f}")

        # Transformer layers with interleaved SDPA
        for i, block in enumerate(model.blocks):
            # Pre-SDPA (graphed)
            self._pre_sdpa_graphs[i].replay()
            device_synchronize()

            # DEBUG: CUDA Graph investigation - layer 0 pre_sdpa outputs
            # Kept for future debugging of CUDA Graph capture issues
            # if os.environ.get("PYGPUKIT_DEBUG_GRAPH") == "1" and i == 0:
            #     import numpy as np
            #     cos_np = self._cos_f32.to_numpy()
            #     sin_np = self._sin_f32.to_numpy()
            #     q_np = buffers.q.to_numpy()
            #     k_np = buffers.k.to_numpy()
            #     print(f"[DEBUG] Layer 0 pre_sdpa:")
            #     print(f"  cos[:5]={cos_np[0, :5]}")
            #     print(f"  sin[:5]={sin_np[0, :5]}")
            #     print(f"  q[:5]={q_np[0, 0, :5]}, q_sum={np.sum(q_np):.4f}")
            #     print(f"  k[:5]={k_np[0, 0, :5]}, k_sum={np.sum(k_np):.4f}")

            # KV cache update (NOT graphed)
            kv_cache_update_gqa(buffers.k, block.attn._k_cache, block.attn.num_heads, position)
            kv_cache_update_gqa(buffers.v, block.attn._v_cache, block.attn.num_heads, position)

            # SDPA (NOT graphed)
            sdpa_causal_fixed_cache(
                buffers.q_t,
                block.attn._k_cache,
                block.attn._v_cache,
                buffers.attn_out,
                context_len,
            )
            device_synchronize()

            # DEBUG: CUDA Graph investigation - SDPA output for layer 0
            # Kept for future debugging of CUDA Graph capture issues
            # if os.environ.get("PYGPUKIT_DEBUG_GRAPH") == "1" and i == 0:
            #     attn_np = buffers.attn_out.to_numpy()
            #     print(f"  attn_out[:5]={attn_np[0, 0, :5]}, attn_sum={np.sum(attn_np):.4f}")

            # Post-SDPA (graphed)
            self._post_sdpa_graphs[i].replay()
            device_synchronize()

        # Final norm + LM head (graphed)
        self._final_graph.replay()
        device_synchronize()

        # DEBUG: CUDA Graph investigation - final logits verification
        # Kept for future debugging of CUDA Graph capture issues
        # if os.environ.get("PYGPUKIT_DEBUG_GRAPH") == "1":
        #     logits_np = buffers.logits.to_numpy()
        #     if logits_np.dtype == np.uint16:
        #         logits_np = (logits_np.astype(np.uint32) << 16).view(np.float32)
        #     print(f"[DEBUG] Final logits[:5]={logits_np[0, :5]}")
        #     print(f"[DEBUG] argmax={np.argmax(logits_np[0])}")

        assert buffers.logits is not None, "logits buffer not allocated"
        return buffers.logits

    @property
    def buffers(self) -> DecodeBuffers | None:
        """Get the decode buffers."""
        return self._decode_buffers
