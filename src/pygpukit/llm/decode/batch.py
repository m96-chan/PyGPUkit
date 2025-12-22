"""Batch decode strategy.

This module provides the DecodeBatch strategy for decoding multiple
tokens at once, with optional CUDA Graph acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.llm.decode.base import DecodeStrategy
from pygpukit.ops.basic import (
    add_inplace,
    copy_to,
    embedding_lookup_batch,
    matmul,
    rmsnorm,
)

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers


class DecodeBatch(DecodeStrategy):
    """Batch decode strategy with optional CUDA Graph support.

    This strategy handles batch decoding (processing multiple tokens at once),
    which is useful for speculative decoding verification.

    CUDA Graph mode pre-captures the decode computation and replays it
    with updated buffer values, eliminating kernel launch overhead.
    """

    def __init__(self, batch_size: int = 8) -> None:
        """Initialize DecodeBatch strategy.

        Args:
            batch_size: Maximum batch size for decode.
        """
        super().__init__()
        self._batch_size = batch_size
        self._batch_decode_graph = None
        self._batch_decode_graph_ready = False
        self._batch_decode_buffers: DecodeBuffers | None = None

        # Numpy buffers for H2D transfers
        self._batch_token_ids_np: np.ndarray | None = None
        self._batch_start_pos_np: np.ndarray | None = None
        self._batch_ctx_len_np: np.ndarray | None = None
        self._batch_graph_max_seq_len: int = 0

    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute a single decode step (delegates to step_batch with single token).

        Args:
            token_id: Current token ID to process.
            position: Position in the sequence.
            context_len: Total context length.
            buffers: Pre-allocated decode buffers.

        Returns:
            Hidden states [1, hidden_size].
        """
        return self.step_batch([token_id], position, context_len, buffers)

    def step_batch(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
        buffers: DecodeBuffers,  # noqa: ARG002
    ) -> GPUArray:
        """Execute batch decode step without CUDA Graph.

        Args:
            token_ids: List of token IDs to decode.
            start_position: Starting position in sequence.
            context_len: Total context length after this batch.
            buffers: Pre-allocated decode buffers (unused, kept for API compat).

        Returns:
            Hidden states [seq_len, hidden_size].
        """
        # Use legacy batch decode which handles bfloat16 RoPE correctly
        return self.model._decode_step_fixed_cache_batch(token_ids, start_position, context_len)

    def init_graph(self, max_seq_len: int = 512) -> None:
        """Initialize CUDA Graph for batch decode.

        Args:
            max_seq_len: Maximum sequence length for RoPE pre-computation.
        """
        import gc

        from pygpukit._native_loader import get_native_module
        from pygpukit.core import default_stream

        CudaGraph = getattr(get_native_module(), "CudaGraph")  # noqa: B009
        from pygpukit.core.factory import from_numpy
        from pygpukit.llm.buffers import DecodeBuffers
        from pygpukit.llm.layers import precompute_freqs_cis

        model = self.model
        batch_size = self._batch_size
        dtype = str(model.embed_tokens.dtype)
        use_qk_norm = model.spec is not None and model.spec.use_qk_norm
        lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
        vocab_size = lm_head.shape[0]

        # Allocate batch decode buffers
        self._batch_decode_buffers = DecodeBuffers.allocate(
            model.config,
            dtype=dtype,
            use_qk_norm=use_qk_norm,
            vocab_size=vocab_size,
            max_batch_size=batch_size,
        )
        buffers = self._batch_decode_buffers

        # Pre-compute RoPE tables
        if model.config.use_rope and not hasattr(model, "_rope_cos_gpu"):
            from pygpukit.ops.basic import cast_f32_to_bf16, cast_f32_to_f16

            cos_np, sin_np = precompute_freqs_cis(
                model.config.head_dim, max_seq_len, model.config.rope_theta
            )
            if dtype == "float16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                model._rope_cos_gpu = cast_f32_to_f16(cos_f32)
                model._rope_sin_gpu = cast_f32_to_f16(sin_f32)
            elif dtype == "bfloat16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                model._rope_cos_gpu = cast_f32_to_bf16(cos_f32)
                model._rope_sin_gpu = cast_f32_to_bf16(sin_f32)
            else:
                model._rope_cos_gpu = from_numpy(cos_np.astype(np.float32))
                model._rope_sin_gpu = from_numpy(sin_np.astype(np.float32))

        # Cache transposed lm_head
        if not hasattr(model, "_lm_head_t_cache"):
            lm_head_np = lm_head.to_numpy()
            model._lm_head_t_cache = from_numpy(lm_head_np.T.copy())

        # Numpy buffers
        self._batch_token_ids_np = np.zeros(batch_size, dtype=np.int32)
        self._batch_start_pos_np = np.array([0], dtype=np.int32)
        self._batch_ctx_len_np = np.array([0], dtype=np.int32)
        self._batch_graph_max_seq_len = max_seq_len

        # Warmup
        print(f"  [Batch CUDA Graph] Warming up with batch_size={batch_size}...")
        self._batch_ctx_len_np[0] = max_seq_len
        buffers.context_len_buf._get_native().copy_from_numpy(self._batch_ctx_len_np)
        for _ in range(3):
            self._step_batch_for_graph(list(range(batch_size)), 0, batch_size, buffers)
        default_stream().synchronize()

        # Capture graph
        print("  [Batch CUDA Graph] Capturing graph...")
        self._batch_decode_graph = CudaGraph()

        # Write initial values
        self._batch_token_ids_np[:] = list(range(batch_size))
        buffers.token_ids_batch_buf._get_native().copy_from_numpy(self._batch_token_ids_np)
        self._batch_start_pos_np[0] = 0
        buffers.start_position_batch_buf._get_native().copy_from_numpy(self._batch_start_pos_np)
        self._batch_ctx_len_np[0] = max_seq_len
        buffers.context_len_buf._get_native().copy_from_numpy(self._batch_ctx_len_np)

        gc.disable()
        try:
            self._batch_decode_graph.begin_capture()

            # Batch embedding lookup
            embedding_lookup_batch(
                model.embed_tokens,
                buffers.hidden_batch,
                buffers.token_ids_batch_buf,
                batch_size,
            )

            # Fixed size views for graph
            hidden = buffers.hidden_batch.slice_rows(batch_size)
            residual_buf = buffers.residual_batch.slice_rows(batch_size)
            norm_out_buf = buffers.norm_out_batch.slice_rows(batch_size)
            mlp_out_buf = buffers.mlp_down_batch.slice_rows(batch_size)

            rope_cos_gpu = getattr(model, "_rope_cos_gpu", None)
            rope_sin_gpu = getattr(model, "_rope_sin_gpu", None)
            start_pos_buf = buffers.start_position_batch_buf

            for block in model.blocks:
                rmsnorm(hidden, block.attn_norm.weight, block.attn_norm.eps, out=norm_out_buf)
                copy_to(hidden, residual_buf)

                attn_out = block.attn.forward_fixed_cache_batch_zero_alloc(
                    norm_out_buf, 0, max_seq_len, buffers, rope_cos_gpu, rope_sin_gpu, start_pos_buf
                )

                add_inplace(residual_buf, attn_out)
                copy_to(residual_buf, hidden)

                rmsnorm(hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=norm_out_buf)
                copy_to(hidden, residual_buf)

                model._mlp_forward_batch_zero_alloc(block.mlp, norm_out_buf, buffers, mlp_out_buf)

                add_inplace(residual_buf, mlp_out_buf)
                copy_to(residual_buf, hidden)

            rmsnorm(hidden, model.final_norm.weight, model.final_norm.eps, out=norm_out_buf)
            matmul(norm_out_buf, model._lm_head_t_cache, out=buffers.logits_batch)

            self._batch_decode_graph.end_capture()
        finally:
            gc.enable()

        self._batch_decode_graph_ready = True
        print(f"  [Batch CUDA Graph] Captured {self._batch_decode_graph.num_nodes} nodes")

        # CRITICAL: Reset KV cache IN-PLACE after warmup/capture to remove pollution
        # Warmup and capture wrote garbage values at position 0
        # Must reset in-place to preserve pointers captured by graph
        print("  [Batch CUDA Graph] Resetting KV cache in-place after capture...")
        for block in model.blocks:
            if block.attn._k_cache is not None:
                block.attn._k_cache._get_native().fill_zeros()
                block.attn._v_cache._get_native().fill_zeros()
        default_stream().synchronize()

    def _step_batch_for_graph(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Batch decode step for graph warmup (matches graph capture path)."""
        model = self.model
        seq_len = len(token_ids)

        # Copy token IDs to GPU buffer
        self._batch_token_ids_np[:seq_len] = token_ids
        buffers.token_ids_batch_buf._get_native().copy_from_numpy(self._batch_token_ids_np)

        self._batch_start_pos_np[0] = start_position
        buffers.start_position_batch_buf._get_native().copy_from_numpy(self._batch_start_pos_np)

        embedding_lookup_batch(
            model.embed_tokens,
            buffers.hidden_batch,
            buffers.token_ids_batch_buf,
            seq_len,
        )

        hidden = buffers.hidden_batch.slice_rows(seq_len)
        residual_buf = buffers.residual_batch.slice_rows(seq_len)
        norm_out_buf = buffers.norm_out_batch.slice_rows(seq_len)
        mlp_out_buf = buffers.mlp_down_batch.slice_rows(seq_len)

        rope_cos_gpu = getattr(model, "_rope_cos_gpu", None)
        rope_sin_gpu = getattr(model, "_rope_sin_gpu", None)
        start_pos_buf = buffers.start_position_batch_buf

        for block in model.blocks:
            rmsnorm(hidden, block.attn_norm.weight, block.attn_norm.eps, out=norm_out_buf)
            copy_to(hidden, residual_buf)

            attn_out = block.attn.forward_fixed_cache_batch_zero_alloc(
                norm_out_buf,
                start_position,
                context_len,
                buffers,
                rope_cos_gpu,
                rope_sin_gpu,
                start_pos_buf,
            )

            add_inplace(residual_buf, attn_out)
            copy_to(residual_buf, hidden)

            rmsnorm(hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=norm_out_buf)
            copy_to(hidden, residual_buf)

            model._mlp_forward_batch_zero_alloc(block.mlp, norm_out_buf, buffers, mlp_out_buf)

            add_inplace(residual_buf, mlp_out_buf)
            copy_to(residual_buf, hidden)

        rmsnorm(hidden, model.final_norm.weight, model.final_norm.eps, out=norm_out_buf)
        return norm_out_buf

    def has_graph(self) -> bool:
        """Check if CUDA Graph is ready."""
        return self._batch_decode_graph_ready

    def step_graph(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
    ) -> GPUArray:
        """Execute batch decode using CUDA Graph replay.

        Args:
            token_ids: List of token IDs (must match captured batch_size).
            start_position: Starting position in sequence.
            context_len: Total context length.

        Returns:
            Logits buffer [batch_size, vocab_size].
        """
        assert self._batch_decode_graph_ready, "Call init_graph() first"
        assert self._batch_decode_buffers is not None

        buffers = self._batch_decode_buffers
        seq_len = len(token_ids)

        if seq_len != self._batch_size:
            raise ValueError(
                f"token_ids length ({seq_len}) must match batch_size ({self._batch_size})"
            )

        # Update GPU buffers
        self._batch_token_ids_np[:seq_len] = token_ids
        buffers.token_ids_batch_buf._get_native().copy_from_numpy(self._batch_token_ids_np)
        self._batch_start_pos_np[0] = start_position
        buffers.start_position_batch_buf._get_native().copy_from_numpy(self._batch_start_pos_np)
        self._batch_ctx_len_np[0] = context_len
        buffers.context_len_buf._get_native().copy_from_numpy(self._batch_ctx_len_np)

        # Synchronize before replay
        from pygpukit.core.backend import get_backend

        get_backend().synchronize()

        # Replay graph
        self._batch_decode_graph.replay()
        self._batch_decode_graph.synchronize()

        assert buffers.logits_batch is not None, "logits_batch buffer not allocated"
        return buffers.logits_batch

    @property
    def buffers(self) -> DecodeBuffers | None:
        """Get the batch decode buffers."""
        return self._batch_decode_buffers

    @property
    def batch_size(self) -> int:
        """Get the configured batch size."""
        return self._batch_size
