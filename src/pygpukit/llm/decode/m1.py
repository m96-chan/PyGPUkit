"""Single-token (M=1) decode strategy.

This module provides the DecodeM1 strategy for single-token decoding,
with optional CUDA Graph acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.llm.decode.base import DecodeStrategy
from pygpukit.ops.basic import (
    add_inplace,
    copy_to,
    embedding_lookup,
    embedding_lookup_ptr,
    matmul,
    rmsnorm,
)

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers


class DecodeM1(DecodeStrategy):
    """Single-token decode strategy with optional CUDA Graph support.

    This strategy handles M=1 decoding (generating one token at a time).
    It supports both standard decode and CUDA Graph accelerated decode.

    CUDA Graph mode pre-captures the decode computation and replays it
    with updated buffer values, eliminating kernel launch overhead.
    """

    def __init__(self) -> None:
        """Initialize DecodeM1 strategy."""
        super().__init__()
        self._decode_graph = None
        self._decode_graph_ready = False
        self._decode_buffers: DecodeBuffers | None = None

        # Numpy buffers for H2D transfers (avoid allocation during decode)
        self._pos_np: np.ndarray | None = None
        self._tok_np: np.ndarray | None = None
        self._ctx_np: np.ndarray | None = None
        self._graph_max_seq_len: int = 0

    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute a single decode step without CUDA Graph.

        Args:
            token_id: Current token ID to process.
            position: Position in the sequence.
            context_len: Total context length (for KV cache attention).
            buffers: Pre-allocated decode buffers.

        Returns:
            Hidden states [1, hidden_size].
        """
        model = self.model

        # Get token embedding directly to hidden
        embedding_lookup(model.embed_tokens, buffers.hidden, token_id)

        # Transformer blocks
        for block in model.blocks:
            # Pre-norm: hidden -> norm_out
            rmsnorm(
                buffers.hidden,
                block.attn_norm.weight,
                block.attn_norm.eps,
                out=buffers.norm_out,
            )

            # Save residual
            copy_to(buffers.hidden, buffers.residual)

            # Attention with fixed cache (writes to buffers.hidden)
            model._attention_forward_zero_alloc(
                block.attn, buffers.norm_out, position, context_len, buffers
            )

            # Add residual: hidden = residual + hidden
            add_inplace(buffers.hidden, buffers.residual)

            # MLP pre-norm
            copy_to(buffers.hidden, buffers.residual)
            rmsnorm(
                buffers.hidden,
                block.mlp_norm.weight,
                block.mlp_norm.eps,
                out=buffers.norm_out,
            )

            # MLP forward (SwiGLU)
            model._mlp_forward_zero_alloc(block.mlp, buffers.norm_out, buffers)

            # Add residual
            add_inplace(buffers.hidden, buffers.residual)

        # Final norm
        rmsnorm(
            buffers.hidden,
            model.final_norm.weight,
            model.final_norm.eps,
            out=buffers.norm_out,
        )
        copy_to(buffers.norm_out, buffers.hidden)

        return buffers.hidden

    def init_graph(self, max_seq_len: int = 512) -> None:
        """Initialize CUDA Graph for single-token decode.

        Pre-allocates buffers, pre-computes RoPE, and captures the decode
        graph for replay.

        IMPORTANT: Call this AFTER prefill and KV cache initialization.

        Args:
            max_seq_len: Maximum sequence length for KV cache.
        """
        import gc

        from pygpukit._pygpukit_native import CudaGraph
        from pygpukit.core.factory import from_numpy
        from pygpukit.llm.buffers import DecodeBuffers
        from pygpukit.llm.layers import precompute_freqs_cis

        model = self.model
        dtype = str(model.embed_tokens.dtype)
        use_qk_norm = model.spec is not None and model.spec.use_qk_norm
        lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
        vocab_size = lm_head.shape[0]

        # Allocate decode buffers with CUDA Graph support
        self._decode_buffers = DecodeBuffers.allocate(
            model.config, dtype=dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
        )

        # Pre-compute RoPE tables on GPU if not already done
        if model.config.use_rope and not hasattr(model, "_rope_cos_gpu"):
            cos_np, sin_np = precompute_freqs_cis(
                model.config.head_dim, max_seq_len, model.config.rope_theta
            )
            np_dtype = np.float16 if dtype == "float16" else np.float32
            model._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
            model._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))

        # Cache transposed lm_head for graph (if not already done)
        if not hasattr(model, "_lm_head_t_cache"):
            lm_head_np = lm_head.to_numpy()
            model._lm_head_t_cache = from_numpy(lm_head_np.T.copy())

        # Numpy buffers for CPU-side updates (reusable, no allocation)
        self._pos_np = np.array([0], dtype=np.int32)
        self._tok_np = np.array([0], dtype=np.int32)
        self._ctx_np = np.array([0], dtype=np.int32)

        # Store max_seq_len for graph replay
        self._graph_max_seq_len = max_seq_len

        # Warmup before capture
        buffers = self._decode_buffers
        self._ctx_np[0] = 1
        buffers.context_len_buf._get_native().copy_from_numpy(self._ctx_np)
        for _ in range(3):
            self.step(0, 0, 1, buffers)

        # Capture the decode graph
        self._decode_graph = CudaGraph()

        # Write initial values to GPU buffers
        self._pos_np[0] = 0
        buffers.position_buf._get_native().copy_from_numpy(self._pos_np)
        self._tok_np[0] = 0
        buffers.token_id_buf._get_native().copy_from_numpy(self._tok_np)
        self._ctx_np[0] = max_seq_len  # Capture with max for shared memory
        buffers.context_len_buf._get_native().copy_from_numpy(self._ctx_np)

        gc.disable()
        try:
            self._decode_graph.begin_capture()

            # Embedding lookup from token_id_buf
            embedding_lookup_ptr(model.embed_tokens, buffers.hidden, buffers.token_id_buf)

            # Transformer blocks
            for block in model.blocks:
                rmsnorm(
                    buffers.hidden,
                    block.attn_norm.weight,
                    block.attn_norm.eps,
                    out=buffers.norm_out,
                )
                copy_to(buffers.hidden, buffers.residual)
                model._attention_forward_zero_alloc(
                    block.attn,
                    buffers.norm_out,
                    0,
                    max_seq_len,
                    buffers,
                    use_position_ptr=True,
                    use_context_len_ptr=True,
                    max_kv_len=max_seq_len,
                )
                add_inplace(buffers.hidden, buffers.residual)
                copy_to(buffers.hidden, buffers.residual)
                rmsnorm(
                    buffers.hidden,
                    block.mlp_norm.weight,
                    block.mlp_norm.eps,
                    out=buffers.norm_out,
                )
                model._mlp_forward_zero_alloc(block.mlp, buffers.norm_out, buffers)
                add_inplace(buffers.hidden, buffers.residual)

            # Final norm
            rmsnorm(
                buffers.hidden,
                model.final_norm.weight,
                model.final_norm.eps,
                out=buffers.norm_out,
            )
            copy_to(buffers.norm_out, buffers.hidden)

            # LM head projection to logits
            matmul(buffers.hidden, model._lm_head_t_cache, out=buffers.logits)

            self._decode_graph.end_capture()
        finally:
            gc.enable()

        self._decode_graph_ready = True
        print(f"  [CUDA Graph] Captured {self._decode_graph.num_nodes} nodes for decode")

    def has_graph(self) -> bool:
        """Check if CUDA Graph is ready."""
        return self._decode_graph_ready

    def step_graph(
        self,
        token_id: int,
        position: int,
        context_len: int,
    ) -> GPUArray:
        """Execute decode step using CUDA Graph replay.

        Updates GPU buffers and replays the captured graph.

        Args:
            token_id: Input token ID.
            position: Position in sequence.
            context_len: Total context length (for KV cache attention).

        Returns:
            Logits buffer [1, vocab_size].
        """
        assert self._decode_graph_ready, "Call init_graph() first"
        assert self._decode_buffers is not None

        buffers = self._decode_buffers

        # Update GPU buffers (outside graph)
        try:
            self._tok_np[0] = token_id
            buffers.token_id_buf._get_native().copy_from_numpy(self._tok_np)
            self._pos_np[0] = position
            buffers.position_buf._get_native().copy_from_numpy(self._pos_np)
            self._ctx_np[0] = context_len
            buffers.context_len_buf._get_native().copy_from_numpy(self._ctx_np)
        except RuntimeError as e:
            raise RuntimeError(
                f"H2D copy failed: tok={token_id}, pos={position}, ctx={context_len}. Error: {e}"
            ) from e

        # Device synchronize to ensure H2D copies are visible to the graph
        from pygpukit.core.backend import get_backend

        get_backend().synchronize()

        # Replay graph
        self._decode_graph.replay()

        # Synchronize graph's stream to ensure replay completes
        try:
            self._decode_graph.synchronize()
        except RuntimeError as e:
            raise RuntimeError(
                f"Graph replay sync failed: tok={token_id}, pos={position}, "
                f"ctx={context_len}. Error: {e}"
            ) from e

        return buffers.logits

    @property
    def buffers(self) -> DecodeBuffers | None:
        """Get the decode buffers (for external access)."""
        return self._decode_buffers
