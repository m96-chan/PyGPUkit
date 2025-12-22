"""Single-token (M=1) decode strategy (non-graph version).

This module provides the DecodeM1 strategy for single-token decoding
without CUDA Graph acceleration.

For CUDA Graph accelerated version, use DecodeM1Graph from m1_graph.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygpukit.llm.decode.base import DecodeStrategy
from pygpukit.ops.basic import (
    add_inplace,
    copy_to,
    embedding_lookup,
    matmul,
    rmsnorm,
)

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers


class DecodeM1(DecodeStrategy):
    """Single-token decode strategy (non-graph version).

    This strategy handles M=1 decoding (generating one token at a time)
    without CUDA Graph acceleration.

    For CUDA Graph support, use DecodeM1Graph instead.
    """

    def __init__(self) -> None:
        """Initialize DecodeM1 strategy."""
        super().__init__()

    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute a single decode step.

        Args:
            token_id: Current token ID to process.
            position: Position in the sequence.
            context_len: Total context length (for KV cache attention).
            buffers: Pre-allocated decode buffers.

        Returns:
            Logits [1, vocab_size].
        """
        model = self.model

        # Cache transposed lm_head for matmul (shared with DecodeM1Graph)
        if not hasattr(model, "_lm_head_t_cache"):
            from pygpukit.core.factory import from_numpy

            lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
            lm_head_np = lm_head.to_numpy()
            model._lm_head_t_cache = from_numpy(lm_head_np.T.copy())

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

            # Attention with fixed cache
            attn_out = block.attn.forward_fixed_cache(
                buffers.norm_out, position, context_len, out=buffers.attn_out
            )
            copy_to(attn_out, buffers.hidden)

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

        # LM head: hidden -> logits
        assert buffers.logits is not None, "logits buffer not allocated"
        matmul(buffers.hidden, model._lm_head_t_cache, out=buffers.logits)

        return buffers.logits
