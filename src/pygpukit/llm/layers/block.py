"""Transformer block implementation for PyGPUkit LLM.

Provides:
- TransformerBlock: Attention + MLP with residual connections
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.ops.basic import add

from .attention import Attention
from .mlp import MLP
from .moe import MoELayer
from .norm import Norm


class TransformerBlock:
    """Unified transformer block.

    Structure:
        Norm -> Attention -> Residual
        Norm -> MLP/MoE -> Residual
    """

    def __init__(
        self,
        attn_norm: Norm,
        attn: Attention,
        mlp_norm: Norm,
        mlp: MLP | MoELayer,
    ):
        self.attn_norm = attn_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp  # Can be MLP or MoELayer

    def __call__(
        self,
        x: GPUArray,
        position_ids: list[int] | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, tuple | None]:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        attn_out, present_kv = self.attn(x, position_ids, past_kv, use_cache)
        x = add(residual, attn_out)

        # MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = add(residual, x)

        return x, present_kv


__all__ = [
    "TransformerBlock",
]
