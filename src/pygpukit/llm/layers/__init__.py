"""Neural network layer implementations for PyGPUkit LLM.

Provides:
- LinearBF16: Dense layer with BF16 weights
- LinearFP8: Dense layer with FP8 weights (online dequantization)
- Norm: RMSNorm and LayerNorm
- Attention: Multi-head attention with RoPE, GQA, QK-Norm, KV cache
- MLP: Feed-forward network (GELU/SwiGLU)
- MoELayer: Mixture of Experts
- TransformerBlock: Attention + MLP with residual connections
- RoPE utilities: precompute_freqs_cis, apply_rotary_pos_emb_numpy
- Repack utilities: repack_weight, repack_linear, repack_norm
"""

from __future__ import annotations

# Attention
from .attention import Attention

# TransformerBlock
from .block import TransformerBlock

# Linear layers
from .linear import (
    Linear,
    LinearBF16,
    LinearFP8,
)

# MLP
from .mlp import MLP

# MoE
from .moe import MoELayer

# Normalization
from .norm import Norm

# RoPE utilities
from .rope import (
    apply_rotary_pos_emb_numpy,
    precompute_freqs_cis,
)

# Repack utilities
from .utils import (
    repack_linear,
    repack_norm,
    repack_weight,
)

__all__ = [
    # Linear layers
    "LinearBF16",
    "LinearFP8",
    "Linear",
    # Normalization
    "Norm",
    # RoPE
    "precompute_freqs_cis",
    "apply_rotary_pos_emb_numpy",
    # Attention
    "Attention",
    # MLP
    "MLP",
    # MoE
    "MoELayer",
    # TransformerBlock
    "TransformerBlock",
    # Repack utilities
    "repack_weight",
    "repack_linear",
    "repack_norm",
]
