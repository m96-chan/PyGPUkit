"""Diffusion-specific operations.

Provides operations required for diffusion models that are not in the main ops module:
- GroupNorm: Group normalization for NCHW tensors
- cross_attention: Non-causal cross-attention
- conv2d: 2D convolution
- timestep_embedding: Sinusoidal timestep embedding
- adaln: Adaptive layer normalization
"""

from __future__ import annotations

from pygpukit.diffusion.ops.adaln import adaln, adaln_zero
from pygpukit.diffusion.ops.conv2d import conv2d, conv2d_transpose
from pygpukit.diffusion.ops.cross_attention import cross_attention
from pygpukit.diffusion.ops.group_norm import group_norm
from pygpukit.diffusion.ops.timestep_embed import sinusoidal_timestep_embedding

__all__ = [
    "group_norm",
    "cross_attention",
    "conv2d",
    "conv2d_transpose",
    "sinusoidal_timestep_embedding",
    "adaln",
    "adaln_zero",
]
