"""Diffusion model implementations.

Provides model implementations for:
- VAE: Variational Autoencoder for image encoding/decoding
- DiT: Diffusion Transformer (used in SD3, Flux, PixArt)
"""

from __future__ import annotations

from pygpukit.diffusion.models.dit import DiT, FluxTransformer, SD3Transformer
from pygpukit.diffusion.models.vae import VAE

__all__ = [
    "VAE",
    "DiT",
    "SD3Transformer",
    "FluxTransformer",
]
