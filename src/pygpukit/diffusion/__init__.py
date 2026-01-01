"""PyGPUkit Diffusion module for image generation.

This module provides support for diffusion models including:
- Stable Diffusion (SD 1.5, SDXL)
- Stable Diffusion 3 (MMDiT)
- Flux.1
- PixArt-Sigma

Architecture:
    Text Encoder (CLIP/T5) -> Text Embeddings
    Noise + Timestep -> UNet/DiT -> Denoised Latents -> VAE Decoder -> Image
"""

from __future__ import annotations

from pygpukit.diffusion.config import (
    DiTSpec,
    FluxSpec,
    PixArtSpec,
    SD3Spec,
    VAESpec,
)
from pygpukit.diffusion.pipeline import Text2ImagePipeline

__all__ = [
    # Configurations
    "DiTSpec",
    "FluxSpec",
    "PixArtSpec",
    "SD3Spec",
    "VAESpec",
    # Pipeline
    "Text2ImagePipeline",
]
