"""Model specifications for diffusion models.

This module defines the architecture specifications for various diffusion models:
- DiT (Diffusion Transformer)
- MMDiT (Multi-Modal DiT, used in SD3)
- Flux
- PixArt
- VAE (Variational Autoencoder)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DiTSpec:
    """Specification for Diffusion Transformer models."""

    name: str

    # Core dimensions
    hidden_size: int
    num_layers: int
    num_heads: int

    # Conditioning
    conditioning_type: Literal["adaln", "adaln_zero", "cross_attn"]
    text_encoder_dim: int

    # Position encoding
    pos_embed_type: Literal["sinusoidal", "rope_2d", "learned"]
    patch_size: int = 2  # Latent patch size

    # Input/output
    in_channels: int = 16  # VAE latent channels
    out_channels: int = 16

    # MMDiT specific
    is_mmdit: bool = False  # Multi-modal DiT (SD3)

    # MLP
    mlp_ratio: float = 4.0

    # Head dimension (auto-computed if not specified)
    head_dim: int | None = None

    def get_head_dim(self) -> int:
        """Get head dimension."""
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class SD3Spec(DiTSpec):
    """Specification for Stable Diffusion 3 (MMDiT)."""

    # SD3 uses joint attention blocks
    joint_attention_dim: int = 4096  # Combined text dim

    # Dual text encoders
    clip_l_dim: int = 768
    clip_g_dim: int = 1280
    t5_dim: int = 4096


@dataclass(frozen=True)
class FluxSpec(DiTSpec):
    """Specification for Flux.1 models."""

    # Flux uses double transformer blocks
    num_double_blocks: int = 19
    num_single_blocks: int = 38

    # Guidance
    guidance_embed: bool = True

    # Resolution
    max_resolution: tuple[int, int] = (1024, 1024)


@dataclass(frozen=True)
class PixArtSpec(DiTSpec):
    """Specification for PixArt models."""

    # PixArt-specific
    cross_attention_dim: int = 4096  # T5-XXL


@dataclass(frozen=True)
class VAESpec:
    """Specification for VAE encoder/decoder."""

    name: str

    # Dimensions
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4

    # Scaling factor (latent -> pixel space)
    scaling_factor: float = 0.18215  # SD 1.5

    # Architecture
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2

    # Normalization
    norm_num_groups: int = 32
    norm_eps: float = 1e-6


# Pre-defined model specifications
SD3_MEDIUM_SPEC = SD3Spec(
    name="sd3_medium",
    hidden_size=1536,
    num_layers=24,
    num_heads=24,
    conditioning_type="adaln_zero",
    text_encoder_dim=4096,
    pos_embed_type="rope_2d",
    in_channels=16,
    out_channels=16,
    is_mmdit=True,
)

SD3_LARGE_SPEC = SD3Spec(
    name="sd3_large",
    hidden_size=2048,
    num_layers=38,
    num_heads=32,
    conditioning_type="adaln_zero",
    text_encoder_dim=4096,
    pos_embed_type="rope_2d",
    in_channels=16,
    out_channels=16,
    is_mmdit=True,
)

FLUX_SCHNELL_SPEC = FluxSpec(
    name="flux_schnell",
    hidden_size=3072,
    num_layers=19,  # Double blocks
    num_heads=24,
    conditioning_type="adaln",
    text_encoder_dim=4096,
    pos_embed_type="rope_2d",
    in_channels=16,
    out_channels=16,
    num_double_blocks=19,
    num_single_blocks=38,
    guidance_embed=False,  # Schnell uses CFG-distillation
)

FLUX_DEV_SPEC = FluxSpec(
    name="flux_dev",
    hidden_size=3072,
    num_layers=19,
    num_heads=24,
    conditioning_type="adaln",
    text_encoder_dim=4096,
    pos_embed_type="rope_2d",
    in_channels=16,
    out_channels=16,
    num_double_blocks=19,
    num_single_blocks=38,
    guidance_embed=True,
)

PIXART_SIGMA_SPEC = PixArtSpec(
    name="pixart_sigma",
    hidden_size=1152,
    num_layers=28,
    num_heads=16,
    conditioning_type="cross_attn",
    text_encoder_dim=4096,
    pos_embed_type="sinusoidal",
    in_channels=4,
    out_channels=4,
    cross_attention_dim=4096,
)

# VAE specifications
SDXL_VAE_SPEC = VAESpec(
    name="sdxl_vae",
    latent_channels=4,
    scaling_factor=0.13025,
    block_out_channels=(128, 256, 512, 512),
)

SD3_VAE_SPEC = VAESpec(
    name="sd3_vae",
    latent_channels=16,  # SD3 uses 16-channel VAE
    scaling_factor=1.5305,  # SD3 scaling
    block_out_channels=(128, 256, 512, 512),
)

FLUX_VAE_SPEC = VAESpec(
    name="flux_vae",
    latent_channels=16,
    scaling_factor=0.3611,
    block_out_channels=(128, 256, 512, 512),
)


__all__ = [
    "DiTSpec",
    "SD3Spec",
    "FluxSpec",
    "PixArtSpec",
    "VAESpec",
    # Pre-defined specs
    "SD3_MEDIUM_SPEC",
    "SD3_LARGE_SPEC",
    "FLUX_SCHNELL_SPEC",
    "FLUX_DEV_SPEC",
    "PIXART_SIGMA_SPEC",
    "SDXL_VAE_SPEC",
    "SD3_VAE_SPEC",
    "FLUX_VAE_SPEC",
]
