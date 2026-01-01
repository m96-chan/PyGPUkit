"""DiT (Diffusion Transformer) models.

Provides:
- DiT: Base Diffusion Transformer
- SD3Transformer: Stable Diffusion 3 MMDiT
- FluxTransformer: Flux.1 model
- PixArtTransformer: PixArt-Sigma implementation
- Attention modules (self_attention, cross_attention)
- FFN modules (geglu_ffn, standard_ffn)
- AdaLN modules
- Embedding modules
"""

# Re-export base classes from dit_base.py
from pygpukit.diffusion.models.dit_base import DiT, FluxTransformer, SD3Transformer

from .adaln import (
    adaln_modulate_mlp,
    adaln_modulation,
    compute_adaln_conditioning,
    layer_norm,
    rms_norm,
)
from .attention import cross_attention, self_attention
from .embeddings import (
    caption_projection,
    patch_embed,
    sinusoidal_embedding,
    timestep_embedding,
    unpatchify,
)
from .ffn import geglu_ffn, gelu, standard_ffn
from .model import PixArtTransformer

__all__ = [
    # Base models
    "DiT",
    "SD3Transformer",
    "FluxTransformer",
    # PixArt model
    "PixArtTransformer",
    # Attention
    "self_attention",
    "cross_attention",
    # FFN
    "geglu_ffn",
    "standard_ffn",
    "gelu",
    # AdaLN
    "rms_norm",
    "layer_norm",
    "adaln_modulation",
    "adaln_modulate_mlp",
    "compute_adaln_conditioning",
    # Embeddings
    "sinusoidal_embedding",
    "patch_embed",
    "timestep_embedding",
    "caption_projection",
    "unpatchify",
]
