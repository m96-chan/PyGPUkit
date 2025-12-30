"""Kokoro-82M TTS model configuration.

Kokoro is a StyleTTS2-based TTS model with 82M parameters.
Architecture: PLBERT -> Style Encoder -> Decoder -> ISTFTNet Vocoder
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KokoroConfig:
    """Configuration for Kokoro-82M TTS model.

    Attributes:
        dim_in: Input dimension for decoder (default: 64)
        hidden_dim: Hidden dimension (default: 512)
        style_dim: Style embedding dimension (default: 128)
        n_mels: Number of mel spectrogram bins (default: 80)
        n_layer: Number of decoder layers (default: 3)
        n_token: Vocabulary size (default: 178)
        max_dur: Maximum duration per token (default: 50)
        dropout: Dropout rate (default: 0.2)
        max_conv_dim: Maximum convolution dimension (default: 512)
        text_encoder_kernel_size: Kernel size for text encoder (default: 5)
        multispeaker: Whether model supports multiple speakers (default: True)
        sample_rate: Audio sample rate in Hz (default: 24000)
    """

    # Core dimensions
    dim_in: int = 64
    hidden_dim: int = 512
    style_dim: int = 128
    n_mels: int = 80
    n_layer: int = 3
    n_token: int = 178
    max_dur: int = 50
    dropout: float = 0.2
    max_conv_dim: int = 512
    text_encoder_kernel_size: int = 5
    multispeaker: bool = True

    # Audio
    sample_rate: int = 24000

    # ISTFTNet vocoder
    upsample_rates: tuple[int, ...] = (10, 6)
    upsample_kernel_sizes: tuple[int, ...] = (20, 12)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_initial_channel: int = 512
    gen_istft_n_fft: int = 20
    gen_istft_hop_size: int = 5

    # PLBERT text encoder
    plbert_hidden_size: int = 768
    plbert_num_attention_heads: int = 12
    plbert_intermediate_size: int = 2048
    plbert_max_position_embeddings: int = 512
    plbert_num_hidden_layers: int = 12
    plbert_dropout: float = 0.1

    # Phoneme vocabulary (loaded from config.json)
    vocab: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> KokoroConfig:
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary (from config.json)

        Returns:
            KokoroConfig instance
        """
        # Extract ISTFTNet config
        istftnet = config_dict.get("istftnet", {})

        # Extract PLBERT config
        plbert = config_dict.get("plbert", {})

        # Convert resblock_dilation_sizes to tuple of tuples
        resblock_dilations = istftnet.get(
            "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )
        resblock_dilations_tuple = tuple(tuple(d) for d in resblock_dilations)

        return cls(
            # Core
            dim_in=config_dict.get("dim_in", 64),
            hidden_dim=config_dict.get("hidden_dim", 512),
            style_dim=config_dict.get("style_dim", 128),
            n_mels=config_dict.get("n_mels", 80),
            n_layer=config_dict.get("n_layer", 3),
            n_token=config_dict.get("n_token", 178),
            max_dur=config_dict.get("max_dur", 50),
            dropout=config_dict.get("dropout", 0.2),
            max_conv_dim=config_dict.get("max_conv_dim", 512),
            text_encoder_kernel_size=config_dict.get("text_encoder_kernel_size", 5),
            multispeaker=config_dict.get("multispeaker", True),
            # ISTFTNet
            upsample_rates=tuple(istftnet.get("upsample_rates", [10, 6])),
            upsample_kernel_sizes=tuple(istftnet.get("upsample_kernel_sizes", [20, 12])),
            resblock_kernel_sizes=tuple(istftnet.get("resblock_kernel_sizes", [3, 7, 11])),
            resblock_dilation_sizes=resblock_dilations_tuple,
            upsample_initial_channel=istftnet.get("upsample_initial_channel", 512),
            gen_istft_n_fft=istftnet.get("gen_istft_n_fft", 20),
            gen_istft_hop_size=istftnet.get("gen_istft_hop_size", 5),
            # PLBERT
            plbert_hidden_size=plbert.get("hidden_size", 768),
            plbert_num_attention_heads=plbert.get("num_attention_heads", 12),
            plbert_intermediate_size=plbert.get("intermediate_size", 2048),
            plbert_max_position_embeddings=plbert.get("max_position_embeddings", 512),
            plbert_num_hidden_layers=plbert.get("num_hidden_layers", 12),
            plbert_dropout=plbert.get("dropout", 0.1),
            # Vocabulary
            vocab=config_dict.get("vocab", {}),
        )

    @classmethod
    def from_json(cls, json_path: str | Path) -> KokoroConfig:
        """Load config from JSON file.

        Args:
            json_path: Path to config.json

        Returns:
            KokoroConfig instance
        """
        with open(json_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> KokoroConfig:
        """Load config from pretrained model directory.

        Args:
            model_path: Path to model directory containing config.json

        Returns:
            KokoroConfig instance
        """
        model_path = Path(model_path)

        # Check for local config.json
        if model_path.is_dir():
            config_path = model_path / "config.json"
            if config_path.exists():
                return cls.from_json(config_path)

        # Try HuggingFace hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=str(model_path), filename="config.json")
            return cls.from_json(config_path)
        except ImportError as err:
            raise ImportError(
                "huggingface_hub is required to download from HuggingFace. "
                "Install with: pip install huggingface_hub"
            ) from err

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dim_in": self.dim_in,
            "hidden_dim": self.hidden_dim,
            "style_dim": self.style_dim,
            "n_mels": self.n_mels,
            "n_layer": self.n_layer,
            "n_token": self.n_token,
            "max_dur": self.max_dur,
            "dropout": self.dropout,
            "max_conv_dim": self.max_conv_dim,
            "text_encoder_kernel_size": self.text_encoder_kernel_size,
            "multispeaker": self.multispeaker,
            "sample_rate": self.sample_rate,
            "istftnet": {
                "upsample_rates": list(self.upsample_rates),
                "upsample_kernel_sizes": list(self.upsample_kernel_sizes),
                "resblock_kernel_sizes": list(self.resblock_kernel_sizes),
                "resblock_dilation_sizes": [list(d) for d in self.resblock_dilation_sizes],
                "upsample_initial_channel": self.upsample_initial_channel,
                "gen_istft_n_fft": self.gen_istft_n_fft,
                "gen_istft_hop_size": self.gen_istft_hop_size,
            },
            "plbert": {
                "hidden_size": self.plbert_hidden_size,
                "num_attention_heads": self.plbert_num_attention_heads,
                "intermediate_size": self.plbert_intermediate_size,
                "max_position_embeddings": self.plbert_max_position_embeddings,
                "num_hidden_layers": self.plbert_num_hidden_layers,
                "dropout": self.plbert_dropout,
            },
            "vocab": self.vocab,
        }

    @property
    def plbert_head_dim(self) -> int:
        """PLBERT attention head dimension."""
        return self.plbert_hidden_size // self.plbert_num_attention_heads

    @property
    def hop_length(self) -> int:
        """Audio hop length (product of upsample rates * istft_hop_size)."""
        hop = self.gen_istft_hop_size
        for rate in self.upsample_rates:
            hop *= rate
        return hop

    def __repr__(self) -> str:
        return (
            f"KokoroConfig(\n"
            f"  dim_in={self.dim_in}, hidden_dim={self.hidden_dim},\n"
            f"  style_dim={self.style_dim}, n_mels={self.n_mels},\n"
            f"  n_layer={self.n_layer}, n_token={self.n_token},\n"
            f"  sample_rate={self.sample_rate},\n"
            f"  plbert_layers={self.plbert_num_hidden_layers},\n"
            f"  upsample_rates={self.upsample_rates}\n"
            f")"
        )


__all__ = ["KokoroConfig"]
