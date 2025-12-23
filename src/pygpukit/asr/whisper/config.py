"""Whisper model configuration.

Supports various Whisper variants:
- OpenAI Whisper (tiny, base, small, medium, large, large-v2, large-v3)
- Distilled Whisper (kotoba-whisper, distil-whisper)
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhisperConfig:
    """Configuration for Whisper models.

    Attributes:
        d_model: Hidden dimension (512-1280 depending on model size)
        encoder_layers: Number of encoder transformer layers
        decoder_layers: Number of decoder transformer layers
        encoder_attention_heads: Number of attention heads in encoder
        decoder_attention_heads: Number of attention heads in decoder
        encoder_ffn_dim: Feed-forward dimension in encoder
        decoder_ffn_dim: Feed-forward dimension in decoder
        vocab_size: Vocabulary size (51865 for multilingual, 51864 for English-only)
        num_mel_bins: Number of mel spectrogram bins (80 or 128)
        max_source_positions: Maximum encoder sequence length (1500 for 30s audio)
        max_target_positions: Maximum decoder sequence length (448 tokens)
        activation_function: Activation function (gelu)
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        activation_dropout: Activation dropout rate
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        decoder_start_token_id: Decoder start token ID
    """

    # Model architecture
    d_model: int = 1280
    encoder_layers: int = 32
    decoder_layers: int = 32
    encoder_attention_heads: int = 20
    decoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_ffn_dim: int = 5120

    # Vocabulary
    vocab_size: int = 51866

    # Audio
    num_mel_bins: int = 128  # 80 for older Whisper, 128 for large-v3

    # Sequence lengths
    max_source_positions: int = 1500  # 30s audio / 160 hop_length / 2
    max_target_positions: int = 448

    # Activation and regularization
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0

    # Special tokens
    bos_token_id: int = 50257
    eos_token_id: int = 50257
    pad_token_id: int = 50256
    decoder_start_token_id: int = 50258

    # Suppress tokens
    begin_suppress_tokens: list = field(default_factory=lambda: [220, 50257])

    # Inference
    use_cache: bool = True
    torch_dtype: str = "bfloat16"

    # Model name
    model_name_or_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "WhisperConfig":
        """Create config from dictionary."""
        # Map HuggingFace config keys to our keys
        key_mapping = {
            "_name_or_path": "model_name_or_path",
        }

        mapped_dict = {}
        for key, value in config_dict.items():
            mapped_key = key_mapping.get(key, key)
            if hasattr(cls, "__dataclass_fields__") and mapped_key in cls.__dataclass_fields__:
                mapped_dict[mapped_key] = value

        return cls(**mapped_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "WhisperConfig":
        """Load config from JSON file."""
        with open(json_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "WhisperConfig":
        """Load config from pretrained model directory or HuggingFace hub."""
        import os

        # Check for local config.json
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                return cls.from_json(config_path)

        # Try HuggingFace hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            return cls.from_json(config_path)
        except ImportError as err:
            raise ImportError(
                "huggingface_hub is required to download from HuggingFace. "
                "Install with: pip install huggingface_hub"
            ) from err

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "d_model": self.d_model,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "encoder_attention_heads": self.encoder_attention_heads,
            "decoder_attention_heads": self.decoder_attention_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "vocab_size": self.vocab_size,
            "num_mel_bins": self.num_mel_bins,
            "max_source_positions": self.max_source_positions,
            "max_target_positions": self.max_target_positions,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.encoder_attention_heads

    @property
    def is_distilled(self) -> bool:
        """Check if this is a distilled model (fewer decoder layers)."""
        return self.decoder_layers < self.encoder_layers

    def __repr__(self) -> str:
        return (
            f"WhisperConfig(\n"
            f"  d_model={self.d_model},\n"
            f"  encoder_layers={self.encoder_layers},\n"
            f"  decoder_layers={self.decoder_layers},\n"
            f"  attention_heads={self.encoder_attention_heads},\n"
            f"  ffn_dim={self.encoder_ffn_dim},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  num_mel_bins={self.num_mel_bins},\n"
            f"  distilled={self.is_distilled}\n"
            f")"
        )


# Predefined configurations for common Whisper variants
WHISPER_CONFIGS = {
    "tiny": WhisperConfig(
        d_model=384,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        num_mel_bins=80,
    ),
    "base": WhisperConfig(
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_mel_bins=80,
    ),
    "small": WhisperConfig(
        d_model=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        num_mel_bins=80,
    ),
    "medium": WhisperConfig(
        d_model=1024,
        encoder_layers=24,
        decoder_layers=24,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        num_mel_bins=80,
    ),
    "large": WhisperConfig(
        d_model=1280,
        encoder_layers=32,
        decoder_layers=32,
        encoder_attention_heads=20,
        decoder_attention_heads=20,
        encoder_ffn_dim=5120,
        decoder_ffn_dim=5120,
        num_mel_bins=80,
    ),
    "large-v3": WhisperConfig(
        d_model=1280,
        encoder_layers=32,
        decoder_layers=32,
        encoder_attention_heads=20,
        decoder_attention_heads=20,
        encoder_ffn_dim=5120,
        decoder_ffn_dim=5120,
        num_mel_bins=128,  # large-v3 uses 128 mel bins
    ),
    "kotoba-v2": WhisperConfig(
        d_model=1280,
        encoder_layers=32,
        decoder_layers=2,  # Distilled!
        encoder_attention_heads=20,
        decoder_attention_heads=20,
        encoder_ffn_dim=5120,
        decoder_ffn_dim=5120,
        num_mel_bins=128,
    ),
}


__all__ = [
    "WhisperConfig",
    "WHISPER_CONFIGS",
]
