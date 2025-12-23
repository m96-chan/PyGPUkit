"""Whisper model implementation for PyGPUkit.

Supports OpenAI Whisper and derived models:
- openai/whisper-large-v3
- kotoba-tech/kotoba-whisper-v2.0 (Japanese ASR)
- distil-whisper variants
"""

from .config import WHISPER_CONFIGS, WhisperConfig
from .loader import WhisperWeights, download_model, load_safetensors, load_whisper_model

__all__ = [
    "WhisperConfig",
    "WHISPER_CONFIGS",
    "WhisperWeights",
    "load_whisper_model",
    "load_safetensors",
    "download_model",
]
