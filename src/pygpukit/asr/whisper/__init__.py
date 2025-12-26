"""Whisper model implementation for PyGPUkit.

Supports OpenAI Whisper and derived models:
- openai/whisper-large-v3
- kotoba-tech/kotoba-whisper-v2.0 (Japanese ASR)
- distil-whisper variants

Example:
    >>> from pygpukit.asr.whisper import WhisperModel
    >>> model = WhisperModel.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")
    >>> result = model.transcribe("audio.wav", language="ja")
    >>> print(result.text)
"""

from .config import WHISPER_CONFIGS, WhisperConfig
from .decoder import WhisperDecoder, WhisperDecoderLayer, create_decoder
from .encoder import WhisperEncoder, WhisperEncoderLayer, create_encoder
from .loader import WhisperWeights, download_model, load_safetensors, load_whisper_model
from .model import TranscriptionResult, TranscriptionSegment, WhisperModel, WhisperTokenizer

__all__ = [
    # High-level API
    "WhisperModel",
    "WhisperTokenizer",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Config
    "WhisperConfig",
    "WHISPER_CONFIGS",
    # Loader
    "WhisperWeights",
    "load_whisper_model",
    "load_safetensors",
    "download_model",
    # Encoder
    "WhisperEncoder",
    "WhisperEncoderLayer",
    "create_encoder",
    # Decoder
    "WhisperDecoder",
    "WhisperDecoderLayer",
    "create_decoder",
]
