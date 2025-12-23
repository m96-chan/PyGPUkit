"""ASR (Automatic Speech Recognition) module for PyGPUkit.

This module provides GPU-accelerated speech recognition models,
starting with Whisper architecture support.

Example:
    >>> from pygpukit.asr import WhisperModel
    >>> model = WhisperModel.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")
    >>> result = model.transcribe("audio.wav", language="ja")
    >>> print(result.text)
"""

from .preprocessing import (
    WHISPER_CHUNK_LENGTH,
    WHISPER_HOP_LENGTH,
    WHISPER_N_FFT,
    WHISPER_N_MELS,
    WHISPER_SAMPLE_RATE,
    normalize_mel,
    pad_or_trim,
    preprocess_audio,
)
from .whisper import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperModel,
)

__all__ = [
    # High-level API
    "WhisperModel",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Preprocessing
    "preprocess_audio",
    "pad_or_trim",
    "normalize_mel",
    # Constants
    "WHISPER_SAMPLE_RATE",
    "WHISPER_N_FFT",
    "WHISPER_HOP_LENGTH",
    "WHISPER_N_MELS",
    "WHISPER_CHUNK_LENGTH",
]
