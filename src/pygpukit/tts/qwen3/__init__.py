"""Qwen3-TTS model implementation for PyGPUkit.

Provides:
- Qwen3TTSModel: Main TTS model with voice design/clone/custom modes
- SpeakerEncoder: ECAPA-TDNN speaker embedding extractor
- SpeechTokenizer: Audio codec for encode/decode
"""

from __future__ import annotations

from pygpukit.tts.qwen3.loader import (
    build_qwen3_tts_config,
    build_talker_model,
    load_config,
    load_qwen3_tts,
    load_safetensors_weights,
)
from pygpukit.tts.qwen3.model import (
    CodePredictor,
    GenerationOutput,
    Qwen3TTSConfig,
    Qwen3TTSModel,
    TalkerModel,
    VoiceClonePromptItem,
)
from pygpukit.tts.qwen3.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig
from pygpukit.tts.qwen3.speech_tokenizer import (
    SpeechTokenizer,
    SpeechTokenizerConfig,
)

__all__ = [
    # Main model
    "Qwen3TTSModel",
    "Qwen3TTSConfig",
    "TalkerModel",
    "CodePredictor",
    "VoiceClonePromptItem",
    "GenerationOutput",
    # Speaker encoder
    "SpeakerEncoder",
    "SpeakerEncoderConfig",
    # Speech tokenizer
    "SpeechTokenizer",
    "SpeechTokenizerConfig",
    # Loader
    "load_qwen3_tts",
    "load_safetensors_weights",
    "load_config",
    "build_qwen3_tts_config",
    "build_talker_model",
]
