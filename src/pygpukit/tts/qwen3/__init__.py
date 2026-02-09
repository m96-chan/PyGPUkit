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
from pygpukit.tts.qwen3.speaker_encoder import (
    SpeakerEncoder,
    SpeakerEncoderConfig,
    compute_mel_spectrogram,
    load_speaker_encoder,
)
from pygpukit.tts.qwen3.speech_decoder import (
    SpeechDecoder,
    SpeechDecoderConfig,
    load_speech_decoder,
)
from pygpukit.tts.qwen3.speech_encoder import (
    SpeechEncoder,
    SpeechEncoderConfig,
)
from pygpukit.tts.qwen3.speech_encoder import (
    load_speech_encoder as load_speech_encoder_icl,
)
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
    "load_speaker_encoder",
    "compute_mel_spectrogram",
    # Speech tokenizer
    "SpeechTokenizer",
    "SpeechTokenizerConfig",
    # Speech encoder (ICL mode)
    "SpeechEncoder",
    "SpeechEncoderConfig",
    "load_speech_encoder_icl",
    # Speech decoder (vocoder)
    "SpeechDecoder",
    "SpeechDecoderConfig",
    "load_speech_decoder",
    # Loader
    "load_qwen3_tts",
    "load_safetensors_weights",
    "load_config",
    "build_qwen3_tts_config",
    "build_talker_model",
]
