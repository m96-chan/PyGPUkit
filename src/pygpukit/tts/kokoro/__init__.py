"""Kokoro-82M TTS model implementation.

Kokoro is a StyleTTS2-based text-to-speech model with 82M parameters.
It achieves high-quality speech synthesis with a compact architecture.

Example:
    >>> from pygpukit.tts.kokoro import KokoroModel
    >>> model = KokoroModel.from_pretrained("hexgrad/Kokoro-82M")
    >>> audio = model.synthesize("Hello, world!")
    >>> audio.to_wav("output.wav")
"""

from pygpukit.tts.kokoro.audio import concatenate_audio, from_wav, resample_audio, to_wav
from pygpukit.tts.kokoro.config import KokoroConfig
from pygpukit.tts.kokoro.loader import (
    list_available_voices,
    load_kokoro_weights,
    load_voice_embedding,
)
from pygpukit.tts.kokoro.model import KokoroModel, SynthesisResult
from pygpukit.tts.kokoro.text import KokoroTokenizer, TokenizerOutput

__all__ = [
    # Model
    "KokoroModel",
    "SynthesisResult",
    # Config
    "KokoroConfig",
    # Tokenizer
    "KokoroTokenizer",
    "TokenizerOutput",
    # Loader
    "load_kokoro_weights",
    "load_voice_embedding",
    "list_available_voices",
    # Audio
    "to_wav",
    "from_wav",
    "resample_audio",
    "concatenate_audio",
]
