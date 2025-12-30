"""PyGPUkit Text-to-Speech module.

Provides GPU-accelerated text-to-speech synthesis using neural network models.

Supported Models:
    - Kokoro-82M: StyleTTS2-based model with 82M parameters

Example:
    >>> from pygpukit.tts import KokoroModel
    >>> model = KokoroModel.from_pretrained("hexgrad/Kokoro-82M")
    >>> audio = model.synthesize("Hello, this is PyGPUkit TTS!")
    >>> audio.to_wav("output.wav")
"""

from pygpukit.tts.kokoro import (
    KokoroConfig,
    KokoroModel,
    KokoroTokenizer,
    SynthesisResult,
    concatenate_audio,
    from_wav,
    list_available_voices,
    load_kokoro_weights,
    load_voice_embedding,
    resample_audio,
    to_wav,
)

__all__ = [
    # Model
    "KokoroModel",
    "SynthesisResult",
    # Config
    "KokoroConfig",
    # Tokenizer
    "KokoroTokenizer",
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
