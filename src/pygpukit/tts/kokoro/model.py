"""Kokoro TTS Model.

High-level API for text-to-speech synthesis using Kokoro-82M.

Example:
    >>> from pygpukit.tts.kokoro import KokoroModel
    >>> model = KokoroModel.from_pretrained("hexgrad/Kokoro-82M")
    >>> audio = model.synthesize("Hello, world!", voice="af_heart")
    >>> audio.to_wav("output.wav")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy, zeros
from pygpukit.ops.audio import AudioBuffer
from pygpukit.tts.kokoro.config import KokoroConfig
from pygpukit.tts.kokoro.loader import (
    list_available_voices,
    load_kokoro_weights,
    load_voice_embedding,
    print_weight_summary,
)
from pygpukit.tts.kokoro.text import KokoroTokenizer, normalize_text, split_sentences

if TYPE_CHECKING:
    from pygpukit.tts.kokoro.layers import Decoder, ISTFTNet, PLBERTEncoder, StyleEncoder


@dataclass
class SynthesisResult:
    """Result from TTS synthesis.

    Attributes:
        audio: Generated audio buffer (24kHz)
        text: Original input text
        phonemes: Phoneme representation
        duration_sec: Audio duration in seconds
    """

    audio: AudioBuffer
    text: str
    phonemes: str
    duration_sec: float

    def to_wav(self, path: str | Path) -> None:
        """Save audio to WAV file.

        Args:
            path: Output file path
        """
        from pygpukit.tts.kokoro.audio import to_wav

        to_wav(self.audio, str(path))

    def to_numpy(self) -> np.ndarray:
        """Get audio as numpy array.

        Returns:
            Float32 audio samples
        """
        return self.audio.to_numpy()


class KokoroModel:
    """Kokoro-82M Text-to-Speech Model.

    A StyleTTS2-based TTS model that generates natural-sounding speech
    from text input.

    Args:
        config: Model configuration
        weights: Model weights dictionary
        tokenizer: Text tokenizer
        voice_embeddings: Dictionary of voice embeddings

    Example:
        >>> model = KokoroModel.from_pretrained("hexgrad/Kokoro-82M")
        >>> result = model.synthesize("Hello, this is a test.")
        >>> result.to_wav("output.wav")
    """

    def __init__(
        self,
        config: KokoroConfig,
        weights: dict[str, GPUArray],
        tokenizer: KokoroTokenizer,
        voice_embeddings: dict[str, GPUArray] | None = None,
    ):
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self.voice_embeddings = voice_embeddings or {}

        # Build model components lazily
        self._plbert: PLBERTEncoder | None = None
        self._style_encoder: StyleEncoder | None = None
        self._decoder: Decoder | None = None
        self._vocoder: ISTFTNet | None = None

        # Default voice
        self._current_voice: str | None = None
        self._current_voice_embedding: GPUArray | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        voice: str = "af_heart",
        dtype: str = "bfloat16",
        load_all_voices: bool = False,
    ) -> KokoroModel:
        """Load model from pretrained checkpoint.

        Args:
            model_path: Path to model directory or HuggingFace repo ID
            voice: Default voice to use (e.g., "af_heart")
            dtype: Weight dtype ("bfloat16" or "float32")
            load_all_voices: Whether to load all voice embeddings

        Returns:
            KokoroModel instance
        """
        model_path = Path(model_path)

        # Load weights and config
        weights, config_dict = load_kokoro_weights(model_path, dtype=dtype)

        # Create config
        config = KokoroConfig.from_dict(config_dict)

        # Create tokenizer
        tokenizer = KokoroTokenizer.from_config(config, use_misaki=True)

        # Load voice embeddings
        voice_embeddings = {}

        if model_path.exists():
            available_voices = list_available_voices(model_path)

            if load_all_voices:
                for voice_name in available_voices:
                    voice_path = model_path / "voices" / f"{voice_name}.pt"
                    if voice_path.exists():
                        voice_embeddings[voice_name] = load_voice_embedding(voice_path)
            elif voice in available_voices:
                voice_path = model_path / "voices" / f"{voice}.pt"
                if voice_path.exists():
                    voice_embeddings[voice] = load_voice_embedding(voice_path)

        model = cls(
            config=config,
            weights=weights,
            tokenizer=tokenizer,
            voice_embeddings=voice_embeddings,
        )

        # Set default voice
        if voice in voice_embeddings:
            model.set_voice(voice)
        elif voice_embeddings:
            model.set_voice(list(voice_embeddings.keys())[0])

        return model

    def set_voice(self, voice: str) -> None:
        """Set the current voice for synthesis.

        Args:
            voice: Voice name (e.g., "af_heart", "bf_emma")
        """
        if voice not in self.voice_embeddings:
            available = list(self.voice_embeddings.keys())
            raise ValueError(f"Voice '{voice}' not loaded. Available: {available}")

        self._current_voice = voice
        self._current_voice_embedding = self.voice_embeddings[voice]

    def load_voice(self, voice_path: str | Path) -> str:
        """Load a voice embedding from file.

        Args:
            voice_path: Path to voice .pt file

        Returns:
            Voice name (file stem)
        """
        voice_path = Path(voice_path)
        voice_name = voice_path.stem
        self.voice_embeddings[voice_name] = load_voice_embedding(voice_path)
        return voice_name

    @property
    def available_voices(self) -> list[str]:
        """List of loaded voice names."""
        return list(self.voice_embeddings.keys())

    @property
    def current_voice(self) -> str | None:
        """Currently selected voice."""
        return self._current_voice

    def _build_components(self) -> None:
        """Build model components from weights (lazy initialization)."""
        if self._plbert is not None:
            return  # Already built

        from pygpukit.tts.kokoro.layers import build_plbert_from_weights

        # Build PLBERT encoder
        # Note: Actual weight prefix may vary depending on checkpoint format
        # This is a placeholder - actual implementation needs weight inspection
        try:
            self._plbert = build_plbert_from_weights(
                self.config, self.weights, prefix="bert"
            )
        except (KeyError, ValueError):
            # Weights might use different naming
            self._plbert = None

        # TODO: Build other components (style encoder, decoder, vocoder)
        # These require inspecting actual Kokoro weight structure

    def _forward_simple(
        self,
        tokens: list[int],
        voice_embedding: GPUArray | None = None,
    ) -> GPUArray:
        """Simple forward pass without full model components.

        This is a placeholder implementation that demonstrates the API.
        Full implementation requires matching Kokoro's exact weight structure.
        """
        # For now, generate placeholder audio
        # Actual implementation would:
        # 1. Embed tokens
        # 2. Run through PLBERT
        # 3. Apply style
        # 4. Decode to mel
        # 5. Vocode to audio

        # Placeholder: generate silence with some noise
        duration_per_token = 0.1  # 100ms per token
        total_duration = len(tokens) * duration_per_token
        num_samples = int(total_duration * self.config.sample_rate)

        # Generate placeholder audio (sine wave for testing)
        t = np.linspace(0, total_duration, num_samples, dtype=np.float32)
        frequency = 440.0  # A4 note
        audio = 0.1 * np.sin(2 * np.pi * frequency * t)

        return from_numpy(audio)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        normalize: bool = True,
    ) -> SynthesisResult:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Voice to use (None for current voice)
            speed: Speech speed multiplier (1.0 = normal)
            normalize: Whether to normalize input text

        Returns:
            SynthesisResult containing audio and metadata
        """
        # Set voice if specified
        if voice is not None and voice != self._current_voice:
            self.set_voice(voice)

        # Normalize text
        if normalize:
            text = normalize_text(text)

        # Tokenize
        tokenizer_output = self.tokenizer.encode(text)
        tokens = tokenizer_output.tokens
        phonemes = tokenizer_output.phonemes

        if not tokens:
            raise ValueError("No tokens generated from input text")

        # Forward pass
        audio_gpu = self._forward_simple(tokens, self._current_voice_embedding)

        # Create AudioBuffer
        audio_np = audio_gpu.to_numpy()
        audio_buffer = AudioBuffer(
            data=audio_gpu,
            sample_rate=self.config.sample_rate,
            channels=1,
        )

        duration_sec = len(audio_np) / self.config.sample_rate

        return SynthesisResult(
            audio=audio_buffer,
            text=text,
            phonemes=phonemes,
            duration_sec=duration_sec,
        )

    def __call__(
        self,
        text: str,
        voice: str | None = None,
        **kwargs,
    ) -> SynthesisResult:
        """Synthesize speech (callable interface).

        Args:
            text: Input text
            voice: Voice to use
            **kwargs: Additional arguments for synthesize()

        Returns:
            SynthesisResult
        """
        return self.synthesize(text, voice=voice, **kwargs)

    def generate_stream(
        self,
        text: str,
        voice: str | None = None,
        chunk_size: int = 4800,  # 200ms at 24kHz
    ):
        """Generate audio in chunks for streaming.

        Args:
            text: Input text
            voice: Voice to use
            chunk_size: Audio chunk size in samples

        Yields:
            AudioBuffer chunks
        """
        # Split into sentences for chunked generation
        sentences = split_sentences(text)

        for sentence in sentences:
            result = self.synthesize(sentence, voice=voice)
            audio_np = result.audio.to_numpy()

            # Yield in chunks
            for i in range(0, len(audio_np), chunk_size):
                chunk = audio_np[i : i + chunk_size]
                chunk_gpu = from_numpy(chunk)
                yield AudioBuffer(
                    data=chunk_gpu,
                    sample_rate=self.config.sample_rate,
                    channels=1,
                )

    def print_info(self) -> None:
        """Print model information."""
        print("=" * 60)
        print("Kokoro-82M TTS Model")
        print("=" * 60)
        print(f"Config: {self.config}")
        print(f"Voices: {self.available_voices}")
        print(f"Current voice: {self._current_voice}")
        print(f"Tokenizer: {self.tokenizer}")
        print("-" * 60)
        print_weight_summary(self.weights)

    def __repr__(self) -> str:
        return (
            f"KokoroModel(\n"
            f"  config={self.config!r},\n"
            f"  voices={self.available_voices},\n"
            f"  current_voice={self._current_voice!r}\n"
            f")"
        )


__all__ = [
    "KokoroModel",
    "SynthesisResult",
]
