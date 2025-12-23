"""Whisper model for speech recognition.

Provides a unified interface for Whisper transcription with support for:
- Single-file transcription
- Streaming/chunked inference for long audio
- Multiple output formats (text, segments with timestamps)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ...core import GPUArray, from_numpy
from ...ops.audio import AudioBuffer
from ..preprocessing import (
    WHISPER_CHUNK_LENGTH,
    WHISPER_HOP_LENGTH,
    WHISPER_SAMPLE_RATE,
    normalize_mel,
    pad_or_trim,
)
from .config import WhisperConfig
from .decoder import WhisperDecoder, create_decoder
from .encoder import WhisperEncoder, create_encoder
from .loader import load_whisper_model


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing information."""

    text: str
    start: float  # seconds
    end: float  # seconds
    tokens: list[int] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None


class WhisperTokenizer:
    """Simple tokenizer wrapper for Whisper models.

    Uses the HuggingFace tokenizers library if available,
    otherwise provides a basic fallback.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        """Load tokenizer from model path."""
        import os

        try:
            from tokenizers import Tokenizer

            tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
            if os.path.exists(tokenizer_path):
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
        except ImportError:
            pass

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self._tokenizer is not None:
            return self._tokenizer.encode(text).ids
        raise RuntimeError("Tokenizer not available")

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        raise RuntimeError("Tokenizer not available")


class WhisperModel:
    """Whisper model for speech recognition.

    Example:
        >>> model = WhisperModel.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")
        >>> result = model.transcribe("audio.wav", language="ja")
        >>> print(result.text)

        # Streaming mode for long audio
        >>> for segment in model.transcribe_streaming(audio_array, language="ja"):
        ...     print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
    """

    def __init__(
        self,
        config: WhisperConfig,
        encoder: WhisperEncoder,
        decoder: WhisperDecoder,
        tokenizer: WhisperTokenizer | None = None,
    ):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_id: str,
        cache_dir: str | None = None,
    ) -> WhisperModel:
        """Load a pretrained Whisper model.

        Args:
            model_path_or_id: Local path or HuggingFace model ID
            cache_dir: Optional cache directory for downloads

        Returns:
            Initialized WhisperModel

        Example:
            >>> model = WhisperModel.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")
        """
        import os

        # Load config and weights
        config, weights = load_whisper_model(model_path_or_id, cache_dir)

        # Create encoder and decoder
        encoder = create_encoder(config, weights)
        decoder = create_decoder(config, weights)

        # Load tokenizer
        tokenizer = None
        if os.path.exists(model_path_or_id):
            tokenizer = WhisperTokenizer(model_path_or_id)
        else:
            # Try to get cached path
            try:
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(
                    repo_id=model_path_or_id,
                    cache_dir=cache_dir,
                    allow_patterns=["tokenizer.*"],
                )
                tokenizer = WhisperTokenizer(model_path)
            except Exception:
                pass

        return cls(config, encoder, decoder, tokenizer)

    def transcribe(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
        language: str | None = None,
        max_length: int = 448,
        temperature: float = 0.0,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform (numpy array) or path to audio file
            sample_rate: Sample rate of input audio (required if not 16kHz)
            language: Optional language code (e.g., "ja", "en")
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            TranscriptionResult with text and optional segments
        """
        # Load audio if path
        if isinstance(audio, str):
            audio = self._load_audio(audio)

        # Resample to 16kHz if needed
        if sample_rate is not None and sample_rate != WHISPER_SAMPLE_RATE:
            audio_gpu = from_numpy(audio.astype(np.float32))
            audio_buf = AudioBuffer(data=audio_gpu, sample_rate=sample_rate, channels=1)
            audio_buf = audio_buf.resample(WHISPER_SAMPLE_RATE)
            audio = audio_buf.data.to_numpy()

        # Preprocess to mel spectrogram
        mel = self._preprocess_audio(audio)

        # Encode audio
        encoder_output = self.encoder(mel)

        # Decode to tokens
        tokens = self.decoder.generate(
            encoder_output,
            max_length=max_length,
            temperature=temperature,
            top_k=None if temperature == 0.0 else 50,
        )

        # Decode tokens to text
        text = self._decode_tokens(tokens)

        return TranscriptionResult(
            text=text,
            segments=[
                TranscriptionSegment(
                    text=text,
                    start=0.0,
                    end=len(audio) / WHISPER_SAMPLE_RATE,
                    tokens=tokens,
                )
            ],
            language=language,
        )

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        language: str | None = None,
        chunk_length: float = WHISPER_CHUNK_LENGTH,
        overlap: float = 0.0,
        max_length: int = 448,
        temperature: float = 0.0,
        **kwargs,
    ) -> Iterator[TranscriptionSegment]:
        """Transcribe long audio in chunks, yielding segments as they're processed.

        Args:
            audio: Audio waveform at 16kHz
            language: Optional language code
            chunk_length: Length of each chunk in seconds (default: 30s)
            overlap: Overlap between chunks in seconds
            max_length: Maximum tokens per chunk
            temperature: Sampling temperature

        Yields:
            TranscriptionSegment for each processed chunk
        """
        samples_per_chunk = int(chunk_length * WHISPER_SAMPLE_RATE)
        overlap_samples = int(overlap * WHISPER_SAMPLE_RATE)
        stride = samples_per_chunk - overlap_samples

        # Process audio in chunks
        start_sample = 0
        while start_sample < len(audio):
            end_sample = min(start_sample + samples_per_chunk, len(audio))
            chunk = audio[start_sample:end_sample]

            # Process chunk
            mel = self._preprocess_audio(chunk)
            encoder_output = self.encoder(mel)

            tokens = self.decoder.generate(
                encoder_output,
                max_length=max_length,
                temperature=temperature,
                top_k=None if temperature == 0.0 else 50,
            )

            text = self._decode_tokens(tokens)

            # Calculate timing
            start_time = start_sample / WHISPER_SAMPLE_RATE
            end_time = end_sample / WHISPER_SAMPLE_RATE

            yield TranscriptionSegment(
                text=text,
                start=start_time,
                end=end_time,
                tokens=tokens,
            )

            start_sample += stride

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio file and resample to 16kHz mono.

        Args:
            path: Path to audio file

        Returns:
            Audio waveform at 16kHz
        """
        try:
            import soundfile as sf

            audio, sr = sf.read(path)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != WHISPER_SAMPLE_RATE:
                try:
                    import resampy

                    audio = resampy.resample(audio, sr, WHISPER_SAMPLE_RATE)
                except ImportError as err:
                    raise RuntimeError(
                        f"Audio sample rate is {sr}Hz but Whisper requires {WHISPER_SAMPLE_RATE}Hz. "
                        "Install resampy to enable automatic resampling: pip install resampy"
                    ) from err

            return audio.astype(np.float32)

        except ImportError as err:
            raise ImportError(
                "soundfile is required to load audio files. Install with: pip install soundfile"
            ) from err

    def _preprocess_audio(self, audio: np.ndarray) -> GPUArray:
        """Convert audio to mel spectrogram.

        Args:
            audio: Audio waveform at 16kHz

        Returns:
            Mel spectrogram [1, n_mels, n_frames]
        """
        # Pad or trim to 30 seconds
        audio_gpu = pad_or_trim(audio)
        audio_np = audio_gpu.to_numpy()

        # Compute mel spectrogram using numpy
        mel = self._compute_mel_spectrogram(audio_np)

        # Normalize (accepts numpy directly)
        mel = normalize_mel(mel)

        # Add batch dimension
        mel_np = mel.to_numpy()
        return from_numpy(mel_np.reshape(1, *mel_np.shape))

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram.

        Args:
            audio: Audio waveform at 16kHz

        Returns:
            Mel spectrogram [n_mels, n_frames]
        """
        from ..preprocessing import WHISPER_N_FFT

        # Use librosa if available, otherwise numpy fallback
        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=WHISPER_SAMPLE_RATE,
                n_fft=WHISPER_N_FFT,
                hop_length=WHISPER_HOP_LENGTH,
                n_mels=self.config.num_mel_bins,
                fmin=0,
                fmax=8000,
            )
            # Convert to log scale
            mel = np.log10(np.clip(mel, a_min=1e-10, a_max=None))

        except ImportError:
            # Numpy fallback (basic STFT + mel filterbank)
            mel = self._compute_mel_numpy(audio)

        return mel.astype(np.float32)

    def _compute_mel_numpy(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram using numpy (fallback).

        Args:
            audio: Audio waveform

        Returns:
            Mel spectrogram
        """
        from ..preprocessing import WHISPER_N_FFT

        n_fft = WHISPER_N_FFT
        hop_length = WHISPER_HOP_LENGTH
        n_mels = self.config.num_mel_bins

        # Pad audio
        audio = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")

        # STFT
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft] * window
            stft[:, i] = np.fft.rfft(frame)

        # Power spectrum
        power = np.abs(stft) ** 2

        # Mel filterbank
        mel_basis = self._create_mel_filterbank(n_mels, n_fft)
        mel = mel_basis @ power

        # Log scale
        mel = np.log10(np.clip(mel, a_min=1e-10, a_max=None))

        return mel

    def _create_mel_filterbank(self, n_mels: int, n_fft: int) -> np.ndarray:
        """Create mel filterbank matrix.

        Args:
            n_mels: Number of mel bands
            n_fft: FFT size

        Returns:
            Mel filterbank [n_mels, n_fft//2+1]
        """
        fmin = 0.0
        fmax = WHISPER_SAMPLE_RATE / 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Mel points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / WHISPER_SAMPLE_RATE).astype(int)

        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising edge
            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)

            # Falling edge
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        # Fallback: just return token IDs as string
        return f"<tokens: {tokens}>"


__all__ = [
    "WhisperModel",
    "WhisperTokenizer",
    "TranscriptionResult",
    "TranscriptionSegment",
]
