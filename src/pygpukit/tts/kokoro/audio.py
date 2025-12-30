"""Audio utilities for Kokoro TTS.

Provides:
- WAV file export
- Audio format conversion
- Playback utilities
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray

if TYPE_CHECKING:
    from pygpukit.ops.audio import AudioBuffer


def to_wav(
    audio: AudioBuffer | GPUArray | np.ndarray,
    path: str | Path,
    sample_rate: int = 24000,
    normalize: bool = True,
) -> None:
    """Export audio to WAV file.

    Writes a standard 16-bit PCM WAV file.

    Args:
        audio: Audio data (AudioBuffer, GPUArray, or numpy array)
        path: Output file path
        sample_rate: Sample rate in Hz (default: 24000 for Kokoro)
        normalize: Whether to normalize audio to prevent clipping

    Example:
        >>> from pygpukit.tts.kokoro import KokoroModel
        >>> model = KokoroModel.from_pretrained("hexgrad/Kokoro-82M")
        >>> result = model.synthesize("Hello!")
        >>> to_wav(result.audio, "output.wav")
    """
    # Convert to numpy array
    if hasattr(audio, "data") and hasattr(audio, "sample_rate"):
        # AudioBuffer
        samples = audio.data.to_numpy()  # type: ignore
        sample_rate = audio.sample_rate  # type: ignore
    elif isinstance(audio, GPUArray):
        samples = audio.to_numpy()
    elif isinstance(audio, np.ndarray):
        samples = audio
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    # Ensure float32
    samples = samples.astype(np.float32)

    # Flatten if needed
    if samples.ndim > 1:
        samples = samples.flatten()

    # Normalize to prevent clipping
    if normalize:
        max_val = np.abs(samples).max()
        if max_val > 0:
            samples = samples / max_val * 0.95

    # Convert to 16-bit PCM
    samples_int16: np.ndarray = (samples * 32767).astype(np.int16)

    # Write WAV file
    path = Path(path)
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        # File size (will be filled later)
        file_size_pos = f.tell()
        f.write(struct.pack("<I", 0))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Chunk size
        f.write(struct.pack("<H", 1))  # Audio format (1 = PCM)
        f.write(struct.pack("<H", 1))  # Number of channels
        f.write(struct.pack("<I", sample_rate))  # Sample rate
        f.write(struct.pack("<I", sample_rate * 2))  # Byte rate
        f.write(struct.pack("<H", 2))  # Block align
        f.write(struct.pack("<H", 16))  # Bits per sample

        # data chunk
        f.write(b"data")
        data_size = len(samples_int16) * 2
        f.write(struct.pack("<I", data_size))
        f.write(samples_int16.tobytes())

        # Update file size
        file_size = f.tell() - 8
        f.seek(file_size_pos)
        f.write(struct.pack("<I", file_size))


def from_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio from WAV file.

    Args:
        path: Path to WAV file

    Returns:
        Tuple of (samples as float32, sample_rate)

    Example:
        >>> samples, sr = from_wav("input.wav")
        >>> print(f"Duration: {len(samples) / sr:.2f}s")
    """
    path = Path(path)

    with open(path, "rb") as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a valid WAV file (missing RIFF header)")

        f.read(4)  # File size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a valid WAV file (missing WAVE header)")

        # Read chunks
        sample_rate = 44100
        num_channels = 1
        bits_per_sample = 16
        audio_data = None

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break

            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                audio_format = struct.unpack("<H", f.read(2))[0]
                num_channels = struct.unpack("<H", f.read(2))[0]
                sample_rate = struct.unpack("<I", f.read(4))[0]
                f.read(4)  # Byte rate
                f.read(2)  # Block align
                bits_per_sample = struct.unpack("<H", f.read(2))[0]

                # Skip any extra format bytes
                extra = chunk_size - 16
                if extra > 0:
                    f.read(extra)

                if audio_format != 1:
                    raise ValueError(f"Unsupported audio format: {audio_format}")

            elif chunk_id == b"data":
                audio_data = f.read(chunk_size)

            else:
                # Skip unknown chunks
                f.read(chunk_size)

    if audio_data is None:
        raise ValueError("No audio data found in WAV file")

    # Convert to numpy array
    if bits_per_sample == 16:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767
    elif bits_per_sample == 8:
        samples = (np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif bits_per_sample == 32:
        samples = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483647
    else:
        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

    # Convert stereo to mono if needed
    if num_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    elif num_channels > 2:
        samples = samples.reshape(-1, num_channels).mean(axis=1)

    return samples, sample_rate


def resample_audio(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.

    Simple linear interpolation resampling.
    For high-quality resampling, use scipy or librosa.

    Args:
        samples: Audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return samples

    # Calculate new length
    duration = len(samples) / orig_sr
    new_length = int(duration * target_sr)

    # Linear interpolation
    old_indices = np.linspace(0, len(samples) - 1, new_length)
    new_samples = np.interp(old_indices, np.arange(len(samples)), samples)

    return new_samples.astype(np.float32)


def concatenate_audio(
    audio_list: list[np.ndarray | GPUArray],
    gap_samples: int = 0,
) -> np.ndarray:
    """Concatenate multiple audio segments.

    Args:
        audio_list: List of audio arrays
        gap_samples: Number of silence samples between segments

    Returns:
        Concatenated audio
    """
    segments = []
    for audio in audio_list:
        if isinstance(audio, GPUArray):
            audio = audio.to_numpy()
        segments.append(audio.flatten())

        if gap_samples > 0:
            segments.append(np.zeros(gap_samples, dtype=np.float32))

    # Remove trailing gap
    if gap_samples > 0 and segments:
        segments = segments[:-1]

    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


__all__ = [
    "to_wav",
    "from_wav",
    "resample_audio",
    "concatenate_audio",
]
