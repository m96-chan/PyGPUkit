"""Audio buffer classes for GPU audio processing.

This module provides:
- AudioBuffer: GPU audio buffer with metadata
- AudioRingBuffer: GPU-side ring buffer for streaming
- AudioStream: High-level streaming audio processor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pygpukit.core import GPUArray
from pygpukit.core import from_numpy as core_from_numpy
from pygpukit.core.dtypes import float32, int16


def _get_native():
    """Get the native module."""
    try:
        from pygpukit._native_loader import get_native_module

        return get_native_module()
    except ImportError:
        from pygpukit import _pygpukit_native

        return _pygpukit_native


@dataclass
class AudioBuffer:
    """GPU audio buffer with metadata.

    Attributes:
        data: GPUArray containing audio samples (float32)
        sample_rate: Sample rate in Hz
        channels: Number of channels (1=mono, 2=stereo)
    """

    data: GPUArray
    sample_rate: int
    channels: int

    def to_mono(self) -> AudioBuffer:
        """Convert stereo audio to mono.

        Returns:
            New AudioBuffer with mono audio (channels=1)

        Raises:
            ValueError: If already mono
        """
        if self.channels == 1:
            return self

        if self.channels != 2:
            raise ValueError(f"to_mono only supports stereo (2 channels), got {self.channels}")

        native = _get_native()
        mono_data = native.audio_stereo_to_mono(self.data._get_native())

        return AudioBuffer(
            data=GPUArray._wrap_native(mono_data),
            sample_rate=self.sample_rate,
            channels=1,
        )

    def resample(self, target_rate: int) -> AudioBuffer:
        """Resample audio to target sample rate.

        Currently supports:
        - 48000 -> 16000 (3:1 decimation for Whisper)

        Args:
            target_rate: Target sample rate in Hz

        Returns:
            New AudioBuffer with resampled audio

        Raises:
            ValueError: If sample rate conversion is not supported
        """
        if self.sample_rate == target_rate:
            return self

        native = _get_native()
        resampled = native.audio_resample(self.data._get_native(), self.sample_rate, target_rate)

        return AudioBuffer(
            data=GPUArray._wrap_native(resampled),
            sample_rate=target_rate,
            channels=self.channels,
        )

    def normalize(self, mode: str = "peak", target_db: float = -20.0) -> AudioBuffer:
        """Normalize audio level.

        Args:
            mode: Normalization mode ("peak" or "rms")
            target_db: Target level in dB (only used for RMS mode)

        Returns:
            Self (in-place normalization)

        Raises:
            ValueError: If mode is not "peak" or "rms"
        """
        native = _get_native()

        if mode == "peak":
            native.audio_normalize_peak(self.data._get_native())
        elif mode == "rms":
            native.audio_normalize_rms(self.data._get_native(), target_db)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}. Use 'peak' or 'rms'.")

        return self

    def to_numpy(self) -> np.ndarray:
        """Convert audio data to NumPy array.

        Returns:
            NumPy array of float32 samples
        """
        return self.data.to_numpy()

    def __repr__(self) -> str:
        return (
            f"AudioBuffer(samples={self.data.shape[0]}, "
            f"sample_rate={self.sample_rate}, channels={self.channels})"
        )


def from_pcm(
    samples: np.ndarray | GPUArray,
    sample_rate: int,
    channels: int = 1,
) -> AudioBuffer:
    """Create AudioBuffer from PCM samples.

    Args:
        samples: PCM samples as int16 or float32 array
        sample_rate: Sample rate in Hz (e.g., 48000, 16000)
        channels: Number of channels (1=mono, 2=stereo)

    Returns:
        AudioBuffer with audio data on GPU

    Example:
        >>> pcm = np.array([0, 16384, -16384], dtype=np.int16)
        >>> buf = from_pcm(pcm, sample_rate=48000)
    """
    native = _get_native()

    # Convert to GPUArray if needed
    if isinstance(samples, np.ndarray):
        gpu_samples = core_from_numpy(samples)
    else:
        gpu_samples = samples

    # Convert int16 PCM to float32
    if gpu_samples.dtype == int16:
        float_data = native.audio_pcm_to_float32(gpu_samples._get_native())
        gpu_data = GPUArray._wrap_native(float_data)
    elif gpu_samples.dtype == float32:
        # Already float32, just use as-is
        gpu_data = gpu_samples
    else:
        raise ValueError(f"Unsupported dtype: {gpu_samples.dtype}. Use int16 or float32.")

    return AudioBuffer(
        data=gpu_data,
        sample_rate=sample_rate,
        channels=channels,
    )


class AudioRingBuffer:
    """GPU-side ring buffer for streaming audio.

    Provides efficient circular buffer operations for real-time audio processing.

    Args:
        capacity: Buffer capacity in samples
        sample_rate: Sample rate in Hz (for metadata)

    Example:
        >>> ring = AudioRingBuffer(capacity=48000, sample_rate=16000)  # 3 sec buffer
        >>> ring.write(chunk1)
        >>> ring.write(chunk2)
        >>> window = ring.read(16000)  # Read 1 second
    """

    def __init__(self, capacity: int, sample_rate: int = 16000):
        from pygpukit.core import zeros

        self._buffer = zeros((capacity,), dtype="float32")
        self._capacity = capacity
        self._sample_rate = sample_rate
        self._write_pos = 0
        self._samples_written = 0

    @property
    def capacity(self) -> int:
        """Buffer capacity in samples."""
        return self._capacity

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def samples_available(self) -> int:
        """Number of samples available for reading."""
        return min(self._samples_written, self._capacity)

    @property
    def duration_available(self) -> float:
        """Duration of available audio in seconds."""
        return self.samples_available / self._sample_rate

    def write(self, samples: np.ndarray | GPUArray) -> int:
        """Write samples to the ring buffer.

        Args:
            samples: Audio samples to write (float32)

        Returns:
            Number of samples written
        """
        native = _get_native()

        # Convert to GPUArray if needed
        if isinstance(samples, np.ndarray):
            gpu_samples = core_from_numpy(samples.astype(np.float32))
        else:
            gpu_samples = samples

        num_samples = gpu_samples.shape[0]

        # Write to ring buffer
        native.audio_ring_buffer_write(
            gpu_samples._get_native(),
            self._buffer._get_native(),
            self._write_pos,
        )

        # Update write position
        self._write_pos = (self._write_pos + num_samples) % self._capacity
        self._samples_written += num_samples

        return num_samples

    def read(self, num_samples: int, offset: int = 0) -> GPUArray:
        """Read samples from the ring buffer.

        Args:
            num_samples: Number of samples to read
            offset: Offset from current read position (0 = most recent)

        Returns:
            GPUArray of audio samples
        """
        native = _get_native()

        # Calculate read position (read from oldest available)
        if self._samples_written <= self._capacity:
            read_pos = offset
        else:
            read_pos = (self._write_pos + offset) % self._capacity

        result = native.audio_ring_buffer_read(
            self._buffer._get_native(),
            read_pos,
            num_samples,
        )

        return GPUArray._wrap_native(result)

    def clear(self) -> None:
        """Clear the buffer."""
        from pygpukit.core import zeros

        self._buffer = zeros((self._capacity,), dtype="float32")
        self._write_pos = 0
        self._samples_written = 0

    def __repr__(self) -> str:
        return (
            f"AudioRingBuffer(capacity={self._capacity}, "
            f"sample_rate={self._sample_rate}, "
            f"available={self.samples_available})"
        )


class AudioStream:
    """High-level streaming audio processor.

    Provides chunked processing with windowing for smooth transitions.
    Suitable for real-time ASR preprocessing.

    Args:
        chunk_size: Processing chunk size in samples (default: 480 = 30ms @ 16kHz)
        hop_size: Hop size between chunks (default: chunk_size // 2 for 50% overlap)
        sample_rate: Sample rate in Hz
        buffer_duration: Ring buffer duration in seconds

    Example:
        >>> stream = AudioStream(chunk_size=480, sample_rate=16000)
        >>> for pcm_chunk in audio_source:
        ...     stream.push(pcm_chunk)
        ...     if stream.has_chunk():
        ...         chunk = stream.pop_chunk()
        ...         # Process chunk for ASR
    """

    def __init__(
        self,
        chunk_size: int = 480,
        hop_size: int | None = None,
        sample_rate: int = 16000,
        buffer_duration: float = 30.0,
    ):
        self._chunk_size = chunk_size
        self._hop_size = hop_size if hop_size is not None else chunk_size // 2
        self._sample_rate = sample_rate

        # Ring buffer for incoming audio
        buffer_samples = int(buffer_duration * sample_rate)
        self._ring_buffer = AudioRingBuffer(buffer_samples, sample_rate)

        # Track chunk position
        self._chunks_processed = 0

    @property
    def chunk_size(self) -> int:
        """Chunk size in samples."""
        return self._chunk_size

    @property
    def hop_size(self) -> int:
        """Hop size in samples."""
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._sample_rate

    def push(self, samples: np.ndarray | GPUArray) -> int:
        """Push audio samples to the stream.

        Args:
            samples: Audio samples (float32)

        Returns:
            Number of samples pushed
        """
        return self._ring_buffer.write(samples)

    def has_chunk(self) -> bool:
        """Check if a full chunk is available."""
        required = self._chunks_processed * self._hop_size + self._chunk_size
        return self._ring_buffer._samples_written >= required

    def pop_chunk(self, apply_window: bool = True) -> GPUArray:
        """Pop the next chunk from the stream.

        Args:
            apply_window: Whether to apply Hann window (default True)

        Returns:
            GPUArray containing the chunk

        Raises:
            RuntimeError: If no chunk is available
        """
        if not self.has_chunk():
            raise RuntimeError("No chunk available. Call has_chunk() first.")

        native = _get_native()

        # Calculate read offset
        read_offset = self._chunks_processed * self._hop_size

        # Read chunk from ring buffer
        chunk = self._ring_buffer.read(self._chunk_size, read_offset)

        # Apply window if requested
        if apply_window:
            native.audio_apply_hann_window(chunk._get_native())

        self._chunks_processed += 1
        return chunk

    def reset(self) -> None:
        """Reset the stream state."""
        self._ring_buffer.clear()
        self._chunks_processed = 0

    @property
    def chunks_available(self) -> int:
        """Number of complete chunks available."""
        if self._ring_buffer._samples_written < self._chunk_size:
            return 0
        available = self._ring_buffer._samples_written - self._chunk_size
        return available // self._hop_size + 1 - self._chunks_processed

    def __repr__(self) -> str:
        return (
            f"AudioStream(chunk_size={self._chunk_size}, "
            f"hop_size={self._hop_size}, "
            f"sample_rate={self._sample_rate}, "
            f"chunks_available={self.chunks_available})"
        )


__all__ = [
    "AudioBuffer",
    "AudioRingBuffer",
    "AudioStream",
    "from_pcm",
]
