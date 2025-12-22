"""GPU Audio Processing Operations.

This module provides GPU-accelerated audio processing for ASR/Whisper preprocessing:
- PCM to float conversion
- Stereo to mono conversion
- Peak/RMS normalization
- Resampling (48kHz -> 16kHz)

Example:
    >>> import numpy as np
    >>> import pygpukit as gk
    >>> from pygpukit.ops import audio
    >>>
    >>> # Load PCM samples (int16)
    >>> pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    >>> buf = audio.from_pcm(pcm, sample_rate=48000)
    >>>
    >>> # Process audio
    >>> buf = buf.to_mono().resample(16000).normalize()
    >>> result = buf.data.to_numpy()
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


@dataclass
class SpeechSegment:
    """Represents a detected speech segment.

    Attributes:
        start_sample: Start sample index
        end_sample: End sample index
        start_time: Start time in seconds
        end_time: End time in seconds
    """

    start_sample: int
    end_sample: int
    start_time: float
    end_time: float


class VAD:
    """GPU-accelerated Voice Activity Detection.

    Detects speech segments in audio using energy and zero-crossing rate features.
    Supports adaptive thresholding and hangover smoothing for robust detection.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        frame_ms: Frame duration in milliseconds (default: 20)
        hop_ms: Hop duration in milliseconds (default: 10)
        energy_threshold: Energy threshold for speech (default: auto)
        hangover_ms: Hangover duration in milliseconds (default: 100)

    Example:
        >>> vad = VAD(sample_rate=16000)
        >>> segments = vad.detect(audio_buffer)
        >>> for seg in segments:
        ...     print(f"Speech: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
        energy_threshold: float | None = None,
        hangover_ms: float = 100.0,
        zcr_low: float = 0.02,
        zcr_high: float = 0.25,
    ):
        self._sample_rate = sample_rate
        self._frame_size = int(frame_ms * sample_rate / 1000)
        self._hop_size = int(hop_ms * sample_rate / 1000)
        self._energy_threshold = energy_threshold
        self._hangover_frames = int(hangover_ms / hop_ms)
        self._zcr_low = zcr_low
        self._zcr_high = zcr_high

        # Adaptive threshold multiplier (above noise floor)
        self._adaptive_multiplier = 3.0

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        """Frame size in samples."""
        return self._frame_size

    @property
    def hop_size(self) -> int:
        """Hop size in samples."""
        return self._hop_size

    def detect(self, audio: AudioBuffer | GPUArray) -> list[SpeechSegment]:
        """Detect speech segments in audio.

        Args:
            audio: AudioBuffer or GPUArray of float32 samples

        Returns:
            List of SpeechSegment objects representing detected speech regions
        """
        native = _get_native()

        # Get audio data
        if isinstance(audio, AudioBuffer):
            data = audio.data
        else:
            data = audio

        # Compute frame features
        energy = native.vad_compute_energy(data._get_native(), self._frame_size, self._hop_size)
        zcr = native.vad_compute_zcr(data._get_native(), self._frame_size, self._hop_size)

        energy_gpu = GPUArray._wrap_native(energy)
        zcr_gpu = GPUArray._wrap_native(zcr)

        # Determine energy threshold
        if self._energy_threshold is not None:
            threshold = self._energy_threshold
        else:
            # Adaptive threshold: multiplier * noise_floor
            noise_floor = native.vad_compute_noise_floor(energy)
            threshold = max(noise_floor * self._adaptive_multiplier, 0.01)

        # VAD decision
        vad_flags = native.vad_decide(
            energy_gpu._get_native(),
            zcr_gpu._get_native(),
            threshold,
            self._zcr_low,
            self._zcr_high,
        )
        vad_flags_gpu = GPUArray._wrap_native(vad_flags)

        # Apply hangover smoothing
        if self._hangover_frames > 0:
            smoothed = native.vad_apply_hangover(vad_flags_gpu._get_native(), self._hangover_frames)
            vad_flags_gpu = GPUArray._wrap_native(smoothed)

        # Convert to segments
        return self._flags_to_segments(vad_flags_gpu)

    def _flags_to_segments(self, vad_flags: GPUArray) -> list[SpeechSegment]:
        """Convert frame-level VAD flags to speech segments."""
        flags: np.ndarray = vad_flags.to_numpy().astype(int)

        segments: list[SpeechSegment] = []
        in_speech = False
        start_frame = 0

        for i, flag in enumerate(flags):
            if flag == 1 and not in_speech:
                # Speech start
                in_speech = True
                start_frame = i
            elif flag == 0 and in_speech:
                # Speech end
                in_speech = False
                segments.append(self._create_segment(start_frame, i))

        # Handle case where speech continues to end
        if in_speech:
            segments.append(self._create_segment(start_frame, len(flags)))

        return segments

    def _create_segment(self, start_frame: int, end_frame: int) -> SpeechSegment:
        """Create a SpeechSegment from frame indices."""
        start_sample = start_frame * self._hop_size
        end_sample = end_frame * self._hop_size + self._frame_size

        return SpeechSegment(
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_sample / self._sample_rate,
            end_time=end_sample / self._sample_rate,
        )

    def get_frame_features(self, audio: AudioBuffer | GPUArray) -> tuple[GPUArray, GPUArray]:
        """Get raw frame features (energy and ZCR) for analysis.

        Args:
            audio: AudioBuffer or GPUArray of float32 samples

        Returns:
            Tuple of (energy, zcr) GPUArrays
        """
        native = _get_native()

        if isinstance(audio, AudioBuffer):
            data = audio.data
        else:
            data = audio

        energy = native.vad_compute_energy(data._get_native(), self._frame_size, self._hop_size)
        zcr = native.vad_compute_zcr(data._get_native(), self._frame_size, self._hop_size)

        return GPUArray._wrap_native(energy), GPUArray._wrap_native(zcr)

    def __repr__(self) -> str:
        return (
            f"VAD(sample_rate={self._sample_rate}, "
            f"frame_size={self._frame_size}, "
            f"hop_size={self._hop_size}, "
            f"hangover_frames={self._hangover_frames})"
        )


__all__ = [
    "AudioBuffer",
    "AudioRingBuffer",
    "AudioStream",
    "SpeechSegment",
    "VAD",
    "from_pcm",
]
