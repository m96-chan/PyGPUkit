"""Voice Activity Detection (VAD) for GPU audio processing.

This module provides:
- VAD: GPU-accelerated Voice Activity Detection
- SpeechSegment: Detected speech segment data class
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pygpukit.core import GPUArray

from .buffer import AudioBuffer


def _get_native():
    """Get the native module."""
    try:
        from pygpukit._native_loader import get_native_module

        return get_native_module()
    except ImportError:
        from pygpukit import _pygpukit_native

        return _pygpukit_native


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
    "SpeechSegment",
    "VAD",
]
