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
from typing import Union

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
    samples: Union[np.ndarray, GPUArray],
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


__all__ = [
    "AudioBuffer",
    "from_pcm",
]
