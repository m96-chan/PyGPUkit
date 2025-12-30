"""Pitch detection functions for GPU audio processing.

This module provides:
- Autocorrelation function
- YIN pitch detection algorithm
"""

from __future__ import annotations

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


def autocorrelation(audio: AudioBuffer | GPUArray, max_lag: int) -> GPUArray:
    """Compute autocorrelation function.

    Args:
        audio: Input audio (float32)
        max_lag: Maximum lag in samples

    Returns:
        Autocorrelation values [max_lag]

    Example:
        >>> acf = autocorrelation(buf, max_lag=400)  # 25ms @ 16kHz
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_autocorrelation(data._get_native(), max_lag)
    return GPUArray._wrap_native(result)


def detect_pitch_yin(
    audio: AudioBuffer | GPUArray,
    sample_rate: int = 16000,
    f_min: float = 50.0,
    f_max: float = 500.0,
    threshold: float = 0.1,
) -> float:
    """Detect pitch using YIN algorithm.

    The YIN algorithm detects the fundamental frequency of a quasi-periodic
    signal using cumulative mean normalized difference function.

    Args:
        audio: Input audio frame (float32)
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency to detect (default 50 Hz)
        f_max: Maximum frequency to detect (default 500 Hz)
        threshold: YIN threshold (default 0.1)

    Returns:
        Detected pitch in Hz (0.0 if unvoiced)

    Example:
        >>> pitch = detect_pitch_yin(audio_frame, sample_rate=16000)
        >>> if pitch > 0:
        ...     print(f"Pitch: {pitch:.1f} Hz")
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    return native.audio_detect_pitch_yin(data._get_native(), sample_rate, f_min, f_max, threshold)


def detect_pitch_yin_frames(
    audio: AudioBuffer | GPUArray,
    sample_rate: int = 16000,
    frame_size: int = 1024,
    hop_size: int = 256,
    f_min: float = 50.0,
    f_max: float = 500.0,
    threshold: float = 0.1,
) -> GPUArray:
    """Detect pitch for each frame using YIN algorithm.

    Args:
        audio: Input audio (float32)
        sample_rate: Sample rate in Hz
        frame_size: Frame size in samples (default 1024)
        hop_size: Hop size in samples (default 256)
        f_min: Minimum frequency to detect (default 50 Hz)
        f_max: Maximum frequency to detect (default 500 Hz)
        threshold: YIN threshold (default 0.1)

    Returns:
        Pitch values for each frame [n_frames]

    Example:
        >>> pitches = detect_pitch_yin_frames(buf, sample_rate=16000)
        >>> voiced = pitches.to_numpy() > 0
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_detect_pitch_yin_frames(
        data._get_native(), sample_rate, frame_size, hop_size, f_min, f_max, threshold
    )
    return GPUArray._wrap_native(result)


__all__ = [
    "autocorrelation",
    "detect_pitch_yin",
    "detect_pitch_yin_frames",
]
