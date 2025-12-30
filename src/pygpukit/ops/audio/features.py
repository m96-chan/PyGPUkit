"""Spectral feature extraction for GPU audio processing.

This module provides:
- Spectral centroid, bandwidth, rolloff, flatness, contrast
- Zero-crossing rate
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


def spectral_centroid(
    spectrum: GPUArray,
    sample_rate: int = 16000,
) -> GPUArray:
    """Compute spectral centroid for each frame.

    The spectral centroid indicates the "center of mass" of the spectrum.

    Args:
        spectrum: Magnitude or power spectrum [n_frames, n_freq]
        sample_rate: Sample rate in Hz

    Returns:
        Spectral centroid in Hz for each frame [n_frames]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> centroid = spectral_centroid(mag, sample_rate=16000)
    """
    native = _get_native()
    result = native.audio_spectral_centroid(spectrum._get_native(), sample_rate)
    return GPUArray._wrap_native(result)


def spectral_bandwidth(
    spectrum: GPUArray,
    centroids: GPUArray,
    sample_rate: int = 16000,
    p: int = 2,
) -> GPUArray:
    """Compute spectral bandwidth for each frame.

    Spectral bandwidth is the weighted standard deviation of frequencies
    around the spectral centroid.

    Args:
        spectrum: Magnitude or power spectrum [n_frames, n_freq]
        centroids: Pre-computed spectral centroids [n_frames]
        sample_rate: Sample rate in Hz
        p: Order for bandwidth computation (default 2)

    Returns:
        Spectral bandwidth in Hz for each frame [n_frames]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> centroid = spectral_centroid(mag, sample_rate=16000)
        >>> bandwidth = spectral_bandwidth(mag, centroid, sample_rate=16000)
    """
    native = _get_native()
    result = native.audio_spectral_bandwidth(
        spectrum._get_native(), centroids._get_native(), sample_rate, p
    )
    return GPUArray._wrap_native(result)


def spectral_rolloff(
    spectrum: GPUArray,
    sample_rate: int = 16000,
    roll_percent: float = 0.85,
) -> GPUArray:
    """Compute spectral rolloff for each frame.

    The rolloff frequency is the frequency below which roll_percent of
    the total spectral energy is contained.

    Args:
        spectrum: Magnitude or power spectrum [n_frames, n_freq]
        sample_rate: Sample rate in Hz
        roll_percent: Percentage of energy (default 0.85)

    Returns:
        Rolloff frequency in Hz for each frame [n_frames]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> rolloff = spectral_rolloff(mag, sample_rate=16000, roll_percent=0.85)
    """
    native = _get_native()
    result = native.audio_spectral_rolloff(spectrum._get_native(), sample_rate, roll_percent)
    return GPUArray._wrap_native(result)


def spectral_flatness(spectrum: GPUArray) -> GPUArray:
    """Compute spectral flatness for each frame.

    Spectral flatness measures how tone-like vs noise-like a sound is.
    Values close to 1 indicate noise, values close to 0 indicate tonal content.

    Computed as: geometric_mean / arithmetic_mean

    Args:
        spectrum: Magnitude or power spectrum [n_frames, n_freq]

    Returns:
        Spectral flatness for each frame [n_frames] (0 to 1)

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> flatness = spectral_flatness(mag)
    """
    native = _get_native()
    result = native.audio_spectral_flatness(spectrum._get_native())
    return GPUArray._wrap_native(result)


def spectral_contrast(
    spectrum: GPUArray,
    n_bands: int = 6,
    alpha: float = 0.2,
) -> GPUArray:
    """Compute spectral contrast for each frame.

    Spectral contrast measures the difference between peaks and valleys
    in the spectrum, divided into frequency bands.

    Args:
        spectrum: Magnitude or power spectrum [n_frames, n_freq]
        n_bands: Number of frequency bands (default 6)
        alpha: Percentile for peak/valley estimation (default 0.2)

    Returns:
        Spectral contrast [n_frames, n_bands]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> contrast = spectral_contrast(mag, n_bands=6)
    """
    native = _get_native()
    result = native.audio_spectral_contrast(spectrum._get_native(), n_bands, alpha)
    return GPUArray._wrap_native(result)


def zero_crossing_rate(
    audio: AudioBuffer | GPUArray,
    frame_size: int = 512,
    hop_size: int = 256,
) -> GPUArray:
    """Compute zero-crossing rate for each frame.

    ZCR counts the number of times the signal crosses zero per frame,
    normalized by frame size.

    Args:
        audio: Input audio (float32)
        frame_size: Frame size in samples (default 512)
        hop_size: Hop size in samples (default 256)

    Returns:
        Zero-crossing rate for each frame [n_frames]

    Example:
        >>> zcr = zero_crossing_rate(buf, frame_size=512, hop_size=256)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_zero_crossing_rate(data._get_native(), frame_size, hop_size)
    return GPUArray._wrap_native(result)


__all__ = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_contrast",
    "zero_crossing_rate",
]
