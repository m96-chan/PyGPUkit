"""Harmonic-Percussive Source Separation (HPSS) for GPU audio processing.

This module provides:
- HPSS (Harmonic-Percussive Source Separation)
- Harmonic and percussive component extraction
"""

from __future__ import annotations

from pygpukit.core import GPUArray


def _get_native():
    """Get the native module."""
    try:
        from pygpukit._native_loader import get_native_module

        return get_native_module()
    except ImportError:
        from pygpukit import _pygpukit_native

        return _pygpukit_native


def hpss(
    stft_magnitude_input: GPUArray,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> tuple[GPUArray, GPUArray]:
    """Harmonic-Percussive Source Separation using median filtering.

    Separates audio into harmonic (tonal) and percussive (transient) components
    using median filtering in time and frequency directions.

    Args:
        stft_magnitude_input: STFT magnitude [n_frames, n_freq]
        kernel_size: Median filter kernel size (default 31)
        power: Power for spectrogram (default 2.0)
        margin: Margin for soft masking (default 1.0)

    Returns:
        Tuple of (harmonic_magnitude, percussive_magnitude)

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> harmonic, percussive = hpss(mag)
    """
    native = _get_native()
    h, p = native.audio_hpss(stft_magnitude_input._get_native(), kernel_size, power, margin)
    return GPUArray._wrap_native(h), GPUArray._wrap_native(p)


def harmonic(
    stft_magnitude_input: GPUArray,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> GPUArray:
    """Extract harmonic component using HPSS.

    Args:
        stft_magnitude_input: STFT magnitude [n_frames, n_freq]
        kernel_size: Median filter kernel size (default 31)
        power: Power for spectrogram (default 2.0)
        margin: Margin for soft masking (default 1.0)

    Returns:
        Harmonic magnitude [n_frames, n_freq]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> harm = harmonic(mag)
    """
    h, _ = hpss(stft_magnitude_input, kernel_size, power, margin)
    return h


def percussive(
    stft_magnitude_input: GPUArray,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> GPUArray:
    """Extract percussive component using HPSS.

    Args:
        stft_magnitude_input: STFT magnitude [n_frames, n_freq]
        kernel_size: Median filter kernel size (default 31)
        power: Power for spectrogram (default 2.0)
        margin: Margin for soft masking (default 1.0)

    Returns:
        Percussive magnitude [n_frames, n_freq]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> perc = percussive(mag)
    """
    _, p = hpss(stft_magnitude_input, kernel_size, power, margin)
    return p


__all__ = [
    "hpss",
    "harmonic",
    "percussive",
]
