"""Phase reconstruction functions for GPU audio processing.

This module provides:
- ISTFT (Inverse Short-Time Fourier Transform)
- Griffin-Lim algorithm for phase reconstruction
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


def istft(
    stft_output: GPUArray,
    hop_length: int = 160,
    win_length: int = -1,
    center: bool = True,
    length: int = -1,
) -> GPUArray:
    """Compute Inverse Short-Time Fourier Transform (ISTFT).

    Reconstructs time-domain signal from complex STFT representation
    using overlap-add with window sum normalization.

    Args:
        stft_output: Complex STFT [n_frames, n_freq, 2] (real, imag)
        hop_length: Hop size (default 160)
        win_length: Window length (default: (n_freq-1)*2)
        center: Whether input was centered (default True)
        length: Output length (-1 for automatic)

    Returns:
        Time-domain signal [n_samples]

    Example:
        >>> stft_out = stft(buf, n_fft=512, hop_length=160)
        >>> reconstructed = istft(stft_out, hop_length=160)
    """
    native = _get_native()
    result = native.audio_istft(stft_output._get_native(), hop_length, win_length, center, length)
    return GPUArray._wrap_native(result)


def griffin_lim(
    magnitude: GPUArray,
    n_iter: int = 32,
    hop_length: int = 160,
    win_length: int = -1,
) -> GPUArray:
    """Griffin-Lim algorithm for phase reconstruction.

    Reconstructs time-domain signal from magnitude spectrogram only,
    iteratively estimating phase using STFT/ISTFT consistency.

    Args:
        magnitude: Magnitude spectrogram [n_frames, n_freq]
        n_iter: Number of iterations (default 32)
        hop_length: Hop size (default 160)
        win_length: Window length (default: (n_freq-1)*2)

    Returns:
        Reconstructed time-domain signal [n_samples]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> reconstructed = griffin_lim(mag, n_iter=32)
    """
    native = _get_native()
    result = native.audio_griffin_lim(magnitude._get_native(), n_iter, hop_length, win_length)
    return GPUArray._wrap_native(result)


__all__ = [
    "istft",
    "griffin_lim",
]
