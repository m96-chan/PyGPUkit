"""Constant-Q Transform and Chromagram for GPU audio processing.

This module provides:
- CQT (Constant-Q Transform)
- Chromagram from STFT and CQT
"""

from __future__ import annotations

from pygpukit.core import GPUArray

from .buffer import AudioBuffer
from .spectral import magnitude_spectrum


def _get_native():
    """Get the native module."""
    try:
        from pygpukit._native_loader import get_native_module

        return get_native_module()
    except ImportError:
        from pygpukit import _pygpukit_native

        return _pygpukit_native


def cqt(
    audio: AudioBuffer | GPUArray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    f_min: float = 32.7,
    n_bins: int = 84,
    bins_per_octave: int = 12,
) -> GPUArray:
    """Compute Constant-Q Transform (CQT).

    CQT provides logarithmically-spaced frequency resolution, useful for
    music analysis where notes are logarithmically distributed.

    This implementation uses STFT-based approximation for efficiency.

    Args:
        audio: Input audio (float32)
        sample_rate: Sample rate in Hz
        hop_length: Hop size (default 160)
        f_min: Minimum frequency (default 32.7 Hz = C1)
        n_bins: Number of frequency bins (default 84 = 7 octaves)
        bins_per_octave: Bins per octave (default 12)

    Returns:
        Complex CQT [n_frames, n_bins, 2] (real, imag)

    Example:
        >>> cqt_out = cqt(buf, sample_rate=16000, n_bins=84)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_cqt(
        data._get_native(), sample_rate, hop_length, f_min, n_bins, bins_per_octave
    )
    return GPUArray._wrap_native(result)


def cqt_magnitude(
    audio: AudioBuffer | GPUArray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    f_min: float = 32.7,
    n_bins: int = 84,
    bins_per_octave: int = 12,
) -> GPUArray:
    """Compute CQT magnitude spectrogram.

    Convenience function that computes CQT and returns magnitude.

    Args:
        audio: Input audio (float32)
        sample_rate: Sample rate in Hz
        hop_length: Hop size (default 160)
        f_min: Minimum frequency (default 32.7 Hz = C1)
        n_bins: Number of frequency bins (default 84)
        bins_per_octave: Bins per octave (default 12)

    Returns:
        CQT magnitude [n_frames, n_bins]

    Example:
        >>> cqt_mag = cqt_magnitude(buf, sample_rate=16000)
    """
    cqt_out = cqt(audio, sample_rate, hop_length, f_min, n_bins, bins_per_octave)
    return magnitude_spectrum(cqt_out)


def chroma_stft(
    spectrum: GPUArray,
    sample_rate: int = 16000,
    n_chroma: int = 12,
    tuning: float = 0.0,
) -> GPUArray:
    """Compute chromagram from STFT magnitude spectrum.

    Maps the spectrum to 12 pitch classes (C, C#, D, ..., B).

    Args:
        spectrum: Magnitude spectrum [n_frames, n_freq]
        sample_rate: Sample rate in Hz
        n_chroma: Number of chroma bins (default 12)
        tuning: Tuning deviation in fractions of a chroma bin (default 0)

    Returns:
        Chromagram [n_frames, n_chroma]

    Example:
        >>> mag = magnitude_spectrum(stft_out)
        >>> chroma = chroma_stft(mag, sample_rate=16000)
    """
    native = _get_native()
    result = native.audio_chroma_stft(spectrum._get_native(), sample_rate, n_chroma, tuning)
    return GPUArray._wrap_native(result)


def chroma_cqt(
    cqt_magnitude_input: GPUArray,
    bins_per_octave: int = 12,
) -> GPUArray:
    """Compute chromagram from CQT magnitude.

    Args:
        cqt_magnitude_input: CQT magnitude [n_frames, n_bins]
        bins_per_octave: Bins per octave in CQT (default 12)

    Returns:
        Chromagram [n_frames, bins_per_octave]

    Example:
        >>> cqt_mag = cqt_magnitude(buf, bins_per_octave=12)
        >>> chroma = chroma_cqt(cqt_mag, bins_per_octave=12)
    """
    native = _get_native()
    result = native.audio_chroma_cqt(cqt_magnitude_input._get_native(), bins_per_octave)
    return GPUArray._wrap_native(result)


__all__ = [
    "cqt",
    "cqt_magnitude",
    "chroma_stft",
    "chroma_cqt",
]
