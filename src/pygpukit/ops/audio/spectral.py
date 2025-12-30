"""Spectral processing functions for GPU audio processing.

This module provides:
- STFT (Short-Time Fourier Transform)
- Power and magnitude spectrum
- Mel filterbank operations
- Log-mel spectrogram
- MFCC
- Delta features
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


def stft(
    audio: AudioBuffer | GPUArray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = -1,
    center: bool = True,
) -> GPUArray:
    """Compute Short-Time Fourier Transform (STFT).

    Uses a custom Radix-2 FFT implementation (no cuFFT dependency).

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        n_fft: FFT size (must be power of 2, default 512)
        hop_length: Hop size (default 160)
        win_length: Window length (default n_fft)
        center: Whether to pad input with reflection (default True)

    Returns:
        Complex STFT output [n_frames, n_fft/2+1, 2] (real, imag)

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> stft_out = stft(buf, n_fft=512, hop_length=160)
        >>> print(f"STFT shape: {stft_out.shape}")  # [n_frames, 257, 2]
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_stft(data._get_native(), n_fft, hop_length, win_length, center)
    return GPUArray._wrap_native(result)


def power_spectrum(stft_output: GPUArray) -> GPUArray:
    """Compute power spectrogram from STFT output.

    power = real^2 + imag^2

    Args:
        stft_output: STFT output [n_frames, n_freq, 2]

    Returns:
        Power spectrogram [n_frames, n_freq]

    Example:
        >>> stft_out = stft(buf, n_fft=512)
        >>> power = power_spectrum(stft_out)
    """
    native = _get_native()
    result = native.audio_power_spectrum(stft_output._get_native())
    return GPUArray._wrap_native(result)


def magnitude_spectrum(stft_output: GPUArray) -> GPUArray:
    """Compute magnitude spectrogram from STFT output.

    magnitude = sqrt(real^2 + imag^2)

    Args:
        stft_output: STFT output [n_frames, n_freq, 2]

    Returns:
        Magnitude spectrogram [n_frames, n_freq]

    Example:
        >>> stft_out = stft(buf, n_fft=512)
        >>> mag = magnitude_spectrum(stft_out)
    """
    native = _get_native()
    result = native.audio_magnitude_spectrum(stft_output._get_native())
    return GPUArray._wrap_native(result)


def create_mel_filterbank(
    n_mels: int = 80,
    n_fft: int = 512,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float = -1.0,
) -> GPUArray:
    """Create Mel filterbank matrix.

    Args:
        n_mels: Number of mel bands (default 80 for Whisper)
        n_fft: FFT size
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency (default 0)
        f_max: Maximum frequency (default sample_rate/2)

    Returns:
        Mel filterbank matrix [n_mels, n_fft/2+1]

    Example:
        >>> mel_fb = create_mel_filterbank(n_mels=80, n_fft=512, sample_rate=16000)
    """
    native = _get_native()
    result = native.audio_create_mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
    return GPUArray._wrap_native(result)


def apply_mel_filterbank(spectrogram: GPUArray, mel_filterbank: GPUArray) -> GPUArray:
    """Apply Mel filterbank to power/magnitude spectrogram.

    Args:
        spectrogram: Input spectrogram [n_frames, n_fft/2+1]
        mel_filterbank: Mel filterbank [n_mels, n_fft/2+1]

    Returns:
        Mel spectrogram [n_frames, n_mels]

    Example:
        >>> power = power_spectrum(stft_out)
        >>> mel_fb = create_mel_filterbank(n_mels=80, n_fft=512)
        >>> mel = apply_mel_filterbank(power, mel_fb)
    """
    native = _get_native()
    result = native.audio_apply_mel_filterbank(
        spectrogram._get_native(), mel_filterbank._get_native()
    )
    return GPUArray._wrap_native(result)


def log_mel(mel_spectrogram: GPUArray, eps: float = 1e-10) -> GPUArray:
    """Compute log-mel spectrogram.

    log_mel = log(mel + eps)

    Args:
        mel_spectrogram: Mel spectrogram [n_frames, n_mels]
        eps: Small constant for numerical stability (default 1e-10)

    Returns:
        Log-mel spectrogram [n_frames, n_mels]

    Example:
        >>> log_mel_spec = log_mel(mel_spectrogram)
    """
    native = _get_native()
    result = native.audio_log_mel_spectrogram(mel_spectrogram._get_native(), eps)
    return GPUArray._wrap_native(result)


def to_decibels(audio: AudioBuffer | GPUArray, eps: float = 1e-10) -> GPUArray:
    """Convert to decibels.

    dB = 10 * log10(x + eps)

    Args:
        audio: Input array (power values)
        eps: Small constant for numerical stability (default 1e-10)

    Returns:
        dB values

    Example:
        >>> power = power_spectrum(stft_out)
        >>> db = to_decibels(power)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_to_decibels(data._get_native(), eps)
    return GPUArray._wrap_native(result)


def mfcc(log_mel_input: GPUArray, n_mfcc: int = 13) -> GPUArray:
    """Compute MFCC from log-mel spectrogram using DCT-II.

    Args:
        log_mel_input: Log-mel spectrogram [n_frames, n_mels]
        n_mfcc: Number of MFCC coefficients (default 13)

    Returns:
        MFCC [n_frames, n_mfcc]

    Example:
        >>> log_mel_spec = log_mel(mel_spectrogram)
        >>> mfcc_features = mfcc(log_mel_spec, n_mfcc=13)
    """
    native = _get_native()
    result = native.audio_mfcc(log_mel_input._get_native(), n_mfcc)
    return GPUArray._wrap_native(result)


def delta(features: GPUArray, order: int = 1, width: int = 2) -> GPUArray:
    """Compute delta (differential) features.

    Args:
        features: Input features [n_frames, n_features]
        order: Delta order (1 for delta, 2 for delta-delta)
        width: Window width for computation (default 2)

    Returns:
        Delta features [n_frames, n_features]

    Example:
        >>> mfcc_features = mfcc(log_mel_spec)
        >>> delta_mfcc = delta(mfcc_features, order=1)
        >>> delta_delta_mfcc = delta(mfcc_features, order=2)
    """
    native = _get_native()
    result = native.audio_delta_features(features._get_native(), order, width)
    return GPUArray._wrap_native(result)


def mel_spectrogram(
    audio: AudioBuffer | GPUArray,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float = -1.0,
) -> GPUArray:
    """Compute mel spectrogram.

    Combines: STFT -> power -> mel filterbank

    Args:
        audio: Input audio (float32)
        n_fft: FFT size (must be power of 2)
        hop_length: Hop size
        n_mels: Number of mel bands
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency
        f_max: Maximum frequency (-1 for sample_rate/2)

    Returns:
        Mel spectrogram [n_frames, n_mels]

    Example:
        >>> mel = mel_spectrogram(buf, n_fft=512, hop_length=160, n_mels=80)
    """
    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    # STFT
    stft_out = stft(data, n_fft=n_fft, hop_length=hop_length, center=True)

    # Power spectrum
    power = power_spectrum(stft_out)

    # Create and apply mel filterbank
    mel_fb = create_mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
    mel = apply_mel_filterbank(power, mel_fb)

    return mel


def log_mel_spectrogram(
    audio: AudioBuffer | GPUArray,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float = -1.0,
    eps: float = 1e-10,
) -> GPUArray:
    """Compute log-mel spectrogram (Whisper-compatible).

    Combines: STFT -> power -> mel filterbank -> log

    Args:
        audio: Input audio (float32, 16kHz expected for Whisper)
        n_fft: FFT size (must be power of 2)
        hop_length: Hop size
        n_mels: Number of mel bands (80 for Whisper)
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency
        f_max: Maximum frequency (-1 for sample_rate/2)
        eps: Small constant for log stability

    Returns:
        Log-mel spectrogram [n_frames, n_mels]

    Example:
        >>> # Whisper-style mel spectrogram
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> log_mel = log_mel_spectrogram(buf, n_fft=512, hop_length=160, n_mels=80)
    """
    mel = mel_spectrogram(audio, n_fft, hop_length, n_mels, sample_rate, f_min, f_max)
    return log_mel(mel, eps)


__all__ = [
    "stft",
    "power_spectrum",
    "magnitude_spectrum",
    "create_mel_filterbank",
    "apply_mel_filterbank",
    "log_mel",
    "to_decibels",
    "mfcc",
    "delta",
    "mel_spectrogram",
    "log_mel_spectrogram",
]
