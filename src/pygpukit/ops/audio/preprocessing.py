"""Audio preprocessing functions for GPU audio processing.

This module provides:
- Pre-emphasis and de-emphasis filters
- DC removal
- High-pass filtering
- Noise gate and spectral gate
- Short-term energy computation
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


def preemphasis(audio: AudioBuffer | GPUArray, alpha: float = 0.97) -> AudioBuffer | GPUArray:
    """Apply pre-emphasis filter to emphasize high-frequency components.

    Pre-emphasis is commonly used in speech processing to boost high frequencies
    that are typically attenuated during recording.

    Formula: y[n] = x[n] - alpha * x[n-1]

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        alpha: Pre-emphasis coefficient (default 0.97)

    Returns:
        Same type as input (modified in-place)

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> preemphasis(buf, alpha=0.97)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        native.audio_preemphasis(audio.data._get_native(), alpha)
        return audio
    else:
        native.audio_preemphasis(audio._get_native(), alpha)
        return audio


def deemphasis(audio: AudioBuffer | GPUArray, alpha: float = 0.97) -> AudioBuffer | GPUArray:
    """Apply de-emphasis filter (inverse of pre-emphasis).

    Used to restore the original spectral balance after pre-emphasis.

    Formula: y[n] = x[n] + alpha * y[n-1]

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        alpha: De-emphasis coefficient (default 0.97)

    Returns:
        Same type as input (modified in-place)

    Example:
        >>> buf = preemphasis(buf)
        >>> # ... processing ...
        >>> deemphasis(buf)  # Restore original balance
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        native.audio_deemphasis(audio.data._get_native(), alpha)
        return audio
    else:
        native.audio_deemphasis(audio._get_native(), alpha)
        return audio


def remove_dc(audio: AudioBuffer | GPUArray) -> AudioBuffer | GPUArray:
    """Remove DC offset from audio signal.

    Subtracts the mean value from all samples, centering the signal at zero.
    This is a simple but effective way to remove DC bias.

    Args:
        audio: AudioBuffer or GPUArray of float32 samples

    Returns:
        Same type as input (modified in-place)

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> remove_dc(buf)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        native.audio_remove_dc(audio.data._get_native())
        return audio
    else:
        native.audio_remove_dc(audio._get_native())
        return audio


def highpass_filter(
    audio: AudioBuffer | GPUArray,
    cutoff_hz: float = 20.0,
    sample_rate: int | None = None,
) -> AudioBuffer | GPUArray:
    """Apply high-pass filter for DC removal.

    Uses a single-pole IIR high-pass filter, which is more effective than
    simple mean subtraction for removing low-frequency noise.

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        cutoff_hz: Cutoff frequency in Hz (default 20.0)
        sample_rate: Sample rate in Hz (auto-detected from AudioBuffer)

    Returns:
        Same type as input (modified in-place)

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> highpass_filter(buf, cutoff_hz=50.0)  # Remove hum
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        sr = sample_rate if sample_rate is not None else audio.sample_rate
        native.audio_highpass_filter(audio.data._get_native(), cutoff_hz, sr)
        return audio
    else:
        sr = sample_rate if sample_rate is not None else 16000
        native.audio_highpass_filter(audio._get_native(), cutoff_hz, sr)
        return audio


def noise_gate(audio: AudioBuffer | GPUArray, threshold: float = 0.01) -> AudioBuffer | GPUArray:
    """Apply simple noise gate.

    Zeros samples with absolute value below threshold. This is a hard gate
    that completely silences quiet sections.

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        threshold: Amplitude threshold (default 0.01)

    Returns:
        Same type as input (modified in-place)

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> noise_gate(buf, threshold=0.02)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        native.audio_noise_gate(audio.data._get_native(), threshold)
        return audio
    else:
        native.audio_noise_gate(audio._get_native(), threshold)
        return audio


def spectral_gate(
    audio: AudioBuffer | GPUArray,
    threshold: float = 0.01,
    attack_samples: int = 64,
    release_samples: int = 256,
) -> AudioBuffer | GPUArray:
    """Apply spectral gate for noise reduction.

    A softer noise gate that attenuates (rather than silences) quiet sections
    based on short-term frame energy. Provides smoother transitions than
    a hard noise gate.

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        threshold: Energy threshold (linear scale, default 0.01)
        attack_samples: Frame size for energy computation (default 64)
        release_samples: Smoothing release in samples (default 256)

    Returns:
        Same type as input (modified in-place)

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> spectral_gate(buf, threshold=0.005)  # Subtle noise reduction
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        native.audio_spectral_gate(
            audio.data._get_native(), threshold, attack_samples, release_samples
        )
        return audio
    else:
        native.audio_spectral_gate(audio._get_native(), threshold, attack_samples, release_samples)
        return audio


def compute_short_term_energy(audio: AudioBuffer | GPUArray, frame_size: int = 256) -> GPUArray:
    """Compute short-term energy for analysis or adaptive processing.

    Divides the audio into non-overlapping frames and computes the mean
    energy (sum of squares / frame_size) for each frame.

    Args:
        audio: AudioBuffer or GPUArray of float32 samples
        frame_size: Frame size in samples (default 256)

    Returns:
        GPUArray of frame energies

    Example:
        >>> buf = from_pcm(pcm_data, sample_rate=16000)
        >>> energy = compute_short_term_energy(buf, frame_size=320)  # 20ms @ 16kHz
        >>> print(f"Max energy: {energy.to_numpy().max():.4f}")
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_compute_short_term_energy(data._get_native(), frame_size)
    return GPUArray._wrap_native(result)


__all__ = [
    "preemphasis",
    "deemphasis",
    "remove_dc",
    "highpass_filter",
    "noise_gate",
    "spectral_gate",
    "compute_short_term_energy",
]
