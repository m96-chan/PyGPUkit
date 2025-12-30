"""Audio effects for GPU audio processing.

This module provides:
- Time stretching using phase vocoder
- Pitch shifting
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


def time_stretch(
    audio: AudioBuffer | GPUArray,
    rate: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> GPUArray:
    """Time stretch audio using phase vocoder.

    Changes the duration of audio without changing its pitch.

    Args:
        audio: Input audio (float32)
        rate: Stretch factor (>1 = faster/shorter, <1 = slower/longer)
        n_fft: FFT size (default 2048)
        hop_length: Hop size (default 512)

    Returns:
        Time-stretched audio [n_samples * rate]

    Example:
        >>> # Slow down to half speed
        >>> slow = time_stretch(buf, rate=0.5)
        >>> # Speed up to double speed
        >>> fast = time_stretch(buf, rate=2.0)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_time_stretch(data._get_native(), rate, n_fft, hop_length)
    return GPUArray._wrap_native(result)


def pitch_shift(
    audio: AudioBuffer | GPUArray,
    sample_rate: int,
    n_steps: float,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> GPUArray:
    """Pitch shift audio using phase vocoder and resampling.

    Changes the pitch of audio without changing its duration.

    Args:
        audio: Input audio (float32)
        sample_rate: Sample rate in Hz
        n_steps: Number of semitones to shift (positive = up, negative = down)
        n_fft: FFT size (default 2048)
        hop_length: Hop size (default 512)

    Returns:
        Pitch-shifted audio [n_samples]

    Example:
        >>> # Shift up one octave
        >>> higher = pitch_shift(buf, sample_rate=16000, n_steps=12)
        >>> # Shift down a perfect fifth
        >>> lower = pitch_shift(buf, sample_rate=16000, n_steps=-7)
    """
    native = _get_native()

    if isinstance(audio, AudioBuffer):
        data = audio.data
    else:
        data = audio

    result = native.audio_pitch_shift(data._get_native(), sample_rate, n_steps, n_fft, hop_length)
    return GPUArray._wrap_native(result)


__all__ = [
    "time_stretch",
    "pitch_shift",
]
