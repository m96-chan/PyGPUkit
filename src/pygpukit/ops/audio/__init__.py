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

Corresponds to native/ops/audio/.
"""

from __future__ import annotations

# Buffer classes
from .buffer import (
    AudioBuffer,
    AudioRingBuffer,
    AudioStream,
    from_pcm,
)

# CQT and Chromagram
from .cqt import (
    chroma_cqt,
    chroma_stft,
    cqt,
    cqt_magnitude,
)

# Audio effects
from .effects import (
    pitch_shift,
    time_stretch,
)

# Spectral features
from .features import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_contrast,
    spectral_flatness,
    spectral_rolloff,
    zero_crossing_rate,
)

# HPSS
from .hpss import (
    harmonic,
    hpss,
    percussive,
)

# Phase reconstruction
from .phase import (
    griffin_lim,
    istft,
)

# Pitch detection
from .pitch import (
    autocorrelation,
    detect_pitch_yin,
    detect_pitch_yin_frames,
)

# Preprocessing functions
from .preprocessing import (
    compute_short_term_energy,
    deemphasis,
    highpass_filter,
    noise_gate,
    preemphasis,
    remove_dc,
    spectral_gate,
)

# Spectral processing
from .spectral import (
    apply_mel_filterbank,
    create_mel_filterbank,
    delta,
    log_mel,
    log_mel_spectrogram,
    magnitude_spectrum,
    mel_spectrogram,
    mfcc,
    power_spectrum,
    stft,
    to_decibels,
)

# VAD
from .vad import (
    VAD,
    SpeechSegment,
)

__all__ = [
    # Classes
    "AudioBuffer",
    "AudioRingBuffer",
    "AudioStream",
    "SpeechSegment",
    "VAD",
    # Basic functions
    "from_pcm",
    # Preprocessing functions
    "preemphasis",
    "deemphasis",
    "remove_dc",
    "highpass_filter",
    "noise_gate",
    "spectral_gate",
    "compute_short_term_energy",
    # Spectral processing
    "stft",
    "power_spectrum",
    "magnitude_spectrum",
    "create_mel_filterbank",
    "apply_mel_filterbank",
    "log_mel",
    "to_decibels",
    "mfcc",
    "delta",
    # High-level functions
    "mel_spectrogram",
    "log_mel_spectrogram",
    # Inverse STFT and phase reconstruction
    "istft",
    "griffin_lim",
    # Pitch detection
    "autocorrelation",
    "detect_pitch_yin",
    "detect_pitch_yin_frames",
    # Spectral features
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_contrast",
    "zero_crossing_rate",
    # CQT and Chromagram
    "cqt",
    "cqt_magnitude",
    "chroma_stft",
    "chroma_cqt",
    # HPSS
    "hpss",
    "harmonic",
    "percussive",
    # Time stretching and pitch shifting
    "time_stretch",
    "pitch_shift",
]
