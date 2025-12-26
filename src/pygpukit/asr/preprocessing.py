"""Whisper-compatible audio preprocessing.

This module provides GPU-accelerated audio preprocessing compatible with
OpenAI Whisper and derived models (kotoba-whisper, faster-whisper, etc.).

Whisper Preprocessing Pipeline:
    1. Resample to 16kHz (if needed)
    2. Pad/trim to 30 seconds (480,000 samples)
    3. STFT: n_fft=400, hop_length=160, window=hann
    4. Mel filterbank: 80 channels, fmin=0, fmax=8000
    5. Log-mel: log10(max(mel, 1e-10))
    6. Normalize: (log_mel + 4.0) / 4.0

Reference:
    https://github.com/openai/whisper/blob/main/whisper/audio.py
"""

from typing import Optional, Union

import numpy as np

from ..core import GPUArray, from_numpy
from ..ops import audio

# Whisper audio constants
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_N_MELS = 80
WHISPER_CHUNK_LENGTH = 30  # seconds
WHISPER_N_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH  # 480000
WHISPER_N_FRAMES = WHISPER_N_SAMPLES // WHISPER_HOP_LENGTH  # 3000


def pad_or_trim(
    audio_data: Union[GPUArray, np.ndarray],
    length: int = WHISPER_N_SAMPLES,
) -> GPUArray:
    """Pad or trim audio to exact length.

    Args:
        audio_data: Input audio samples (float32)
        length: Target length in samples (default: 480000 for 30s @ 16kHz)

    Returns:
        GPUArray of exact length, zero-padded or trimmed
    """
    # Convert to GPUArray if numpy
    if isinstance(audio_data, np.ndarray):
        audio_data = from_numpy(audio_data.astype(np.float32))

    current_length = audio_data.shape[0]

    if current_length == length:
        return audio_data

    if current_length > length:
        # Trim
        return audio_data[:length]
    else:
        # Pad with zeros
        pad_length = length - current_length
        padding = from_numpy(np.zeros(pad_length, dtype=np.float32))
        # Concatenate on GPU
        result_np = np.concatenate([audio_data.to_numpy(), padding.to_numpy()])
        return from_numpy(result_np)


def normalize_mel(log_mel: Union[GPUArray, np.ndarray]) -> GPUArray:
    """Apply Whisper-style normalization to log-mel spectrogram.

    Whisper normalization: (log_mel + 4.0) / 4.0

    This centers the values around 0 and scales them to roughly [-1, 1] range.

    Args:
        log_mel: Log-mel spectrogram [n_mels, n_frames] or [n_frames, n_mels]

    Returns:
        Normalized log-mel spectrogram as GPUArray
    """
    # Convert to GPUArray if numpy
    if isinstance(log_mel, np.ndarray):
        log_mel = from_numpy(log_mel.astype(np.float32))

    # (log_mel + 4.0) / 4.0
    return (log_mel + 4.0) / 4.0


def preprocess_audio(
    audio_input: Union[GPUArray, np.ndarray, str],
    sample_rate: Optional[int] = None,
    n_mels: int = WHISPER_N_MELS,
    padding: bool = True,
) -> GPUArray:
    """Preprocess audio for Whisper model inference.

    Complete preprocessing pipeline:
    1. Load audio (if path provided)
    2. Resample to 16kHz (if needed)
    3. Pad/trim to 30 seconds
    4. Compute log-mel spectrogram
    5. Apply Whisper normalization

    Args:
        audio_input: Audio samples (GPUArray/ndarray) or file path
        sample_rate: Sample rate of input audio (required if not 16kHz)
        n_mels: Number of mel bands (default: 80)
        padding: Whether to pad short audio to 30s (default: True)

    Returns:
        Preprocessed mel spectrogram [n_mels, n_frames] ready for encoder
        Shape: [80, 3000] for 30s audio

    Example:
        >>> mel = preprocess_audio("audio.wav")
        >>> print(mel.shape)  # [80, 3000]
        >>> # Feed to encoder
        >>> encoder_output = encoder(mel.unsqueeze(0))
    """
    # Handle file path input
    if isinstance(audio_input, str):
        # Load audio file using audio module
        audio_buf = audio.load_audio(audio_input)
        samples = audio_buf
        input_sample_rate = WHISPER_SAMPLE_RATE  # Assume load_audio resamples
    elif isinstance(audio_input, np.ndarray):
        samples = from_numpy(audio_input.astype(np.float32))
        input_sample_rate = sample_rate or WHISPER_SAMPLE_RATE
    elif isinstance(audio_input, GPUArray):
        samples = audio_input
        input_sample_rate = sample_rate or WHISPER_SAMPLE_RATE
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Resample if needed
    if input_sample_rate != WHISPER_SAMPLE_RATE:
        samples = audio.resample(samples, input_sample_rate, WHISPER_SAMPLE_RATE)

    # Pad or trim to 30 seconds
    if padding:
        samples = pad_or_trim(samples, WHISPER_N_SAMPLES)

    # Compute STFT
    stft_out = audio.stft(
        samples,
        n_fft=WHISPER_N_FFT,
        hop_length=WHISPER_HOP_LENGTH,
        center=True,
    )

    # Compute power spectrum
    power = audio.power_spectrum(stft_out)

    # Create and apply mel filterbank
    mel_fb = audio.create_mel_filterbank(
        n_mels=n_mels,
        n_fft=WHISPER_N_FFT,
        sample_rate=WHISPER_SAMPLE_RATE,
        f_min=0.0,
        f_max=8000.0,
    )
    mel = audio.apply_mel_filterbank(power, mel_fb)

    # Log-mel
    log_mel = audio.log_mel(mel, eps=1e-10)

    # Whisper normalization
    normalized = normalize_mel(log_mel)

    # Transpose to [n_mels, n_frames] for encoder input
    # Current shape: [n_frames, n_mels]
    # Target shape: [n_mels, n_frames]
    result_np = normalized.to_numpy().T
    return from_numpy(result_np.astype(np.float32))


def preprocess_audio_batch(
    audio_list: list,
    sample_rate: Optional[int] = None,
    n_mels: int = WHISPER_N_MELS,
) -> GPUArray:
    """Preprocess multiple audio samples as a batch.

    Args:
        audio_list: List of audio samples (GPUArray/ndarray) or file paths
        sample_rate: Sample rate of input audio
        n_mels: Number of mel bands

    Returns:
        Batch of preprocessed mel spectrograms [batch, n_mels, n_frames]
    """
    mels = []
    for audio_input in audio_list:
        mel = preprocess_audio(audio_input, sample_rate, n_mels)
        mels.append(mel.to_numpy())

    batch = np.stack(mels, axis=0)
    return from_numpy(batch)


__all__ = [
    "preprocess_audio",
    "preprocess_audio_batch",
    "pad_or_trim",
    "normalize_mel",
    "WHISPER_SAMPLE_RATE",
    "WHISPER_N_FFT",
    "WHISPER_HOP_LENGTH",
    "WHISPER_N_MELS",
    "WHISPER_CHUNK_LENGTH",
    "WHISPER_N_SAMPLES",
    "WHISPER_N_FRAMES",
]
