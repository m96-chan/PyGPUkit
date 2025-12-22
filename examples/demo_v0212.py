#!/usr/bin/env python3
"""
PyGPUkit v0.2.12 - Audio Processing Demo

Demonstrates the comprehensive audio processing capabilities:
1. STFT/ISTFT - Short-Time Fourier Transform and inverse
2. Griffin-Lim - Phase reconstruction from magnitude
3. Spectral Features - Centroid, bandwidth, rolloff, flatness, contrast
4. Pitch Detection - YIN algorithm for fundamental frequency
5. CQT/Chromagram - Constant-Q Transform and pitch class mapping
6. HPSS - Harmonic-Percussive Source Separation
7. Time Stretch/Pitch Shift - Phase vocoder manipulation

All kernels are Driver-Only (no cuFFT dependency).

Usage:
    python demo_v0212.py

Requirements:
    - PyGPUkit v0.2.12+
    - CUDA capable GPU (SM >= 80)
"""

from __future__ import annotations

import time

import numpy as np


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    """Print subsection header."""
    print()
    print(f"--- {title} ---")


def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio with multiple frequency components."""
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    # Mix of frequencies: 440Hz (A4), 880Hz (A5), 1320Hz (E6)
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.2 * np.sin(2 * np.pi * 1320 * t)
    )
    return audio.astype(np.float32)


def demo_stft_istft():
    """Demonstrate STFT and ISTFT roundtrip."""
    section("1. STFT / ISTFT Roundtrip")

    from pygpukit.ops import audio

    # Generate test signal
    samples = generate_test_audio(duration=1.0, sample_rate=16000)
    buf = audio.from_pcm(samples, sample_rate=16000)
    print(f"Input: {len(samples)} samples ({len(samples) / 16000:.2f}s)")

    # STFT
    start = time.perf_counter()
    stft_out = audio.stft(buf, n_fft=512, hop_length=160)
    stft_time = (time.perf_counter() - start) * 1000
    print(f"STFT shape: {stft_out.shape} (n_frames, n_freq, 2)")
    print(f"STFT time: {stft_time:.2f} ms")

    # ISTFT
    start = time.perf_counter()
    reconstructed = audio.istft(stft_out, hop_length=160)
    istft_time = (time.perf_counter() - start) * 1000
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"ISTFT time: {istft_time:.2f} ms")

    # Verify reconstruction
    recon_np = reconstructed.to_numpy()
    min_len = min(len(samples), len(recon_np))
    error = np.abs(samples[:min_len] - recon_np[:min_len]).mean()
    print(f"Mean reconstruction error: {error:.6f}")


def demo_griffin_lim():
    """Demonstrate Griffin-Lim phase reconstruction."""
    section("2. Griffin-Lim Phase Reconstruction")

    from pygpukit.ops import audio

    samples = generate_test_audio(duration=0.5, sample_rate=16000)
    buf = audio.from_pcm(samples, sample_rate=16000)

    # Get magnitude spectrogram (discard phase)
    stft_out = audio.stft(buf, n_fft=512, hop_length=160)
    magnitude = audio.magnitude_spectrum(stft_out)
    print(f"Magnitude shape: {magnitude.shape}")

    # Reconstruct with Griffin-Lim
    start = time.perf_counter()
    reconstructed = audio.griffin_lim(magnitude, n_iter=32, hop_length=160)
    gl_time = (time.perf_counter() - start) * 1000
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Griffin-Lim time (32 iterations): {gl_time:.2f} ms")


def demo_spectral_features():
    """Demonstrate spectral feature extraction."""
    section("3. Spectral Features")

    from pygpukit.ops import audio

    samples = generate_test_audio(duration=1.0, sample_rate=16000)
    buf = audio.from_pcm(samples, sample_rate=16000)

    # Compute STFT and magnitude
    stft_out = audio.stft(buf, n_fft=512, hop_length=160)
    mag = audio.magnitude_spectrum(stft_out)
    n_frames = mag.shape[0]

    subsection("Spectral Centroid")
    centroid = audio.spectral_centroid(mag, sample_rate=16000)
    centroid_np = centroid.to_numpy()
    print(f"Shape: {centroid.shape}")
    print(f"Mean: {centroid_np.mean():.2f} Hz")
    print(f"Range: {centroid_np.min():.2f} - {centroid_np.max():.2f} Hz")

    subsection("Spectral Bandwidth")
    bandwidth = audio.spectral_bandwidth(mag, centroid, sample_rate=16000)
    bandwidth_np = bandwidth.to_numpy()
    print(f"Shape: {bandwidth.shape}")
    print(f"Mean: {bandwidth_np.mean():.2f} Hz")

    subsection("Spectral Rolloff (85%)")
    rolloff = audio.spectral_rolloff(mag, sample_rate=16000, roll_percent=0.85)
    rolloff_np = rolloff.to_numpy()
    print(f"Shape: {rolloff.shape}")
    print(f"Mean: {rolloff_np.mean():.2f} Hz")

    subsection("Spectral Flatness")
    flatness = audio.spectral_flatness(mag)
    flatness_np = flatness.to_numpy()
    print(f"Shape: {flatness.shape}")
    print(f"Mean: {flatness_np.mean():.4f} (0=tonal, 1=noise)")

    subsection("Spectral Contrast")
    contrast = audio.spectral_contrast(mag, n_bands=6, alpha=0.2)
    contrast_np = contrast.to_numpy()
    print(f"Shape: {contrast.shape} (n_frames, n_bands)")
    print(f"Mean per band: {contrast_np.mean(axis=0)}")


def demo_pitch_detection():
    """Demonstrate pitch detection with YIN algorithm."""
    section("4. Pitch Detection (YIN Algorithm)")

    from pygpukit.ops import audio

    # Generate pure tone at 440 Hz (A4)
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    tone_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = audio.from_pcm(tone_440, sample_rate=sample_rate)

    subsection("Single Frame Detection")
    # Use a segment for pitch detection
    segment = audio.from_pcm(tone_440[:2048], sample_rate=sample_rate)
    pitch = audio.detect_pitch_yin(segment, sample_rate=sample_rate)
    print("Expected: 440.0 Hz")
    print(f"Detected: {pitch:.1f} Hz")
    print(f"Error: {abs(440.0 - pitch):.1f} Hz")

    subsection("Frame-by-Frame Detection")
    pitches = audio.detect_pitch_yin_frames(
        buf, sample_rate=sample_rate, frame_size=1024, hop_size=256
    )
    pitches_np = pitches.to_numpy()
    voiced = pitches_np[pitches_np > 0]
    print(f"Total frames: {len(pitches_np)}")
    print(f"Voiced frames: {len(voiced)}")
    if len(voiced) > 0:
        print(f"Mean pitch (voiced): {voiced.mean():.1f} Hz")


def demo_zero_crossing_rate():
    """Demonstrate zero-crossing rate computation."""
    section("5. Zero-Crossing Rate")

    from pygpukit.ops import audio

    # Compare ZCR of low and high frequency signals
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)

    # Low frequency (100 Hz)
    low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    buf_low = audio.from_pcm(low_freq, sample_rate=sample_rate)
    zcr_low = audio.zero_crossing_rate(buf_low, frame_size=512, hop_size=256)

    # High frequency (2000 Hz)
    high_freq = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    buf_high = audio.from_pcm(high_freq, sample_rate=sample_rate)
    zcr_high = audio.zero_crossing_rate(buf_high, frame_size=512, hop_size=256)

    print(f"100 Hz signal - Mean ZCR: {zcr_low.to_numpy().mean():.4f}")
    print(f"2000 Hz signal - Mean ZCR: {zcr_high.to_numpy().mean():.4f}")
    print("(Higher frequency = higher ZCR)")


def demo_cqt_chromagram():
    """Demonstrate CQT and Chromagram."""
    section("6. CQT and Chromagram")

    from pygpukit.ops import audio

    samples = generate_test_audio(duration=1.0, sample_rate=16000)
    buf = audio.from_pcm(samples, sample_rate=16000)

    subsection("Constant-Q Transform")
    start = time.perf_counter()
    cqt_out = audio.cqt(buf, sample_rate=16000, hop_length=160, n_bins=84, bins_per_octave=12)
    cqt_time = (time.perf_counter() - start) * 1000
    print(f"CQT shape: {cqt_out.shape} (n_frames, n_bins, 2)")
    print(f"CQT time: {cqt_time:.2f} ms")
    print("Frequency range: 7 octaves (84 bins / 12 per octave)")

    subsection("Chromagram from CQT")
    cqt_mag = audio.cqt_magnitude(buf, sample_rate=16000, hop_length=160, n_bins=84)
    chroma = audio.chroma_cqt(cqt_mag, bins_per_octave=12)
    chroma_np = chroma.to_numpy()
    print(f"Chroma shape: {chroma.shape} (n_frames, 12 pitch classes)")
    print("Pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B")
    print(f"Mean energy per class: {chroma_np.mean(axis=0).round(3)}")


def demo_hpss():
    """Demonstrate Harmonic-Percussive Source Separation."""
    section("7. HPSS (Harmonic-Percussive Separation)")

    from pygpukit.ops import audio

    # Generate mixed signal: tone + noise bursts
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    harmonic = np.sin(2 * np.pi * 440 * t)  # Pure tone (harmonic)
    percussive = np.zeros_like(t)
    # Add click sounds (percussive)
    for i in range(0, sample_rate, sample_rate // 4):
        percussive[i : i + 100] = np.random.randn(100) * 0.5
    mixed = (harmonic + percussive).astype(np.float32)

    buf = audio.from_pcm(mixed, sample_rate=sample_rate)
    stft_out = audio.stft(buf, n_fft=512, hop_length=160)
    mag = audio.magnitude_spectrum(stft_out)

    start = time.perf_counter()
    harmonic_mag, percussive_mag = audio.hpss(mag, kernel_size=17)
    hpss_time = (time.perf_counter() - start) * 1000

    print(f"Input magnitude shape: {mag.shape}")
    print(f"Harmonic component shape: {harmonic_mag.shape}")
    print(f"Percussive component shape: {percussive_mag.shape}")
    print(f"HPSS time: {hpss_time:.2f} ms")

    # Compare energy
    total_energy = mag.to_numpy().sum()
    harm_energy = harmonic_mag.to_numpy().sum()
    perc_energy = percussive_mag.to_numpy().sum()
    print(f"Harmonic energy: {harm_energy / total_energy * 100:.1f}%")
    print(f"Percussive energy: {perc_energy / total_energy * 100:.1f}%")


def demo_time_stretch_pitch_shift():
    """Demonstrate time stretching and pitch shifting."""
    section("8. Time Stretch / Pitch Shift (Phase Vocoder)")

    from pygpukit.ops import audio

    samples = generate_test_audio(duration=0.5, sample_rate=16000)
    buf = audio.from_pcm(samples, sample_rate=16000)
    original_len = len(samples)

    subsection("Time Stretch")
    # Slow down (rate < 1)
    start = time.perf_counter()
    slow = audio.time_stretch(buf, rate=0.5, n_fft=1024, hop_length=256)
    slow_time = (time.perf_counter() - start) * 1000
    print(f"Original: {original_len} samples")
    print(f"Slow (0.5x): {slow.shape[0]} samples (expected ~{original_len * 2})")
    print(f"Time: {slow_time:.2f} ms")

    # Speed up (rate > 1)
    start = time.perf_counter()
    fast = audio.time_stretch(buf, rate=2.0, n_fft=1024, hop_length=256)
    fast_time = (time.perf_counter() - start) * 1000
    print(f"Fast (2.0x): {fast.shape[0]} samples (expected ~{original_len // 2})")
    print(f"Time: {fast_time:.2f} ms")

    subsection("Pitch Shift")
    # Shift up by 12 semitones (one octave)
    start = time.perf_counter()
    higher = audio.pitch_shift(buf, sample_rate=16000, n_steps=12.0)
    up_time = (time.perf_counter() - start) * 1000
    print(f"Original length: {original_len}")
    print(f"+12 semitones (1 octave up): {higher.shape[0]} samples")
    print(f"Time: {up_time:.2f} ms")

    # Shift down by 7 semitones (perfect fifth)
    start = time.perf_counter()
    lower = audio.pitch_shift(buf, sample_rate=16000, n_steps=-7.0)
    down_time = (time.perf_counter() - start) * 1000
    print(f"-7 semitones (5th down): {lower.shape[0]} samples")
    print(f"Time: {down_time:.2f} ms")


def demo_autocorrelation():
    """Demonstrate autocorrelation computation."""
    section("9. Autocorrelation")

    from pygpukit.ops import audio

    # Generate periodic signal
    sample_rate = 16000
    freq = 200  # 200 Hz
    t = np.linspace(0, 0.1, int(0.1 * sample_rate), dtype=np.float32)
    periodic = np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = audio.from_pcm(periodic, sample_rate=sample_rate)

    max_lag = sample_rate // 50  # Up to 50 Hz minimum
    acf = audio.autocorrelation(buf, max_lag=max_lag)
    acf_np = acf.to_numpy()

    print(f"Signal: {freq} Hz sine wave")
    print(f"ACF shape: {acf.shape}")
    print(f"Expected period: {sample_rate / freq:.1f} samples")

    # Find first peak after lag 0
    peaks = []
    for i in range(1, len(acf_np) - 1):
        if acf_np[i] > acf_np[i - 1] and acf_np[i] > acf_np[i + 1]:
            peaks.append(i)
    if peaks:
        print(f"First ACF peak at lag: {peaks[0]} samples")
        print(f"Estimated frequency: {sample_rate / peaks[0]:.1f} Hz")


def main():
    """Run all demos."""
    print()
    print("=" * 70)
    print(" PyGPUkit v0.2.12 - Audio Processing Demo")
    print(" Driver-Only Mode (no cuFFT dependency)")
    print("=" * 70)

    import pygpukit as gk

    print(f"\nCUDA Available: {gk.is_cuda_available()}")
    if gk.is_cuda_available():
        try:
            caps = gk.get_device_capabilities()
            if hasattr(caps, "sm_major"):
                print(f"GPU: SM {caps.sm_major}.{caps.sm_minor}")
        except Exception:
            pass

    try:
        demo_stft_istft()
        demo_griffin_lim()
        demo_spectral_features()
        demo_pitch_detection()
        demo_zero_crossing_rate()
        demo_cqt_chromagram()
        demo_hpss()
        demo_time_stretch_pitch_shift()
        demo_autocorrelation()

        section("Summary")
        print("All audio processing features demonstrated successfully!")
        print()
        print("Features available in pygpukit.ops.audio:")
        print("  - STFT/ISTFT: Time-frequency analysis")
        print("  - Griffin-Lim: Phase reconstruction")
        print("  - Spectral features: centroid, bandwidth, rolloff, flatness, contrast")
        print("  - Pitch detection: YIN algorithm")
        print("  - Zero-crossing rate")
        print("  - CQT: Constant-Q Transform")
        print("  - Chromagram: Pitch class distribution")
        print("  - HPSS: Harmonic-percussive separation")
        print("  - Time stretch / Pitch shift: Phase vocoder")
        print("  - Autocorrelation")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
