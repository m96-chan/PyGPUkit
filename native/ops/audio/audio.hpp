/**
 * GPU Audio Processing Operations
 *
 * Header file for audio processing ops.
 */
#pragma once

#include "../../core/memory.hpp"

namespace pygpukit {
namespace ops {
namespace audio {

/**
 * Convert int16 PCM samples to float32.
 * @param input Input GPUArray of int16 samples
 * @return GPUArray of float32 samples normalized to [-1.0, 1.0]
 */
GPUArray pcm_to_float32(const GPUArray& input);

/**
 * Convert stereo audio to mono by averaging channels.
 * @param input Input GPUArray of interleaved stereo samples [L,R,L,R,...]
 * @return GPUArray of mono samples
 */
GPUArray stereo_to_mono(const GPUArray& input);

/**
 * Peak normalize audio to [-1.0, 1.0] range.
 * @param input Input GPUArray to normalize (modified in-place)
 */
void normalize_peak(GPUArray& input);

/**
 * RMS normalize audio to target dB level.
 * @param input Input GPUArray to normalize (modified in-place)
 * @param target_db Target RMS level in dB (default -20.0)
 */
void normalize_rms(GPUArray& input, float target_db = -20.0f);

/**
 * Resample audio from source to target sample rate.
 * Currently supports 48kHz -> 16kHz (3:1 decimation).
 * @param input Input GPUArray of audio samples
 * @param src_rate Source sample rate (e.g., 48000)
 * @param dst_rate Target sample rate (e.g., 16000)
 * @return Resampled GPUArray
 */
GPUArray resample(const GPUArray& input, int src_rate, int dst_rate);

// ============================================================================
// Streaming Operations
// ============================================================================

/**
 * Write samples to a ring buffer with wrap-around.
 * @param input Input samples to write
 * @param ring_buffer Ring buffer GPUArray
 * @param write_pos Current write position (updated after write)
 */
void ring_buffer_write(const GPUArray& input, GPUArray& ring_buffer, int write_pos);

/**
 * Read samples from a ring buffer (linearized).
 * @param ring_buffer Ring buffer GPUArray
 * @param read_pos Read position
 * @param num_samples Number of samples to read
 * @return Linearized GPUArray
 */
GPUArray ring_buffer_read(const GPUArray& ring_buffer, int read_pos, int num_samples);

/**
 * Apply Hann window to audio data (in-place).
 * @param data Audio data to window (modified in-place)
 */
void apply_hann_window(GPUArray& data);

/**
 * Overlap-add: add windowed chunk to output buffer.
 * @param input Windowed input chunk
 * @param output Output buffer (accumulated)
 * @param output_offset Offset in output buffer
 */
void overlap_add(const GPUArray& input, GPUArray& output, int output_offset);

// ============================================================================
// Voice Activity Detection (VAD)
// ============================================================================

/**
 * Compute frame-level energy (RMS) for VAD.
 * @param audio Input audio samples (float32)
 * @param frame_size Frame size in samples
 * @param hop_size Hop size in samples
 * @return GPUArray of frame energies
 */
GPUArray vad_compute_energy(const GPUArray& audio, int frame_size, int hop_size);

/**
 * Compute frame-level zero-crossing rate for VAD.
 * @param audio Input audio samples (float32)
 * @param frame_size Frame size in samples
 * @param hop_size Hop size in samples
 * @return GPUArray of frame ZCR values [0, 1]
 */
GPUArray vad_compute_zcr(const GPUArray& audio, int frame_size, int hop_size);

/**
 * Apply threshold-based VAD decision.
 * @param frame_energy Frame energy values
 * @param frame_zcr Frame ZCR values
 * @param energy_threshold Energy threshold for speech detection
 * @param zcr_low Lower ZCR bound for voiced speech
 * @param zcr_high Upper ZCR bound (above = unvoiced or noise)
 * @return GPUArray of int32 VAD flags (0=silence, 1=speech)
 */
GPUArray vad_decide(
    const GPUArray& frame_energy,
    const GPUArray& frame_zcr,
    float energy_threshold,
    float zcr_low,
    float zcr_high);

/**
 * Apply hangover smoothing to VAD output.
 * Extends speech regions by hangover_frames after speech ends.
 * @param vad_input Input VAD flags
 * @param hangover_frames Number of frames to extend
 * @return Smoothed VAD flags
 */
GPUArray vad_apply_hangover(const GPUArray& vad_input, int hangover_frames);

/**
 * Compute noise floor (minimum energy) for adaptive thresholding.
 * @param frame_energy Frame energy values
 * @return Minimum energy value (scalar)
 */
float vad_compute_noise_floor(const GPUArray& frame_energy);

// ============================================================================
// Audio Preprocessing (Priority: Medium)
// ============================================================================

/**
 * Apply pre-emphasis filter to emphasize high-frequency components.
 * y[n] = x[n] - alpha * x[n-1]
 * @param input Input GPUArray (modified in-place)
 * @param alpha Pre-emphasis coefficient (default 0.97)
 */
void preemphasis(GPUArray& input, float alpha = 0.97f);

/**
 * Apply de-emphasis filter (inverse of pre-emphasis).
 * y[n] = x[n] + alpha * y[n-1]
 * @param input Input GPUArray (modified in-place)
 * @param alpha De-emphasis coefficient (default 0.97)
 */
void deemphasis(GPUArray& input, float alpha = 0.97f);

/**
 * Remove DC offset from audio signal.
 * Subtracts the mean value from all samples.
 * @param input Input GPUArray (modified in-place)
 */
void remove_dc(GPUArray& input);

/**
 * Apply high-pass filter for DC removal (IIR).
 * Uses single-pole high-pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
 * @param input Input GPUArray (modified in-place)
 * @param cutoff_hz Cutoff frequency in Hz (default 20.0)
 * @param sample_rate Sample rate in Hz (default 16000)
 */
void highpass_filter(GPUArray& input, float cutoff_hz = 20.0f, int sample_rate = 16000);

/**
 * Apply spectral gate for noise reduction.
 * Attenuates samples with energy below threshold.
 * @param input Input GPUArray (modified in-place)
 * @param threshold Energy threshold (linear scale, default 0.01)
 * @param attack_samples Smoothing attack in samples (default 64)
 * @param release_samples Smoothing release in samples (default 256)
 */
void spectral_gate(GPUArray& input, float threshold = 0.01f,
                   int attack_samples = 64, int release_samples = 256);

/**
 * Apply simple noise gate (hard gate).
 * Zeros samples with absolute value below threshold.
 * @param input Input GPUArray (modified in-place)
 * @param threshold Amplitude threshold (default 0.01)
 */
void noise_gate(GPUArray& input, float threshold = 0.01f);

/**
 * Compute short-term energy for adaptive noise gating.
 * @param input Input audio samples
 * @param frame_size Frame size for energy computation
 * @return GPUArray of frame energies
 */
GPUArray compute_short_term_energy(const GPUArray& input, int frame_size);

// ============================================================================
// Spectral Processing (Priority: High - Whisper/ASR)
// ============================================================================

/**
 * Compute Short-Time Fourier Transform (STFT) using cuFFT.
 * @param input Input audio samples (float32)
 * @param n_fft FFT size (default 400 for Whisper)
 * @param hop_length Hop size (default 160 for Whisper)
 * @param win_length Window length (default n_fft)
 * @param center Whether to pad input (default true)
 * @return Complex STFT output [n_frames, n_fft/2+1, 2] (real, imag)
 */
GPUArray stft(const GPUArray& input, int n_fft = 400, int hop_length = 160,
              int win_length = -1, bool center = true);

/**
 * Compute power spectrogram from STFT output.
 * power = real^2 + imag^2
 * @param stft_output STFT output [n_frames, n_fft/2+1, 2]
 * @return Power spectrogram [n_frames, n_fft/2+1]
 */
GPUArray power_spectrum(const GPUArray& stft_output);

/**
 * Compute magnitude spectrogram from STFT output.
 * magnitude = sqrt(real^2 + imag^2)
 * @param stft_output STFT output [n_frames, n_fft/2+1, 2]
 * @return Magnitude spectrogram [n_frames, n_fft/2+1]
 */
GPUArray magnitude_spectrum(const GPUArray& stft_output);

/**
 * Create Mel filterbank matrix.
 * @param n_mels Number of mel bands (default 80 for Whisper)
 * @param n_fft FFT size
 * @param sample_rate Sample rate in Hz
 * @param f_min Minimum frequency (default 0)
 * @param f_max Maximum frequency (default sample_rate/2)
 * @return Mel filterbank matrix [n_mels, n_fft/2+1]
 */
GPUArray create_mel_filterbank(int n_mels, int n_fft, int sample_rate,
                                float f_min = 0.0f, float f_max = -1.0f);

/**
 * Apply Mel filterbank to power/magnitude spectrogram.
 * @param spectrogram Input spectrogram [n_frames, n_fft/2+1]
 * @param mel_filterbank Mel filterbank [n_mels, n_fft/2+1]
 * @return Mel spectrogram [n_frames, n_mels]
 */
GPUArray apply_mel_filterbank(const GPUArray& spectrogram,
                               const GPUArray& mel_filterbank);

/**
 * Compute log-mel spectrogram (Whisper-compatible).
 * log_mel = log(mel + eps)
 * @param mel_spectrogram Mel spectrogram [n_frames, n_mels]
 * @param eps Small constant for numerical stability (default 1e-10)
 * @return Log-mel spectrogram [n_frames, n_mels]
 */
GPUArray log_mel_spectrogram(const GPUArray& mel_spectrogram, float eps = 1e-10f);

/**
 * Convert to decibels.
 * dB = 10 * log10(x + eps)
 * @param input Input array
 * @param eps Small constant for numerical stability (default 1e-10)
 * @return dB values
 */
GPUArray to_decibels(const GPUArray& input, float eps = 1e-10f);

/**
 * Compute MFCC from log-mel spectrogram using DCT-II.
 * @param log_mel Log-mel spectrogram [n_frames, n_mels]
 * @param n_mfcc Number of MFCC coefficients (default 13)
 * @return MFCC [n_frames, n_mfcc]
 */
GPUArray mfcc(const GPUArray& log_mel, int n_mfcc = 13);

/**
 * Compute delta (differential) features.
 * @param features Input features [n_frames, n_features]
 * @param order Delta order (1 for delta, 2 for delta-delta)
 * @param width Window width for computation (default 2)
 * @return Delta features [n_frames, n_features]
 */
GPUArray delta_features(const GPUArray& features, int order = 1, int width = 2);

// ============================================================================
// High-level Convenience Functions
// ============================================================================

/**
 * Compute Whisper-compatible log-mel spectrogram in one call.
 * Combines: STFT -> power -> mel filterbank -> log
 * @param input Input audio (float32, 16kHz expected)
 * @param n_fft FFT size (default 400)
 * @param hop_length Hop size (default 160)
 * @param n_mels Number of mel bands (default 80)
 * @return Log-mel spectrogram [n_frames, n_mels]
 */
GPUArray whisper_mel_spectrogram(const GPUArray& input,
                                  int n_fft = 400,
                                  int hop_length = 160,
                                  int n_mels = 80);

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
