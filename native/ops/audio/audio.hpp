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

// ============================================================================
// Inverse STFT
// ============================================================================

/**
 * Compute Inverse Short-Time Fourier Transform (ISTFT).
 * @param stft_output STFT output [n_frames, n_fft/2+1, 2] (real, imag)
 * @param hop_length Hop size (default 160)
 * @param win_length Window length (default n_fft)
 * @param center Whether input was padded (default true)
 * @param length Expected output length (optional, -1 for auto)
 * @return Reconstructed audio signal
 */
GPUArray istft(const GPUArray& stft_output, int hop_length = 160,
               int win_length = -1, bool center = true, int length = -1);

// ============================================================================
// Griffin-Lim Algorithm
// ============================================================================

/**
 * Griffin-Lim phase reconstruction algorithm.
 * Reconstructs audio from magnitude spectrogram.
 * @param magnitude Magnitude spectrogram [n_frames, n_fft/2+1]
 * @param n_iter Number of iterations (default 32)
 * @param hop_length Hop size (default 160)
 * @param win_length Window length (default n_fft * 2 - 2)
 * @return Reconstructed audio signal
 */
GPUArray griffin_lim(const GPUArray& magnitude, int n_iter = 32,
                     int hop_length = 160, int win_length = -1);

// ============================================================================
// Pitch Detection
// ============================================================================

/**
 * Compute autocorrelation of signal.
 * @param input Input audio samples
 * @param max_lag Maximum lag to compute
 * @return Autocorrelation values [max_lag]
 */
GPUArray autocorrelation(const GPUArray& input, int max_lag);

/**
 * Detect pitch using YIN algorithm.
 * @param input Input audio samples (single frame)
 * @param sample_rate Sample rate in Hz
 * @param f_min Minimum frequency (default 50 Hz)
 * @param f_max Maximum frequency (default 2000 Hz)
 * @param threshold YIN threshold (default 0.1)
 * @return Detected pitch in Hz (0 if unvoiced)
 */
float detect_pitch_yin(const GPUArray& input, int sample_rate,
                       float f_min = 50.0f, float f_max = 2000.0f,
                       float threshold = 0.1f);

/**
 * Detect pitch for multiple frames using YIN algorithm.
 * @param input Input audio samples
 * @param sample_rate Sample rate in Hz
 * @param frame_size Frame size in samples
 * @param hop_size Hop size in samples
 * @param f_min Minimum frequency (default 50 Hz)
 * @param f_max Maximum frequency (default 2000 Hz)
 * @param threshold YIN threshold (default 0.1)
 * @return Detected pitches [n_frames] in Hz (0 if unvoiced)
 */
GPUArray detect_pitch_yin_frames(const GPUArray& input, int sample_rate,
                                  int frame_size, int hop_size,
                                  float f_min = 50.0f, float f_max = 2000.0f,
                                  float threshold = 0.1f);

// ============================================================================
// Spectral Features
// ============================================================================

/**
 * Compute spectral centroid (center of mass of spectrum).
 * @param spectrum Magnitude/power spectrogram [n_frames, n_freq]
 * @param sample_rate Sample rate in Hz
 * @return Spectral centroid per frame [n_frames] in Hz
 */
GPUArray spectral_centroid(const GPUArray& spectrum, int sample_rate);

/**
 * Compute spectral bandwidth.
 * @param spectrum Magnitude/power spectrogram [n_frames, n_freq]
 * @param centroids Pre-computed centroids [n_frames]
 * @param sample_rate Sample rate in Hz
 * @param p Order of the bandwidth norm (default 2)
 * @return Spectral bandwidth per frame [n_frames] in Hz
 */
GPUArray spectral_bandwidth(const GPUArray& spectrum,
                             const GPUArray& centroids,
                             int sample_rate, int p = 2);

/**
 * Compute spectral rolloff point.
 * @param spectrum Magnitude/power spectrogram [n_frames, n_freq]
 * @param sample_rate Sample rate in Hz
 * @param roll_percent Rolloff percentage (default 0.85 = 85%)
 * @return Rolloff frequency per frame [n_frames] in Hz
 */
GPUArray spectral_rolloff(const GPUArray& spectrum, int sample_rate,
                           float roll_percent = 0.85f);

/**
 * Compute spectral flatness (Wiener entropy).
 * @param spectrum Magnitude/power spectrogram [n_frames, n_freq]
 * @return Flatness per frame [n_frames] in [0, 1]
 */
GPUArray spectral_flatness(const GPUArray& spectrum);

/**
 * Compute spectral contrast.
 * @param spectrum Magnitude/power spectrogram [n_frames, n_freq]
 * @param n_bands Number of frequency bands (default 6)
 * @param alpha Percentile for peak/valley (default 0.02 = 2%)
 * @return Spectral contrast [n_frames, n_bands]
 */
GPUArray spectral_contrast(const GPUArray& spectrum, int n_bands = 6,
                            float alpha = 0.02f);

/**
 * Compute zero-crossing rate.
 * @param input Input audio samples
 * @param frame_size Frame size in samples
 * @param hop_size Hop size in samples
 * @return ZCR per frame [n_frames] in [0, 1]
 */
GPUArray zero_crossing_rate(const GPUArray& input, int frame_size, int hop_size);

// ============================================================================
// CQT (Constant-Q Transform)
// ============================================================================

/**
 * Compute Constant-Q Transform.
 * @param input Input audio samples
 * @param sample_rate Sample rate in Hz
 * @param hop_length Hop size (default 512)
 * @param f_min Minimum frequency (default 32.7 Hz, C1)
 * @param n_bins Number of CQT bins (default 84, 7 octaves)
 * @param bins_per_octave Bins per octave (default 12)
 * @return Complex CQT output [n_frames, n_bins, 2]
 */
GPUArray cqt(const GPUArray& input, int sample_rate, int hop_length = 512,
             float f_min = 32.7f, int n_bins = 84, int bins_per_octave = 12);

/**
 * Compute CQT magnitude spectrogram.
 * @param cqt_output CQT output [n_frames, n_bins, 2]
 * @return Magnitude spectrogram [n_frames, n_bins]
 */
GPUArray cqt_magnitude(const GPUArray& cqt_output);

// ============================================================================
// Chromagram
// ============================================================================

/**
 * Compute chromagram from STFT.
 * @param spectrum Power/magnitude spectrogram [n_frames, n_freq]
 * @param sample_rate Sample rate in Hz
 * @param n_chroma Number of chroma bins (default 12)
 * @param tuning Tuning deviation from A440 in cents (default 0)
 * @return Chromagram [n_frames, n_chroma]
 */
GPUArray chroma_stft(const GPUArray& spectrum, int sample_rate,
                     int n_chroma = 12, float tuning = 0.0f);

/**
 * Compute chromagram from CQT.
 * @param cqt_mag CQT magnitude [n_frames, n_bins]
 * @param bins_per_octave Bins per octave (must match CQT, default 12)
 * @return Chromagram [n_frames, 12]
 */
GPUArray chroma_cqt(const GPUArray& cqt_mag, int bins_per_octave = 12);

// ============================================================================
// HPSS (Harmonic-Percussive Source Separation)
// ============================================================================

/**
 * Harmonic-percussive source separation.
 * @param stft_magnitude STFT magnitude [n_frames, n_freq]
 * @param kernel_size Median filter kernel size (default 31)
 * @param power Mask power for softness (default 2.0)
 * @param margin Margin for separation (default 1.0)
 * @return Pair of (harmonic_magnitude, percussive_magnitude)
 */
std::pair<GPUArray, GPUArray> hpss(const GPUArray& stft_magnitude,
                                    int kernel_size = 31,
                                    float power = 2.0f,
                                    float margin = 1.0f);

/**
 * Get harmonic component only from HPSS.
 */
GPUArray harmonic(const GPUArray& stft_magnitude, int kernel_size = 31,
                  float power = 2.0f, float margin = 1.0f);

/**
 * Get percussive component only from HPSS.
 */
GPUArray percussive(const GPUArray& stft_magnitude, int kernel_size = 31,
                    float power = 2.0f, float margin = 1.0f);

// ============================================================================
// Time Stretch / Pitch Shift (Phase Vocoder)
// ============================================================================

/**
 * Time-stretch audio using phase vocoder.
 * @param input Input audio samples
 * @param rate Time stretch rate (>1 = slower, <1 = faster)
 * @param n_fft FFT size (default 2048)
 * @param hop_length Hop size (default n_fft/4)
 * @return Time-stretched audio
 */
GPUArray time_stretch(const GPUArray& input, float rate,
                      int n_fft = 2048, int hop_length = -1);

/**
 * Pitch-shift audio.
 * @param input Input audio samples
 * @param sample_rate Sample rate in Hz
 * @param n_steps Number of semitones to shift
 * @param n_fft FFT size (default 2048)
 * @param hop_length Hop size (default n_fft/4)
 * @return Pitch-shifted audio
 */
GPUArray pitch_shift(const GPUArray& input, int sample_rate, float n_steps,
                     int n_fft = 2048, int hop_length = -1);

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
