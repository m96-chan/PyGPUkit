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

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
