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

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
