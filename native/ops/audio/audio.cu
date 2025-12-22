/**
 * GPU Audio Processing Operations Dispatch
 */
#include "audio_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>

namespace pygpukit {
namespace ops {
namespace audio {

// ============================================================================
// PCM to Float Conversion
// ============================================================================

GPUArray pcm_to_float32(const GPUArray& input) {
    if (input.dtype() != DataType::Int16) {
        throw std::runtime_error("pcm_to_float32: input must be Int16");
    }

    size_t n = input.size();
    GPUArray output(input.shape(), DataType::Float32);

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    pcm_int16_to_f32_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const int16_t*>(input.data()),
        static_cast<float*>(output.data()),
        n);

    sync_and_check("pcm_to_float32 kernel failed");
    return output;
}

// ============================================================================
// Stereo to Mono Conversion
// ============================================================================

GPUArray stereo_to_mono(const GPUArray& input) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("stereo_to_mono: input must be Float32");
    }

    size_t total_samples = input.size();
    if (total_samples % 2 != 0) {
        throw std::runtime_error("stereo_to_mono: input size must be even (stereo pairs)");
    }

    size_t mono_samples = total_samples / 2;

    // Output shape: flatten to 1D mono
    GPUArray output({mono_samples}, DataType::Float32);

    const int block_size = 256;
    int num_blocks = (mono_samples + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    stereo_to_mono_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        mono_samples);

    sync_and_check("stereo_to_mono kernel failed");
    return output;
}

// ============================================================================
// Peak Normalization
// ============================================================================

void normalize_peak(GPUArray& input) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("normalize_peak: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    // Allocate temp buffer for block maximums
    GPUArray block_max({static_cast<size_t>(num_blocks)}, DataType::Float32);

    // First pass: find max per block
    find_max_abs_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(block_max.data()),
        n);

    sync_and_check("find_max_abs kernel failed");

    // Copy block results to host and find global max
    std::vector<float> host_max(num_blocks);
    memcpy_device_to_host(host_max.data(), block_max.data(), num_blocks * sizeof(float));

    float global_max = 0.0f;
    for (int i = 0; i < num_blocks; ++i) {
        global_max = std::max(global_max, host_max[i]);
    }

    // Apply scale if max is non-zero
    if (global_max > 1e-8f) {
        float scale = 1.0f / global_max;
        apply_scale_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(input.data()),
            n,
            scale);
        sync_and_check("apply_scale kernel failed");
    }
}

// ============================================================================
// RMS Normalization
// ============================================================================

void normalize_rms(GPUArray& input, float target_db) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("normalize_rms: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    // Allocate temp buffer for block sums
    GPUArray block_sum({static_cast<size_t>(num_blocks)}, DataType::Float32);

    // First pass: compute sum of squares per block
    sum_of_squares_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(block_sum.data()),
        n);

    sync_and_check("sum_of_squares kernel failed");

    // Copy block results to host and compute global RMS
    std::vector<float> host_sum(num_blocks);
    memcpy_device_to_host(host_sum.data(), block_sum.data(), num_blocks * sizeof(float));

    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += host_sum[i];
    }

    double current_rms = std::sqrt(total_sum / n);

    // Convert target dB to linear
    // dB = 20 * log10(rms), so rms = 10^(dB/20)
    double target_rms = std::pow(10.0, target_db / 20.0);

    // Apply scale if current RMS is non-zero
    if (current_rms > 1e-8) {
        float scale = static_cast<float>(target_rms / current_rms);
        apply_scale_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(input.data()),
            n,
            scale);
        sync_and_check("apply_scale kernel failed");
    }
}

// ============================================================================
// Resampling
// ============================================================================

GPUArray resample(const GPUArray& input, int src_rate, int dst_rate) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("resample: input must be Float32");
    }

    // Currently only support 48kHz -> 16kHz (3:1 decimation)
    if (src_rate != 48000 || dst_rate != 16000) {
        throw std::runtime_error("resample: currently only 48000 -> 16000 is supported");
    }

    int in_len = static_cast<int>(input.size());
    int out_len = in_len / 3;  // 3:1 decimation

    GPUArray output({static_cast<size_t>(out_len)}, DataType::Float32);

    const int block_size = 256;
    int num_blocks = (out_len + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    resample_polyphase_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        in_len,
        out_len);

    sync_and_check("resample_polyphase kernel failed");
    return output;
}

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
