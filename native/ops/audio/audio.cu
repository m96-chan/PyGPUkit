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

// ============================================================================
// Streaming Operations
// ============================================================================

void ring_buffer_write(const GPUArray& input, GPUArray& ring_buffer, int write_pos) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("ring_buffer_write: input must be Float32");
    }
    if (ring_buffer.dtype() != DataType::Float32) {
        throw std::runtime_error("ring_buffer_write: ring_buffer must be Float32");
    }

    int num_samples = static_cast<int>(input.size());
    int ring_size = static_cast<int>(ring_buffer.size());

    const int block_size = 256;
    int num_blocks = (num_samples + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    ring_buffer_write_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(ring_buffer.data()),
        ring_size,
        write_pos,
        num_samples);

    sync_and_check("ring_buffer_write kernel failed");
}

GPUArray ring_buffer_read(const GPUArray& ring_buffer, int read_pos, int num_samples) {
    if (ring_buffer.dtype() != DataType::Float32) {
        throw std::runtime_error("ring_buffer_read: ring_buffer must be Float32");
    }

    int ring_size = static_cast<int>(ring_buffer.size());

    GPUArray output({static_cast<size_t>(num_samples)}, DataType::Float32);

    const int block_size = 256;
    int num_blocks = (num_samples + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    ring_buffer_read_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(ring_buffer.data()),
        static_cast<float*>(output.data()),
        ring_size,
        read_pos,
        num_samples);

    sync_and_check("ring_buffer_read kernel failed");
    return output;
}

void apply_hann_window(GPUArray& data) {
    if (data.dtype() != DataType::Float32) {
        throw std::runtime_error("apply_hann_window: data must be Float32");
    }

    int window_size = static_cast<int>(data.size());

    const int block_size = 256;
    int num_blocks = (window_size + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    apply_hann_window_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(data.data()),
        window_size);

    sync_and_check("apply_hann_window kernel failed");
}

void overlap_add(const GPUArray& input, GPUArray& output, int output_offset) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("overlap_add: input must be Float32");
    }
    if (output.dtype() != DataType::Float32) {
        throw std::runtime_error("overlap_add: output must be Float32");
    }

    int chunk_size = static_cast<int>(input.size());

    const int block_size = 256;
    int num_blocks = (chunk_size + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    overlap_add_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        output_offset,
        chunk_size);

    sync_and_check("overlap_add kernel failed");
}

// ============================================================================
// Voice Activity Detection (VAD)
// ============================================================================

GPUArray vad_compute_energy(const GPUArray& audio, int frame_size, int hop_size) {
    if (audio.dtype() != DataType::Float32) {
        throw std::runtime_error("vad_compute_energy: input must be Float32");
    }

    int audio_len = static_cast<int>(audio.size());
    int num_frames = (audio_len - frame_size) / hop_size + 1;
    if (num_frames <= 0) {
        throw std::runtime_error("vad_compute_energy: audio too short for given frame_size");
    }

    GPUArray output({static_cast<size_t>(num_frames)}, DataType::Float32);

    const int block_size = 256;
    cudaStream_t stream = internal::get_capture_stream();

    // One block per frame
    vad_frame_energy_kernel<<<num_frames, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(audio.data()),
        static_cast<float*>(output.data()),
        audio_len,
        frame_size,
        hop_size,
        num_frames);

    sync_and_check("vad_frame_energy kernel failed");
    return output;
}

GPUArray vad_compute_zcr(const GPUArray& audio, int frame_size, int hop_size) {
    if (audio.dtype() != DataType::Float32) {
        throw std::runtime_error("vad_compute_zcr: input must be Float32");
    }

    int audio_len = static_cast<int>(audio.size());
    int num_frames = (audio_len - frame_size) / hop_size + 1;
    if (num_frames <= 0) {
        throw std::runtime_error("vad_compute_zcr: audio too short for given frame_size");
    }

    GPUArray output({static_cast<size_t>(num_frames)}, DataType::Float32);

    const int block_size = 256;
    cudaStream_t stream = internal::get_capture_stream();

    // One block per frame
    vad_zero_crossing_kernel<<<num_frames, block_size, block_size * sizeof(int), stream>>>(
        static_cast<const float*>(audio.data()),
        static_cast<float*>(output.data()),
        audio_len,
        frame_size,
        hop_size,
        num_frames);

    sync_and_check("vad_zero_crossing kernel failed");
    return output;
}

GPUArray vad_decide(
    const GPUArray& frame_energy,
    const GPUArray& frame_zcr,
    float energy_threshold,
    float zcr_low,
    float zcr_high)
{
    if (frame_energy.dtype() != DataType::Float32) {
        throw std::runtime_error("vad_decide: frame_energy must be Float32");
    }
    if (frame_zcr.dtype() != DataType::Float32) {
        throw std::runtime_error("vad_decide: frame_zcr must be Float32");
    }
    if (frame_energy.size() != frame_zcr.size()) {
        throw std::runtime_error("vad_decide: frame_energy and frame_zcr must have same size");
    }

    int num_frames = static_cast<int>(frame_energy.size());
    GPUArray output({static_cast<size_t>(num_frames)}, DataType::Int32);

    const int block_size = 256;
    int num_blocks = (num_frames + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    vad_decision_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(frame_energy.data()),
        static_cast<const float*>(frame_zcr.data()),
        static_cast<int*>(output.data()),
        num_frames,
        energy_threshold,
        zcr_low,
        zcr_high);

    sync_and_check("vad_decision kernel failed");
    return output;
}

GPUArray vad_apply_hangover(const GPUArray& vad_input, int hangover_frames) {
    if (vad_input.dtype() != DataType::Int32) {
        throw std::runtime_error("vad_apply_hangover: input must be Int32");
    }

    int num_frames = static_cast<int>(vad_input.size());
    GPUArray output({static_cast<size_t>(num_frames)}, DataType::Int32);

    const int block_size = 256;
    int num_blocks = (num_frames + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    vad_hangover_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const int*>(vad_input.data()),
        static_cast<int*>(output.data()),
        num_frames,
        hangover_frames);

    sync_and_check("vad_hangover kernel failed");
    return output;
}

float vad_compute_noise_floor(const GPUArray& frame_energy) {
    if (frame_energy.dtype() != DataType::Float32) {
        throw std::runtime_error("vad_compute_noise_floor: input must be Float32");
    }

    int num_frames = static_cast<int>(frame_energy.size());
    if (num_frames == 0) return 0.0f;

    const int block_size = 256;
    int num_blocks = (num_frames + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    GPUArray block_min({static_cast<size_t>(num_blocks)}, DataType::Float32);

    vad_compute_noise_floor_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(frame_energy.data()),
        static_cast<float*>(block_min.data()),
        num_frames);

    sync_and_check("vad_compute_noise_floor kernel failed");

    // Copy to host and find global minimum
    std::vector<float> host_min(num_blocks);
    memcpy_device_to_host(host_min.data(), block_min.data(), num_blocks * sizeof(float));

    float global_min = host_min[0];
    for (int i = 1; i < num_blocks; ++i) {
        global_min = std::min(global_min, host_min[i]);
    }

    return global_min;
}

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
