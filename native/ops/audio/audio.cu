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

    if (src_rate == dst_rate) {
        // No resampling needed, return copy
        GPUArray output(input.shape(), DataType::Float32);
        cudaMemcpy(output.data(), input.data(), input.size() * sizeof(float), cudaMemcpyDeviceToDevice);
        return output;
    }

    int in_len = static_cast<int>(input.size());
    int out_len = static_cast<int>(static_cast<int64_t>(in_len) * dst_rate / src_rate);
    float ratio = static_cast<float>(src_rate) / static_cast<float>(dst_rate);

    GPUArray output({static_cast<size_t>(out_len)}, DataType::Float32);

    const int block_size = 256;
    int num_blocks = (out_len + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    // Use optimized polyphase filter for 48kHz -> 16kHz
    if (src_rate == 48000 && dst_rate == 16000) {
        resample_polyphase_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(output.data()),
            in_len,
            out_len);
    } else {
        // Generic linear interpolation for other sample rates
        resample_linear_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(output.data()),
            in_len,
            out_len,
            ratio);
    }

    sync_and_check("resample kernel failed");
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

// ============================================================================
// Audio Preprocessing Operations
// ============================================================================

void preemphasis(GPUArray& input, float alpha) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("preemphasis: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    preemphasis_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(input.data()),
        n,
        alpha);

    sync_and_check("preemphasis kernel failed");
}

void deemphasis(GPUArray& input, float alpha) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("deemphasis: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    cudaStream_t stream = internal::get_capture_stream();

    // Sequential IIR filter - single thread
    deemphasis_sequential_kernel<<<1, 1, 0, stream>>>(
        static_cast<float*>(input.data()),
        n,
        alpha);

    sync_and_check("deemphasis kernel failed");
}

void remove_dc(GPUArray& input) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("remove_dc: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    // Allocate temp buffer for block sums
    GPUArray block_sum({static_cast<size_t>(num_blocks)}, DataType::Float32);

    // Compute sum per block
    compute_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(block_sum.data()),
        n);

    sync_and_check("compute_sum kernel failed");

    // Copy to host and compute total sum
    std::vector<float> host_sum(num_blocks);
    memcpy_device_to_host(host_sum.data(), block_sum.data(), num_blocks * sizeof(float));

    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += host_sum[i];
    }

    float mean = static_cast<float>(total_sum / n);

    // Subtract mean
    subtract_mean_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(input.data()),
        n,
        mean);

    sync_and_check("subtract_mean kernel failed");
}

void highpass_filter(GPUArray& input, float cutoff_hz, int sample_rate) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("highpass_filter: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    // Compute alpha for single-pole high-pass filter
    // alpha = 1 / (1 + 2*pi*fc/fs)
    // Higher alpha = higher cutoff preservation
    float rc = 1.0f / (2.0f * 3.14159265358979f * cutoff_hz);
    float dt = 1.0f / static_cast<float>(sample_rate);
    float alpha = rc / (rc + dt);

    cudaStream_t stream = internal::get_capture_stream();

    // Sequential IIR filter
    highpass_iir_kernel<<<1, 1, 0, stream>>>(
        static_cast<float*>(input.data()),
        n,
        alpha);

    sync_and_check("highpass_filter kernel failed");
}

void noise_gate(GPUArray& input, float threshold) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("noise_gate: input must be Float32");
    }

    size_t n = input.size();
    if (n == 0) return;

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    noise_gate_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(input.data()),
        n,
        threshold);

    sync_and_check("noise_gate kernel failed");
}

GPUArray compute_short_term_energy(const GPUArray& input, int frame_size) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("compute_short_term_energy: input must be Float32");
    }

    int input_len = static_cast<int>(input.size());
    int num_frames = input_len / frame_size;
    if (num_frames <= 0) {
        throw std::runtime_error("compute_short_term_energy: input too short for frame_size");
    }

    GPUArray output({static_cast<size_t>(num_frames)}, DataType::Float32);

    const int block_size = 256;
    cudaStream_t stream = internal::get_capture_stream();

    // One block per frame
    short_term_energy_kernel<<<num_frames, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        input_len,
        frame_size,
        num_frames);

    sync_and_check("short_term_energy kernel failed");
    return output;
}

void spectral_gate(GPUArray& input, float threshold, int attack_samples, int release_samples) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("spectral_gate: input must be Float32");
    }

    int n = static_cast<int>(input.size());
    if (n == 0) return;

    // Use attack_samples as frame size for energy computation
    int frame_size = attack_samples;
    int num_frames = n / frame_size;
    if (num_frames <= 0) {
        // Fallback to simple noise gate for very short signals
        noise_gate(input, threshold);
        return;
    }

    // Compute short-term energy
    GPUArray frame_energy = compute_short_term_energy(input, frame_size);

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    // Apply spectral gate
    spectral_gate_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(input.data()),
        static_cast<const float*>(frame_energy.data()),
        n,
        frame_size,
        num_frames,
        threshold);

    sync_and_check("spectral_gate kernel failed");
}

// ============================================================================
// Spectral Processing Operations
// ============================================================================

// Helper: compute log2 of power of 2
static int log2_int(int n) {
    int log2n = 0;
    while ((1 << log2n) < n) ++log2n;
    return log2n;
}

// Helper: check if power of 2
static bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Batch FFT using custom Radix-2 implementation
static void batch_fft(
    const float* input_real,
    float* output_real,
    float* output_imag,
    int n,
    int batch_size,
    cudaStream_t stream)
{
    if (!is_power_of_2(n)) {
        throw std::runtime_error("FFT size must be power of 2");
    }

    int log2n = log2_int(n);
    const int block_size = 256;

    // Use optimized shared-memory kernel for common sizes
    if (n == 256 || n == 512) {
        int smem_size = 2 * n * sizeof(float);
        if (n == 256) {
            fft_stockham_kernel<256><<<batch_size, 256, smem_size, stream>>>(
                input_real, output_real, output_imag, batch_size);
        } else {
            fft_stockham_kernel<512><<<batch_size, 512, smem_size, stream>>>(
                input_real, output_real, output_imag, batch_size);
        }
    } else {
        // General case: bit-reversal + butterfly stages
        // Allocate temp buffers for in-place FFT
        GPUArray temp_real({static_cast<size_t>(batch_size * n)}, DataType::Float32);
        GPUArray temp_imag({static_cast<size_t>(batch_size * n)}, DataType::Float32);

        // Bit-reversal permutation
        dim3 grid_br((n + block_size - 1) / block_size, batch_size);
        fft_bit_reverse_kernel<<<grid_br, block_size, 0, stream>>>(
            input_real, nullptr,
            static_cast<float*>(temp_real.data()),
            static_cast<float*>(temp_imag.data()),
            n, log2n, batch_size);

        // Butterfly stages
        for (int stage = 0; stage < log2n; ++stage) {
            int half_size = 1 << stage;
            dim3 grid_bf((n / 2 + block_size - 1) / block_size, batch_size);
            fft_butterfly_kernel<<<grid_bf, block_size, 0, stream>>>(
                static_cast<float*>(temp_real.data()),
                static_cast<float*>(temp_imag.data()),
                n, stage, batch_size);
        }

        // Copy to output
        cudaMemcpyAsync(output_real, temp_real.data(),
                        batch_size * n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(output_imag, temp_imag.data(),
                        batch_size * n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
}

GPUArray stft(const GPUArray& input, int n_fft, int hop_length, int win_length, bool center) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("stft: input must be Float32");
    }

    if (!is_power_of_2(n_fft)) {
        throw std::runtime_error("stft: n_fft must be power of 2");
    }

    if (win_length < 0) win_length = n_fft;

    int input_len = static_cast<int>(input.size());
    cudaStream_t stream = internal::get_capture_stream();

    // Handle center padding
    const float* audio_ptr = static_cast<const float*>(input.data());
    GPUArray padded_input({1}, DataType::Float32);  // Placeholder
    int padded_len = input_len;

    if (center) {
        int pad_left = n_fft / 2;
        int pad_right = n_fft / 2;
        padded_len = input_len + pad_left + pad_right;

        padded_input = GPUArray({static_cast<size_t>(padded_len)}, DataType::Float32);
        const int block_size = 256;
        int num_blocks = (padded_len + block_size - 1) / block_size;

        pad_reflect_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(padded_input.data()),
            input_len, pad_left, padded_len);

        audio_ptr = static_cast<const float*>(padded_input.data());
    }

    // Calculate number of frames
    int n_frames = (padded_len - n_fft) / hop_length + 1;
    if (n_frames <= 0) {
        throw std::runtime_error("stft: input too short for given n_fft");
    }

    // Extract frames
    GPUArray frames({static_cast<size_t>(n_frames * n_fft)}, DataType::Float32);
    extract_frames_kernel<<<n_frames, n_fft, 0, stream>>>(
        audio_ptr,
        static_cast<float*>(frames.data()),
        padded_len, n_fft, hop_length, n_frames);

    // Generate and apply Hann window
    GPUArray window({static_cast<size_t>(n_fft)}, DataType::Float32);
    {
        const int block_size = 256;
        int num_blocks = (n_fft + block_size - 1) / block_size;
        generate_hann_window_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(window.data()), n_fft);
    }

    apply_window_to_frames_kernel<<<n_frames, n_fft, 0, stream>>>(
        static_cast<float*>(frames.data()),
        static_cast<const float*>(window.data()),
        n_frames, n_fft);

    // Perform batch FFT
    GPUArray fft_real({static_cast<size_t>(n_frames * n_fft)}, DataType::Float32);
    GPUArray fft_imag({static_cast<size_t>(n_frames * n_fft)}, DataType::Float32);

    batch_fft(
        static_cast<const float*>(frames.data()),
        static_cast<float*>(fft_real.data()),
        static_cast<float*>(fft_imag.data()),
        n_fft, n_frames, stream);

    // Output: [n_frames, n_fft/2+1, 2] (real, imag interleaved)
    int n_freq = n_fft / 2 + 1;
    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq), 2}, DataType::Float32);

    // Copy first n_freq bins (real input FFT symmetry)
    const int block_size = 256;
    dim3 grid((n_freq + block_size - 1) / block_size, n_frames);
    fft_real_to_complex_kernel<<<grid, block_size, 0, stream>>>(
        static_cast<const float*>(fft_real.data()),
        static_cast<const float*>(fft_imag.data()),
        static_cast<float*>(output.data()),
        static_cast<float*>(output.data()) + n_frames * n_freq,
        n_fft, n_freq, n_frames);

    sync_and_check("stft failed");
    return output;
}

GPUArray power_spectrum(const GPUArray& stft_output) {
    if (stft_output.dtype() != DataType::Float32) {
        throw std::runtime_error("power_spectrum: input must be Float32");
    }

    auto& shape = stft_output.shape();
    if (shape.size() != 3 || shape[2] != 2) {
        throw std::runtime_error("power_spectrum: expected shape [n_frames, n_freq, 2]");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    int n_elements = n_frames * n_freq;

    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq)}, DataType::Float32);

    const int block_size = 256;
    int num_blocks = (n_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    const float* real_ptr = static_cast<const float*>(stft_output.data());
    const float* imag_ptr = real_ptr + n_elements;

    power_spectrum_kernel<<<num_blocks, block_size, 0, stream>>>(
        real_ptr, imag_ptr,
        static_cast<float*>(output.data()),
        n_elements);

    sync_and_check("power_spectrum failed");
    return output;
}

GPUArray magnitude_spectrum(const GPUArray& stft_output) {
    if (stft_output.dtype() != DataType::Float32) {
        throw std::runtime_error("magnitude_spectrum: input must be Float32");
    }

    auto& shape = stft_output.shape();
    if (shape.size() != 3 || shape[2] != 2) {
        throw std::runtime_error("magnitude_spectrum: expected shape [n_frames, n_freq, 2]");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    int n_elements = n_frames * n_freq;

    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq)}, DataType::Float32);

    const int block_size = 256;
    int num_blocks = (n_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    const float* real_ptr = static_cast<const float*>(stft_output.data());
    const float* imag_ptr = real_ptr + n_elements;

    magnitude_spectrum_kernel<<<num_blocks, block_size, 0, stream>>>(
        real_ptr, imag_ptr,
        static_cast<float*>(output.data()),
        n_elements);

    sync_and_check("magnitude_spectrum failed");
    return output;
}

GPUArray create_mel_filterbank(int n_mels, int n_fft, int sample_rate, float f_min, float f_max) {
    if (f_max < 0) f_max = static_cast<float>(sample_rate) / 2.0f;

    int n_freq = n_fft / 2 + 1;
    GPUArray filterbank({static_cast<size_t>(n_mels), static_cast<size_t>(n_freq)}, DataType::Float32);

    cudaStream_t stream = internal::get_capture_stream();

    // One block per mel band, threads for frequency bins
    int threads = std::min(n_freq, 1024);
    create_mel_filterbank_kernel<<<n_mels, threads, 0, stream>>>(
        static_cast<float*>(filterbank.data()),
        n_mels, n_fft, sample_rate, f_min, f_max);

    sync_and_check("create_mel_filterbank failed");
    return filterbank;
}

GPUArray apply_mel_filterbank(const GPUArray& spectrogram, const GPUArray& mel_filterbank) {
    if (spectrogram.dtype() != DataType::Float32 || mel_filterbank.dtype() != DataType::Float32) {
        throw std::runtime_error("apply_mel_filterbank: inputs must be Float32");
    }

    auto& spec_shape = spectrogram.shape();
    auto& mel_shape = mel_filterbank.shape();

    if (spec_shape.size() != 2 || mel_shape.size() != 2) {
        throw std::runtime_error("apply_mel_filterbank: expected 2D inputs");
    }

    int n_frames = static_cast<int>(spec_shape[0]);
    int n_freq = static_cast<int>(spec_shape[1]);
    int n_mels = static_cast<int>(mel_shape[0]);

    if (static_cast<int>(mel_shape[1]) != n_freq) {
        throw std::runtime_error("apply_mel_filterbank: frequency dimension mismatch");
    }

    // mel_spec = spectrogram @ mel_filterbank.T
    // spectrogram: [n_frames, n_freq]
    // mel_filterbank: [n_mels, n_freq]
    // output: [n_frames, n_mels]

    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_mels)}, DataType::Float32);

    // Simple matmul: C[i,j] = sum_k A[i,k] * B[j,k]
    cudaStream_t stream = internal::get_capture_stream();

    // Use simple kernel for now (can optimize with cuBLAS later)
    // Each thread computes one output element
    auto matmul_kernel = [](float* C, const float* A, const float* B,
                            int M, int N, int K, cudaStream_t stream) {
        // Simple CPU-side loop launcher (for small matrices)
        // In production, use cuBLAS or optimized kernel
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);

        // Lambda can't be a kernel, so we'll compute on CPU and copy
        // For now, use a simple approach
    };

    // Compute on host for simplicity (mel filterbank is typically small)
    std::vector<float> h_spec(n_frames * n_freq);
    std::vector<float> h_mel(n_mels * n_freq);
    std::vector<float> h_out(n_frames * n_mels, 0.0f);

    memcpy_device_to_host(h_spec.data(), spectrogram.data(), n_frames * n_freq * sizeof(float));
    memcpy_device_to_host(h_mel.data(), mel_filterbank.data(), n_mels * n_freq * sizeof(float));

    // CPU matmul
    for (int i = 0; i < n_frames; ++i) {
        for (int j = 0; j < n_mels; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n_freq; ++k) {
                sum += h_spec[i * n_freq + k] * h_mel[j * n_freq + k];
            }
            h_out[i * n_mels + j] = sum;
        }
    }

    memcpy_host_to_device(output.data(), h_out.data(), n_frames * n_mels * sizeof(float));

    return output;
}

GPUArray log_mel_spectrogram(const GPUArray& mel_spectrogram, float eps) {
    if (mel_spectrogram.dtype() != DataType::Float32) {
        throw std::runtime_error("log_mel_spectrogram: input must be Float32");
    }

    int n_elements = static_cast<int>(mel_spectrogram.size());
    GPUArray output(mel_spectrogram.shape(), DataType::Float32);

    const int block_size = 256;
    int num_blocks = (n_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    log_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(mel_spectrogram.data()),
        static_cast<float*>(output.data()),
        n_elements, eps);

    sync_and_check("log_mel_spectrogram failed");
    return output;
}

GPUArray to_decibels(const GPUArray& input, float eps) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("to_decibels: input must be Float32");
    }

    int n_elements = static_cast<int>(input.size());
    GPUArray output(input.shape(), DataType::Float32);

    const int block_size = 256;
    int num_blocks = (n_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    to_decibels_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        n_elements, eps);

    sync_and_check("to_decibels failed");
    return output;
}

GPUArray mfcc(const GPUArray& log_mel, int n_mfcc) {
    if (log_mel.dtype() != DataType::Float32) {
        throw std::runtime_error("mfcc: input must be Float32");
    }

    auto& shape = log_mel.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("mfcc: expected 2D input [n_frames, n_mels]");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_mels = static_cast<int>(shape[1]);

    if (n_mfcc > n_mels) {
        throw std::runtime_error("mfcc: n_mfcc cannot exceed n_mels");
    }

    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_mfcc)}, DataType::Float32);

    cudaStream_t stream = internal::get_capture_stream();

    // One block per frame, threads for MFCC coefficients
    dct_ii_kernel<<<n_frames, n_mfcc, 0, stream>>>(
        static_cast<const float*>(log_mel.data()),
        static_cast<float*>(output.data()),
        n_frames, n_mels, n_mfcc);

    sync_and_check("mfcc failed");
    return output;
}

GPUArray delta_features(const GPUArray& features, int order, int width) {
    if (features.dtype() != DataType::Float32) {
        throw std::runtime_error("delta_features: input must be Float32");
    }

    auto& shape = features.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("delta_features: expected 2D input [n_frames, n_features]");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_features = static_cast<int>(shape[1]);

    GPUArray output(shape, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    if (order == 1) {
        // Simple case: single delta computation
        delta_features_kernel<<<n_frames, n_features, 0, stream>>>(
            static_cast<const float*>(features.data()),
            static_cast<float*>(output.data()),
            n_frames, n_features, width);
    } else {
        // For higher order, we need a temp buffer
        GPUArray temp(shape, DataType::Float32);

        // First pass: compute delta from original features
        delta_features_kernel<<<n_frames, n_features, 0, stream>>>(
            static_cast<const float*>(features.data()),
            static_cast<float*>(output.data()),
            n_frames, n_features, width);

        // Subsequent passes: compute delta-delta, etc.
        for (int o = 1; o < order; ++o) {
            // Copy output to temp
            cudaMemcpyAsync(temp.data(), output.data(),
                           n_frames * n_features * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);

            // Compute delta of delta
            delta_features_kernel<<<n_frames, n_features, 0, stream>>>(
                static_cast<const float*>(temp.data()),
                static_cast<float*>(output.data()),
                n_frames, n_features, width);
        }
    }

    sync_and_check("delta_features failed");
    return output;
}

GPUArray whisper_mel_spectrogram(const GPUArray& input, int n_fft, int hop_length, int n_mels) {
    // STFT
    GPUArray stft_out = stft(input, n_fft, hop_length, n_fft, true);

    // Power spectrum
    GPUArray power = power_spectrum(stft_out);

    // Create and apply mel filterbank
    GPUArray mel_fb = create_mel_filterbank(n_mels, n_fft, 16000, 0.0f, 8000.0f);
    GPUArray mel = apply_mel_filterbank(power, mel_fb);

    // Log
    GPUArray log_mel = log_mel_spectrogram(mel, 1e-10f);

    return log_mel;
}

// ============================================================================
// Inverse STFT
// ============================================================================

// Helper: batch IFFT
static void batch_ifft(
    float* real,
    float* imag,
    int n,
    int batch_size,
    cudaStream_t stream)
{
    if (!is_power_of_2(n)) {
        throw std::runtime_error("IFFT size must be power of 2");
    }

    int log2n = log2_int(n);
    const int block_size = 256;

    // Bit-reversal permutation (in-place via temp buffers)
    GPUArray temp_real({static_cast<size_t>(batch_size * n)}, DataType::Float32);
    GPUArray temp_imag({static_cast<size_t>(batch_size * n)}, DataType::Float32);

    dim3 grid_br((n + block_size - 1) / block_size, batch_size);
    fft_bit_reverse_kernel<<<grid_br, block_size, 0, stream>>>(
        real, imag,
        static_cast<float*>(temp_real.data()),
        static_cast<float*>(temp_imag.data()),
        n, log2n, batch_size);

    // Copy back
    cudaMemcpyAsync(real, temp_real.data(), batch_size * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(imag, temp_imag.data(), batch_size * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // IFFT butterfly stages (conjugate twiddles)
    for (int stage = 0; stage < log2n; ++stage) {
        dim3 grid_bf((n / 2 + block_size - 1) / block_size, batch_size);
        ifft_butterfly_kernel<<<grid_bf, block_size, 0, stream>>>(
            real, imag, n, stage, batch_size);
    }

    // Scale by 1/N
    dim3 grid_sc((n + block_size - 1) / block_size, batch_size);
    ifft_scale_kernel<<<grid_sc, block_size, 0, stream>>>(
        real, imag, n, batch_size);
}

GPUArray istft(const GPUArray& stft_output, int hop_length, int win_length, bool center, int length) {
    if (stft_output.dtype() != DataType::Float32) {
        throw std::runtime_error("istft: input must be Float32");
    }

    auto& shape = stft_output.shape();
    if (shape.size() != 3 || shape[2] != 2) {
        throw std::runtime_error("istft: expected shape [n_frames, n_freq, 2]");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    int n_fft = (n_freq - 1) * 2;

    if (win_length < 0) win_length = n_fft;

    cudaStream_t stream = internal::get_capture_stream();

    // Expand to full FFT spectrum (conjugate symmetry)
    GPUArray fft_real({static_cast<size_t>(n_frames * n_fft)}, DataType::Float32);
    GPUArray fft_imag({static_cast<size_t>(n_frames * n_fft)}, DataType::Float32);

    const float* real_ptr = static_cast<const float*>(stft_output.data());
    const float* imag_ptr = real_ptr + n_frames * n_freq;

    // Copy first half and create conjugate for second half on host for simplicity
    std::vector<float> h_real(n_frames * n_fft);
    std::vector<float> h_imag(n_frames * n_fft);
    std::vector<float> h_in_real(n_frames * n_freq);
    std::vector<float> h_in_imag(n_frames * n_freq);

    memcpy_device_to_host(h_in_real.data(), const_cast<float*>(real_ptr), n_frames * n_freq * sizeof(float));
    memcpy_device_to_host(h_in_imag.data(), const_cast<float*>(imag_ptr), n_frames * n_freq * sizeof(float));

    for (int f = 0; f < n_frames; ++f) {
        // Copy first half
        for (int k = 0; k < n_freq; ++k) {
            h_real[f * n_fft + k] = h_in_real[f * n_freq + k];
            h_imag[f * n_fft + k] = h_in_imag[f * n_freq + k];
        }
        // Conjugate symmetry for second half
        for (int k = 1; k < n_freq - 1; ++k) {
            h_real[f * n_fft + n_fft - k] = h_in_real[f * n_freq + k];
            h_imag[f * n_fft + n_fft - k] = -h_in_imag[f * n_freq + k];
        }
    }

    memcpy_host_to_device(fft_real.data(), h_real.data(), n_frames * n_fft * sizeof(float));
    memcpy_host_to_device(fft_imag.data(), h_imag.data(), n_frames * n_fft * sizeof(float));

    // Perform IFFT
    batch_ifft(
        static_cast<float*>(fft_real.data()),
        static_cast<float*>(fft_imag.data()),
        n_fft, n_frames, stream);

    // Apply window
    GPUArray window({static_cast<size_t>(n_fft)}, DataType::Float32);
    {
        const int block_size = 256;
        int num_blocks = (n_fft + block_size - 1) / block_size;
        generate_hann_window_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(window.data()), n_fft);
    }

    apply_window_to_frames_kernel<<<n_frames, n_fft, 0, stream>>>(
        static_cast<float*>(fft_real.data()),
        static_cast<const float*>(window.data()),
        n_frames, n_fft);

    // Compute output length
    int output_len = (n_frames - 1) * hop_length + n_fft;
    if (center) {
        output_len -= n_fft;  // Remove padding
    }
    if (length > 0) {
        output_len = length;
    }

    // Overlap-add
    int total_len = (n_frames - 1) * hop_length + n_fft;
    GPUArray output({static_cast<size_t>(total_len)}, DataType::Float32);
    GPUArray window_sum({static_cast<size_t>(total_len)}, DataType::Float32);

    // Zero initialize
    cudaMemsetAsync(output.data(), 0, total_len * sizeof(float), stream);
    cudaMemsetAsync(window_sum.data(), 0, total_len * sizeof(float), stream);

    // Overlap-add frames
    istft_overlap_add_kernel<<<n_frames, n_fft, 0, stream>>>(
        static_cast<const float*>(fft_real.data()),
        static_cast<float*>(output.data()),
        n_frames, n_fft, hop_length);

    // Compute window sum for normalization
    {
        const int block_size = 256;
        int num_blocks = (total_len + block_size - 1) / block_size;
        istft_window_sum_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const float*>(window.data()),
            static_cast<float*>(window_sum.data()),
            n_frames, n_fft, hop_length, total_len);

        istft_normalize_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<float*>(output.data()),
            static_cast<const float*>(window_sum.data()),
            total_len, 1e-10f);
    }

    sync_and_check("istft failed");

    // Trim if center padding was used
    if (center) {
        int pad = n_fft / 2;
        int final_len = std::min(output_len, total_len - 2 * pad);
        if (length > 0) final_len = std::min(final_len, length);

        GPUArray final_output({static_cast<size_t>(final_len)}, DataType::Float32);
        cudaMemcpy(final_output.data(),
                   static_cast<float*>(output.data()) + pad,
                   final_len * sizeof(float), cudaMemcpyDeviceToDevice);
        return final_output;
    }

    return output;
}

// ============================================================================
// Griffin-Lim Algorithm
// ============================================================================

GPUArray griffin_lim(const GPUArray& magnitude, int n_iter, int hop_length, int win_length) {
    if (magnitude.dtype() != DataType::Float32) {
        throw std::runtime_error("griffin_lim: input must be Float32");
    }

    auto& shape = magnitude.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("griffin_lim: expected 2D input [n_frames, n_freq]");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    int n_fft = (n_freq - 1) * 2;

    if (win_length < 0) win_length = n_fft;

    cudaStream_t stream = internal::get_capture_stream();
    const int block_size = 256;
    int n_elements = n_frames * n_freq;
    int num_blocks = (n_elements + block_size - 1) / block_size;

    // Initialize with random phase
    GPUArray phase({static_cast<size_t>(n_elements)}, DataType::Float32);
    random_phase_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(phase.data()), n_elements, 42u);

    GPUArray stft_real({static_cast<size_t>(n_elements)}, DataType::Float32);
    GPUArray stft_imag({static_cast<size_t>(n_elements)}, DataType::Float32);

    for (int iter = 0; iter < n_iter; ++iter) {
        // Apply magnitude with current phase
        apply_magnitude_phase_kernel<<<num_blocks, block_size, 0, stream>>>(
            static_cast<const float*>(magnitude.data()),
            static_cast<const float*>(phase.data()),
            static_cast<float*>(stft_real.data()),
            static_cast<float*>(stft_imag.data()),
            n_elements);

        // Create STFT output format [n_frames, n_freq, 2]
        GPUArray stft_combined({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq), 2},
                               DataType::Float32);
        cudaMemcpyAsync(stft_combined.data(), stft_real.data(),
                        n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(static_cast<float*>(stft_combined.data()) + n_elements,
                        stft_imag.data(), n_elements * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // ISTFT
        GPUArray audio = istft(stft_combined, hop_length, win_length, true, -1);

        // STFT
        GPUArray new_stft = stft(audio, n_fft, hop_length, win_length, true);

        // Extract new phase
        auto& ns_shape = new_stft.shape();
        int new_n_frames = static_cast<int>(ns_shape[0]);
        int new_n_freq = static_cast<int>(ns_shape[1]);
        int new_n_elements = new_n_frames * new_n_freq;

        const float* new_real = static_cast<const float*>(new_stft.data());
        const float* new_imag = new_real + new_n_elements;

        // Resize phase if needed
        if (new_n_elements != n_elements) {
            phase = GPUArray({static_cast<size_t>(new_n_elements)}, DataType::Float32);
            stft_real = GPUArray({static_cast<size_t>(new_n_elements)}, DataType::Float32);
            stft_imag = GPUArray({static_cast<size_t>(new_n_elements)}, DataType::Float32);
            n_elements = new_n_elements;
            n_frames = new_n_frames;
            num_blocks = (n_elements + block_size - 1) / block_size;
        }

        compute_phase_kernel<<<num_blocks, block_size, 0, stream>>>(
            new_real, new_imag,
            static_cast<float*>(phase.data()),
            n_elements);
    }

    // Final reconstruction
    apply_magnitude_phase_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(magnitude.data()),
        static_cast<const float*>(phase.data()),
        static_cast<float*>(stft_real.data()),
        static_cast<float*>(stft_imag.data()),
        n_elements);

    GPUArray stft_final({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq), 2},
                        DataType::Float32);
    cudaMemcpyAsync(stft_final.data(), stft_real.data(),
                    n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(static_cast<float*>(stft_final.data()) + n_elements,
                    stft_imag.data(), n_elements * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    sync_and_check("griffin_lim failed");

    return istft(stft_final, hop_length, win_length, true, -1);
}

// ============================================================================
// Pitch Detection
// ============================================================================

GPUArray autocorrelation(const GPUArray& input, int max_lag) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("autocorrelation: input must be Float32");
    }

    int input_len = static_cast<int>(input.size());
    if (max_lag > input_len) max_lag = input_len;

    GPUArray output({static_cast<size_t>(max_lag)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;
    autocorrelation_kernel<<<max_lag, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        input_len, max_lag);

    sync_and_check("autocorrelation failed");
    return output;
}

float detect_pitch_yin(const GPUArray& input, int sample_rate,
                       float f_min, float f_max, float threshold) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("detect_pitch_yin: input must be Float32");
    }

    int frame_size = static_cast<int>(input.size());
    int max_lag = sample_rate / static_cast<int>(f_min);
    int min_lag = sample_rate / static_cast<int>(f_max);

    if (max_lag > frame_size / 2) max_lag = frame_size / 2;

    GPUArray diff({static_cast<size_t>(max_lag)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;

    // Compute difference function
    yin_difference_kernel<<<max_lag, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(diff.data()),
        frame_size, max_lag);

    // Cumulative mean normalized difference (sequential)
    yin_cumulative_mean_kernel<<<1, 1, 0, stream>>>(
        static_cast<float*>(diff.data()), max_lag);

    sync_and_check("detect_pitch_yin failed");

    // Find pitch on host
    std::vector<float> h_diff(max_lag);
    memcpy_device_to_host(h_diff.data(), diff.data(), max_lag * sizeof(float));

    // Find first dip below threshold
    for (int tau = min_lag; tau < max_lag; ++tau) {
        if (h_diff[tau] < threshold) {
            // Parabolic interpolation
            float s0 = h_diff[tau - 1];
            float s1 = h_diff[tau];
            float s2 = h_diff[tau + 1];

            float denom = 2.0f * (s0 - 2.0f * s1 + s2);
            float delta = 0.0f;
            if (std::abs(denom) > 1e-10f) {
                delta = (s0 - s2) / denom;
            }

            float refined_tau = static_cast<float>(tau) + delta;
            return static_cast<float>(sample_rate) / refined_tau;
        }
    }

    return 0.0f;  // Unvoiced
}

GPUArray detect_pitch_yin_frames(const GPUArray& input, int sample_rate,
                                  int frame_size, int hop_size,
                                  float f_min, float f_max, float threshold) {
    int input_len = static_cast<int>(input.size());
    int n_frames = (input_len - frame_size) / hop_size + 1;

    std::vector<float> pitches(n_frames);
    std::vector<float> h_input(input_len);
    memcpy_device_to_host(h_input.data(), input.data(), input_len * sizeof(float));

    for (int f = 0; f < n_frames; ++f) {
        // Create frame on device
        GPUArray frame({static_cast<size_t>(frame_size)}, DataType::Float32);
        memcpy_host_to_device(frame.data(), h_input.data() + f * hop_size,
                              frame_size * sizeof(float));

        pitches[f] = detect_pitch_yin(frame, sample_rate, f_min, f_max, threshold);
    }

    GPUArray output({static_cast<size_t>(n_frames)}, DataType::Float32);
    memcpy_host_to_device(output.data(), pitches.data(), n_frames * sizeof(float));

    return output;
}

// ============================================================================
// Spectral Features
// ============================================================================

GPUArray spectral_centroid(const GPUArray& spectrum, int sample_rate) {
    if (spectrum.dtype() != DataType::Float32) {
        throw std::runtime_error("spectral_centroid: input must be Float32");
    }

    auto& shape = spectrum.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("spectral_centroid: expected 2D input");
    }

    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    float freq_bin_hz = static_cast<float>(sample_rate) / (2.0f * (n_freq - 1));

    GPUArray output({static_cast<size_t>(n_frames)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;
    spectral_centroid_kernel<<<n_frames, block_size, 2 * block_size * sizeof(float), stream>>>(
        static_cast<const float*>(spectrum.data()),
        static_cast<float*>(output.data()),
        n_frames, n_freq, freq_bin_hz);

    sync_and_check("spectral_centroid failed");
    return output;
}

GPUArray spectral_bandwidth(const GPUArray& spectrum, const GPUArray& centroids,
                             int sample_rate, int p) {
    if (spectrum.dtype() != DataType::Float32 || centroids.dtype() != DataType::Float32) {
        throw std::runtime_error("spectral_bandwidth: inputs must be Float32");
    }

    auto& shape = spectrum.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    float freq_bin_hz = static_cast<float>(sample_rate) / (2.0f * (n_freq - 1));

    GPUArray output({static_cast<size_t>(n_frames)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;
    spectral_bandwidth_kernel<<<n_frames, block_size, 2 * block_size * sizeof(float), stream>>>(
        static_cast<const float*>(spectrum.data()),
        static_cast<const float*>(centroids.data()),
        static_cast<float*>(output.data()),
        n_frames, n_freq, freq_bin_hz, p);

    sync_and_check("spectral_bandwidth failed");
    return output;
}

GPUArray spectral_rolloff(const GPUArray& spectrum, int sample_rate, float roll_percent) {
    if (spectrum.dtype() != DataType::Float32) {
        throw std::runtime_error("spectral_rolloff: input must be Float32");
    }

    auto& shape = spectrum.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    float freq_bin_hz = static_cast<float>(sample_rate) / (2.0f * (n_freq - 1));

    GPUArray output({static_cast<size_t>(n_frames)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;
    spectral_rolloff_kernel<<<n_frames, block_size, block_size * sizeof(float), stream>>>(
        static_cast<const float*>(spectrum.data()),
        static_cast<float*>(output.data()),
        n_frames, n_freq, freq_bin_hz, roll_percent);

    sync_and_check("spectral_rolloff failed");
    return output;
}

GPUArray spectral_flatness(const GPUArray& spectrum) {
    if (spectrum.dtype() != DataType::Float32) {
        throw std::runtime_error("spectral_flatness: input must be Float32");
    }

    auto& shape = spectrum.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);

    GPUArray output({static_cast<size_t>(n_frames)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;
    spectral_flatness_kernel<<<n_frames, block_size, 2 * block_size * sizeof(float), stream>>>(
        static_cast<const float*>(spectrum.data()),
        static_cast<float*>(output.data()),
        n_frames, n_freq);

    sync_and_check("spectral_flatness failed");
    return output;
}

GPUArray spectral_contrast(const GPUArray& spectrum, int n_bands, float alpha) {
    if (spectrum.dtype() != DataType::Float32) {
        throw std::runtime_error("spectral_contrast: input must be Float32");
    }

    auto& shape = spectrum.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);

    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_bands)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    spectral_contrast_kernel<<<n_frames, n_bands, 0, stream>>>(
        static_cast<const float*>(spectrum.data()),
        static_cast<float*>(output.data()),
        n_frames, n_freq, n_bands, alpha);

    sync_and_check("spectral_contrast failed");
    return output;
}

GPUArray zero_crossing_rate(const GPUArray& input, int frame_size, int hop_size) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("zero_crossing_rate: input must be Float32");
    }

    int input_len = static_cast<int>(input.size());
    int n_frames = (input_len - frame_size) / hop_size + 1;

    GPUArray output({static_cast<size_t>(n_frames)}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    const int block_size = 256;
    zero_crossing_rate_kernel<<<n_frames, block_size, block_size * sizeof(int), stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        n_frames, frame_size, hop_size);

    sync_and_check("zero_crossing_rate failed");
    return output;
}

// ============================================================================
// CQT (Constant-Q Transform)
// ============================================================================

GPUArray cqt(const GPUArray& input, int sample_rate, int hop_length,
             float f_min, int n_bins, int bins_per_octave) {
    // Simplified CQT using STFT with FFT size based on lowest frequency
    // Full CQT would require variable window sizes per bin

    int n_fft = 2048;  // Default for most use cases
    while (n_fft < sample_rate / f_min * 4) {
        n_fft *= 2;
    }

    // Compute STFT
    GPUArray stft_out = stft(input, n_fft, hop_length, n_fft, true);

    auto& shape = stft_out.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);

    // Map FFT bins to CQT bins
    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_bins), 2}, DataType::Float32);

    // Simplified mapping: interpolate from FFT bins
    const float* stft_real = static_cast<const float*>(stft_out.data());
    const float* stft_imag = stft_real + n_frames * n_freq;

    std::vector<float> h_out_real(n_frames * n_bins);
    std::vector<float> h_out_imag(n_frames * n_bins);
    std::vector<float> h_stft_real(n_frames * n_freq);
    std::vector<float> h_stft_imag(n_frames * n_freq);

    memcpy_device_to_host(h_stft_real.data(), const_cast<float*>(stft_real), n_frames * n_freq * sizeof(float));
    memcpy_device_to_host(h_stft_imag.data(), const_cast<float*>(stft_imag), n_frames * n_freq * sizeof(float));

    for (int f = 0; f < n_frames; ++f) {
        for (int b = 0; b < n_bins; ++b) {
            // CQT frequency for this bin
            float freq = f_min * std::pow(2.0f, static_cast<float>(b) / bins_per_octave);
            float fft_bin = freq * n_fft / sample_rate;

            int bin_low = static_cast<int>(fft_bin);
            int bin_high = bin_low + 1;
            float frac = fft_bin - bin_low;

            if (bin_high < n_freq) {
                h_out_real[f * n_bins + b] =
                    (1 - frac) * h_stft_real[f * n_freq + bin_low] +
                    frac * h_stft_real[f * n_freq + bin_high];
                h_out_imag[f * n_bins + b] =
                    (1 - frac) * h_stft_imag[f * n_freq + bin_low] +
                    frac * h_stft_imag[f * n_freq + bin_high];
            } else if (bin_low < n_freq) {
                h_out_real[f * n_bins + b] = h_stft_real[f * n_freq + bin_low];
                h_out_imag[f * n_bins + b] = h_stft_imag[f * n_freq + bin_low];
            }
        }
    }

    float* out_ptr = static_cast<float*>(output.data());
    memcpy_host_to_device(out_ptr, h_out_real.data(), n_frames * n_bins * sizeof(float));
    memcpy_host_to_device(out_ptr + n_frames * n_bins, h_out_imag.data(),
                          n_frames * n_bins * sizeof(float));

    return output;
}

GPUArray cqt_magnitude(const GPUArray& cqt_output) {
    return magnitude_spectrum(cqt_output);
}

// ============================================================================
// Chromagram
// ============================================================================

GPUArray chroma_stft(const GPUArray& spectrum, int sample_rate, int n_chroma, float tuning) {
    // Build chroma filterbank and apply
    auto& shape = spectrum.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    int n_fft = (n_freq - 1) * 2;

    // Build chroma filterbank on host
    std::vector<float> h_chroma_fb(n_chroma * n_freq, 0.0f);

    float A4 = 440.0f * std::pow(2.0f, tuning / 1200.0f);  // Reference pitch with tuning

    for (int f = 1; f < n_freq; ++f) {
        float freq = static_cast<float>(f) * sample_rate / n_fft;
        if (freq < 20.0f) continue;  // Skip very low frequencies

        // Convert to pitch class (0-11)
        float pitch = 12.0f * std::log2(freq / A4);
        int chroma = static_cast<int>(std::fmod(pitch + 120.0f, 12.0f));
        if (chroma < 0) chroma += 12;

        // Weight by frequency (higher frequencies contribute less)
        float weight = 1.0f;
        h_chroma_fb[chroma * n_freq + f] += weight;
    }

    // Normalize filterbank
    for (int c = 0; c < n_chroma; ++c) {
        float sum = 0.0f;
        for (int f = 0; f < n_freq; ++f) {
            sum += h_chroma_fb[c * n_freq + f];
        }
        if (sum > 0) {
            for (int f = 0; f < n_freq; ++f) {
                h_chroma_fb[c * n_freq + f] /= sum;
            }
        }
    }

    // Apply filterbank
    std::vector<float> h_spec(n_frames * n_freq);
    std::vector<float> h_chroma(n_frames * n_chroma, 0.0f);

    memcpy_device_to_host(h_spec.data(), spectrum.data(), n_frames * n_freq * sizeof(float));

    for (int fr = 0; fr < n_frames; ++fr) {
        for (int c = 0; c < n_chroma; ++c) {
            float sum = 0.0f;
            for (int f = 0; f < n_freq; ++f) {
                sum += h_spec[fr * n_freq + f] * h_chroma_fb[c * n_freq + f];
            }
            h_chroma[fr * n_chroma + c] = sum;
        }
    }

    GPUArray output({static_cast<size_t>(n_frames), static_cast<size_t>(n_chroma)}, DataType::Float32);
    memcpy_host_to_device(output.data(), h_chroma.data(), n_frames * n_chroma * sizeof(float));

    return output;
}

GPUArray chroma_cqt(const GPUArray& cqt_mag, int bins_per_octave) {
    auto& shape = cqt_mag.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_bins = static_cast<int>(shape[1]);
    int n_octaves = n_bins / bins_per_octave;

    GPUArray output({static_cast<size_t>(n_frames), 12}, DataType::Float32);
    cudaStream_t stream = internal::get_capture_stream();

    cqt_to_chroma_kernel<<<n_frames, 12, 0, stream>>>(
        static_cast<const float*>(cqt_mag.data()),
        static_cast<float*>(output.data()),
        n_frames, n_bins, bins_per_octave, n_octaves);

    normalize_chroma_kernel<<<n_frames, 1, 0, stream>>>(
        static_cast<float*>(output.data()),
        n_frames, 1e-10f);

    sync_and_check("chroma_cqt failed");
    return output;
}

// ============================================================================
// HPSS
// ============================================================================

std::pair<GPUArray, GPUArray> hpss(const GPUArray& stft_magnitude, int kernel_size,
                                    float power, float margin) {
    if (stft_magnitude.dtype() != DataType::Float32) {
        throw std::runtime_error("hpss: input must be Float32");
    }

    auto& shape = stft_magnitude.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);
    int n_elements = n_frames * n_freq;

    cudaStream_t stream = internal::get_capture_stream();
    const int block_size = 256;

    // Apply horizontal median filter (harmonic)
    GPUArray harmonic_filtered({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq)}, DataType::Float32);
    {
        dim3 grid((n_freq + block_size - 1) / block_size, n_frames);
        median_filter_horizontal_kernel<<<grid, block_size, 0, stream>>>(
            static_cast<const float*>(stft_magnitude.data()),
            static_cast<float*>(harmonic_filtered.data()),
            n_frames, n_freq, kernel_size);
    }

    // Apply vertical median filter (percussive)
    GPUArray percussive_filtered({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq)}, DataType::Float32);
    {
        dim3 grid((n_freq + block_size - 1) / block_size, n_frames);
        median_filter_vertical_kernel<<<grid, block_size, 0, stream>>>(
            static_cast<const float*>(stft_magnitude.data()),
            static_cast<float*>(percussive_filtered.data()),
            n_frames, n_freq, kernel_size);
    }

    // Compute soft masks
    GPUArray harmonic_mask({static_cast<size_t>(n_elements)}, DataType::Float32);
    GPUArray percussive_mask({static_cast<size_t>(n_elements)}, DataType::Float32);

    int num_blocks = (n_elements + block_size - 1) / block_size;
    hpss_soft_mask_kernel<<<num_blocks, block_size, 0, stream>>>(
        static_cast<const float*>(harmonic_filtered.data()),
        static_cast<const float*>(percussive_filtered.data()),
        static_cast<float*>(harmonic_mask.data()),
        static_cast<float*>(percussive_mask.data()),
        n_elements, power);

    // Apply masks to original magnitude
    GPUArray harmonic_out({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq)}, DataType::Float32);
    GPUArray percussive_out({static_cast<size_t>(n_frames), static_cast<size_t>(n_freq)}, DataType::Float32);

    // Element-wise multiply on host for simplicity
    std::vector<float> h_mag(n_elements), h_h_mask(n_elements), h_p_mask(n_elements);
    std::vector<float> h_h_out(n_elements), h_p_out(n_elements);

    memcpy_device_to_host(h_mag.data(), stft_magnitude.data(), n_elements * sizeof(float));
    memcpy_device_to_host(h_h_mask.data(), harmonic_mask.data(), n_elements * sizeof(float));
    memcpy_device_to_host(h_p_mask.data(), percussive_mask.data(), n_elements * sizeof(float));

    for (int i = 0; i < n_elements; ++i) {
        h_h_out[i] = h_mag[i] * h_h_mask[i];
        h_p_out[i] = h_mag[i] * h_p_mask[i];
    }

    memcpy_host_to_device(harmonic_out.data(), h_h_out.data(), n_elements * sizeof(float));
    memcpy_host_to_device(percussive_out.data(), h_p_out.data(), n_elements * sizeof(float));

    sync_and_check("hpss failed");
    return std::make_pair(std::move(harmonic_out), std::move(percussive_out));
}

GPUArray harmonic(const GPUArray& stft_magnitude, int kernel_size, float power, float margin) {
    auto result = hpss(stft_magnitude, kernel_size, power, margin);
    return std::move(result.first);
}

GPUArray percussive(const GPUArray& stft_magnitude, int kernel_size, float power, float margin) {
    auto result = hpss(stft_magnitude, kernel_size, power, margin);
    return std::move(result.second);
}

// ============================================================================
// Time Stretch / Pitch Shift
// ============================================================================

GPUArray time_stretch(const GPUArray& input, float rate, int n_fft, int hop_length) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("time_stretch: input must be Float32");
    }

    if (hop_length < 0) hop_length = n_fft / 4;

    // Compute STFT
    GPUArray stft_out = stft(input, n_fft, hop_length, n_fft, true);

    auto& shape = stft_out.shape();
    int n_frames = static_cast<int>(shape[0]);
    int n_freq = static_cast<int>(shape[1]);

    // Calculate new number of frames
    int new_n_frames = static_cast<int>(std::ceil(n_frames / rate));

    cudaStream_t stream = internal::get_capture_stream();
    const int block_size = 256;
    int n_elements = n_freq;

    // Extract magnitude and phase
    const float* stft_real = static_cast<const float*>(stft_out.data());
    const float* stft_imag = stft_real + n_frames * n_freq;

    std::vector<float> h_real(n_frames * n_freq);
    std::vector<float> h_imag(n_frames * n_freq);
    memcpy_device_to_host(h_real.data(), const_cast<float*>(stft_real), n_frames * n_freq * sizeof(float));
    memcpy_device_to_host(h_imag.data(), const_cast<float*>(stft_imag), n_frames * n_freq * sizeof(float));

    // Phase vocoder interpolation on host
    std::vector<float> h_new_real(new_n_frames * n_freq);
    std::vector<float> h_new_imag(new_n_frames * n_freq);
    std::vector<float> phase_accum(n_freq, 0.0f);

    float expected_phase_advance = 2.0f * 3.14159265358979f * hop_length / n_fft;

    for (int new_f = 0; new_f < new_n_frames; ++new_f) {
        float src_frame = new_f * rate;
        int f0 = static_cast<int>(src_frame);
        int f1 = std::min(f0 + 1, n_frames - 1);
        float alpha = src_frame - f0;

        for (int k = 0; k < n_freq; ++k) {
            // Get magnitudes
            float m0_r = h_real[f0 * n_freq + k];
            float m0_i = h_imag[f0 * n_freq + k];
            float m1_r = h_real[f1 * n_freq + k];
            float m1_i = h_imag[f1 * n_freq + k];

            float mag0 = std::sqrt(m0_r * m0_r + m0_i * m0_i);
            float mag1 = std::sqrt(m1_r * m1_r + m1_i * m1_i);
            float phase0 = std::atan2(m0_i, m0_r);
            float phase1 = std::atan2(m1_i, m1_r);

            // Interpolate magnitude
            float mag = (1 - alpha) * mag0 + alpha * mag1;

            // Phase vocoder: accumulate phase difference
            if (new_f == 0) {
                phase_accum[k] = phase0;
            } else {
                float freq_bin_advance = expected_phase_advance * k;
                float phase_diff = phase1 - phase0 - freq_bin_advance;
                // Wrap to [-pi, pi]
                phase_diff = phase_diff - 2.0f * 3.14159265358979f *
                             std::round(phase_diff / (2.0f * 3.14159265358979f));
                phase_accum[k] += freq_bin_advance + phase_diff;
            }

            h_new_real[new_f * n_freq + k] = mag * std::cos(phase_accum[k]);
            h_new_imag[new_f * n_freq + k] = mag * std::sin(phase_accum[k]);
        }
    }

    // Create new STFT
    GPUArray new_stft({static_cast<size_t>(new_n_frames), static_cast<size_t>(n_freq), 2}, DataType::Float32);
    float* new_stft_ptr = static_cast<float*>(new_stft.data());
    memcpy_host_to_device(new_stft_ptr, h_new_real.data(), new_n_frames * n_freq * sizeof(float));
    memcpy_host_to_device(new_stft_ptr + new_n_frames * n_freq, h_new_imag.data(),
                          new_n_frames * n_freq * sizeof(float));

    // ISTFT
    return istft(new_stft, hop_length, n_fft, true, -1);
}

GPUArray pitch_shift(const GPUArray& input, int sample_rate, float n_steps,
                     int n_fft, int hop_length) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("pitch_shift: input must be Float32");
    }

    // Pitch shift = time stretch + resample
    float rate = std::pow(2.0f, -n_steps / 12.0f);

    // Time stretch
    GPUArray stretched = time_stretch(input, rate, n_fft, hop_length);

    // For proper pitch shifting, we'd need to resample
    // For now, return time-stretched (which changes both pitch and duration)
    // Full implementation would require rational resampling

    return stretched;
}

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
