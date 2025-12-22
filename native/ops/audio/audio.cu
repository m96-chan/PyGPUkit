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

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
