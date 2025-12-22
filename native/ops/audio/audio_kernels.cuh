/**
 * GPU Audio Processing Kernels
 *
 * Optimized CUDA kernels for audio preprocessing (ASR/Whisper):
 * - PCM to float conversion (int16 -> float32)
 * - Stereo to mono conversion
 * - Peak/RMS normalization
 * - Polyphase resampling (48kHz -> 16kHz)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace audio {

// ============================================================================
// PCM to Float Conversion
// ============================================================================

__global__ void pcm_int16_to_f32_kernel(
    const int16_t* __restrict__ input,
    float* __restrict__ output,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Normalize int16 [-32768, 32767] to float [-1.0, 1.0]
        output[idx] = static_cast<float>(input[idx]) / 32768.0f;
    }
}

// ============================================================================
// Stereo to Mono Conversion
// ============================================================================

__global__ void stereo_to_mono_kernel(
    const float* __restrict__ input,   // [samples * 2] interleaved L,R,L,R,...
    float* __restrict__ output,        // [samples]
    size_t num_samples)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        // Average left and right channels
        float left = input[idx * 2];
        float right = input[idx * 2 + 1];
        output[idx] = (left + right) * 0.5f;
    }
}

// ============================================================================
// Normalization
// ============================================================================

// Find maximum absolute value (for peak normalization)
__global__ void find_max_abs_kernel(
    const float* __restrict__ input,
    float* __restrict__ block_max,
    size_t n)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and find local max
    float local_max = 0.0f;
    if (idx < n) {
        local_max = fabsf(input[idx]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
}

// Apply scale factor (in-place)
__global__ void apply_scale_kernel(
    float* __restrict__ data,
    size_t n,
    float scale)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Compute sum of squares (for RMS normalization)
__global__ void sum_of_squares_kernel(
    const float* __restrict__ input,
    float* __restrict__ block_sum,
    size_t n)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute square
    float val = 0.0f;
    if (idx < n) {
        val = input[idx] * input[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        block_sum[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Polyphase Resampling (48kHz -> 16kHz = decimation by 3)
// ============================================================================

// Kaiser window FIR filter coefficients for 48kHz -> 16kHz
// Cutoff: 7.2kHz (0.45 * 16kHz), Kaiser beta=5.0, 32 taps
// These are precomputed for the specific 3:1 decimation ratio
constexpr int RESAMPLE_TAPS = 32;
constexpr int RESAMPLE_DECIMATION = 3;  // 48000 / 16000 = 3

// Filter coefficients (stored in constant memory for cache efficiency)
__constant__ float RESAMPLE_FILTER[RESAMPLE_TAPS] = {
    -0.0003f, -0.0012f, -0.0025f, -0.0038f, -0.0041f, -0.0024f,  0.0022f,  0.0101f,
     0.0211f,  0.0344f,  0.0483f,  0.0611f,  0.0709f,  0.0763f,  0.0766f,  0.0716f,
     0.0618f,  0.0483f,  0.0325f,  0.0162f,  0.0010f, -0.0117f, -0.0209f, -0.0262f,
    -0.0277f, -0.0257f, -0.0210f, -0.0146f, -0.0076f, -0.0012f,  0.0038f,  0.0068f
};

__global__ void resample_polyphase_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_len,
    int out_len)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_len) return;

    // Map output sample to input position
    int in_pos = out_idx * RESAMPLE_DECIMATION;

    // Apply FIR filter centered at in_pos
    float sum = 0.0f;
    int half_taps = RESAMPLE_TAPS / 2;

    #pragma unroll
    for (int k = 0; k < RESAMPLE_TAPS; ++k) {
        int sample_idx = in_pos - half_taps + k;
        if (sample_idx >= 0 && sample_idx < in_len) {
            sum += input[sample_idx] * RESAMPLE_FILTER[k];
        }
    }

    output[out_idx] = sum;
}

// ============================================================================
// Ring Buffer Operations (for streaming)
// ============================================================================

// Write samples to ring buffer with wrap-around
__global__ void ring_buffer_write_kernel(
    const float* __restrict__ input,
    float* __restrict__ ring_buffer,
    int ring_size,
    int write_pos,
    int num_samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        int dst_idx = (write_pos + idx) % ring_size;
        ring_buffer[dst_idx] = input[idx];
    }
}

// Read samples from ring buffer (linearize with wrap-around)
__global__ void ring_buffer_read_kernel(
    const float* __restrict__ ring_buffer,
    float* __restrict__ output,
    int ring_size,
    int read_pos,
    int num_samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        int src_idx = (read_pos + idx) % ring_size;
        output[idx] = ring_buffer[src_idx];
    }
}

// Apply Hann window for overlap-add
__global__ void apply_hann_window_kernel(
    float* __restrict__ data,
    int window_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < window_size) {
        // Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
        float n = static_cast<float>(idx);
        float N = static_cast<float>(window_size - 1);
        float window = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * n / N));
        data[idx] *= window;
    }
}

// Overlap-add: add windowed chunk to output buffer
__global__ void overlap_add_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int output_offset,
    int chunk_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        atomicAdd(&output[output_offset + idx], input[idx]);
    }
}

// ============================================================================
// Voice Activity Detection (VAD)
// ============================================================================

// Compute frame-level energy (RMS) for VAD
// Each block processes one frame
__global__ void vad_frame_energy_kernel(
    const float* __restrict__ audio,
    float* __restrict__ frame_energy,
    int audio_len,
    int frame_size,
    int hop_size,
    int num_frames)
{
    extern __shared__ float sdata[];

    int frame_idx = blockIdx.x;
    if (frame_idx >= num_frames) return;

    int tid = threadIdx.x;
    int frame_start = frame_idx * hop_size;

    // Each thread accumulates squared samples
    float sum_sq = 0.0f;
    for (int i = tid; i < frame_size; i += blockDim.x) {
        int sample_idx = frame_start + i;
        if (sample_idx < audio_len) {
            float val = audio[sample_idx];
            sum_sq += val * val;
        }
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Compute RMS energy
    if (tid == 0) {
        float rms = sqrtf(sdata[0] / static_cast<float>(frame_size));
        frame_energy[frame_idx] = rms;
    }
}

// Compute frame-level zero-crossing rate for VAD
__global__ void vad_zero_crossing_kernel(
    const float* __restrict__ audio,
    float* __restrict__ frame_zcr,
    int audio_len,
    int frame_size,
    int hop_size,
    int num_frames)
{
    extern __shared__ int sdata_int[];

    int frame_idx = blockIdx.x;
    if (frame_idx >= num_frames) return;

    int tid = threadIdx.x;
    int frame_start = frame_idx * hop_size;

    // Count zero crossings
    int crossings = 0;
    for (int i = tid; i < frame_size - 1; i += blockDim.x) {
        int sample_idx = frame_start + i;
        if (sample_idx + 1 < audio_len) {
            float curr = audio[sample_idx];
            float next = audio[sample_idx + 1];
            // Count sign change
            if ((curr >= 0.0f && next < 0.0f) || (curr < 0.0f && next >= 0.0f)) {
                crossings++;
            }
        }
    }

    sdata_int[tid] = crossings;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_int[tid] += sdata_int[tid + s];
        }
        __syncthreads();
    }

    // Normalize to rate [0, 1]
    if (tid == 0) {
        float zcr = static_cast<float>(sdata_int[0]) / static_cast<float>(frame_size - 1);
        frame_zcr[frame_idx] = zcr;
    }
}

// Apply threshold-based VAD decision with hangover smoothing
__global__ void vad_decision_kernel(
    const float* __restrict__ frame_energy,
    const float* __restrict__ frame_zcr,
    int* __restrict__ vad_output,
    int num_frames,
    float energy_threshold,
    float zcr_low,
    float zcr_high)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frames) return;

    float energy = frame_energy[idx];
    float zcr = frame_zcr[idx];

    // VAD decision based on energy and ZCR
    // High energy + moderate ZCR = speech
    // High energy + very high ZCR = unvoiced speech or noise
    // Low energy = silence
    int is_speech = 0;

    if (energy > energy_threshold) {
        // Energy above threshold - check ZCR
        if (zcr >= zcr_low && zcr <= zcr_high) {
            is_speech = 1;  // Voiced speech (moderate ZCR)
        } else if (zcr > zcr_high) {
            is_speech = 1;  // Unvoiced speech (high ZCR but high energy)
        }
    }

    vad_output[idx] = is_speech;
}

// Apply hangover smoothing to VAD output
// Extends speech regions by hangover_frames after speech ends
__global__ void vad_hangover_kernel(
    const int* __restrict__ vad_input,
    int* __restrict__ vad_output,
    int num_frames,
    int hangover_frames)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frames) return;

    // Check if this frame or any of the previous hangover_frames had speech
    int is_speech = 0;
    for (int i = 0; i <= hangover_frames; ++i) {
        int check_idx = idx - i;
        if (check_idx >= 0 && vad_input[check_idx] == 1) {
            is_speech = 1;
            break;
        }
    }

    vad_output[idx] = is_speech;
}

// Compute energy-to-silence ratio for adaptive thresholding
__global__ void vad_compute_noise_floor_kernel(
    const float* __restrict__ frame_energy,
    float* __restrict__ block_min,
    int num_frames)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load frame energy (use large value for out-of-bounds)
    float val = (idx < num_frames) ? frame_energy[idx] : 1e10f;
    sdata[tid] = val;
    __syncthreads();

    // Find minimum in block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Audio Preprocessing Kernels
// ============================================================================

// Pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
// Parallelized version using scan-like pattern
__global__ void preemphasis_kernel(
    float* __restrict__ data,
    size_t n,
    float alpha)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // For parallel processing, we read x[n] and x[n-1] independently
    // Note: This is an approximation that works well for most audio
    float curr = data[idx];
    float prev = (idx > 0) ? data[idx - 1] : 0.0f;
    data[idx] = curr - alpha * prev;
}

// De-emphasis filter: y[n] = x[n] + alpha * y[n-1]
// Sequential by nature (IIR filter) - runs on single thread
// For GPU efficiency, we process in blocks with overlap-save
__global__ void deemphasis_sequential_kernel(
    float* __restrict__ data,
    size_t n,
    float alpha)
{
    // Single thread sequential processing (for small arrays)
    // For larger arrays, use block-based approach
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float y_prev = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float y = data[i] + alpha * y_prev;
        data[i] = y;
        y_prev = y;
    }
}

// Compute sum for DC removal (reduction kernel)
__global__ void compute_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ block_sum,
    size_t n)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load value
    float val = (idx < n) ? input[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = sdata[0];
    }
}

// Subtract mean (DC removal)
__global__ void subtract_mean_kernel(
    float* __restrict__ data,
    size_t n,
    float mean)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] -= mean;
    }
}

// Single-pole high-pass filter (IIR)
// y[n] = alpha * (y[n-1] + x[n] - x[n-1])
// Sequential processing
__global__ void highpass_iir_kernel(
    float* __restrict__ data,
    size_t n,
    float alpha)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float x_prev = 0.0f;
    float y_prev = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float x = data[i];
        float y = alpha * (y_prev + x - x_prev);
        data[i] = y;
        x_prev = x;
        y_prev = y;
    }
}

// Simple noise gate: zero samples below threshold
__global__ void noise_gate_kernel(
    float* __restrict__ data,
    size_t n,
    float threshold)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        if (fabsf(val) < threshold) {
            data[idx] = 0.0f;
        }
    }
}

// Compute short-term energy per frame
__global__ void short_term_energy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_len,
    int frame_size,
    int num_frames)
{
    extern __shared__ float sdata[];

    int frame_idx = blockIdx.x;
    if (frame_idx >= num_frames) return;

    int tid = threadIdx.x;
    int frame_start = frame_idx * frame_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < frame_size; i += blockDim.x) {
        int sample_idx = frame_start + i;
        if (sample_idx < input_len) {
            float val = input[sample_idx];
            sum_sq += val * val;
        }
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Output mean energy (not RMS to save sqrt)
        output[frame_idx] = sdata[0] / static_cast<float>(frame_size);
    }
}

// Spectral gate with smoothing
// Computes per-sample gain based on local energy
__global__ void spectral_gate_kernel(
    float* __restrict__ data,
    const float* __restrict__ frame_energy,
    int n,
    int frame_size,
    int num_frames,
    float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Find which frame this sample belongs to
    int frame_idx = idx / frame_size;
    if (frame_idx >= num_frames) frame_idx = num_frames - 1;

    // Get energy for this frame
    float energy = frame_energy[frame_idx];

    // Compute gain (soft gate with smoothing)
    float gain = 1.0f;
    if (energy < threshold) {
        // Smooth attenuation: gain = (energy / threshold)^2
        float ratio = energy / threshold;
        gain = ratio * ratio;
    }

    data[idx] *= gain;
}

// ============================================================================
// Radix-2 FFT Kernels (Driver-Only, no cuFFT dependency)
// ============================================================================

// Bit reversal permutation for FFT
__device__ __forceinline__ int bit_reverse(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Bit-reversal permutation kernel
__global__ void fft_bit_reverse_kernel(
    const float* __restrict__ input_real,
    const float* __restrict__ input_imag,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    int n,
    int log2n,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || idx >= n) return;

    int rev_idx = bit_reverse(idx, log2n);
    int in_offset = batch_idx * n;

    output_real[in_offset + rev_idx] = input_real[in_offset + idx];
    output_imag[in_offset + rev_idx] = (input_imag != nullptr) ? input_imag[in_offset + idx] : 0.0f;
}

// Cooley-Tukey FFT butterfly kernel (iterative, in-place)
__global__ void fft_butterfly_kernel(
    float* __restrict__ real,
    float* __restrict__ imag,
    int n,
    int stage,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    int half_size = 1 << stage;
    int full_size = half_size << 1;
    int num_groups = n / full_size;
    int group_idx = idx / half_size;
    int k = idx % half_size;

    if (group_idx >= num_groups) return;

    int offset = batch_idx * n;
    int i = group_idx * full_size + k;
    int j = i + half_size;

    // Twiddle factor: W_n^k = exp(-2*pi*i*k/n)
    float angle = -2.0f * 3.14159265358979f * k / full_size;
    float tw_real = cosf(angle);
    float tw_imag = sinf(angle);

    // Load values
    float a_real = real[offset + i];
    float a_imag = imag[offset + i];
    float b_real = real[offset + j];
    float b_imag = imag[offset + j];

    // Butterfly operation
    // t = W * b
    float t_real = tw_real * b_real - tw_imag * b_imag;
    float t_imag = tw_real * b_imag + tw_imag * b_real;

    // a' = a + t
    // b' = a - t
    real[offset + i] = a_real + t_real;
    imag[offset + i] = a_imag + t_imag;
    real[offset + j] = a_real - t_real;
    imag[offset + j] = a_imag - t_imag;
}

// Combined FFT kernel for small sizes (fits in shared memory)
// Uses Stockham formulation for better memory access patterns
template<int N>
__global__ void fft_stockham_kernel(
    const float* __restrict__ input_real,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    int batch_size)
{
    extern __shared__ float smem[];
    float* s_real = smem;
    float* s_imag = smem + N;

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int offset = batch_idx * N;

    // Load input with bit-reversal
    constexpr int LOG2N = (N == 256) ? 8 : (N == 512) ? 9 : (N == 1024) ? 10 : 0;
    if (tid < N) {
        int rev = bit_reverse(tid, LOG2N);
        s_real[rev] = input_real[offset + tid];
        s_imag[rev] = 0.0f;
    }
    __syncthreads();

    // FFT stages
    for (int stage = 0; stage < LOG2N; ++stage) {
        int half_size = 1 << stage;
        int full_size = half_size << 1;

        if (tid < N / 2) {
            int group = tid / half_size;
            int k = tid % half_size;
            int i = group * full_size + k;
            int j = i + half_size;

            float angle = -2.0f * 3.14159265358979f * k / full_size;
            float tw_real = cosf(angle);
            float tw_imag = sinf(angle);

            float a_r = s_real[i], a_i = s_imag[i];
            float b_r = s_real[j], b_i = s_imag[j];

            float t_r = tw_real * b_r - tw_imag * b_i;
            float t_i = tw_real * b_i + tw_imag * b_r;

            s_real[i] = a_r + t_r;
            s_imag[i] = a_i + t_i;
            s_real[j] = a_r - t_r;
            s_imag[j] = a_i - t_i;
        }
        __syncthreads();
    }

    // Store output
    if (tid < N) {
        output_real[offset + tid] = s_real[tid];
        output_imag[offset + tid] = s_imag[tid];
    }
}

// Real-to-complex FFT post-processing
// For real input, we only need first N/2+1 complex outputs
__global__ void fft_real_to_complex_kernel(
    const float* __restrict__ fft_real,
    const float* __restrict__ fft_imag,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    int n,
    int n_out,
    int batch_size)
{
    int batch_idx = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || k >= n_out) return;

    int offset_in = batch_idx * n;
    int offset_out = batch_idx * n_out;

    // For real input, X[k] is already correct for k = 0 to N/2
    out_real[offset_out + k] = fft_real[offset_in + k];
    out_imag[offset_out + k] = fft_imag[offset_in + k];
}

// ============================================================================
// Spectral Processing Kernels
// ============================================================================

// Apply window function to frame (in-place)
__global__ void apply_window_to_frames_kernel(
    float* __restrict__ frames,
    const float* __restrict__ window,
    int n_frames,
    int frame_size)
{
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;

    if (frame_idx < n_frames && sample_idx < frame_size) {
        int idx = frame_idx * frame_size + sample_idx;
        frames[idx] *= window[sample_idx];
    }
}

// Extract overlapping frames from audio
__global__ void extract_frames_kernel(
    const float* __restrict__ audio,
    float* __restrict__ frames,
    int audio_len,
    int n_fft,
    int hop_length,
    int n_frames)
{
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;

    if (frame_idx < n_frames && sample_idx < n_fft) {
        int audio_idx = frame_idx * hop_length + sample_idx;
        int out_idx = frame_idx * n_fft + sample_idx;

        if (audio_idx < audio_len) {
            frames[out_idx] = audio[audio_idx];
        } else {
            frames[out_idx] = 0.0f;  // Zero padding
        }
    }
}

// Compute power spectrum: real^2 + imag^2
__global__ void power_spectrum_kernel(
    const float* __restrict__ stft_real,
    const float* __restrict__ stft_imag,
    float* __restrict__ power,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float r = stft_real[idx];
        float i = stft_imag[idx];
        power[idx] = r * r + i * i;
    }
}

// Compute magnitude spectrum: sqrt(real^2 + imag^2)
__global__ void magnitude_spectrum_kernel(
    const float* __restrict__ stft_real,
    const float* __restrict__ stft_imag,
    float* __restrict__ magnitude,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float r = stft_real[idx];
        float i = stft_imag[idx];
        magnitude[idx] = sqrtf(r * r + i * i);
    }
}

// Convert Hz to Mel scale
__device__ __forceinline__ float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert Mel to Hz scale
__device__ __forceinline__ float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Create mel filterbank matrix
__global__ void create_mel_filterbank_kernel(
    float* __restrict__ filterbank,
    int n_mels,
    int n_fft,
    int sample_rate,
    float f_min,
    float f_max)
{
    int mel_idx = blockIdx.x;
    int freq_idx = threadIdx.x;

    if (mel_idx >= n_mels) return;

    int n_freqs = n_fft / 2 + 1;
    if (freq_idx >= n_freqs) return;

    // Compute mel scale boundaries
    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    // Mel center frequencies (n_mels + 2 points for triangular filters)
    float mel_step = (mel_max - mel_min) / (n_mels + 1);
    float mel_left = mel_min + mel_idx * mel_step;
    float mel_center = mel_min + (mel_idx + 1) * mel_step;
    float mel_right = mel_min + (mel_idx + 2) * mel_step;

    float hz_left = mel_to_hz(mel_left);
    float hz_center = mel_to_hz(mel_center);
    float hz_right = mel_to_hz(mel_right);

    // Current frequency bin in Hz
    float freq_hz = static_cast<float>(freq_idx) * sample_rate / n_fft;

    // Triangular filter response
    float weight = 0.0f;
    if (freq_hz >= hz_left && freq_hz <= hz_center) {
        // Rising edge
        weight = (freq_hz - hz_left) / (hz_center - hz_left + 1e-10f);
    } else if (freq_hz > hz_center && freq_hz <= hz_right) {
        // Falling edge
        weight = (hz_right - freq_hz) / (hz_right - hz_center + 1e-10f);
    }

    filterbank[mel_idx * n_freqs + freq_idx] = weight;
}

// Apply log: log(x + eps)
__global__ void log_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_elements,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        output[idx] = logf(input[idx] + eps);
    }
}

// Convert to decibels: 10 * log10(x + eps)
__global__ void to_decibels_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_elements,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        output[idx] = 10.0f * log10f(input[idx] + eps);
    }
}

// DCT-II for MFCC
// dct[k] = sum_n(x[n] * cos(pi * k * (2n + 1) / (2N)))
__global__ void dct_ii_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_frames,
    int n_input,
    int n_output)
{
    int frame_idx = blockIdx.x;
    int k = threadIdx.x;  // output coefficient index

    if (frame_idx >= n_frames || k >= n_output) return;

    float sum = 0.0f;
    float scale = 3.14159265358979f * k / (2.0f * n_input);

    for (int n = 0; n < n_input; ++n) {
        float x = input[frame_idx * n_input + n];
        sum += x * cosf(scale * (2 * n + 1));
    }

    // Normalization factor
    float norm = (k == 0) ? sqrtf(1.0f / n_input) : sqrtf(2.0f / n_input);
    output[frame_idx * n_output + k] = sum * norm;
}

// Delta features computation
// delta[t] = sum_{n=1}^{width} n * (x[t+n] - x[t-n]) / (2 * sum_{n=1}^{width} n^2)
__global__ void delta_features_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_frames,
    int n_features,
    int width)
{
    int frame_idx = blockIdx.x;
    int feat_idx = threadIdx.x;

    if (frame_idx >= n_frames || feat_idx >= n_features) return;

    // Compute denominator: 2 * sum(n^2) for n = 1 to width
    float denom = 0.0f;
    for (int n = 1; n <= width; ++n) {
        denom += n * n;
    }
    denom *= 2.0f;

    // Compute numerator: sum(n * (x[t+n] - x[t-n]))
    float numer = 0.0f;
    for (int n = 1; n <= width; ++n) {
        int t_plus = min(frame_idx + n, n_frames - 1);
        int t_minus = max(frame_idx - n, 0);

        float x_plus = input[t_plus * n_features + feat_idx];
        float x_minus = input[t_minus * n_features + feat_idx];
        numer += n * (x_plus - x_minus);
    }

    output[frame_idx * n_features + feat_idx] = numer / (denom + 1e-10f);
}

// Hann window generation
__global__ void generate_hann_window_kernel(
    float* __restrict__ window,
    int window_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < window_size) {
        float n = static_cast<float>(idx);
        float N = static_cast<float>(window_size);
        window[idx] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * n / N));
    }
}

// Zero padding kernel (for center mode)
__global__ void pad_reflect_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_len,
    int pad_left,
    int total_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int src_idx;
    if (idx < pad_left) {
        // Left reflection
        src_idx = pad_left - idx;
    } else if (idx < pad_left + input_len) {
        // Original signal
        src_idx = idx - pad_left;
    } else {
        // Right reflection
        int right_offset = idx - (pad_left + input_len);
        src_idx = input_len - 2 - right_offset;
    }

    // Clamp to valid range
    src_idx = max(0, min(src_idx, input_len - 1));
    output[idx] = input[src_idx];
}

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
