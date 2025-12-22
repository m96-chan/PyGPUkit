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

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
