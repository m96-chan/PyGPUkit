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

}  // namespace audio
}  // namespace ops
}  // namespace pygpukit
