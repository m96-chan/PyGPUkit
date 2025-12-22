/**
 * Softmax kernels
 *
 * Refactored from nn_kernels.cuh for better modularity.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// Softmax
// ============================================================================

// Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
// Applied row-wise: input [batch, features] -> output [batch, features]
// Uses online softmax algorithm for numerical stability

__global__ void softmax_f32_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    size_t batch_size,
                                    size_t features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * features;
    float* row_output = output + row * features;

    // Step 1: Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // Warp-level reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float exp_val = expf(row_input[i] - row_max);
        row_output[i] = exp_val;  // Store temporarily
        sum += exp_val;
    }

    // Warp-level reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum;
    }
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}

__global__ void softmax_f64_kernel(const double* __restrict__ input,
                                    double* __restrict__ output,
                                    size_t batch_size,
                                    size_t features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const double* row_input = input + row * features;
    double* row_output = output + row * features;

    double max_val = -INFINITY;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        max_val = fmax(max_val, row_input[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmax(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ double shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmax(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }

    __shared__ double row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();

    double sum = 0.0;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        double exp_val = exp(row_input[i] - row_max);
        row_output[i] = exp_val;
        sum += exp_val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ double shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ double row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum;
    }
    __syncthreads();

    double inv_sum = 1.0 / row_sum;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}

__global__ void softmax_f16_kernel(const __half* __restrict__ input,
                                    __half* __restrict__ output,
                                    size_t batch_size,
                                    size_t features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* row_input = input + row * features;
    __half* row_output = output + row * features;

    // Compute in FP32 for precision
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        max_val = fmaxf(max_val, __half2float(row_input[i]));
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float exp_val = expf(__half2float(row_input[i]) - row_max);
        row_output[i] = __float2half(exp_val);
        sum += exp_val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        row_output[i] = __float2half(__half2float(row_output[i]) * inv_sum);
    }
}

__global__ void softmax_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                     __nv_bfloat16* __restrict__ output,
                                     size_t batch_size,
                                     size_t features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + row * features;
    __nv_bfloat16* row_output = output + row * features;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        max_val = fmaxf(max_val, __bfloat162float(row_input[i]));
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float exp_val = expf(__bfloat162float(row_input[i]) - row_max);
        row_output[i] = __float2bfloat16(exp_val);
        sum += exp_val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        row_output[i] = __float2bfloat16(__bfloat162float(row_output[i]) * inv_sum);
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
