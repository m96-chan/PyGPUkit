/**
 * Normalization kernels (LayerNorm, RMSNorm)
 *
 * Refactored from nn_kernels.cuh for better modularity.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// LayerNorm
// ============================================================================

// Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Input: [batch, features], normalize over features dimension

// Single-pass mean and variance using Welford's algorithm
__device__ __forceinline__ void welford_update(float& mean, float& m2, float val, int count) {
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
}

// LayerNorm kernel - one warp per row for small feature sizes
__global__ void layernorm_f32_kernel(const float* __restrict__ input,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ output,
                                      size_t batch_size,
                                      size_t features,
                                      float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * features;
    float* row_output = output + row * features;

    // Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        sum += row_input[i];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction using shared memory
    __shared__ float shared_sum[32];  // Max 32 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = sum / features;
    }
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }

    // Warp reduction for variance
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum / features + eps);
    }
    __syncthreads();

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = row_input[i];
        float normalized = (x - mean) * inv_std;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}

// Double precision LayerNorm
__global__ void layernorm_f64_kernel(const double* __restrict__ input,
                                      const double* __restrict__ gamma,
                                      const double* __restrict__ beta,
                                      double* __restrict__ output,
                                      size_t batch_size,
                                      size_t features,
                                      double eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const double* row_input = input + row * features;
    double* row_output = output + row * features;

    // Compute mean
    double sum = 0.0;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        sum += row_input[i];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ double shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

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

    __shared__ double mean;
    if (threadIdx.x == 0) {
        mean = sum / features;
    }
    __syncthreads();

    // Compute variance
    double var_sum = 0.0;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        double diff = row_input[i] - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ double inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrt(var_sum / features + eps);
    }
    __syncthreads();

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        double x = row_input[i];
        double normalized = (x - mean) * inv_std;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}

// FP16 LayerNorm (compute in FP32 for precision)
__global__ void layernorm_f16_kernel(const __half* __restrict__ input,
                                      const __half* __restrict__ gamma,
                                      const __half* __restrict__ beta,
                                      __half* __restrict__ output,
                                      size_t batch_size,
                                      size_t features,
                                      float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* row_input = input + row * features;
    __half* row_output = output + row * features;

    // Compute mean in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        sum += __half2float(row_input[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

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

    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = sum / features;
    }
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float diff = __half2float(row_input[i]) - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __half2float(row_input[i]);
        float normalized = (x - mean) * inv_std;
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        row_output[i] = __float2half(normalized * g + b);
    }
}

// BF16 LayerNorm (compute in FP32 for precision)
__global__ void layernorm_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                       const __nv_bfloat16* __restrict__ gamma,
                                       const __nv_bfloat16* __restrict__ beta,
                                       __nv_bfloat16* __restrict__ output,
                                       size_t batch_size,
                                       size_t features,
                                       float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + row * features;
    __nv_bfloat16* row_output = output + row * features;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        sum += __bfloat162float(row_input[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

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

    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = sum / features;
    }
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float diff = __bfloat162float(row_input[i]) - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        float normalized = (x - mean) * inv_std;
        float g = __bfloat162float(gamma[i]);
        float b = __bfloat162float(beta[i]);
        row_output[i] = __float2bfloat16(normalized * g + b);
    }
}

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// ============================================================================

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
// Input: [batch, features], normalize over features dimension
// Simpler than LayerNorm: no mean subtraction, no beta

__global__ void rmsnorm_f32_kernel(const float* __restrict__ input,
                                    const float* __restrict__ gamma,
                                    float* __restrict__ output,
                                    size_t batch_size,
                                    size_t features,
                                    float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * features;
    float* row_output = output + row * features;

    // Compute sum of squares using parallel reduction
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block-level reduction using shared memory
    __shared__ float shared_sum[32];  // Max 32 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        // RMS = sqrt(mean(x^2) + eps)
        inv_rms = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    // Normalize and apply scale (gamma)
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = row_input[i];
        row_output[i] = x * inv_rms * gamma[i];
    }
}

// Double precision RMSNorm
__global__ void rmsnorm_f64_kernel(const double* __restrict__ input,
                                    const double* __restrict__ gamma,
                                    double* __restrict__ output,
                                    size_t batch_size,
                                    size_t features,
                                    double eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const double* row_input = input + row * features;
    double* row_output = output + row * features;

    double sum_sq = 0.0;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        double val = row_input[i];
        sum_sq += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ double shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ double inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrt(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        double x = row_input[i];
        row_output[i] = x * inv_rms * gamma[i];
    }
}

// FP16 RMSNorm (compute in FP32 for precision)
__global__ void rmsnorm_f16_kernel(const __half* __restrict__ input,
                                    const __half* __restrict__ gamma,
                                    __half* __restrict__ output,
                                    size_t batch_size,
                                    size_t features,
                                    float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* row_input = input + row * features;
    __half* row_output = output + row * features;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __half2float(row_input[i]);
        float g = __half2float(gamma[i]);
        row_output[i] = __float2half(x * inv_rms * g);
    }
}

// BF16 RMSNorm (compute in FP32 for precision)
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                     const __nv_bfloat16* __restrict__ gamma,
                                     __nv_bfloat16* __restrict__ output,
                                     size_t batch_size,
                                     size_t features,
                                     float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + row * features;
    __nv_bfloat16* row_output = output + row * features;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = __bfloat162float(row_input[i]);
        sum_sq += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        float g = __bfloat162float(gamma[i]);
        row_output[i] = __float2bfloat16(x * inv_rms * g);
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
