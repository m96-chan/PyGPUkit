/**
 * Neural Network operation kernels
 *
 * Provides: Linear (matmul + bias), LayerNorm, GELU
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
// GELU Activation
// ============================================================================

// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// tanh-based approximation (faster, close to exact)
__device__ __forceinline__ float gelu_f32(float x) {
    const float c1 = 0.7978845608f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    return x * 0.5f * (1.0f + tanhf(c1 * (x + c2 * x3)));
}

__device__ __forceinline__ double gelu_f64(double x) {
    const double c1 = 0.7978845608028654;  // sqrt(2/pi)
    const double c2 = 0.044715;
    double x3 = x * x * x;
    return x * 0.5 * (1.0 + tanh(c1 * (x + c2 * x3)));
}

__global__ void gelu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_f32(input[idx]);
    }
}

__global__ void gelu_f64_kernel(const double* __restrict__ input,
                                 double* __restrict__ output,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_f64(input[idx]);
    }
}

__global__ void gelu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(gelu_f32(x));
    }
}

__global__ void gelu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(gelu_f32(x));
    }
}

// ============================================================================
// Bias Add (for Linear layer: y = Wx + b)
// ============================================================================

// Add bias to each row of output [batch, features]
// output[i,j] += bias[j]
__global__ void bias_add_f32_kernel(float* __restrict__ output,
                                     const float* __restrict__ bias,
                                     size_t batch_size,
                                     size_t features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        size_t j = idx % features;
        output[idx] += bias[j];
    }
}

__global__ void bias_add_f64_kernel(double* __restrict__ output,
                                     const double* __restrict__ bias,
                                     size_t batch_size,
                                     size_t features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        size_t j = idx % features;
        output[idx] += bias[j];
    }
}

__global__ void bias_add_f16_kernel(__half* __restrict__ output,
                                     const __half* __restrict__ bias,
                                     size_t batch_size,
                                     size_t features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        size_t j = idx % features;
        float out_val = __half2float(output[idx]);
        float bias_val = __half2float(bias[j]);
        output[idx] = __float2half(out_val + bias_val);
    }
}

__global__ void bias_add_bf16_kernel(__nv_bfloat16* __restrict__ output,
                                      const __nv_bfloat16* __restrict__ bias,
                                      size_t batch_size,
                                      size_t features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        size_t j = idx % features;
        float out_val = __bfloat162float(output[idx]);
        float bias_val = __bfloat162float(bias[j]);
        output[idx] = __float2bfloat16(out_val + bias_val);
    }
}

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

// ============================================================================
// Matrix Transpose
// ============================================================================

// Transpose kernel using shared memory for coalesced access
// Input: [rows, cols], Output: [cols, rows]
// Uses 32x32 tile with padding to avoid bank conflicts

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_f32_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Transpose indices for output
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // swapped
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // swapped

    // Write transposed tile (coalesced write)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_f64_kernel(const double* __restrict__ input,
                                      double* __restrict__ output,
                                      int rows, int cols) {
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_f16_kernel(const __half* __restrict__ input,
                                      __half* __restrict__ output,
                                      int rows, int cols) {
    __shared__ __half tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                       __nv_bfloat16* __restrict__ output,
                                       int rows, int cols) {
    __shared__ __nv_bfloat16 tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Tensor Manipulation Operations
// ============================================================================

// Concat two tensors along axis 0
// src1: [dim0_1, dim1, dim2], src2: [dim0_2, dim1, dim2]
// dst: [dim0_1 + dim0_2, dim1, dim2]
__global__ void concat_axis0_f32_kernel(
    const float* __restrict__ src1,
    const float* __restrict__ src2,
    float* __restrict__ dst,
    size_t dim0_1,      // First tensor's dim0
    size_t dim0_2,      // Second tensor's dim0
    size_t stride       // dim1 * dim2 (elements per row)
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// FP16 concat along axis 0
__global__ void concat_axis0_f16_kernel(
    const __half* __restrict__ src1,
    const __half* __restrict__ src2,
    __half* __restrict__ dst,
    size_t dim0_1,
    size_t dim0_2,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// BF16 concat along axis 0
__global__ void concat_axis0_bf16_kernel(
    const __nv_bfloat16* __restrict__ src1,
    const __nv_bfloat16* __restrict__ src2,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0_1,
    size_t dim0_2,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// Repeat tensor along axis 1 (for GQA expansion)
// src: [dim0, dim1, dim2] -> dst: [dim0, dim1 * repeats, dim2]
// Each element in dim1 is repeated 'repeats' times
__global__ void repeat_interleave_axis1_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t repeats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * repeats * dim2;

    if (idx < total) {
        // Compute output coordinates
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1_out = remaining % (dim1 * repeats);
        size_t d0 = remaining / (dim1 * repeats);

        // Map output d1 to input d1 (integer division gives the source index)
        size_t d1_in = d1_out / repeats;

        // Compute source index
        size_t src_idx = d0 * dim1 * dim2 + d1_in * dim2 + d2;
        dst[idx] = src[src_idx];
    }
}

// FP16 repeat interleave along axis 1
__global__ void repeat_interleave_axis1_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t repeats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * repeats * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1_out = remaining % (dim1 * repeats);
        size_t d0 = remaining / (dim1 * repeats);
        size_t d1_in = d1_out / repeats;
        size_t src_idx = d0 * dim1 * dim2 + d1_in * dim2 + d2;
        dst[idx] = src[src_idx];
    }
}

// BF16 repeat interleave along axis 1
__global__ void repeat_interleave_axis1_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t repeats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * repeats * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1_out = remaining % (dim1 * repeats);
        size_t d0 = remaining / (dim1 * repeats);
        size_t d1_in = d1_out / repeats;
        size_t src_idx = d0 * dim1 * dim2 + d1_in * dim2 + d2;
        dst[idx] = src[src_idx];
    }
}

// Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]
// Swaps axes 0 and 1
__global__ void transpose_021_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        // Compute source coordinates [d0, d1, d2]
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        // Compute destination index [d1, d0, d2]
        size_t dst_idx = d1 * dim0 * dim2 + d0 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// Transpose 3D FP16: [d0, d1, d2] -> [d1, d0, d2]
__global__ void transpose_021_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d1 * dim0 * dim2 + d0 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// Transpose 3D BF16: [d0, d1, d2] -> [d1, d0, d2]
__global__ void transpose_021_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d1 * dim0 * dim2 + d0 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// Reshape with copy (ensures contiguous output)
// Simply copies data - reshape is handled by changing shape metadata
__global__ void copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// FP16 copy kernel
__global__ void copy_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// BF16 copy kernel
__global__ void copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================
//
// Applies rotary position embeddings to Q and K tensors
// q, k: [seq_len, n_heads, head_dim] - input tensors (modified in-place)
// cos, sin: [seq_len, head_dim] - precomputed rotary frequencies
//
// For each position i and head h:
//   q_rot[i,h,0:d/2] = q[i,h,0:d/2] * cos[i,0:d/2] - q[i,h,d/2:d] * sin[i,0:d/2]
//   q_rot[i,h,d/2:d] = q[i,h,d/2:d] * cos[i,0:d/2] + q[i,h,0:d/2] * sin[i,0:d/2]

__global__ void rope_f32_kernel(
    float* __restrict__ q,      // [seq_len, n_heads_q, head_dim] - modified in-place
    float* __restrict__ k,      // [seq_len, n_heads_k, head_dim] - modified in-place
    const float* __restrict__ cos,  // [seq_len, head_dim]
    const float* __restrict__ sin,  // [seq_len, head_dim]
    int seq_len,
    int n_heads_q,
    int n_heads_k,
    int head_dim
) {
    int half_dim = head_dim / 2;

    // Each thread handles one (seq_pos, head, dim_pair)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = seq_len * n_heads_q * half_dim;
    int total_k = seq_len * n_heads_k * half_dim;

    // Process Q tensor
    if (idx < total_q) {
        int d = idx % half_dim;  // Which pair (0 to half_dim-1)
        int remaining = idx / half_dim;
        int h = remaining % n_heads_q;
        int s = remaining / n_heads_q;

        int base = s * n_heads_q * head_dim + h * head_dim;
        float q0 = q[base + d];
        float q1 = q[base + d + half_dim];

        int cos_idx = s * head_dim + d;
        float c = cos[cos_idx];
        float sn = sin[cos_idx];

        q[base + d] = q0 * c - q1 * sn;
        q[base + d + half_dim] = q1 * c + q0 * sn;
    }

    // Process K tensor (may have fewer heads than Q due to GQA)
    if (idx < total_k) {
        int d = idx % half_dim;
        int remaining = idx / half_dim;
        int h = remaining % n_heads_k;
        int s = remaining / n_heads_k;

        int base = s * n_heads_k * head_dim + h * head_dim;
        float k0 = k[base + d];
        float k1 = k[base + d + half_dim];

        int cos_idx = s * head_dim + d;
        float c = cos[cos_idx];
        float sn = sin[cos_idx];

        k[base + d] = k0 * c - k1 * sn;
        k[base + d + half_dim] = k1 * c + k0 * sn;
    }
}

// FP16 RoPE kernel (compute in FP32 for precision, store in FP16)
__global__ void rope_f16_kernel(
    __half* __restrict__ q,      // [seq_len, n_heads_q, head_dim] - modified in-place
    __half* __restrict__ k,      // [seq_len, n_heads_k, head_dim] - modified in-place
    const __half* __restrict__ cos,  // [seq_len, head_dim]
    const __half* __restrict__ sin,  // [seq_len, head_dim]
    int seq_len,
    int n_heads_q,
    int n_heads_k,
    int head_dim
) {
    int half_dim = head_dim / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = seq_len * n_heads_q * half_dim;
    int total_k = seq_len * n_heads_k * half_dim;

    // Process Q tensor
    if (idx < total_q) {
        int d = idx % half_dim;
        int remaining = idx / half_dim;
        int h = remaining % n_heads_q;
        int s = remaining / n_heads_q;

        int base = s * n_heads_q * head_dim + h * head_dim;
        float q0 = __half2float(q[base + d]);
        float q1 = __half2float(q[base + d + half_dim]);

        int cos_idx = s * head_dim + d;
        float c = __half2float(cos[cos_idx]);
        float sn = __half2float(sin[cos_idx]);

        q[base + d] = __float2half(q0 * c - q1 * sn);
        q[base + d + half_dim] = __float2half(q1 * c + q0 * sn);
    }

    // Process K tensor
    if (idx < total_k) {
        int d = idx % half_dim;
        int remaining = idx / half_dim;
        int h = remaining % n_heads_k;
        int s = remaining / n_heads_k;

        int base = s * n_heads_k * head_dim + h * head_dim;
        float k0 = __half2float(k[base + d]);
        float k1 = __half2float(k[base + d + half_dim]);

        int cos_idx = s * head_dim + d;
        float c = __half2float(cos[cos_idx]);
        float sn = __half2float(sin[cos_idx]);

        k[base + d] = __float2half(k0 * c - k1 * sn);
        k[base + d + half_dim] = __float2half(k1 * c + k0 * sn);
    }
}

// BF16 RoPE kernel (compute in FP32 for precision, store in BF16)
__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    int seq_len,
    int n_heads_q,
    int n_heads_k,
    int head_dim
) {
    int half_dim = head_dim / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = seq_len * n_heads_q * half_dim;
    int total_k = seq_len * n_heads_k * half_dim;

    // Process Q tensor
    if (idx < total_q) {
        int d = idx % half_dim;
        int remaining = idx / half_dim;
        int h = remaining % n_heads_q;
        int s = remaining / n_heads_q;

        int base = s * n_heads_q * head_dim + h * head_dim;
        float q0 = __bfloat162float(q[base + d]);
        float q1 = __bfloat162float(q[base + d + half_dim]);

        int cos_idx = s * head_dim + d;
        float c = __bfloat162float(cos[cos_idx]);
        float sn = __bfloat162float(sin[cos_idx]);

        q[base + d] = __float2bfloat16(q0 * c - q1 * sn);
        q[base + d + half_dim] = __float2bfloat16(q1 * c + q0 * sn);
    }

    // Process K tensor
    if (idx < total_k) {
        int d = idx % half_dim;
        int remaining = idx / half_dim;
        int h = remaining % n_heads_k;
        int s = remaining / n_heads_k;

        int base = s * n_heads_k * head_dim + h * head_dim;
        float k0 = __bfloat162float(k[base + d]);
        float k1 = __bfloat162float(k[base + d + half_dim]);

        int cos_idx = s * head_dim + d;
        float c = __bfloat162float(cos[cos_idx]);
        float sn = __bfloat162float(sin[cos_idx]);

        k[base + d] = __float2bfloat16(k0 * c - k1 * sn);
        k[base + d + half_dim] = __float2bfloat16(k1 * c + k0 * sn);
    }
}

// ============================================================================
// SiLU (Swish) Activation: x * sigmoid(x)
// ============================================================================

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = silu_f32(input[idx]);
    }
}

__global__ void silu_f64_kernel(const double* __restrict__ input,
                                 double* __restrict__ output,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = input[idx];
        output[idx] = x / (1.0 + exp(-x));
    }
}

__global__ void silu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(silu_f32(x));
    }
}

__global__ void silu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(silu_f32(x));
    }
}

// ============================================================================
// Scaled Dot-Product Attention (SDPA) with Causal Mask
// ============================================================================
//
// For multi-head attention:
//   Q: [n_heads, q_len, head_dim]
//   K: [n_heads, kv_len, head_dim]
//   V: [n_heads, kv_len, head_dim]
//   Output: [n_heads, q_len, head_dim]
//
// Algorithm:
//   1. scores = Q @ K^T / sqrt(head_dim)  -> [n_heads, q_len, kv_len]
//   2. Apply causal mask (future positions = -inf)
//   3. weights = softmax(scores, dim=-1)
//   4. output = weights @ V               -> [n_heads, q_len, head_dim]
//
// This kernel handles one (head, query_position) pair per block.
// Each block computes attention for one query position in one head.

__global__ void sdpa_causal_f32_kernel(
    const float* __restrict__ Q,      // [n_heads, q_len, head_dim]
    const float* __restrict__ K,      // [n_heads, kv_len, head_dim]
    const float* __restrict__ V,      // [n_heads, kv_len, head_dim]
    float* __restrict__ output,       // [n_heads, q_len, head_dim]
    int n_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,                      // 1/sqrt(head_dim)
    int causal_offset                 // kv_len - q_len (for proper causal masking)
) {
    // Each block handles one (head, query_pos) pair
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    // Pointers for this head
    const float* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const float* K_head = K + head_idx * kv_len * head_dim;
    const float* V_head = V + head_idx * kv_len * head_dim;
    float* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    // Causal mask: query at position q_pos can attend to positions 0..(causal_offset + q_pos)
    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    // Step 1: Compute attention scores and find max (for numerical stability)
    extern __shared__ float shared[];
    float* scores = shared;  // [kv_len]

    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        if (kv_pos < max_attend) {
            // Dot product Q[q_pos] @ K[kv_pos]
            for (int d = 0; d < head_dim; d++) {
                score += Q_head[d] * K_head[kv_pos * head_dim + d];
            }
            score *= scale;
        } else {
            score = -INFINITY;  // Masked position
        }
        scores[kv_pos] = score;
        if (score > max_score) max_score = score;
    }

    // Reduce max across threads
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, max_score, offset);
        max_score = fmaxf(max_score, other);
    }

    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    // Step 2: Compute exp(score - max) and sum
    float sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum += exp_score;
    }

    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    // Step 3: Normalize scores to get attention weights
    float inv_sum = 1.0f / row_sum;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Compute output = weights @ V
    // Each thread handles a subset of head_dim
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            out_val += scores[kv_pos] * V_head[kv_pos * head_dim + d];
        }
        out_head[d] = out_val;
    }
}

// FP16 SDPA (compute in FP32 for precision)
__global__ void sdpa_causal_f16_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ output,
    int n_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    int causal_offset
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    const __half* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __half* K_head = K + head_idx * kv_len * head_dim;
    const __half* V_head = V + head_idx * kv_len * head_dim;
    __half* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    extern __shared__ float shared[];
    float* scores = shared;

    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        if (kv_pos < max_attend) {
            for (int d = 0; d < head_dim; d++) {
                score += __half2float(Q_head[d]) * __half2float(K_head[kv_pos * head_dim + d]);
            }
            score *= scale;
        } else {
            score = -INFINITY;
        }
        scores[kv_pos] = score;
        if (score > max_score) max_score = score;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    }

    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    float sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum += exp_score;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            out_val += scores[kv_pos] * __half2float(V_head[kv_pos * head_dim + d]);
        }
        out_head[d] = __float2half(out_val);
    }
}

// BF16 SDPA
__global__ void sdpa_causal_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ output,
    int n_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    int causal_offset
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    const __nv_bfloat16* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __nv_bfloat16* K_head = K + head_idx * kv_len * head_dim;
    const __nv_bfloat16* V_head = V + head_idx * kv_len * head_dim;
    __nv_bfloat16* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    extern __shared__ float shared[];
    float* scores = shared;

    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        if (kv_pos < max_attend) {
            for (int d = 0; d < head_dim; d++) {
                score += __bfloat162float(Q_head[d]) * __bfloat162float(K_head[kv_pos * head_dim + d]);
            }
            score *= scale;
        } else {
            score = -INFINITY;
        }
        scores[kv_pos] = score;
        if (score > max_score) max_score = score;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    }

    __shared__ float shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    float sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum += exp_score;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            out_val += scores[kv_pos] * __bfloat162float(V_head[kv_pos * head_dim + d]);
        }
        out_head[d] = __float2bfloat16(out_val);
    }
}

// ============================================================================
// KV Cache Update Kernel (Fixed-Length KV Cache for CUDA Graph)
// ============================================================================

// Copy new K/V values to position in fixed-length cache
// new_kv: [1, num_kv_heads, head_dim] - single token K or V
// cache: [max_seq_len, num_kv_heads, head_dim] - pre-allocated cache
// position: where to write in cache (0-indexed)
template <typename T>
__global__ void kv_cache_update_kernel(
    const T* __restrict__ new_kv,
    T* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    // Total elements per position: num_kv_heads * head_dim
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // new_kv is [1, num_kv_heads, head_dim], so offset is just idx
        // cache is [max_seq_len, num_kv_heads, head_dim]
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// FP16 version
__global__ void kv_cache_update_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// BF16 version
__global__ void kv_cache_update_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// FP32 version
__global__ void kv_cache_update_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// Prefill version: Copy multiple tokens from prefill K/V to cache
// new_kv: [seq_len, num_kv_heads, head_dim]
// cache: [max_seq_len, num_kv_heads, head_dim]
// start_pos: where to start writing in cache
// seq_len: number of tokens to copy
__global__ void kv_cache_prefill_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len
) {
    int elements_per_pos = num_kv_heads * head_dim;
    int total_elements = seq_len * elements_per_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int seq_pos = idx / elements_per_pos;
        int elem_idx = idx % elements_per_pos;
        int cache_offset = (start_pos + seq_pos) * elements_per_pos + elem_idx;
        cache[cache_offset] = new_kv[idx];
    }
}

__global__ void kv_cache_prefill_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len
) {
    int elements_per_pos = num_kv_heads * head_dim;
    int total_elements = seq_len * elements_per_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int seq_pos = idx / elements_per_pos;
        int elem_idx = idx % elements_per_pos;
        int cache_offset = (start_pos + seq_pos) * elements_per_pos + elem_idx;
        cache[cache_offset] = new_kv[idx];
    }
}

__global__ void kv_cache_prefill_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len
) {
    int elements_per_pos = num_kv_heads * head_dim;
    int total_elements = seq_len * elements_per_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int seq_pos = idx / elements_per_pos;
        int elem_idx = idx % elements_per_pos;
        int cache_offset = (start_pos + seq_pos) * elements_per_pos + elem_idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// ============================================================================
// GQA-expanded KV Cache Update (for CUDA Graph optimization)
// ============================================================================
// These kernels write to a transposed, GQA-expanded cache layout:
// Input: new_kv [1, num_kv_heads, head_dim] or [seq_len, num_kv_heads, head_dim]
// Cache: [num_heads, max_seq_len, head_dim] (transposed and expanded)
// This eliminates per-step transpose and GQA expansion overhead.

// Single token update with GQA expansion
// new_kv: [1, num_kv_heads, head_dim]
// cache: [num_heads, max_seq_len, head_dim]
__global__ void kv_cache_update_gqa_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    // Total output elements: num_heads * head_dim
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;

        // GQA: find source kv_head
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;

        // Source: new_kv[0, kv_head, d] = new_kv[kv_head * head_dim + d]
        int src_offset = kv_head * head_dim + d;

        // Dest: cache[head, position, d] = cache[head * max_seq_len * head_dim + position * head_dim + d]
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;

        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_update_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_update_gqa_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

// Prefill with GQA expansion
// new_kv: [seq_len, num_kv_heads, head_dim]
// cache: [num_heads, max_seq_len, head_dim]
__global__ void kv_cache_prefill_gqa_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int seq_len
) {
    // Total output elements: seq_len * num_heads * head_dim
    int total_elements = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int elements_per_seq = num_heads * head_dim;
        int seq_pos = idx / elements_per_seq;
        int remaining = idx % elements_per_seq;
        int head = remaining / head_dim;
        int d = remaining % head_dim;

        // GQA: find source kv_head
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;

        // Source: new_kv[seq_pos, kv_head, d]
        int src_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + d;

        // Dest: cache[head, start_pos + seq_pos, d]
        int dst_offset = head * max_seq_len * head_dim + (start_pos + seq_pos) * head_dim + d;

        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_prefill_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int seq_len
) {
    int total_elements = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int elements_per_seq = num_heads * head_dim;
        int seq_pos = idx / elements_per_seq;
        int remaining = idx % elements_per_seq;
        int head = remaining / head_dim;
        int d = remaining % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + (start_pos + seq_pos) * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_prefill_gqa_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int seq_len
) {
    int total_elements = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int elements_per_seq = num_heads * head_dim;
        int seq_pos = idx / elements_per_seq;
        int remaining = idx % elements_per_seq;
        int head = remaining / head_dim;
        int d = remaining % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + (start_pos + seq_pos) * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

} // namespace nn
} // namespace ops
} // namespace pygpukit
