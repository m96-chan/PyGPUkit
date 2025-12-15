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

} // namespace nn
} // namespace ops
} // namespace pygpukit
