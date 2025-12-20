/**
 * Activation function kernels (GELU, SiLU)
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

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
