/**
 * Activation function kernels (GELU, SiLU)
 *
 * Refactored from nn_kernels.cuh for better modularity.
 *
 * Usage:
 * - Include this header for declarations only (most files)
 * - Define PYGPUKIT_IMPLEMENT_NN_KERNELS before including to get definitions
 *   (only in nn_kernels.cu)
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
// Device helper functions (always inline, safe to include multiple times)
// ============================================================================

__device__ __forceinline__ float gelu_f32(float x) {
    const float c1 = 0.7978845608f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    return x * 0.5f * (1.0f + tanhf(c1 * (x + c2 * x3)));
}

__device__ __forceinline__ double gelu_f64(double x) {
    const double c1 = 0.7978845608028654;
    const double c2 = 0.044715;
    double x3 = x * x * x;
    return x * 0.5 * (1.0 + tanh(c1 * (x + c2 * x3)));
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float relu2_f32(float x) {
    float relu_val = fmaxf(0.0f, x);
    return relu_val * relu_val;
}

// ============================================================================
// Kernel declarations (always available)
// ============================================================================

__global__ void gelu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n);
__global__ void gelu_f64_kernel(const double* __restrict__ input,
                                 double* __restrict__ output, size_t n);
__global__ void gelu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n);
__global__ void gelu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n);

__global__ void silu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n);
__global__ void silu_f64_kernel(const double* __restrict__ input,
                                 double* __restrict__ output, size_t n);
__global__ void silu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n);
__global__ void silu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n);

__global__ void relu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n);
__global__ void relu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n);
__global__ void relu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n);

__global__ void sigmoid_f32_kernel(const float* __restrict__ input,
                                    float* __restrict__ output, size_t n);
__global__ void sigmoid_f16_kernel(const __half* __restrict__ input,
                                    __half* __restrict__ output, size_t n);
__global__ void sigmoid_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                     __nv_bfloat16* __restrict__ output, size_t n);

__global__ void tanh_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n);
__global__ void tanh_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n);
__global__ void tanh_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n);

// ReLU squared (Primer paper)
__global__ void relu2_f32_kernel(const float* __restrict__ input,
                                  float* __restrict__ output, size_t n);
__global__ void relu2_f16_kernel(const __half* __restrict__ input,
                                  __half* __restrict__ output, size_t n);
__global__ void relu2_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                   __nv_bfloat16* __restrict__ output, size_t n);

// ============================================================================
// Kernel definitions (only when PYGPUKIT_IMPLEMENT_NN_KERNELS is defined)
// ============================================================================

#ifdef PYGPUKIT_IMPLEMENT_NN_KERNELS

__global__ void gelu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = gelu_f32(input[idx]);
}

__global__ void gelu_f64_kernel(const double* __restrict__ input,
                                 double* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = gelu_f64(input[idx]);
}

__global__ void gelu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(gelu_f32(x));
    }
}

__global__ void gelu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(gelu_f32(x));
    }
}

__global__ void silu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = silu_f32(input[idx]);
}

__global__ void silu_f64_kernel(const double* __restrict__ input,
                                 double* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = input[idx];
        output[idx] = x / (1.0 + exp(-x));
    }
}

__global__ void silu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(silu_f32(x));
    }
}

__global__ void silu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(silu_f32(x));
    }
}

__global__ void relu_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void relu_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(fmaxf(0.0f, x));
    }
}

__global__ void relu_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(fmaxf(0.0f, x));
    }
}

__global__ void sigmoid_f32_kernel(const float* __restrict__ input,
                                    float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = sigmoid_f32(input[idx]);
}

__global__ void sigmoid_f16_kernel(const __half* __restrict__ input,
                                    __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(sigmoid_f32(x));
    }
}

__global__ void sigmoid_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                     __nv_bfloat16* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(sigmoid_f32(x));
    }
}

__global__ void tanh_f32_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = tanhf(input[idx]);
}

__global__ void tanh_f16_kernel(const __half* __restrict__ input,
                                 __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(tanhf(x));
    }
}

__global__ void tanh_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(tanhf(x));
    }
}

// ReLU squared kernels
__global__ void relu2_f32_kernel(const float* __restrict__ input,
                                  float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = relu2_f32(input[idx]);
}

__global__ void relu2_f16_kernel(const __half* __restrict__ input,
                                  __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(relu2_f32(x));
    }
}

__global__ void relu2_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                   __nv_bfloat16* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(relu2_f32(x));
    }
}

#endif  // PYGPUKIT_IMPLEMENT_NN_KERNELS

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
