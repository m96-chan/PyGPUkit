/**
 * Unary operation kernels (exp, log, relu)
 */
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include "../common/types.cuh"

namespace pygpukit {
namespace ops {
namespace unary {

// ============================================================================
// Exp kernels
// ============================================================================

__global__ void exp_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = expf(a[idx]);
    }
}

__global__ void exp_f64_kernel(const double* a, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = ::exp(a[idx]);
    }
}

__global__ void exp_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(expf(__half2float(a[idx])));
    }
}

__global__ void exp_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(expf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// Log kernels
// ============================================================================

__global__ void log_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = logf(a[idx]);
    }
}

__global__ void log_f64_kernel(const double* a, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = ::log(a[idx]);
    }
}

__global__ void log_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(logf(__half2float(a[idx])));
    }
}

__global__ void log_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(logf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// ReLU kernels
// ============================================================================

__global__ void relu_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(0.0f, a[idx]);
    }
}

__global__ void relu_f64_kernel(const double* a, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmax(0.0, a[idx]);
    }
}

__global__ void relu_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(a[idx]);
        c[idx] = __float2half(val > 0.0f ? val : 0.0f);
    }
}

__global__ void relu_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = bf16_to_float(a[idx]);
        c[idx] = float_to_bf16(val > 0.0f ? val : 0.0f);
    }
}

} // namespace unary
} // namespace ops
} // namespace pygpukit
