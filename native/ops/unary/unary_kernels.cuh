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

// ============================================================================
// Sin kernels
// ============================================================================

__global__ void sin_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sinf(a[idx]);
    }
}

__global__ void sin_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(sinf(__half2float(a[idx])));
    }
}

__global__ void sin_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(sinf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// Cos kernels
// ============================================================================

__global__ void cos_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = cosf(a[idx]);
    }
}

__global__ void cos_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(cosf(__half2float(a[idx])));
    }
}

__global__ void cos_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(cosf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// Sqrt kernels
// ============================================================================

__global__ void sqrt_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrtf(a[idx]);
    }
}

__global__ void sqrt_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(sqrtf(__half2float(a[idx])));
    }
}

__global__ void sqrt_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(sqrtf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// Rsqrt kernels (reciprocal sqrt: 1/sqrt(x))
// ============================================================================

__global__ void rsqrt_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = rsqrtf(a[idx]);
    }
}

__global__ void rsqrt_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(rsqrtf(__half2float(a[idx])));
    }
}

__global__ void rsqrt_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(rsqrtf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// Abs kernels
// ============================================================================

__global__ void abs_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fabsf(a[idx]);
    }
}

__global__ void abs_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(fabsf(__half2float(a[idx])));
    }
}

__global__ void abs_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(fabsf(bf16_to_float(a[idx])));
    }
}

// ============================================================================
// Neg kernels (negate: -x)
// ============================================================================

__global__ void neg_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = -a[idx];
    }
}

__global__ void neg_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hneg(a[idx]);
    }
}

__global__ void neg_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hneg(a[idx]);
    }
}

} // namespace unary
} // namespace ops
} // namespace pygpukit
