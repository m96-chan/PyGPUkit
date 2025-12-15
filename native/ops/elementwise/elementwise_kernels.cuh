/**
 * Elementwise binary operation kernels (add, mul, sub, div)
 */
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "../common/types.cuh"

namespace pygpukit {
namespace ops {
namespace elementwise {

// ============================================================================
// Add kernels
// ============================================================================

__global__ void add_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void add_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) + bf16_to_float(b[idx]));
    }
}

// ============================================================================
// Mul kernels
// ============================================================================

__global__ void mul_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void mul_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) * bf16_to_float(b[idx]));
    }
}

// ============================================================================
// Sub kernels
// ============================================================================

__global__ void sub_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void sub_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) - bf16_to_float(b[idx]));
    }
}

// ============================================================================
// Div kernels
// ============================================================================

__global__ void div_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(__half2float(a[idx]) / __half2float(b[idx]));
    }
}

__global__ void div_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) / bf16_to_float(b[idx]));
    }
}

} // namespace elementwise
} // namespace ops
} // namespace pygpukit
