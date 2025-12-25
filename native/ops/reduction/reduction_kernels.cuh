/**
 * Reduction operation kernels (sum, mean, max)
 * Uses warp-level shuffle for efficient parallel reduction
 */
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include "../common/types.cuh"

namespace pygpukit {
namespace ops {
namespace reduction {

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ============================================================================
// Sum reduction kernels
// ============================================================================

__global__ void reduce_sum_f32_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    __shared__ float shared[32];  // One value per warp

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop to accumulate
    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    // Write warp result to shared memory
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

__global__ void reduce_sum_f64_kernel(const double* __restrict__ input, double* __restrict__ output, size_t n) {
    __shared__ double shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }

    sum = warp_reduce_sum_f64(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0;
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) {
            // atomicAdd for double requires sm_60+
            atomicAdd(output, sum);
        }
    }
}

// FP16 reduction - accumulate in FP32 for numerical stability
__global__ void reduce_sum_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output, size_t n) {
    __shared__ float shared[32];  // Accumulate in FP32

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += __half2float(input[i]);
    }

    sum = warp_reduce_sum(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            // Atomic add in FP32, then convert back
            float old_val = __half2float(*output);
            *output = __float2half(old_val + sum);
        }
    }
}

// BF16 reduction - accumulate in FP32 for numerical stability
__global__ void reduce_sum_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += bf16_to_float(input[i]);
    }

    sum = warp_reduce_sum(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            float old_val = bf16_to_float(*output);
            *output = float_to_bf16(old_val + sum);
        }
    }
}

// ============================================================================
// Max reduction kernels
// ============================================================================

__global__ void reduce_max_f32_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, input[i]);
    }

    max_val = warp_reduce_max(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            // Atomic max for float - use atomicMax with int cast trick
            int* addr = (int*)output;
            int expected = *addr;
            while (max_val > __int_as_float(expected)) {
                int old = atomicCAS(addr, expected, __float_as_int(max_val));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

__global__ void reduce_max_f64_kernel(const double* __restrict__ input, double* __restrict__ output, size_t n) {
    __shared__ double shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmax(max_val, input[i]);
    }

    max_val = warp_reduce_max_f64(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_f64(max_val);
        if (lane == 0) {
            // Atomic max for double using CAS
            unsigned long long* addr = (unsigned long long*)output;
            unsigned long long expected = *addr;
            while (max_val > __longlong_as_double(expected)) {
                unsigned long long old = atomicCAS(addr, expected, __double_as_longlong(max_val));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

__global__ void reduce_max_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, __half2float(input[i]));
    }

    max_val = warp_reduce_max(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            float old_val = __half2float(*output);
            if (max_val > old_val) {
                *output = __float2half(max_val);
            }
        }
    }
}

__global__ void reduce_max_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, bf16_to_float(input[i]));
    }

    max_val = warp_reduce_max(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            float old_val = bf16_to_float(*output);
            if (max_val > old_val) {
                *output = float_to_bf16(max_val);
            }
        }
    }
}

// ============================================================================
// Min reduction kernels
// ============================================================================

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void reduce_min_f32_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float min_val = INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        min_val = fminf(min_val, input[i]);
    }

    min_val = warp_reduce_min(min_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min(min_val);
        if (lane == 0) {
            int* addr = (int*)output;
            int expected = *addr;
            while (min_val < __int_as_float(expected)) {
                int old = atomicCAS(addr, expected, __float_as_int(min_val));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

__global__ void reduce_min_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float min_val = INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        min_val = fminf(min_val, __half2float(input[i]));
    }

    min_val = warp_reduce_min(min_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min(min_val);
        if (lane == 0) {
            float old_val = __half2float(*output);
            if (min_val < old_val) {
                *output = __float2half(min_val);
            }
        }
    }
}

__global__ void reduce_min_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float min_val = INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        min_val = fminf(min_val, bf16_to_float(input[i]));
    }

    min_val = warp_reduce_min(min_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min(min_val);
        if (lane == 0) {
            float old_val = bf16_to_float(*output);
            if (min_val < old_val) {
                *output = float_to_bf16(min_val);
            }
        }
    }
}

__global__ void init_min_f32_kernel(float* output) { *output = INFINITY; }
__global__ void init_min_f16_kernel(__half* output) { *output = __float2half(INFINITY); }
__global__ void init_min_bf16_kernel(__nv_bfloat16* output) { *output = float_to_bf16(INFINITY); }

// ============================================================================
// Sum with axis kernels - reduce along specified axis
// For 2D tensor [M, N]: axis=0 reduces to [N], axis=1 reduces to [M]
// ============================================================================

// Sum along axis 0: [M, N] -> [N]
__global__ void sum_axis0_f32_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        sum += input[m * N + n];
    }
    output[n] = sum;
}

__global__ void sum_axis0_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output,
                                      int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        sum += __half2float(input[m * N + n]);
    }
    output[n] = __float2half(sum);
}

__global__ void sum_axis0_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
                                       int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        sum += bf16_to_float(input[m * N + n]);
    }
    output[n] = float_to_bf16(sum);
}

// Sum along axis 1: [M, N] -> [M]
__global__ void sum_axis1_f32_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += input[m * N + n];
    }
    output[m] = sum;
}

__global__ void sum_axis1_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output,
                                      int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += __half2float(input[m * N + n]);
    }
    output[m] = __float2half(sum);
}

__global__ void sum_axis1_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
                                       int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += bf16_to_float(input[m * N + n]);
    }
    output[m] = float_to_bf16(sum);
}

// ============================================================================
// Argmax reduction kernels - find index of maximum value
// ============================================================================

// Warp-level argmax primitive
__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__global__ void argmax_f32_kernel(const float* __restrict__ input, int64_t* __restrict__ output, size_t n) {
    __shared__ float shared_val[32];
    __shared__ int shared_idx[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (size_t i = idx; i < n; i += stride) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = static_cast<int>(i);
        }
    }

    warp_reduce_argmax(max_val, max_idx);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared_val[warp_id] = max_val;
        shared_idx[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared_val[lane] : -INFINITY;
        max_idx = (tid < (blockDim.x + 31) / 32) ? shared_idx[lane] : 0;
        warp_reduce_argmax(max_val, max_idx);
        if (lane == 0) {
            *output = static_cast<int64_t>(max_idx);
        }
    }
}

__global__ void argmax_f16_kernel(const __half* __restrict__ input, int64_t* __restrict__ output, size_t n) {
    __shared__ float shared_val[32];
    __shared__ int shared_idx[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (size_t i = idx; i < n; i += stride) {
        float v = __half2float(input[i]);
        if (v > max_val) {
            max_val = v;
            max_idx = static_cast<int>(i);
        }
    }

    warp_reduce_argmax(max_val, max_idx);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared_val[warp_id] = max_val;
        shared_idx[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared_val[lane] : -INFINITY;
        max_idx = (tid < (blockDim.x + 31) / 32) ? shared_idx[lane] : 0;
        warp_reduce_argmax(max_val, max_idx);
        if (lane == 0) {
            *output = static_cast<int64_t>(max_idx);
        }
    }
}

__global__ void argmax_bf16_kernel(const __nv_bfloat16* __restrict__ input, int64_t* __restrict__ output, size_t n) {
    __shared__ float shared_val[32];
    __shared__ int shared_idx[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (size_t i = idx; i < n; i += stride) {
        float v = bf16_to_float(input[i]);
        if (v > max_val) {
            max_val = v;
            max_idx = static_cast<int>(i);
        }
    }

    warp_reduce_argmax(max_val, max_idx);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared_val[warp_id] = max_val;
        shared_idx[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared_val[lane] : -INFINITY;
        max_idx = (tid < (blockDim.x + 31) / 32) ? shared_idx[lane] : 0;
        warp_reduce_argmax(max_val, max_idx);
        if (lane == 0) {
            *output = static_cast<int64_t>(max_idx);
        }
    }
}

// ============================================================================
// Output initialization kernels
// ============================================================================

__global__ void init_sum_f32_kernel(float* output) { *output = 0.0f; }
__global__ void init_sum_f64_kernel(double* output) { *output = 0.0; }
__global__ void init_sum_f16_kernel(__half* output) { *output = __float2half(0.0f); }
__global__ void init_sum_bf16_kernel(__nv_bfloat16* output) { *output = float_to_bf16(0.0f); }
__global__ void init_max_f32_kernel(float* output) { *output = -INFINITY; }
__global__ void init_max_f64_kernel(double* output) { *output = -INFINITY; }
__global__ void init_max_f16_kernel(__half* output) { *output = __float2half(-INFINITY); }
__global__ void init_max_bf16_kernel(__nv_bfloat16* output) { *output = float_to_bf16(-INFINITY); }

// ============================================================================
// Scale kernels (for mean calculation)
// ============================================================================

__global__ void scale_f32_kernel(float* data, float scale) {
    *data *= scale;
}

__global__ void scale_f64_kernel(double* data, double scale) {
    *data *= scale;
}

__global__ void scale_f16_kernel(__half* data, float scale) {
    *data = __float2half(__half2float(*data) * scale);
}

__global__ void scale_bf16_kernel(__nv_bfloat16* data, float scale) {
    *data = float_to_bf16(bf16_to_float(*data) * scale);
}

} // namespace reduction
} // namespace ops
} // namespace pygpukit
