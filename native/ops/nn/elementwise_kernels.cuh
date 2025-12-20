/**
 * Elementwise and bias operation kernels
 *
 * Provides: Bias Add, RoPE, Add/Mul In-place, Split QKV
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

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
// cos/sin are also BF16
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

// BF16 RoPE kernel with FP32 cos/sin tables (higher precision, no intermediate allocation)
// Q/K are BF16, cos/sin are FP32 - compute in FP32, write back BF16
__global__ void rope_bf16_f32table_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
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
        float c = cos[cos_idx];
        float sn = sin[cos_idx];

        q[base + d] = __float2bfloat16_rn(q0 * c - q1 * sn);
        q[base + d + half_dim] = __float2bfloat16_rn(q1 * c + q0 * sn);
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
        float c = cos[cos_idx];
        float sn = sin[cos_idx];

        k[base + d] = __float2bfloat16_rn(k0 * c - k1 * sn);
        k[base + d + half_dim] = __float2bfloat16_rn(k1 * c + k0 * sn);
    }
}

// FP16 RoPE kernel with FP32 cos/sin tables (higher precision, no intermediate allocation)
// Q/K are FP16, cos/sin are FP32 - compute in FP32, write back FP16
__global__ void rope_f16_f32table_kernel(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
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
        float c = cos[cos_idx];
        float sn = sin[cos_idx];

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
        float c = cos[cos_idx];
        float sn = sin[cos_idx];

        k[base + d] = __float2half(k0 * c - k1 * sn);
        k[base + d + half_dim] = __float2half(k1 * c + k0 * sn);
    }
}

// ============================================================================
// Add In-place (for CUDA Graph - no allocation)
// ============================================================================
// a += b (element-wise)

__global__ void add_inplace_f16_kernel(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void add_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void add_inplace_f32_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] + b[idx];
    }
}

__global__ void add_inplace_f64_kernel(
    double* __restrict__ a,
    const double* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// In-place multiply kernels: a *= b
// ============================================================================

__global__ void mul_inplace_f16_kernel(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void mul_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void mul_inplace_f32_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_inplace_f64_kernel(
    double* __restrict__ a,
    const double* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * b[idx];
    }
}

// ============================================================================
// Split QKV Batch Kernels
// Splits fused QKV projection output [seq_len, q_dim + k_dim + v_dim]
// into separate Q, K, V tensors for batch decode
// ============================================================================

template<typename T>
__global__ void split_qkv_batch_kernel(
    const T* __restrict__ qkv,      // [seq_len, q_dim + k_dim + v_dim]
    T* __restrict__ q,              // [seq_len, q_dim]
    T* __restrict__ k,              // [seq_len, k_dim]
    T* __restrict__ v,              // [seq_len, v_dim]
    int seq_len,
    int q_dim,
    int k_dim,
    int v_dim
) {
    // Each thread handles one element
    int total_qkv = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * total_qkv;

    if (idx >= total_elements) return;

    int row = idx / total_qkv;
    int col = idx % total_qkv;

    T val = qkv[idx];

    if (col < q_dim) {
        // Q region
        q[row * q_dim + col] = val;
    } else if (col < q_dim + k_dim) {
        // K region
        k[row * k_dim + (col - q_dim)] = val;
    } else {
        // V region
        v[row * v_dim + (col - q_dim - k_dim)] = val;
    }
}

// Explicit instantiations
__global__ void split_qkv_batch_f16_kernel(
    const __half* __restrict__ qkv,
    __half* __restrict__ q,
    __half* __restrict__ k,
    __half* __restrict__ v,
    int seq_len,
    int q_dim,
    int k_dim,
    int v_dim
) {
    int total_qkv = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * total_qkv;

    if (idx >= total_elements) return;

    int row = idx / total_qkv;
    int col = idx % total_qkv;

    __half val = qkv[idx];

    if (col < q_dim) {
        q[row * q_dim + col] = val;
    } else if (col < q_dim + k_dim) {
        k[row * k_dim + (col - q_dim)] = val;
    } else {
        v[row * v_dim + (col - q_dim - k_dim)] = val;
    }
}

__global__ void split_qkv_batch_f32_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int seq_len,
    int q_dim,
    int k_dim,
    int v_dim
) {
    int total_qkv = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * total_qkv;

    if (idx >= total_elements) return;

    int row = idx / total_qkv;
    int col = idx % total_qkv;

    float val = qkv[idx];

    if (col < q_dim) {
        q[row * q_dim + col] = val;
    } else if (col < q_dim + k_dim) {
        k[row * k_dim + (col - q_dim)] = val;
    } else {
        v[row * v_dim + (col - q_dim - k_dim)] = val;
    }
}

__global__ void split_qkv_batch_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int seq_len,
    int q_dim,
    int k_dim,
    int v_dim
) {
    int total_qkv = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * total_qkv;

    if (idx >= total_elements) return;

    int row = idx / total_qkv;
    int col = idx % total_qkv;

    __nv_bfloat16 val = qkv[idx];

    if (col < q_dim) {
        q[row * q_dim + col] = val;
    } else if (col < q_dim + k_dim) {
        k[row * k_dim + (col - q_dim)] = val;
    } else {
        v[row * v_dim + (col - q_dim - k_dim)] = val;
    }
}

// ============================================================================
// Dtype Cast Kernels
// ============================================================================

// Cast float32 to bfloat16 (round to nearest even)
__global__ void cast_f32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16_rn(src[idx]);
    }
}

// Cast float32 to float16 (round to nearest)
__global__ void cast_f32_to_f16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Cast bfloat16 to float32
__global__ void cast_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

// Cast float16 to float32
__global__ void cast_f16_to_f32_kernel(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

} // namespace nn
} // namespace ops
} // namespace pygpukit
