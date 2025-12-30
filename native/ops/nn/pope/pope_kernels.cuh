/**
 * PoPE (Positional Encoding) kernels
 *
 * Additive positional encoding as an alternative to RoPE.
 * Adds sinusoidal or learned position embeddings to Q/K.
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
// PoPE Init Encoding - Generate sinusoidal position embeddings
// ============================================================================

__global__ void pope_init_sinusoidal_f32_kernel(
    float* __restrict__ encoding,
    int max_seq_len,
    int head_dim,
    float base
) {
    // encoding: [max_seq_len, head_dim]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * head_dim;

    if (idx < total) {
        int pos = idx / head_dim;
        int dim = idx % head_dim;

        // Sinusoidal encoding: PE(pos, 2i) = sin(pos / base^(2i/d))
        //                      PE(pos, 2i+1) = cos(pos / base^(2i/d))
        float freq = 1.0f / powf(base, (float)(dim / 2 * 2) / (float)head_dim);
        float angle = (float)pos * freq;

        if (dim % 2 == 0) {
            encoding[idx] = sinf(angle);
        } else {
            encoding[idx] = cosf(angle);
        }
    }
}

// ============================================================================
// PoPE Apply - Add position encoding to Q/K
// ============================================================================

// F32 kernel
__global__ void pope_apply_f32_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ encoding,
    int seq_len,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int start_pos
) {
    // q: [seq_len, n_heads_q, head_dim]
    // k: [seq_len, n_heads_k, head_dim]
    // encoding: [max_seq_len, head_dim]

    int total_q = seq_len * n_heads_q * head_dim;
    int total_k = seq_len * n_heads_k * head_dim;
    int total_work = max(total_q, total_k);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_work) return;

    // Process Q
    if (idx < total_q) {
        int s = idx / (n_heads_q * head_dim);
        int h = (idx / head_dim) % n_heads_q;
        int d = idx % head_dim;

        int pos = start_pos + s;
        float pe = encoding[pos * head_dim + d];
        q[idx] += pe;
    }

    // Process K
    if (idx < total_k) {
        int s = idx / (n_heads_k * head_dim);
        int h = (idx / head_dim) % n_heads_k;
        int d = idx % head_dim;

        int pos = start_pos + s;
        float pe = encoding[pos * head_dim + d];
        k[idx] += pe;
    }
}

// F16 kernel
__global__ void pope_apply_f16_kernel(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const float* __restrict__ encoding,  // Keep encoding in f32 for precision
    int seq_len,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int start_pos
) {
    int total_q = seq_len * n_heads_q * head_dim;
    int total_k = seq_len * n_heads_k * head_dim;
    int total_work = max(total_q, total_k);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_work) return;

    // Process Q
    if (idx < total_q) {
        int s = idx / (n_heads_q * head_dim);
        int d = idx % head_dim;
        int pos = start_pos + s;
        float pe = encoding[pos * head_dim + d];
        float val = __half2float(q[idx]) + pe;
        q[idx] = __float2half(val);
    }

    // Process K
    if (idx < total_k) {
        int s = idx / (n_heads_k * head_dim);
        int d = idx % head_dim;
        int pos = start_pos + s;
        float pe = encoding[pos * head_dim + d];
        float val = __half2float(k[idx]) + pe;
        k[idx] = __float2half(val);
    }
}

// BF16 kernel
__global__ void pope_apply_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ encoding,
    int seq_len,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int start_pos
) {
    int total_q = seq_len * n_heads_q * head_dim;
    int total_k = seq_len * n_heads_k * head_dim;
    int total_work = max(total_q, total_k);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_work) return;

    // Process Q
    if (idx < total_q) {
        int s = idx / (n_heads_q * head_dim);
        int d = idx % head_dim;
        int pos = start_pos + s;
        float pe = encoding[pos * head_dim + d];
        float val = __bfloat162float(q[idx]) + pe;
        q[idx] = __float2bfloat16(val);
    }

    // Process K
    if (idx < total_k) {
        int s = idx / (n_heads_k * head_dim);
        int d = idx % head_dim;
        int pos = start_pos + s;
        float pe = encoding[pos * head_dim + d];
        float val = __bfloat162float(k[idx]) + pe;
        k[idx] = __float2bfloat16(val);
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
