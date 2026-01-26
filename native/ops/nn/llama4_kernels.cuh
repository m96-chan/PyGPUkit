/**
 * Llama 4 architecture specific kernels
 *
 * Implements:
 * - L2 Norm (Llama4TextL2Norm): y = x * rsqrt(mean(x^2) + eps)
 * - SDPA with iRoPE temperature scaling
 *
 * Reference: HuggingFace Transformers Llama4
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
// L2 Norm (Llama4TextL2Norm)
// ============================================================================
//
// Formula: y = x * rsqrt(mean(x^2) + eps)
// Unlike RMSNorm, no gamma scaling is applied.
// Used for QK normalization in Llama 4 attention.
//
// Input: [batch, features] or flattened [seq_len * num_heads, head_dim]
// Output: same shape as input

__global__ void l2norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t batch_size,
    size_t features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + row * features;
    __nv_bfloat16* row_output = output + row * features;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = __bfloat162float(row_input[i]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block-level reduction
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
                     ? shared_sum[threadIdx.x]
                     : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float scale;
    if (threadIdx.x == 0) {
        // L2 norm: rsqrt(mean(x^2) + eps)
        scale = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    // Normalize (no gamma, unlike RMSNorm)
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        row_output[i] = __float2bfloat16(x * scale);
    }
}

__global__ void l2norm_f16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    size_t batch_size,
    size_t features,
    float eps
) {
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
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
                     ? shared_sum[threadIdx.x]
                     : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __half2float(row_input[i]);
        row_output[i] = __float2half(x * scale);
    }
}

__global__ void l2norm_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t batch_size,
    size_t features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * features;
    float* row_output = output + row * features;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = row_input[i];
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
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
                     ? shared_sum[threadIdx.x]
                     : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = row_input[i];
        row_output[i] = x * scale;
    }
}

// ============================================================================
// iRoPE Temperature Scaling
// ============================================================================
//
// Llama 4 uses position-dependent temperature scaling instead of RoPE
// for NoPE (No Positional Encoding) layers.
//
// Formula: scale = log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1.0
// Applied to Q before attention: Q_scaled = Q * scale
//
// Input Q: [seq_len, num_heads, head_dim]
// positions: [seq_len]
// Output: [seq_len, num_heads, head_dim]

__global__ void irope_scale_q_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const int64_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ Q_out,
    int seq_len,
    int num_heads,
    int head_dim,
    float attn_scale,
    float floor_scale
) {
    // Each block handles one (seq_pos, head) pair
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq_len || head_idx >= num_heads) return;

    // Compute temperature scale for this position
    int64_t pos = positions[seq_idx];
    float temp_scale = log1pf(floorf((float)(pos + 1) / floor_scale)) * attn_scale + 1.0f;

    // Pointers to this head's Q vector
    int offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    const __nv_bfloat16* q_in = Q + offset;
    __nv_bfloat16* q_out = Q_out + offset;

    // Scale Q by temperature
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float q_val = __bfloat162float(q_in[d]);
        q_out[d] = __float2bfloat16(q_val * temp_scale);
    }
}

__global__ void irope_scale_q_f16_kernel(
    const __half* __restrict__ Q,
    const int64_t* __restrict__ positions,
    __half* __restrict__ Q_out,
    int seq_len,
    int num_heads,
    int head_dim,
    float attn_scale,
    float floor_scale
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq_len || head_idx >= num_heads) return;

    int64_t pos = positions[seq_idx];
    float temp_scale = log1pf(floorf((float)(pos + 1) / floor_scale)) * attn_scale + 1.0f;

    int offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    const __half* q_in = Q + offset;
    __half* q_out = Q_out + offset;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float q_val = __half2float(q_in[d]);
        q_out[d] = __float2half(q_val * temp_scale);
    }
}

// ============================================================================
// SDPA with iRoPE (fused temperature scaling)
// ============================================================================
//
// Fused SDPA kernel for Llama 4 NoPE layers.
// Applies temperature scaling to Q during attention computation.
//
// Q: [n_heads, q_len, head_dim]
// K: [n_kv_heads, kv_len, head_dim]
// V: [n_kv_heads, kv_len, head_dim]
// positions: [q_len]
// Output: [n_heads, q_len, head_dim]

__global__ void sdpa_irope_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int64_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ output,
    int n_heads,
    int n_kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float attn_scale,
    float floor_scale,
    int causal_offset
) {
    // Each block handles one (head, query_pos) pair
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    // GQA: map query head to KV head
    int kv_head = head_idx * n_kv_heads / n_heads;

    // Compute temperature scale for this position
    int64_t pos = positions[q_pos];
    float temp_scale = log1pf(floorf((float)(pos + 1) / floor_scale)) * attn_scale + 1.0f;

    // Pointers
    const __nv_bfloat16* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __nv_bfloat16* K_head = K + kv_head * kv_len * head_dim;
    const __nv_bfloat16* V_head = V + kv_head * kv_len * head_dim;
    __nv_bfloat16* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    // Causal mask: query at position q_pos can attend to positions 0..(causal_offset + q_pos)
    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    // Shared memory for scores
    extern __shared__ float shared[];
    float* scores = shared;

    // Step 1: Compute attention scores with temperature scaling
    // Formula: score = Q @ K^T * temp_scale / sqrt(head_dim)
    float scale_factor = temp_scale * rsqrtf((float)head_dim);
    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        if (kv_pos < max_attend) {
            // Dot product Q[q_pos] @ K[kv_pos] * scale_factor
            for (int d = 0; d < head_dim; d++) {
                float q_val = __bfloat162float(Q_head[d]);
                float k_val = __bfloat162float(K_head[kv_pos * head_dim + d]);
                score += q_val * k_val;
            }
            score *= scale_factor;
        } else {
            score = -INFINITY;
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
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
                        ? shared_max[threadIdx.x]
                        : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    // Step 2: Compute softmax weights
    float sum_exp = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum_exp += exp_score;
    }

    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
                      ? shared_sum[threadIdx.x]
                      : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
    }

    __shared__ float inv_sum;
    if (threadIdx.x == 0) inv_sum = 1.0f / sum_exp;
    __syncthreads();

    // Normalize weights
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            float weight = scores[kv_pos];
            float v_val = __bfloat162float(V_head[kv_pos * head_dim + d]);
            acc += weight * v_val;
        }
        out_head[d] = __float2bfloat16(acc);
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
