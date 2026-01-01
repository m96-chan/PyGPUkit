/**
 * Cross-Attention kernels for diffusion models
 *
 * Cross-attention for text-to-image conditioning:
 *   Q: [n_heads, q_len, head_dim] (from image latents)
 *   K: [n_heads, kv_len, head_dim] (from text embeddings)
 *   V: [n_heads, kv_len, head_dim] (from text embeddings)
 *   Output: [n_heads, q_len, head_dim]
 *
 * Unlike self-attention, there is NO causal mask.
 * Each query position can attend to all key positions.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace pygpukit {
namespace ops {
namespace nn {

// Cross-attention kernel (no causal mask)
// Each block handles one (head, query_position) pair
__global__ void cross_attention_f32_kernel(
    const float* __restrict__ Q,      // [n_heads, q_len, head_dim]
    const float* __restrict__ K,      // [n_heads, kv_len, head_dim]
    const float* __restrict__ V,      // [n_heads, kv_len, head_dim]
    float* __restrict__ output,       // [n_heads, q_len, head_dim]
    int n_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale                       // 1/sqrt(head_dim)
) {
    // Each block handles one (head, query_pos) pair
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    // Pointers for this head
    const float* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const float* K_head = K + head_idx * kv_len * head_dim;
    const float* V_head = V + head_idx * kv_len * head_dim;
    float* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    // Shared memory for scores
    extern __shared__ float shared[];
    float* scores = shared;  // [kv_len]

    // Step 1: Compute attention scores and find max
    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        // Dot product Q[q_pos] @ K[kv_pos]
        for (int d = 0; d < head_dim; d++) {
            score += Q_head[d] * K_head[kv_pos * head_dim + d];
        }
        score *= scale;
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
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    // Step 2: Compute exp(score - max) and sum
    float sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum += exp_score;
    }

    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    // Step 3: Normalize scores
    float inv_sum = 1.0f / row_sum;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Compute output = weights @ V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            out_val += scores[kv_pos] * V_head[kv_pos * head_dim + d];
        }
        out_head[d] = out_val;
    }
}

// BF16 Cross-attention (compute in FP32)
__global__ void cross_attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ output,
    int n_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    const __nv_bfloat16* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __nv_bfloat16* K_head = K + head_idx * kv_len * head_dim;
    const __nv_bfloat16* V_head = V + head_idx * kv_len * head_dim;
    __nv_bfloat16* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    extern __shared__ float shared[];
    float* scores = shared;

    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(Q_head[d]) * __bfloat162float(K_head[kv_pos * head_dim + d]);
        }
        score *= scale;
        scores[kv_pos] = score;
        if (score > max_score) max_score = score;
    }

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
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    float sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum += exp_score;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            out_val += scores[kv_pos] * __bfloat162float(V_head[kv_pos * head_dim + d]);
        }
        out_head[d] = __float2bfloat16(out_val);
    }
}

// FP16 Cross-attention (compute in FP32)
__global__ void cross_attention_f16_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ output,
    int n_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    const __half* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __half* K_head = K + head_idx * kv_len * head_dim;
    const __half* V_head = V + head_idx * kv_len * head_dim;
    __half* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    extern __shared__ float shared[];
    float* scores = shared;

    float max_score = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(Q_head[d]) * __half2float(K_head[kv_pos * head_dim + d]);
        }
        score *= scale;
        scores[kv_pos] = score;
        if (score > max_score) max_score = score;
    }

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
        max_score = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[threadIdx.x] : -INFINITY;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }

    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_score;
    __syncthreads();

    float sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - row_max);
        scores[kv_pos] = exp_score;
        sum += exp_score;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float row_sum;
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int kv_pos = threadIdx.x; kv_pos < kv_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            out_val += scores[kv_pos] * __half2float(V_head[kv_pos * head_dim + d]);
        }
        out_head[d] = __float2half(out_val);
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit

#endif  // PYGPUKIT_CROSS_ATTENTION_KERNELS_CUH
