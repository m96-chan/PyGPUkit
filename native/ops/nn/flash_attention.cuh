/**
 * Flash Attention 2 Implementation
 *
 * Memory-efficient attention using tiled computation with online softmax.
 * Reduces memory complexity from O(n²) to O(n) by not materializing the full
 * attention matrix.
 *
 * Reference: "FlashAttention-2: Faster Attention with Better Parallelism
 *            and Work Partitioning" (Dao, 2023)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace pygpukit {
namespace ops {
namespace nn {

// Tile size for K/V chunking
// TILE_KV: Number of KV positions processed per iteration
// Should fit in shared memory along with Q tile
// For head_dim=128: smem = 4 * (128 + 2*32*128 + 32) = 33KB (fits in 48KB limit)
constexpr int FLASH_TILE_KV = 32;

/**
 * Flash Attention 2 kernel - FP32
 *
 * Uses online softmax algorithm to compute attention without materializing
 * the full N×N attention matrix. Processes KV in tiles of FLASH_TILE_KV.
 *
 * Memory usage: O(TILE_KV * head_dim) per block instead of O(kv_len)
 *
 * Grid: (n_heads, q_len)
 * Block: (BLOCK_SIZE) where BLOCK_SIZE handles head_dim elements
 */
__global__ void flash_attention_f32_kernel(
    const float* __restrict__ Q,      // [n_heads, q_len, head_dim]
    const float* __restrict__ K,      // [n_heads, kv_stride, head_dim]
    const float* __restrict__ V,      // [n_heads, kv_stride, head_dim]
    float* __restrict__ output,       // [n_heads, q_len, head_dim]
    int n_heads,
    int q_len,
    int kv_len,                       // Number of KV positions to attend to
    int kv_stride,                    // Actual K/V tensor size (for pointer arithmetic)
    int head_dim,
    float scale,                      // 1/sqrt(head_dim)
    int causal_offset                 // kv_len - q_len (for proper causal masking)
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    // Pointers for this head/query position (use kv_stride for K/V, not kv_len)
    const float* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const float* K_head = K + head_idx * kv_stride * head_dim;
    const float* V_head = V + head_idx * kv_stride * head_dim;
    float* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    // Causal mask: can attend to positions 0..(causal_offset + q_pos)
    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    // Shared memory layout:
    // - Q tile: [head_dim] - query vector for this position
    // - K tile: [FLASH_TILE_KV, head_dim] - current K tile
    // - V tile: [FLASH_TILE_KV, head_dim] - current V tile
    // - scores: [FLASH_TILE_KV] - attention scores for current tile
    extern __shared__ float smem[];

    float* Q_tile = smem;                                           // [head_dim]
    float* K_tile = Q_tile + head_dim;                              // [FLASH_TILE_KV * head_dim]
    float* V_tile = K_tile + FLASH_TILE_KV * head_dim;              // [FLASH_TILE_KV * head_dim]
    float* tile_scores = V_tile + FLASH_TILE_KV * head_dim;         // [FLASH_TILE_KV]

    // Load Q into shared memory (one-time load)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        Q_tile[d] = Q_head[d];
    }
    __syncthreads();

    // Online softmax state (per-thread accumulator for different head_dim elements)
    // We use registers for output accumulation
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Output accumulator - each thread handles some dimensions
    // For simplicity, accumulate in registers then write
    float out_acc[128];  // Assuming head_dim <= 128 (common for most models)
    for (int d = 0; d < head_dim && d < 128; d++) {
        out_acc[d] = 0.0f;
    }

    // Process KV in tiles
    int num_tiles = (max_attend + FLASH_TILE_KV - 1) / FLASH_TILE_KV;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * FLASH_TILE_KV;
        int tile_end = min(tile_start + FLASH_TILE_KV, max_attend);
        int tile_size = tile_end - tile_start;

        // Load K tile into shared memory
        for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + kv_local;
            K_tile[kv_local * head_dim + d] = K_head[kv_pos * head_dim + d];
        }

        // Load V tile into shared memory
        for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + kv_local;
            V_tile[kv_local * head_dim + d] = V_head[kv_pos * head_dim + d];
        }
        __syncthreads();

        // Compute attention scores for this tile: Q @ K^T
        // Each thread computes scores for some KV positions
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q_tile[d] * K_tile[kv_local * head_dim + d];
            }
            tile_scores[kv_local] = score * scale;
        }
        __syncthreads();

        // Find max in this tile (for online softmax)
        float tile_max = -INFINITY;
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            tile_max = fmaxf(tile_max, tile_scores[kv_local]);
        }

        // Warp reduction for tile max
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }

        __shared__ float shared_max[32];
        int lane = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;
        if (lane == 0) shared_max[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : -INFINITY;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
        }

        __shared__ float block_tile_max;
        if (threadIdx.x == 0) block_tile_max = tile_max;
        __syncthreads();
        tile_max = block_tile_max;

        // Online softmax update
        // new_max = max(running_max, tile_max)
        // correction = exp(running_max - new_max)
        // running_sum = running_sum * correction + sum(exp(scores - new_max))
        // output = output * correction + weighted_values

        float new_max = fmaxf(running_max, tile_max);
        float correction = expf(running_max - new_max);

        // Compute exp(scores - new_max) and sum
        float tile_sum = 0.0f;
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            float exp_score = expf(tile_scores[kv_local] - new_max);
            tile_scores[kv_local] = exp_score;  // Store normalized score
            tile_sum += exp_score;
        }

        // Reduce tile sum
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
        }

        __shared__ float shared_sum[32];
        if (lane == 0) shared_sum[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
        }

        __shared__ float block_tile_sum;
        if (threadIdx.x == 0) block_tile_sum = tile_sum;
        __syncthreads();
        tile_sum = block_tile_sum;

        // Update running state
        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        // Compute weighted V and accumulate (with correction factor)
        // Each thread handles subset of head_dim
        __syncthreads();  // Ensure tile_scores is ready

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float weighted_v = 0.0f;
            for (int kv_local = 0; kv_local < tile_size; kv_local++) {
                weighted_v += tile_scores[kv_local] * V_tile[kv_local * head_dim + d];
            }
            out_acc[d] = out_acc[d] * correction + weighted_v;
        }

        __syncthreads();  // Before loading next tile
    }

    // Final normalization and write output
    float inv_sum = 1.0f / running_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out_head[d] = out_acc[d] * inv_sum;
    }
}

/**
 * Flash Attention 2 kernel - FP16 (compute in FP32 for precision)
 */
__global__ void flash_attention_f16_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ output,
    int n_heads,
    int q_len,
    int kv_len,
    int kv_stride,
    int head_dim,
    float scale,
    int causal_offset
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    const __half* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __half* K_head = K + head_idx * kv_stride * head_dim;
    const __half* V_head = V + head_idx * kv_stride * head_dim;
    __half* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    extern __shared__ float smem[];

    float* Q_tile = smem;
    float* K_tile = Q_tile + head_dim;
    float* V_tile = K_tile + FLASH_TILE_KV * head_dim;
    float* tile_scores = V_tile + FLASH_TILE_KV * head_dim;

    // Load Q into shared memory (convert to FP32)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        Q_tile[d] = __half2float(Q_head[d]);
    }
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    float out_acc[128];
    for (int d = 0; d < head_dim && d < 128; d++) {
        out_acc[d] = 0.0f;
    }

    int num_tiles = (max_attend + FLASH_TILE_KV - 1) / FLASH_TILE_KV;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * FLASH_TILE_KV;
        int tile_end = min(tile_start + FLASH_TILE_KV, max_attend);
        int tile_size = tile_end - tile_start;

        // Load K tile (convert to FP32)
        for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + kv_local;
            K_tile[kv_local * head_dim + d] = __half2float(K_head[kv_pos * head_dim + d]);
        }

        // Load V tile (convert to FP32)
        for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + kv_local;
            V_tile[kv_local * head_dim + d] = __half2float(V_head[kv_pos * head_dim + d]);
        }
        __syncthreads();

        // Compute scores
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q_tile[d] * K_tile[kv_local * head_dim + d];
            }
            tile_scores[kv_local] = score * scale;
        }
        __syncthreads();

        // Find tile max
        float tile_max = -INFINITY;
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            tile_max = fmaxf(tile_max, tile_scores[kv_local]);
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }

        __shared__ float shared_max[32];
        int lane = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;
        if (lane == 0) shared_max[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : -INFINITY;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
        }

        __shared__ float block_tile_max;
        if (threadIdx.x == 0) block_tile_max = tile_max;
        __syncthreads();
        tile_max = block_tile_max;

        float new_max = fmaxf(running_max, tile_max);
        float correction = expf(running_max - new_max);

        float tile_sum = 0.0f;
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            float exp_score = expf(tile_scores[kv_local] - new_max);
            tile_scores[kv_local] = exp_score;
            tile_sum += exp_score;
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
        }

        __shared__ float shared_sum[32];
        if (lane == 0) shared_sum[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
        }

        __shared__ float block_tile_sum;
        if (threadIdx.x == 0) block_tile_sum = tile_sum;
        __syncthreads();
        tile_sum = block_tile_sum;

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        __syncthreads();

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float weighted_v = 0.0f;
            for (int kv_local = 0; kv_local < tile_size; kv_local++) {
                weighted_v += tile_scores[kv_local] * V_tile[kv_local * head_dim + d];
            }
            out_acc[d] = out_acc[d] * correction + weighted_v;
        }

        __syncthreads();
    }

    float inv_sum = 1.0f / running_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out_head[d] = __float2half(out_acc[d] * inv_sum);
    }
}

/**
 * Flash Attention 2 kernel - BF16 (compute in FP32 for precision)
 */
__global__ void flash_attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ output,
    int n_heads,
    int q_len,
    int kv_len,
    int kv_stride,
    int head_dim,
    float scale,
    int causal_offset
) {
    int head_idx = blockIdx.x;
    int q_pos = blockIdx.y;

    if (head_idx >= n_heads || q_pos >= q_len) return;

    const __nv_bfloat16* Q_head = Q + head_idx * q_len * head_dim + q_pos * head_dim;
    const __nv_bfloat16* K_head = K + head_idx * kv_stride * head_dim;
    const __nv_bfloat16* V_head = V + head_idx * kv_stride * head_dim;
    __nv_bfloat16* out_head = output + head_idx * q_len * head_dim + q_pos * head_dim;

    int max_attend = causal_offset + q_pos + 1;
    if (max_attend > kv_len) max_attend = kv_len;

    extern __shared__ float smem[];

    float* Q_tile = smem;
    float* K_tile = Q_tile + head_dim;
    float* V_tile = K_tile + FLASH_TILE_KV * head_dim;
    float* tile_scores = V_tile + FLASH_TILE_KV * head_dim;

    // Load Q into shared memory (convert to FP32)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        Q_tile[d] = __bfloat162float(Q_head[d]);
    }
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    float out_acc[128];
    for (int d = 0; d < head_dim && d < 128; d++) {
        out_acc[d] = 0.0f;
    }

    int num_tiles = (max_attend + FLASH_TILE_KV - 1) / FLASH_TILE_KV;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * FLASH_TILE_KV;
        int tile_end = min(tile_start + FLASH_TILE_KV, max_attend);
        int tile_size = tile_end - tile_start;

        // Load K tile (convert to FP32)
        for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + kv_local;
            K_tile[kv_local * head_dim + d] = __bfloat162float(K_head[kv_pos * head_dim + d]);
        }

        // Load V tile (convert to FP32)
        for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
            int kv_local = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + kv_local;
            V_tile[kv_local * head_dim + d] = __bfloat162float(V_head[kv_pos * head_dim + d]);
        }
        __syncthreads();

        // Compute scores
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q_tile[d] * K_tile[kv_local * head_dim + d];
            }
            tile_scores[kv_local] = score * scale;
        }
        __syncthreads();

        // Find tile max
        float tile_max = -INFINITY;
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            tile_max = fmaxf(tile_max, tile_scores[kv_local]);
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }

        __shared__ float shared_max[32];
        int lane = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;
        if (lane == 0) shared_max[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : -INFINITY;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
        }

        __shared__ float block_tile_max;
        if (threadIdx.x == 0) block_tile_max = tile_max;
        __syncthreads();
        tile_max = block_tile_max;

        float new_max = fmaxf(running_max, tile_max);
        float correction = expf(running_max - new_max);

        float tile_sum = 0.0f;
        for (int kv_local = threadIdx.x; kv_local < tile_size; kv_local += blockDim.x) {
            float exp_score = expf(tile_scores[kv_local] - new_max);
            tile_scores[kv_local] = exp_score;
            tile_sum += exp_score;
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
        }

        __shared__ float shared_sum[32];
        if (lane == 0) shared_sum[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
        }

        __shared__ float block_tile_sum;
        if (threadIdx.x == 0) block_tile_sum = tile_sum;
        __syncthreads();
        tile_sum = block_tile_sum;

        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;

        __syncthreads();

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float weighted_v = 0.0f;
            for (int kv_local = 0; kv_local < tile_size; kv_local++) {
                weighted_v += tile_scores[kv_local] * V_tile[kv_local * head_dim + d];
            }
            out_acc[d] = out_acc[d] * correction + weighted_v;
        }

        __syncthreads();
    }

    float inv_sum = 1.0f / running_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out_head[d] = __float2bfloat16(out_acc[d] * inv_sum);
    }
}

/**
 * Calculate shared memory size needed for Flash Attention
 */
inline size_t flash_attention_smem_size(int head_dim) {
    // Q_tile: head_dim
    // K_tile: FLASH_TILE_KV * head_dim
    // V_tile: FLASH_TILE_KV * head_dim
    // tile_scores: FLASH_TILE_KV
    return sizeof(float) * (head_dim + 2 * FLASH_TILE_KV * head_dim + FLASH_TILE_KV);
}

} // namespace nn
} // namespace ops
} // namespace pygpukit
