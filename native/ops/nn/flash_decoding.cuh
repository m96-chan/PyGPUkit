/**
 * Flash-Decoding: Decode-specific Attention Optimization
 *
 * For decode phase (q_len=1, batch=1), standard SDPA underutilizes GPU:
 * - Only n_heads blocks (e.g., 32 for Qwen3-8B)
 * - RTX 3090 Ti has 84 SMs → massive underutilization
 *
 * Flash-Decoding parallelizes over KV sequence length:
 * - Phase 1: Each block handles one (head, chunk) pair
 *   - num_blocks = n_heads * num_chunks
 *   - Each block computes partial softmax and weighted sum
 * - Phase 2: Reduction combines partial results
 *   - Uses log-sum-exp trick for numerical stability
 *
 * Reference: Flash-Decoding (Tri Dao et al.)
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace pygpukit {
namespace ops {
namespace flash_decoding {

// Configuration
constexpr int CHUNK_SIZE = 256;      // KV elements per chunk
constexpr int BLOCK_SIZE = 256;      // Threads per block
constexpr int WARP_SIZE = 32;

//-----------------------------------------------------------------------------
// Warp-level reduction utilities
//-----------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

//-----------------------------------------------------------------------------
// Phase 1: Chunk-parallel attention kernel (FP16)
//
// Each block processes one (head, chunk) pair:
// - Computes QK^T for chunk elements
// - Applies max-trick for numerical stability
// - Computes partial softmax and weighted sum with V
//
// Input layout (matches existing SDPA):
// - Q: [n_heads, 1, head_dim]
// - K_cache: [n_heads, kv_stride, head_dim]  (kv_stride = max_seq_len)
// - V_cache: [n_heads, kv_stride, head_dim]
//
// Grid: (num_chunks, n_heads, 1)
// Block: (BLOCK_SIZE, 1, 1)
//
// Outputs:
// - partial_out: [n_heads, num_chunks, head_dim] - weighted sums
// - partial_max: [n_heads, num_chunks] - max scores per chunk
// - partial_sum: [n_heads, num_chunks] - sum of exp(score - max)
//-----------------------------------------------------------------------------

__global__ void flash_decoding_phase1_f16_kernel(
    const __half* __restrict__ Q,        // [n_heads, 1, head_dim]
    const __half* __restrict__ K_cache,  // [n_heads, kv_stride, head_dim]
    const __half* __restrict__ V_cache,  // [n_heads, kv_stride, head_dim]
    float* __restrict__ partial_out,     // [n_heads, num_chunks, head_dim]
    float* __restrict__ partial_max,     // [n_heads, num_chunks]
    float* __restrict__ partial_sum,     // [n_heads, num_chunks]
    int n_heads,
    int head_dim,
    int kv_len,                          // Actual KV sequence length (context_len)
    int kv_stride,                       // Max sequence length (cache dimension)
    int num_chunks,
    float scale                          // 1/sqrt(head_dim)
) {
    const int chunk_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Chunk boundaries
    const int chunk_start = chunk_idx * CHUNK_SIZE;
    const int chunk_end = min(chunk_start + CHUNK_SIZE, kv_len);
    const int chunk_len = chunk_end - chunk_start;

    // Output index for this (head, chunk) pair
    const int out_idx = head_idx * num_chunks + chunk_idx;

    // Early exit for empty chunks
    if (chunk_len <= 0) {
        if (tid == 0) {
            partial_max[out_idx] = -FLT_MAX;
            partial_sum[out_idx] = 0.0f;
        }
        // Zero out partial output
        float* out_ptr = partial_out + out_idx * head_dim;
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            out_ptr[d] = 0.0f;
        }
        return;
    }

    // Shared memory layout:
    // [0, head_dim): s_q - query vector
    // [head_dim, head_dim + CHUNK_SIZE): s_scores - attention scores
    // [head_dim + CHUNK_SIZE, 2*head_dim + CHUNK_SIZE): s_out - output accumulator
    extern __shared__ char smem[];
    float* s_q = reinterpret_cast<float*>(smem);
    float* s_scores = s_q + head_dim;
    float* s_out = s_scores + CHUNK_SIZE;

    // Load Q into shared memory (coalesced read)
    // Q layout: [n_heads, 1, head_dim] -> q_ptr = Q + head_idx * head_dim
    const __half* q_ptr = Q + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        s_q[d] = __half2float(q_ptr[d]);
    }

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        s_out[d] = 0.0f;
    }
    __syncthreads();

    // K/V base pointers for this head
    // K_cache layout: [n_heads, kv_stride, head_dim]
    const __half* k_base = K_cache + head_idx * kv_stride * head_dim;
    const __half* v_base = V_cache + head_idx * kv_stride * head_dim;

    // Phase 1a: Compute attention scores for this chunk
    // Each thread handles multiple KV positions
    float thread_max = -FLT_MAX;

    for (int kv_local = tid; kv_local < chunk_len; kv_local += BLOCK_SIZE) {
        const int kv_pos = chunk_start + kv_local;

        // K at position kv_pos: k_base + kv_pos * head_dim
        const __half* k_ptr = k_base + kv_pos * head_dim;

        // Dot product: Q · K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += s_q[d] * __half2float(k_ptr[d]);
        }
        score *= scale;

        s_scores[kv_local] = score;
        thread_max = fmaxf(thread_max, score);
    }
    __syncthreads();

    // Phase 1b: Reduce max across threads
    // Warp-level reduction first
    float warp_max = warp_reduce_max(thread_max);

    // Store warp maxes in shared memory (reuse end of s_out)
    __shared__ float s_warp_max[BLOCK_SIZE / WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // Final reduction by first warp
    float block_max = -FLT_MAX;
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        block_max = s_warp_max[tid];
    }
    block_max = warp_reduce_max(block_max);

    // Broadcast max to all threads
    if (tid == 0) {
        s_warp_max[0] = block_max;
    }
    __syncthreads();
    block_max = s_warp_max[0];

    // Phase 1c: Compute exp(score - max) and sum
    float thread_sum = 0.0f;
    for (int kv_local = tid; kv_local < chunk_len; kv_local += BLOCK_SIZE) {
        float exp_score = expf(s_scores[kv_local] - block_max);
        s_scores[kv_local] = exp_score;  // Reuse for weighted sum
        thread_sum += exp_score;
    }
    __syncthreads();

    // Reduce sum across threads
    float warp_sum = warp_reduce_sum(thread_sum);

    __shared__ float s_warp_sum[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        block_sum = s_warp_sum[tid];
    }
    block_sum = warp_reduce_sum(block_sum);

    // Phase 1d: Compute weighted sum: attn_weight * V
    // Sequential over kv positions (already in shared mem), parallel over head_dim
    for (int kv_local = 0; kv_local < chunk_len; kv_local++) {
        const int kv_pos = chunk_start + kv_local;
        const float attn_weight = s_scores[kv_local];  // exp(score - max)

        // V at position kv_pos
        const __half* v_ptr = v_base + kv_pos * head_dim;

        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            s_out[d] += attn_weight * __half2float(v_ptr[d]);
        }
    }
    __syncthreads();

    // Store partial results
    float* out_ptr = partial_out + out_idx * head_dim;
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        out_ptr[d] = s_out[d];
    }

    // Store max and sum for reduction phase
    if (tid == 0) {
        partial_max[out_idx] = block_max;
        partial_sum[out_idx] = block_sum;
    }
}

//-----------------------------------------------------------------------------
// Phase 2: Reduction kernel
//
// Combines partial results from all chunks using log-sum-exp trick:
// - Find global max across all chunks
// - Rescale each chunk's sum: sum_i * exp(max_i - global_max)
// - Combine weighted sums with rescaling
//
// Grid: (n_heads, 1, 1)
// Block: (128 or head_dim, 1, 1)
//
// Output: [n_heads, 1, head_dim]
//-----------------------------------------------------------------------------

__global__ void flash_decoding_phase2_f16_kernel(
    const float* __restrict__ partial_out,   // [n_heads, num_chunks, head_dim]
    const float* __restrict__ partial_max,   // [n_heads, num_chunks]
    const float* __restrict__ partial_sum,   // [n_heads, num_chunks]
    __half* __restrict__ output,             // [n_heads, 1, head_dim]
    int n_heads,
    int num_chunks,
    int head_dim
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Shared memory for reduction
    extern __shared__ char smem2[];
    float* s_out = reinterpret_cast<float*>(smem2);  // [head_dim]

    // Load partial max and sum for this head
    const float* max_ptr = partial_max + head_idx * num_chunks;
    const float* sum_ptr = partial_sum + head_idx * num_chunks;

    // Find global max across all chunks (single thread does this, small num_chunks)
    float global_max = -FLT_MAX;
    for (int c = 0; c < num_chunks; c++) {
        global_max = fmaxf(global_max, max_ptr[c]);
    }

    // Compute total sum with rescaling
    float total_sum = 0.0f;
    for (int c = 0; c < num_chunks; c++) {
        float rescale = expf(max_ptr[c] - global_max);
        total_sum += sum_ptr[c] * rescale;
    }

    // Inverse for final normalization
    float inv_total_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_out[d] = 0.0f;
    }
    __syncthreads();

    // Combine weighted sums with rescaling
    for (int c = 0; c < num_chunks; c++) {
        const float* chunk_out = partial_out + (head_idx * num_chunks + c) * head_dim;
        float rescale = expf(max_ptr[c] - global_max);

        for (int d = tid; d < head_dim; d += blockDim.x) {
            s_out[d] += chunk_out[d] * rescale;
        }
    }
    __syncthreads();

    // Final normalization and write output
    // Output layout: [n_heads, 1, head_dim] -> out_ptr = output + head_idx * head_dim
    __half* out_ptr = output + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = __float2half(s_out[d] * inv_total_sum);
    }
}

//-----------------------------------------------------------------------------
// Host-callable dispatch function
//-----------------------------------------------------------------------------

inline cudaError_t flash_decoding_f16(
    const __half* Q,          // [n_heads, 1, head_dim]
    const __half* K_cache,    // [n_heads, kv_stride, head_dim]
    const __half* V_cache,    // [n_heads, kv_stride, head_dim]
    __half* output,           // [n_heads, 1, head_dim]
    float* workspace,         // Temporary workspace for partial results
    int n_heads,
    int head_dim,
    int kv_len,               // Actual context length
    int kv_stride,            // Max sequence length (cache dimension)
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int num_chunks = (kv_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Workspace layout:
    // - partial_out: n_heads * num_chunks * head_dim floats
    // - partial_max: n_heads * num_chunks floats
    // - partial_sum: n_heads * num_chunks floats
    float* partial_out = workspace;
    float* partial_max = partial_out + n_heads * num_chunks * head_dim;
    float* partial_sum = partial_max + n_heads * num_chunks;

    // Phase 1: Chunk-parallel attention
    {
        dim3 grid(num_chunks, n_heads, 1);
        dim3 block(BLOCK_SIZE, 1, 1);

        // Shared memory: s_q[head_dim] + s_scores[CHUNK_SIZE] + s_out[head_dim]
        size_t smem_size = (head_dim + CHUNK_SIZE + head_dim) * sizeof(float);

        flash_decoding_phase1_f16_kernel<<<grid, block, smem_size, stream>>>(
            Q, K_cache, V_cache,
            partial_out, partial_max, partial_sum,
            n_heads, head_dim, kv_len, kv_stride, num_chunks, scale
        );
    }

    // Phase 2: Reduction
    {
        dim3 grid2(n_heads, 1, 1);
        dim3 block2(min(head_dim, 128), 1, 1);

        // Shared memory: s_out[head_dim]
        size_t smem_size2 = head_dim * sizeof(float);

        flash_decoding_phase2_f16_kernel<<<grid2, block2, smem_size2, stream>>>(
            partial_out, partial_max, partial_sum,
            output,
            n_heads, num_chunks, head_dim
        );
    }

    return cudaGetLastError();
}

// Calculate required workspace size in bytes
inline size_t flash_decoding_workspace_size(int n_heads, int head_dim, int kv_len) {
    const int num_chunks = (kv_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    // partial_out: n_heads * num_chunks * head_dim floats
    // partial_max: n_heads * num_chunks floats
    // partial_sum: n_heads * num_chunks floats
    return sizeof(float) * (n_heads * num_chunks * head_dim + n_heads * num_chunks * 2);
}

// Get number of chunks for given kv_len
inline int get_num_chunks(int kv_len) {
    return (kv_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
}

}  // namespace flash_decoding
}  // namespace ops
}  // namespace pygpukit
