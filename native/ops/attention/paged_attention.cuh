/**
 * Paged Attention Kernels for PyGPUkit (#87)
 *
 * Implements vLLM-style paged attention for efficient KV cache management.
 * Memory is organized into fixed-size pages (blocks) that can be allocated
 * and deallocated dynamically.
 *
 * Key concepts:
 * - Block: A fixed-size memory region (e.g., 16 tokens per block)
 * - Page Table: Maps logical token positions to physical block indices
 * - Block Table: Per-sequence mapping from logical block index to physical block
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace paged {

// Default configuration
constexpr int DEFAULT_BLOCK_SIZE = 16;    // Tokens per block
constexpr int WARP_SIZE = 32;

// ============================================================================
// Paged Attention v1: Single-query attention with paged KV cache
// ============================================================================

/**
 * Paged Attention v1 Kernel (FP16)
 *
 * For each query position, computes attention over paged KV cache.
 * Used during decode phase (one new token per sequence).
 *
 * Q: [num_seqs, num_heads, head_dim] - queries for current tokens
 * K_cache: [num_blocks, num_kv_heads, block_size, head_dim] - paged key cache
 * V_cache: [num_blocks, num_kv_heads, block_size, head_dim] - paged value cache
 * block_tables: [num_seqs, max_num_blocks_per_seq] - maps seq to physical blocks
 * context_lens: [num_seqs] - actual sequence lengths
 * output: [num_seqs, num_heads, head_dim] - attention output
 *
 * Scale: 1/sqrt(head_dim)
 */
__global__ void paged_attention_v1_kernel(
    const __half* __restrict__ Q,           // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ K_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    const __half* __restrict__ V_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,  // [num_seqs]
    __half* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_num_blocks_per_seq,
    float scale
) {
    // Each block handles one (sequence, head) pair
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= num_seqs) return;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);  // GQA support
    int context_len = context_lens[seq_idx];

    // Shared memory for Q vector and partial results
    extern __shared__ float smem[];
    float* q_shared = smem;                           // [head_dim]
    float* logits_shared = q_shared + head_dim;       // [max_context_len] - attention scores

    // Load Q to shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        q_shared[d] = __half2float(Q[q_offset]);
    }
    __syncthreads();

    // Compute attention scores for each KV position
    int num_blocks = (context_len + block_size - 1) / block_size;

    for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += blockDim.x) {
        int block_idx = token_idx / block_size;
        int block_offset = token_idx % block_size;

        // Get physical block index from block table
        int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

        // Compute Q @ K^T for this position
        float score = 0.0f;
        int k_base = physical_block * num_kv_heads * block_size * head_dim +
                     kv_head_idx * block_size * head_dim +
                     block_offset * head_dim;

        for (int d = 0; d < head_dim; d++) {
            score += q_shared[d] * __half2float(K_cache[k_base + d]);
        }

        logits_shared[token_idx] = score * scale;
    }
    __syncthreads();

    // Softmax over attention scores
    // Find max for numerical stability
    float max_logit = -1e20f;
    for (int i = threadIdx.x; i < context_len; i += blockDim.x) {
        max_logit = fmaxf(max_logit, logits_shared[i]);
    }

    // Reduce max across threads
    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffff, max_logit, offset));
    }
    if (lane == 0) shared_max[warp_id] = max_logit;
    __syncthreads();

    if (threadIdx.x == 0) {
        max_logit = shared_max[0];
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 1; i < num_warps; i++) {
            max_logit = fmaxf(max_logit, shared_max[i]);
        }
        shared_max[0] = max_logit;
    }
    __syncthreads();
    max_logit = shared_max[0];

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < context_len; i += blockDim.x) {
        float exp_val = expf(logits_shared[i] - max_logit);
        logits_shared[i] = exp_val;
        sum_exp += exp_val;
    }

    // Reduce sum across threads
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();

    if (threadIdx.x == 0) {
        sum_exp = shared_sum[0];
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 1; i < num_warps; i++) {
            sum_exp += shared_sum[i];
        }
        shared_sum[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = shared_sum[0];

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = threadIdx.x; i < context_len; i += blockDim.x) {
        logits_shared[i] *= inv_sum;
    }
    __syncthreads();

    // Compute output = attention_weights @ V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;

        for (int token_idx = 0; token_idx < context_len; token_idx++) {
            int block_idx = token_idx / block_size;
            int block_offset = token_idx % block_size;
            int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

            int v_offset = physical_block * num_kv_heads * block_size * head_dim +
                           kv_head_idx * block_size * head_dim +
                           block_offset * head_dim + d;

            out_val += logits_shared[token_idx] * __half2float(V_cache[v_offset]);
        }

        int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        output[out_offset] = __float2half(out_val);
    }
}

// ============================================================================
// KV Cache Management Kernels
// ============================================================================

/**
 * Copy new KV entries to paged cache
 *
 * Used after computing K, V for new tokens to store them in the paged cache.
 *
 * K_new: [num_seqs, num_kv_heads, head_dim] - new key vectors
 * V_new: [num_seqs, num_kv_heads, head_dim] - new value vectors
 * K_cache: [num_blocks, num_kv_heads, block_size, head_dim]
 * V_cache: [num_blocks, num_kv_heads, block_size, head_dim]
 * slot_mapping: [num_seqs] - physical slot index for each new token
 */
__global__ void copy_to_paged_cache_kernel(
    const __half* __restrict__ K_new,
    const __half* __restrict__ V_new,
    __half* __restrict__ K_cache,
    __half* __restrict__ V_cache,
    const int32_t* __restrict__ slot_mapping,
    int num_seqs,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    int seq_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;

    if (seq_idx >= num_seqs || kv_head_idx >= num_kv_heads) return;

    int slot = slot_mapping[seq_idx];
    int block_idx = slot / block_size;
    int block_offset = slot % block_size;

    // Compute cache offset
    int cache_offset = block_idx * num_kv_heads * block_size * head_dim +
                       kv_head_idx * block_size * head_dim +
                       block_offset * head_dim;

    // Input offset
    int input_offset = seq_idx * num_kv_heads * head_dim + kv_head_idx * head_dim;

    // Copy K and V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        K_cache[cache_offset + d] = K_new[input_offset + d];
        V_cache[cache_offset + d] = V_new[input_offset + d];
    }
}

/**
 * Reshape and copy KV from prefill format to paged cache
 *
 * During prefill, K/V are computed as [batch, seq_len, num_kv_heads, head_dim].
 * This kernel copies them to the paged cache format.
 */
__global__ void reshape_and_cache_kernel(
    const __half* __restrict__ K,           // [batch, seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ V,           // [batch, seq_len, num_kv_heads, head_dim]
    __half* __restrict__ K_cache,           // [num_blocks, num_kv_heads, block_size, head_dim]
    __half* __restrict__ V_cache,           // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* __restrict__ slot_mapping,  // [total_tokens]
    int total_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    int token_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;

    if (token_idx >= total_tokens || kv_head_idx >= num_kv_heads) return;

    int slot = slot_mapping[token_idx];
    int block_idx = slot / block_size;
    int block_offset = slot % block_size;

    int cache_offset = block_idx * num_kv_heads * block_size * head_dim +
                       kv_head_idx * block_size * head_dim +
                       block_offset * head_dim;

    // Input format: [batch, seq_len, num_kv_heads, head_dim] flattened
    // token_idx is the flattened index
    int input_offset = token_idx * num_kv_heads * head_dim + kv_head_idx * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        K_cache[cache_offset + d] = K[input_offset + d];
        V_cache[cache_offset + d] = V[input_offset + d];
    }
}

} // namespace paged
} // namespace ops
} // namespace pygpukit
