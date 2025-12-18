/**
 * Continuous Batching Infrastructure for PyGPUkit (#86)
 *
 * Enables vLLM-style iteration-level batching for efficient multi-request inference.
 *
 * Key concepts:
 * - Request: A single inference request with input tokens
 * - Sequence: Generated output for a request
 * - Batch: Group of sequences processed together
 * - Iteration: One forward pass (prefill or decode step)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace pygpukit {
namespace ops {
namespace batch {

// ============================================================================
// Constants
// ============================================================================

constexpr int MAX_BATCH_SIZE = 256;      // Max sequences per batch
constexpr int MAX_SEQ_LEN = 8192;        // Max sequence length
constexpr int DEFAULT_BLOCK_SIZE = 16;   // KV cache block size

// ============================================================================
// Request/Sequence State
// ============================================================================

enum class SequenceStatus : int32_t {
    WAITING = 0,        // Waiting for prefill
    RUNNING = 1,        // Currently generating
    FINISHED = 2,       // Generation complete
    SWAPPED = 3,        // Swapped out to CPU
};

/**
 * Sequence metadata (per-sequence state)
 */
struct SequenceMetadata {
    int32_t seq_id;             // Unique sequence ID
    int32_t prompt_len;         // Original prompt length
    int32_t output_len;         // Generated output length
    int32_t max_output_len;     // Maximum output length
    SequenceStatus status;      // Current status
    int32_t block_table_offset; // Offset in block tables array
    int32_t num_blocks;         // Number of allocated blocks
};

// ============================================================================
// Batch Formation Kernels
// ============================================================================

/**
 * Gather token embeddings for a batch of sequences
 *
 * Used to prepare input for a forward pass by gathering tokens from
 * different sequences into a contiguous batch.
 *
 * token_ids: [total_tokens] - flattened token IDs for all sequences
 * embeddings: [vocab_size, hidden_size] - embedding table
 * output: [total_tokens, hidden_size] - gathered embeddings
 * seq_lens: [batch_size] - length of each sequence in this iteration
 */
__global__ void gather_embeddings_kernel(
    const int32_t* __restrict__ token_ids,
    const __half* __restrict__ embeddings,
    __half* __restrict__ output,
    int total_tokens,
    int hidden_size,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= vocab_size) return;

    // Copy embedding for this token
    int emb_offset = token_id * hidden_size;
    int out_offset = token_idx * hidden_size;

    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        output[out_offset + d] = embeddings[emb_offset + d];
    }
}

/**
 * Scatter logits to sequence outputs
 *
 * After forward pass, scatter the output logits to per-sequence buffers.
 *
 * logits: [batch_tokens, vocab_size] - model output
 * output_logits: [batch_size, vocab_size] - per-sequence last-token logits
 * seq_start_positions: [batch_size] - start position of each sequence
 * seq_lens: [batch_size] - length of each sequence (last token position = start + len - 1)
 */
__global__ void scatter_last_token_logits_kernel(
    const __half* __restrict__ logits,
    __half* __restrict__ output_logits,
    const int32_t* __restrict__ seq_start_positions,
    const int32_t* __restrict__ seq_lens,
    int batch_size,
    int vocab_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= batch_size) return;

    // Get the last token position for this sequence
    int start = seq_start_positions[seq_idx];
    int len = seq_lens[seq_idx];
    int last_token_pos = start + len - 1;

    // Copy logits for last token
    int in_offset = last_token_pos * vocab_size;
    int out_offset = seq_idx * vocab_size;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        output_logits[out_offset + v] = logits[in_offset + v];
    }
}

/**
 * Prepare position IDs for rotary embedding
 *
 * For prefill: positions are [0, 1, 2, ..., seq_len-1]
 * For decode: position is context_len (the new token position)
 *
 * position_ids: [total_tokens] - output position IDs
 * seq_start_positions: [batch_size] - cumulative start positions
 * seq_context_lens: [batch_size] - context length for each sequence
 * is_prefill: [batch_size] - whether each sequence is in prefill mode
 */
__global__ void prepare_position_ids_kernel(
    int32_t* __restrict__ position_ids,
    const int32_t* __restrict__ seq_start_positions,
    const int32_t* __restrict__ seq_context_lens,
    const int32_t* __restrict__ is_prefill,
    const int32_t* __restrict__ input_lens,
    int batch_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= batch_size) return;

    int start = seq_start_positions[seq_idx];
    int context_len = seq_context_lens[seq_idx];
    int input_len = input_lens[seq_idx];
    bool prefill = is_prefill[seq_idx] != 0;

    for (int i = threadIdx.x; i < input_len; i += blockDim.x) {
        if (prefill) {
            // Prefill: position = token index within sequence
            position_ids[start + i] = i;
        } else {
            // Decode: position = context_len (all decode tokens use same position)
            position_ids[start + i] = context_len;
        }
    }
}

// ============================================================================
// Sampling Utilities
// ============================================================================

/**
 * Argmax sampling kernel
 *
 * For each sequence, find the token with highest logit.
 *
 * logits: [batch_size, vocab_size] - per-sequence logits
 * output_tokens: [batch_size] - sampled token IDs
 */
__global__ void argmax_sampling_kernel(
    const __half* __restrict__ logits,
    int32_t* __restrict__ output_tokens,
    int batch_size,
    int vocab_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= batch_size) return;

    // Find max in this sequence's logits
    extern __shared__ float smem[];
    float* shared_max = smem;
    int* shared_idx = (int*)(shared_max + blockDim.x);

    float thread_max = -1e20f;
    int thread_idx = 0;

    int offset = seq_idx * vocab_size;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float val = __half2float(logits[offset + v]);
        if (val > thread_max) {
            thread_max = val;
            thread_idx = v;
        }
    }

    shared_max[threadIdx.x] = thread_max;
    shared_idx[threadIdx.x] = thread_idx;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
                shared_idx[threadIdx.x] = shared_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output_tokens[seq_idx] = shared_idx[0];
    }
}

/**
 * Check for EOS tokens
 *
 * finished: [batch_size] - output: 1 if EOS found, 0 otherwise
 * tokens: [batch_size] - sampled tokens
 * eos_token_id: EOS token ID to check for
 */
__global__ void check_eos_kernel(
    int32_t* __restrict__ finished,
    const int32_t* __restrict__ tokens,
    int batch_size,
    int eos_token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        finished[idx] = (tokens[idx] == eos_token_id) ? 1 : 0;
    }
}

} // namespace batch
} // namespace ops
} // namespace pygpukit
