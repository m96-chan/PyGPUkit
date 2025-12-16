/**
 * Paged Attention dispatch implementations (#87)
 */
#include "paged_attention.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"

namespace pygpukit {
namespace ops {

using namespace paged;

// Default block size for paged attention
constexpr int PAGED_BLOCK_SIZE = 16;

// ============================================================================
// Paged Attention v1
// ============================================================================

GPUArray paged_attention_v1(
    const GPUArray& Q,              // [num_seqs, num_heads, head_dim]
    const GPUArray& K_cache,        // [num_blocks, num_kv_heads, block_size, head_dim]
    const GPUArray& V_cache,        // [num_blocks, num_kv_heads, block_size, head_dim]
    const GPUArray& block_tables,   // [num_seqs, max_num_blocks_per_seq] int32
    const GPUArray& context_lens,   // [num_seqs] int32
    float scale
) {
    // Validate inputs
    if (Q.dtype() != DataType::Float16) {
        throw std::runtime_error("paged_attention_v1: Q must be Float16");
    }
    if (K_cache.dtype() != DataType::Float16 || V_cache.dtype() != DataType::Float16) {
        throw std::runtime_error("paged_attention_v1: K_cache and V_cache must be Float16");
    }
    if (block_tables.dtype() != DataType::Int32 || context_lens.dtype() != DataType::Int32) {
        throw std::runtime_error("paged_attention_v1: block_tables and context_lens must be Int32");
    }

    if (Q.ndim() != 3) {
        throw std::runtime_error("paged_attention_v1: Q must be 3D [num_seqs, num_heads, head_dim]");
    }
    if (K_cache.ndim() != 4 || V_cache.ndim() != 4) {
        throw std::runtime_error("paged_attention_v1: K_cache/V_cache must be 4D [num_blocks, num_kv_heads, block_size, head_dim]");
    }

    int num_seqs = Q.shape()[0];
    int num_heads = Q.shape()[1];
    int head_dim = Q.shape()[2];
    int num_blocks = K_cache.shape()[0];
    int num_kv_heads = K_cache.shape()[1];
    int block_size = K_cache.shape()[2];
    int max_num_blocks_per_seq = block_tables.shape()[1];

    // Auto-compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    // Allocate output
    GPUArray output({(size_t)num_seqs, (size_t)num_heads, (size_t)head_dim}, DataType::Float16);

    // Find max context length for shared memory allocation
    // For simplicity, use a fixed maximum (can be optimized later)
    int max_context_len = block_size * max_num_blocks_per_seq;

    // Shared memory: Q vector + logits
    size_t smem_size = (head_dim + max_context_len) * sizeof(float);

    // Limit shared memory to 48KB
    if (smem_size > 48 * 1024) {
        max_context_len = (48 * 1024 / sizeof(float)) - head_dim;
        smem_size = (head_dim + max_context_len) * sizeof(float);
    }

    // Launch kernel: one block per (sequence, head)
    dim3 grid(num_seqs, num_heads);
    int block_threads = 256;

    paged_attention_v1_kernel<<<grid, block_threads, smem_size>>>(
        static_cast<const __half*>(Q.data()),
        static_cast<const __half*>(K_cache.data()),
        static_cast<const __half*>(V_cache.data()),
        static_cast<const int32_t*>(block_tables.data()),
        static_cast<const int32_t*>(context_lens.data()),
        static_cast<__half*>(output.data()),
        num_seqs,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks_per_seq,
        scale
    );

    sync_and_check("paged_attention_v1 kernel failed");
    return output;
}

// ============================================================================
// KV Cache Management
// ============================================================================

void copy_to_paged_cache(
    const GPUArray& K_new,          // [num_seqs, num_kv_heads, head_dim]
    const GPUArray& V_new,          // [num_seqs, num_kv_heads, head_dim]
    GPUArray& K_cache,              // [num_blocks, num_kv_heads, block_size, head_dim]
    GPUArray& V_cache,              // [num_blocks, num_kv_heads, block_size, head_dim]
    const GPUArray& slot_mapping    // [num_seqs] int32
) {
    if (K_new.dtype() != DataType::Float16 || V_new.dtype() != DataType::Float16) {
        throw std::runtime_error("copy_to_paged_cache: K_new and V_new must be Float16");
    }
    if (slot_mapping.dtype() != DataType::Int32) {
        throw std::runtime_error("copy_to_paged_cache: slot_mapping must be Int32");
    }

    int num_seqs = K_new.shape()[0];
    int num_kv_heads = K_new.shape()[1];
    int head_dim = K_new.shape()[2];
    int block_size = K_cache.shape()[2];

    dim3 grid(num_seqs, num_kv_heads);
    int block_threads = 128;

    copy_to_paged_cache_kernel<<<grid, block_threads>>>(
        static_cast<const __half*>(K_new.data()),
        static_cast<const __half*>(V_new.data()),
        static_cast<__half*>(K_cache.data()),
        static_cast<__half*>(V_cache.data()),
        static_cast<const int32_t*>(slot_mapping.data()),
        num_seqs,
        num_kv_heads,
        head_dim,
        block_size
    );

    sync_and_check("copy_to_paged_cache kernel failed");
}

void reshape_and_cache(
    const GPUArray& K,              // [batch, seq_len, num_kv_heads, head_dim]
    const GPUArray& V,              // [batch, seq_len, num_kv_heads, head_dim]
    GPUArray& K_cache,              // [num_blocks, num_kv_heads, block_size, head_dim]
    GPUArray& V_cache,              // [num_blocks, num_kv_heads, block_size, head_dim]
    const GPUArray& slot_mapping    // [total_tokens] int32
) {
    if (K.dtype() != DataType::Float16 || V.dtype() != DataType::Float16) {
        throw std::runtime_error("reshape_and_cache: K and V must be Float16");
    }
    if (slot_mapping.dtype() != DataType::Int32) {
        throw std::runtime_error("reshape_and_cache: slot_mapping must be Int32");
    }

    int total_tokens = slot_mapping.shape()[0];
    int num_kv_heads = K_cache.shape()[1];
    int head_dim = K_cache.shape()[3];
    int block_size = K_cache.shape()[2];

    dim3 grid(total_tokens, num_kv_heads);
    int block_threads = 128;

    reshape_and_cache_kernel<<<grid, block_threads>>>(
        static_cast<const __half*>(K.data()),
        static_cast<const __half*>(V.data()),
        static_cast<__half*>(K_cache.data()),
        static_cast<__half*>(V_cache.data()),
        static_cast<const int32_t*>(slot_mapping.data()),
        total_tokens,
        num_kv_heads,
        head_dim,
        block_size
    );

    sync_and_check("reshape_and_cache kernel failed");
}

// ============================================================================
// Block Table Utilities
// ============================================================================

GPUArray allocate_kv_cache(int num_blocks, int num_kv_heads, int block_size, int head_dim) {
    return GPUArray({(size_t)num_blocks, (size_t)num_kv_heads, (size_t)block_size, (size_t)head_dim},
                    DataType::Float16);
}

} // namespace ops
} // namespace pygpukit
