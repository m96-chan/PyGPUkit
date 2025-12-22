/**
 * Continuous Batching dispatch implementations (#86)
 */
#include "continuous_batching.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"

namespace pygpukit {
namespace ops {

using namespace batch;

// ============================================================================
// Batch Formation
// ============================================================================

GPUArray gather_embeddings(
    const GPUArray& token_ids,      // [total_tokens] int32
    const GPUArray& embeddings,     // [vocab_size, hidden_size] FP16
    int total_tokens
) {
    if (token_ids.dtype() != DataType::Int32) {
        throw std::runtime_error("gather_embeddings: token_ids must be Int32");
    }
    if (embeddings.dtype() != DataType::Float16) {
        throw std::runtime_error("gather_embeddings: embeddings must be Float16");
    }

    int vocab_size = embeddings.shape()[0];
    int hidden_size = embeddings.shape()[1];

    GPUArray output({(size_t)total_tokens, (size_t)hidden_size}, DataType::Float16);

    int block_threads = 256;
    gather_embeddings_kernel<<<total_tokens, block_threads>>>(
        static_cast<const int32_t*>(token_ids.data()),
        static_cast<const __half*>(embeddings.data()),
        static_cast<__half*>(output.data()),
        total_tokens,
        hidden_size,
        vocab_size
    );

    sync_and_check("gather_embeddings kernel failed");
    return output;
}

GPUArray scatter_last_token_logits(
    const GPUArray& logits,             // [batch_tokens, vocab_size] FP16
    const GPUArray& seq_start_positions,// [batch_size] int32
    const GPUArray& seq_lens,           // [batch_size] int32
    int batch_size,
    int vocab_size
) {
    if (logits.dtype() != DataType::Float16) {
        throw std::runtime_error("scatter_last_token_logits: logits must be Float16");
    }

    GPUArray output({(size_t)batch_size, (size_t)vocab_size}, DataType::Float16);

    int block_threads = 256;
    scatter_last_token_logits_kernel<<<batch_size, block_threads>>>(
        static_cast<const __half*>(logits.data()),
        static_cast<__half*>(output.data()),
        static_cast<const int32_t*>(seq_start_positions.data()),
        static_cast<const int32_t*>(seq_lens.data()),
        batch_size,
        vocab_size
    );

    sync_and_check("scatter_last_token_logits kernel failed");
    return output;
}

GPUArray prepare_position_ids(
    const GPUArray& seq_start_positions,// [batch_size] int32
    const GPUArray& seq_context_lens,   // [batch_size] int32
    const GPUArray& is_prefill,         // [batch_size] int32 (0 or 1)
    const GPUArray& input_lens,         // [batch_size] int32
    int batch_size,
    int total_tokens
) {
    GPUArray position_ids({(size_t)total_tokens}, DataType::Int32);

    int block_threads = 128;
    prepare_position_ids_kernel<<<batch_size, block_threads>>>(
        static_cast<int32_t*>(position_ids.data()),
        static_cast<const int32_t*>(seq_start_positions.data()),
        static_cast<const int32_t*>(seq_context_lens.data()),
        static_cast<const int32_t*>(is_prefill.data()),
        static_cast<const int32_t*>(input_lens.data()),
        batch_size
    );

    sync_and_check("prepare_position_ids kernel failed");
    return position_ids;
}

// ============================================================================
// Sampling
// ============================================================================

GPUArray argmax_sample(
    const GPUArray& logits,     // [batch_size, vocab_size] FP16
    int batch_size,
    int vocab_size
) {
    if (logits.dtype() != DataType::Float16) {
        throw std::runtime_error("argmax_sample: logits must be Float16");
    }

    GPUArray output_tokens({(size_t)batch_size}, DataType::Int32);

    int block_threads = 256;
    size_t smem_size = block_threads * (sizeof(float) + sizeof(int));

    argmax_sampling_kernel<<<batch_size, block_threads, smem_size>>>(
        static_cast<const __half*>(logits.data()),
        static_cast<int32_t*>(output_tokens.data()),
        batch_size,
        vocab_size
    );

    sync_and_check("argmax_sample kernel failed");
    return output_tokens;
}

GPUArray check_eos(
    const GPUArray& tokens,     // [batch_size] int32
    int eos_token_id
) {
    if (tokens.dtype() != DataType::Int32) {
        throw std::runtime_error("check_eos: tokens must be Int32");
    }

    int batch_size = tokens.shape()[0];
    GPUArray finished({(size_t)batch_size}, DataType::Int32);

    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;

    check_eos_kernel<<<grid_size, block_size>>>(
        static_cast<int32_t*>(finished.data()),
        static_cast<const int32_t*>(tokens.data()),
        batch_size,
        eos_token_id
    );

    sync_and_check("check_eos kernel failed");
    return finished;
}

// ============================================================================
// Batch Utilities
// ============================================================================

GPUArray compute_cumsum(const GPUArray& input) {
    // Simple CPU-side cumsum for small arrays (batch sizes)
    if (input.dtype() != DataType::Int32) {
        throw std::runtime_error("compute_cumsum: input must be Int32");
    }

    int n = input.shape()[0];
    std::vector<int32_t> input_host(n);
    std::vector<int32_t> output_host(n);

    // Copy to host
    memcpy_device_to_host(input_host.data(), input.data(), n * sizeof(int32_t));

    // Compute cumsum (exclusive prefix sum)
    output_host[0] = 0;
    for (int i = 1; i < n; i++) {
        output_host[i] = output_host[i-1] + input_host[i-1];
    }

    // Copy back
    GPUArray output({(size_t)n}, DataType::Int32);
    memcpy_host_to_device(output.data(), output_host.data(), n * sizeof(int32_t));

    return output;
}

std::pair<GPUArray, int> prepare_batch_inputs(
    const std::vector<std::vector<int>>& token_lists  // List of token ID lists
) {
    // Flatten all tokens into a single array
    int total_tokens = 0;
    for (const auto& tokens : token_lists) {
        total_tokens += tokens.size();
    }

    std::vector<int32_t> flat_tokens;
    flat_tokens.reserve(total_tokens);

    for (const auto& tokens : token_lists) {
        for (int tok : tokens) {
            flat_tokens.push_back(tok);
        }
    }

    GPUArray token_ids({(size_t)total_tokens}, DataType::Int32);
    memcpy_host_to_device(token_ids.data(), flat_tokens.data(),
               total_tokens * sizeof(int32_t));

    return {std::move(token_ids), total_tokens};
}

} // namespace ops
} // namespace pygpukit
