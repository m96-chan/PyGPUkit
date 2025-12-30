/**
 * ALiBi (Attention with Linear Biases) dispatch functions
 *
 * Provides:
 * - alibi_init_slopes: Compute head-specific slopes
 * - alibi_compute_bias: Create bias matrix for attention
 * - alibi_add_bias: Add bias to attention scores in-place
 */

#include "alibi_kernels.cuh"

namespace pygpukit {
namespace ops {

GPUArray alibi_init_slopes(int num_heads) {
    // Create slopes tensor: [num_heads]
    GPUArray slopes({(size_t)num_heads}, DataType::Float32);

    const int block_size = 256;
    const int grid_size = (num_heads + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    nn::alibi_init_slopes_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(slopes.data()),
        num_heads);

    sync_and_check("alibi_init_slopes kernel failed");
    return slopes;
}

GPUArray alibi_compute_bias(int seq_len, int num_heads, const GPUArray& slopes, bool causal) {
    // Create bias tensor: [num_heads, seq_len, seq_len]
    if (slopes.dtype() != DataType::Float32) {
        throw std::runtime_error("alibi_compute_bias: slopes must be float32");
    }
    if (slopes.size() != (size_t)num_heads) {
        throw std::runtime_error("alibi_compute_bias: slopes size must match num_heads");
    }

    GPUArray bias({(size_t)num_heads, (size_t)seq_len, (size_t)seq_len}, DataType::Float32);

    int total = num_heads * seq_len * seq_len;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    if (causal) {
        nn::alibi_compute_bias_causal_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(bias.data()),
            static_cast<const float*>(slopes.data()),
            seq_len,
            num_heads);
    } else {
        nn::alibi_compute_bias_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(bias.data()),
            static_cast<const float*>(slopes.data()),
            seq_len,
            num_heads);
    }

    sync_and_check("alibi_compute_bias kernel failed");
    return bias;
}

void alibi_add_bias(GPUArray& scores, const GPUArray& slopes, int start_pos) {
    // scores: [batch, num_heads, q_len, kv_len]
    // slopes: [num_heads]

    if (scores.ndim() != 4) {
        throw std::runtime_error("alibi_add_bias: scores must be 4D [batch, heads, q_len, kv_len]");
    }
    if (scores.dtype() != DataType::Float32) {
        throw std::runtime_error("alibi_add_bias: scores must be float32");
    }
    if (slopes.dtype() != DataType::Float32) {
        throw std::runtime_error("alibi_add_bias: slopes must be float32");
    }

    int batch_size = scores.shape()[0];
    int num_heads = scores.shape()[1];
    int q_len = scores.shape()[2];
    int kv_len = scores.shape()[3];

    if (slopes.size() != (size_t)num_heads) {
        throw std::runtime_error("alibi_add_bias: slopes size must match num_heads");
    }

    int total = batch_size * num_heads * q_len * kv_len;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    nn::alibi_add_bias_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(scores.data()),
        static_cast<const float*>(slopes.data()),
        batch_size,
        num_heads,
        q_len,
        kv_len,
        start_pos);

    sync_and_check("alibi_add_bias kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
