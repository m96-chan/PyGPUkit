/**
 * PoPE (Positional Encoding) - Additive positional embedding
 *
 * Alternative to RoPE using simple addition instead of rotation.
 */

#include "pope_kernels.cuh"

namespace pygpukit {
namespace ops {

GPUArray pope_init_encoding(int max_seq_len, int head_dim, float base) {
    // Create sinusoidal positional encoding table
    // Shape: [max_seq_len, head_dim]
    GPUArray encoding({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);

    int total = max_seq_len * head_dim;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    nn::pope_init_sinusoidal_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(encoding.data()),
        max_seq_len,
        head_dim,
        base);

    sync_and_check("pope_init_encoding kernel failed");
    return encoding;
}

void pope_inplace(GPUArray& q, GPUArray& k, const GPUArray& encoding, int start_pos) {
    // q: [seq_len, n_heads_q, head_dim]
    // k: [seq_len, n_heads_k, head_dim]
    // encoding: [max_seq_len, head_dim] (float32)

    if (q.ndim() != 3 || k.ndim() != 3) {
        throw std::runtime_error("pope_inplace: q and k must be 3D [seq_len, n_heads, head_dim]");
    }
    if (encoding.ndim() != 2 || encoding.dtype() != DataType::Float32) {
        throw std::runtime_error("pope_inplace: encoding must be 2D float32 [max_seq_len, head_dim]");
    }
    if (q.dtype() != k.dtype()) {
        throw std::runtime_error("pope_inplace: q and k dtype mismatch");
    }

    int seq_len = q.shape()[0];
    int n_heads_q = q.shape()[1];
    int n_heads_k = k.shape()[1];
    int head_dim = q.shape()[2];
    int max_seq_len = encoding.shape()[0];

    if (k.shape()[0] != seq_len || k.shape()[2] != head_dim) {
        throw std::runtime_error("pope_inplace: q and k shape mismatch");
    }
    if (encoding.shape()[1] != head_dim) {
        throw std::runtime_error("pope_inplace: encoding head_dim mismatch");
    }
    if (start_pos + seq_len > max_seq_len) {
        throw std::runtime_error("pope_inplace: position exceeds max_seq_len");
    }

    int total_q = seq_len * n_heads_q * head_dim;
    int total_k = seq_len * n_heads_k * head_dim;
    int total_work = std::max(total_q, total_k);

    const int block_size = 256;
    const int grid_size = (total_work + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (q.dtype()) {
        case DataType::Float32:
            nn::pope_apply_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(q.data()),
                static_cast<float*>(k.data()),
                static_cast<const float*>(encoding.data()),
                seq_len, n_heads_q, n_heads_k, head_dim, start_pos);
            break;
        case DataType::Float16:
            nn::pope_apply_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(q.data()),
                static_cast<__half*>(k.data()),
                static_cast<const float*>(encoding.data()),
                seq_len, n_heads_q, n_heads_k, head_dim, start_pos);
            break;
        case DataType::BFloat16:
            nn::pope_apply_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(q.data()),
                static_cast<__nv_bfloat16*>(k.data()),
                static_cast<const float*>(encoding.data()),
                seq_len, n_heads_q, n_heads_k, head_dim, start_pos);
            break;
        default:
            throw std::runtime_error("pope_inplace: unsupported dtype");
    }

    sync_and_check("pope_inplace kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
