/**
 * RoPE (Rotary Position Embedding) - In-place operations
 */

namespace pygpukit {
namespace ops {

void rope_inplace(GPUArray& q, GPUArray& k, const GPUArray& cos, const GPUArray& sin) {
    // q: [seq_len, n_heads_q, head_dim]
    // k: [seq_len, n_heads_k, head_dim]
    // cos, sin: [seq_len, head_dim]

    if (q.ndim() != 3 || k.ndim() != 3 || cos.ndim() != 2 || sin.ndim() != 2) {
        throw std::runtime_error("rope: invalid dimensions");
    }
    if (q.dtype() != k.dtype() || q.dtype() != cos.dtype() || q.dtype() != sin.dtype()) {
        throw std::runtime_error("rope: dtype mismatch between q, k, cos, sin");
    }
    if (q.dtype() != DataType::Float32 && q.dtype() != DataType::Float16 &&
        q.dtype() != DataType::BFloat16) {
        throw std::runtime_error("rope: only float32, float16, bfloat16 supported");
    }

    int seq_len = q.shape()[0];
    int n_heads_q = q.shape()[1];
    int n_heads_k = k.shape()[1];
    int head_dim = q.shape()[2];

    if (k.shape()[0] != seq_len || k.shape()[2] != head_dim) {
        throw std::runtime_error("rope: q and k shape mismatch");
    }
    if (cos.shape()[0] != seq_len || cos.shape()[1] != head_dim) {
        throw std::runtime_error("rope: cos shape mismatch");
    }
    if (sin.shape()[0] != seq_len || sin.shape()[1] != head_dim) {
        throw std::runtime_error("rope: sin shape mismatch");
    }

    // Total work items: max of Q and K
    int half_dim = head_dim / 2;
    int total_q = seq_len * n_heads_q * half_dim;
    int total_k = seq_len * n_heads_k * half_dim;
    int total_work = std::max(total_q, total_k);

    const int block_size = 256;
    const int grid_size = (total_work + block_size - 1) / block_size;

    // Use capture stream if available (for CUDA Graph support)
    cudaStream_t stream = internal::get_capture_stream();

    switch (q.dtype()) {
        case DataType::Float32:
            nn::rope_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(q.data()),
                static_cast<float*>(k.data()),
                static_cast<const float*>(cos.data()),
                static_cast<const float*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        case DataType::Float16:
            nn::rope_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(q.data()),
                static_cast<__half*>(k.data()),
                static_cast<const __half*>(cos.data()),
                static_cast<const __half*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        case DataType::BFloat16:
            nn::rope_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(q.data()),
                static_cast<__nv_bfloat16*>(k.data()),
                static_cast<const __nv_bfloat16*>(cos.data()),
                static_cast<const __nv_bfloat16*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        default:
            break;
    }

    sync_and_check("rope kernel failed");
}

// RoPE with FP32 cos/sin tables (for bf16/f16 Q/K with higher precision)
void rope_inplace_f32table(GPUArray& q, GPUArray& k, const GPUArray& cos, const GPUArray& sin) {
    // q: [seq_len, n_heads_q, head_dim] (bf16 or f16)
    // k: [seq_len, n_heads_k, head_dim] (bf16 or f16)
    // cos, sin: [seq_len, head_dim] (f32)

    if (q.ndim() != 3 || k.ndim() != 3 || cos.ndim() != 2 || sin.ndim() != 2) {
        throw std::runtime_error("rope_f32table: invalid dimensions");
    }
    if (q.dtype() != k.dtype()) {
        throw std::runtime_error("rope_f32table: q and k dtype mismatch");
    }
    if (cos.dtype() != DataType::Float32 || sin.dtype() != DataType::Float32) {
        throw std::runtime_error("rope_f32table: cos/sin must be float32");
    }
    if (q.dtype() != DataType::Float16 && q.dtype() != DataType::BFloat16) {
        throw std::runtime_error("rope_f32table: q/k must be float16 or bfloat16");
    }

    int seq_len = q.shape()[0];
    int n_heads_q = q.shape()[1];
    int n_heads_k = k.shape()[1];
    int head_dim = q.shape()[2];

    if (k.shape()[0] != seq_len || k.shape()[2] != head_dim) {
        throw std::runtime_error("rope_f32table: q and k shape mismatch");
    }
    if (cos.shape()[0] != seq_len || cos.shape()[1] != head_dim) {
        throw std::runtime_error("rope_f32table: cos shape mismatch");
    }
    if (sin.shape()[0] != seq_len || sin.shape()[1] != head_dim) {
        throw std::runtime_error("rope_f32table: sin shape mismatch");
    }

    int half_dim = head_dim / 2;
    int total_q = seq_len * n_heads_q * half_dim;
    int total_k = seq_len * n_heads_k * half_dim;
    int total_work = std::max(total_q, total_k);

    const int block_size = 256;
    const int grid_size = (total_work + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (q.dtype()) {
        case DataType::Float16:
            nn::rope_f16_f32table_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(q.data()),
                static_cast<__half*>(k.data()),
                static_cast<const float*>(cos.data()),
                static_cast<const float*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        case DataType::BFloat16:
            nn::rope_bf16_f32table_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(q.data()),
                static_cast<__nv_bfloat16*>(k.data()),
                static_cast<const float*>(cos.data()),
                static_cast<const float*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        default:
            break;
    }

    sync_and_check("rope_f32table kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
