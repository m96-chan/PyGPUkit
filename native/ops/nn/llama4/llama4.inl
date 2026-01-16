/**
 * Llama4 operations
 *
 * L2 norm and iRoPE temperature scaling for Llama 4 architecture.
 */

#include "../llama4_kernels.cuh"

namespace pygpukit {
namespace ops {

// ============================================================================
// L2 Norm - Llama4TextL2Norm
// Formula: x * rsqrt(mean(x^2) + eps)
// ============================================================================

static void l2norm_dispatch(
    const GPUArray& input,
    GPUArray& output,
    float eps
) {
    size_t features = input.shape().back();
    size_t batch_size = input.size() / features;

    int block_size = std::min((size_t)256, features);
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::l2norm_f32_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(output.data()),
                batch_size, features, eps
            );
            break;
        case DataType::Float16:
            nn::l2norm_f16_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(output.data()),
                batch_size, features, eps
            );
            break;
        case DataType::BFloat16:
            nn::l2norm_bf16_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(output.data()),
                batch_size, features, eps
            );
            break;
        default:
            throw std::runtime_error("l2norm only supports float types");
    }
}

GPUArray l2norm(const GPUArray& input, float eps) {
    if (input.ndim() < 1) {
        throw std::runtime_error("l2norm requires at least 1D input");
    }

    GPUArray output(input.shape(), input.dtype());
    l2norm_dispatch(input, output, eps);
    sync_and_check("l2norm kernel failed");
    return output;
}

void l2norm(const GPUArray& input, GPUArray& out, float eps) {
    if (input.ndim() < 1) {
        throw std::runtime_error("l2norm requires at least 1D input");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("l2norm: dtype mismatch");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("l2norm: input and output shape mismatch");
    }

    l2norm_dispatch(input, out, eps);
    sync_and_check("l2norm kernel failed");
}

// ============================================================================
// iRoPE Q Scaling
// Formula: scale = log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1.0
// ============================================================================

static void irope_scale_q_dispatch(
    const GPUArray& Q,
    const GPUArray& positions,
    GPUArray& Q_out,
    float attn_scale,
    float floor_scale
) {
    int seq_len = Q.shape()[0];
    int num_heads = Q.shape()[1];
    int head_dim = Q.shape()[2];

    dim3 grid(seq_len, num_heads);
    int block_size = std::min(128, head_dim);
    cudaStream_t stream = internal::get_capture_stream();

    switch (Q.dtype()) {
        case DataType::Float16:
            nn::irope_scale_q_f16_kernel<<<grid, block_size, 0, stream>>>(
                static_cast<const __half*>(Q.data()),
                static_cast<const int64_t*>(positions.data()),
                static_cast<__half*>(Q_out.data()),
                seq_len, num_heads, head_dim,
                attn_scale, floor_scale
            );
            break;
        case DataType::BFloat16:
            nn::irope_scale_q_bf16_kernel<<<grid, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const int64_t*>(positions.data()),
                static_cast<__nv_bfloat16*>(Q_out.data()),
                seq_len, num_heads, head_dim,
                attn_scale, floor_scale
            );
            break;
        default:
            throw std::runtime_error("irope_scale_q only supports float16/bfloat16");
    }
}

GPUArray irope_scale_q(
    const GPUArray& Q,
    const GPUArray& positions,
    float attn_scale,
    float floor_scale
) {
    if (Q.ndim() != 3) {
        throw std::runtime_error("Q must be 3D: [seq_len, num_heads, head_dim]");
    }

    GPUArray Q_out(Q.shape(), Q.dtype());
    irope_scale_q_dispatch(Q, positions, Q_out, attn_scale, floor_scale);
    sync_and_check("irope_scale_q kernel failed");
    return Q_out;
}

// ============================================================================
// SDPA with iRoPE temperature scaling
// ============================================================================

static void sdpa_irope_dispatch(
    const GPUArray& Q,
    const GPUArray& K,
    const GPUArray& V,
    const GPUArray& positions,
    GPUArray& output,
    float attn_scale,
    float floor_scale,
    int causal_offset
) {
    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];
    int n_kv_heads = K.shape()[0];
    int kv_len = K.shape()[1];

    dim3 grid(n_heads, q_len);
    int block_size = std::min(256, kv_len);
    size_t smem_size = kv_len * sizeof(float);  // scores array
    cudaStream_t stream = internal::get_capture_stream();

    switch (Q.dtype()) {
        case DataType::BFloat16:
            nn::sdpa_irope_bf16_kernel<<<grid, block_size, smem_size, stream>>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<const int64_t*>(positions.data()),
                static_cast<__nv_bfloat16*>(output.data()),
                n_heads, n_kv_heads,
                q_len, kv_len, head_dim,
                attn_scale, floor_scale, causal_offset
            );
            break;
        default:
            throw std::runtime_error("sdpa_irope only supports bfloat16");
    }
}

GPUArray sdpa_irope(
    const GPUArray& Q,
    const GPUArray& K,
    const GPUArray& V,
    const GPUArray& positions,
    float attn_scale,
    float floor_scale,
    int causal_offset
) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) {
        throw std::runtime_error("Q, K, V must be 3D: [heads, seq, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
        throw std::runtime_error("sdpa_irope: Q/K/V dtype mismatch");
    }

    GPUArray output(Q.shape(), Q.dtype());
    sdpa_irope_dispatch(Q, K, V, positions, output, attn_scale, floor_scale, causal_offset);
    sync_and_check("sdpa_irope kernel failed");
    return output;
}

}  // namespace ops
}  // namespace pygpukit
