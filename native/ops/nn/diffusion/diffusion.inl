/**
 * Diffusion model operations dispatch
 *
 * Provides GPUArray wrapper functions for diffusion-specific operations:
 * - GroupNorm
 * - AdaLN / AdaLN-Zero
 * - Cross-Attention
 * - Conv2D (im2col + GEMM)
 */

#include "groupnorm_kernels.cuh"
#include "adaln_kernels.cuh"
#include "cross_attention_kernels.cuh"
#include "conv2d_kernels.cuh"
#include "flux_kernels.cuh"
#include "../../common/error.cuh"
#include "../../../core/memory.hpp"

namespace pygpukit {
namespace ops {

using namespace nn;

// ============================================================================
// GroupNorm
// ============================================================================

GPUArray group_norm(const GPUArray& input, const GPUArray& gamma, const GPUArray& beta,
                    int num_groups, float eps) {
    // input: [N, C, H, W]
    // gamma: [C]
    // beta: [C]

    if (input.ndim() != 4) {
        throw std::runtime_error("group_norm expects 4D input [N, C, H, W]");
    }
    if (gamma.ndim() != 1 || beta.ndim() != 1) {
        throw std::runtime_error("group_norm expects 1D gamma and beta");
    }
    if (input.dtype() != gamma.dtype() || input.dtype() != beta.dtype()) {
        throw std::runtime_error("group_norm: dtype mismatch");
    }

    int N = static_cast<int>(input.shape()[0]);
    int C = static_cast<int>(input.shape()[1]);
    int H = static_cast<int>(input.shape()[2]);
    int W = static_cast<int>(input.shape()[3]);

    if (C % num_groups != 0) {
        throw std::runtime_error("group_norm: C must be divisible by num_groups");
    }
    if (gamma.shape()[0] != static_cast<size_t>(C) || beta.shape()[0] != static_cast<size_t>(C)) {
        throw std::runtime_error("group_norm: gamma/beta size must match C");
    }

    GPUArray result(input.shape(), input.dtype());

    int num_blocks = N * num_groups;
    int threads = 256;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            groupnorm_f32_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(gamma.data()),
                static_cast<const float*>(beta.data()),
                static_cast<float*>(result.data()),
                N, C, H, W, num_groups, eps);
            break;
        case DataType::BFloat16:
            groupnorm_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(gamma.data()),
                static_cast<const __nv_bfloat16*>(beta.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                N, C, H, W, num_groups, eps);
            break;
        case DataType::Float16:
            groupnorm_f16_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(gamma.data()),
                static_cast<const __half*>(beta.data()),
                static_cast<__half*>(result.data()),
                N, C, H, W, num_groups, eps);
            break;
        default:
            throw std::runtime_error("group_norm only supports float types");
    }

    sync_and_check("group_norm kernel failed");
    return result;
}

// ============================================================================
// AdaLN
// ============================================================================

GPUArray adaln(const GPUArray& input, const GPUArray& scale, const GPUArray& shift, float eps) {
    // input: [B, N, D]
    // scale: [B, D]
    // shift: [B, D]

    if (input.ndim() != 3) {
        throw std::runtime_error("adaln expects 3D input [B, N, D]");
    }
    if (scale.ndim() != 2 || shift.ndim() != 2) {
        throw std::runtime_error("adaln expects 2D scale and shift [B, D]");
    }
    if (input.dtype() != scale.dtype() || input.dtype() != shift.dtype()) {
        throw std::runtime_error("adaln: dtype mismatch");
    }

    int B = static_cast<int>(input.shape()[0]);
    int N = static_cast<int>(input.shape()[1]);
    int D = static_cast<int>(input.shape()[2]);

    if (scale.shape()[0] != static_cast<size_t>(B) || scale.shape()[1] != static_cast<size_t>(D)) {
        throw std::runtime_error("adaln: scale shape must be [B, D]");
    }
    if (shift.shape()[0] != static_cast<size_t>(B) || shift.shape()[1] != static_cast<size_t>(D)) {
        throw std::runtime_error("adaln: shift shape must be [B, D]");
    }

    GPUArray result(input.shape(), input.dtype());

    int num_blocks = B * N;
    int threads = 256;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            adaln_f32_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(scale.data()),
                static_cast<const float*>(shift.data()),
                static_cast<float*>(result.data()),
                B, N, D, eps);
            break;
        case DataType::BFloat16:
            adaln_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(scale.data()),
                static_cast<const __nv_bfloat16*>(shift.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                B, N, D, eps);
            break;
        default:
            throw std::runtime_error("adaln only supports float32 and bfloat16");
    }

    sync_and_check("adaln kernel failed");
    return result;
}

GPUArray adaln_zero(const GPUArray& input, const GPUArray& scale, const GPUArray& shift,
                    const GPUArray& gate, const GPUArray& residual, float eps) {
    // input: [B, N, D]
    // scale: [B, D]
    // shift: [B, D]
    // gate: [B, D]
    // residual: [B, N, D]

    if (input.ndim() != 3) {
        throw std::runtime_error("adaln_zero expects 3D input [B, N, D]");
    }
    if (scale.ndim() != 2 || shift.ndim() != 2 || gate.ndim() != 2) {
        throw std::runtime_error("adaln_zero expects 2D scale, shift, and gate [B, D]");
    }
    if (residual.ndim() != 3) {
        throw std::runtime_error("adaln_zero expects 3D residual [B, N, D]");
    }

    int B = static_cast<int>(input.shape()[0]);
    int N = static_cast<int>(input.shape()[1]);
    int D = static_cast<int>(input.shape()[2]);

    GPUArray result(input.shape(), input.dtype());

    int num_blocks = B * N;
    int threads = 256;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            adaln_zero_f32_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(scale.data()),
                static_cast<const float*>(shift.data()),
                static_cast<const float*>(gate.data()),
                static_cast<const float*>(residual.data()),
                static_cast<float*>(result.data()),
                B, N, D, eps);
            break;
        case DataType::BFloat16:
            adaln_zero_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(scale.data()),
                static_cast<const __nv_bfloat16*>(shift.data()),
                static_cast<const __nv_bfloat16*>(gate.data()),
                static_cast<const __nv_bfloat16*>(residual.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                B, N, D, eps);
            break;
        default:
            throw std::runtime_error("adaln_zero only supports float32 and bfloat16");
    }

    sync_and_check("adaln_zero kernel failed");
    return result;
}

// ============================================================================
// Cross-Attention
// ============================================================================

GPUArray cross_attention(const GPUArray& Q, const GPUArray& K, const GPUArray& V, float scale) {
    // Q: [n_heads, q_len, head_dim]
    // K: [n_heads, kv_len, head_dim]
    // V: [n_heads, kv_len, head_dim]

    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) {
        throw std::runtime_error("cross_attention expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
        throw std::runtime_error("cross_attention: dtype mismatch");
    }

    int n_heads = static_cast<int>(Q.shape()[0]);
    int q_len = static_cast<int>(Q.shape()[1]);
    int head_dim = static_cast<int>(Q.shape()[2]);
    int kv_len = static_cast<int>(K.shape()[1]);

    if (K.shape()[0] != static_cast<size_t>(n_heads) || V.shape()[0] != static_cast<size_t>(n_heads)) {
        throw std::runtime_error("cross_attention: n_heads mismatch");
    }
    if (K.shape()[2] != static_cast<size_t>(head_dim) || V.shape()[2] != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("cross_attention: head_dim mismatch");
    }

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    GPUArray result({static_cast<size_t>(n_heads), static_cast<size_t>(q_len), static_cast<size_t>(head_dim)}, Q.dtype());

    dim3 grid(n_heads, q_len);
    int threads = 128;
    size_t shared_mem = kv_len * sizeof(float);
    cudaStream_t stream = internal::get_capture_stream();

    switch (Q.dtype()) {
        case DataType::Float32:
            cross_attention_f32_kernel<<<grid, threads, shared_mem, stream>>>(
                static_cast<const float*>(Q.data()),
                static_cast<const float*>(K.data()),
                static_cast<const float*>(V.data()),
                static_cast<float*>(result.data()),
                n_heads, q_len, kv_len, head_dim, scale);
            break;
        case DataType::BFloat16:
            cross_attention_bf16_kernel<<<grid, threads, shared_mem, stream>>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n_heads, q_len, kv_len, head_dim, scale);
            break;
        case DataType::Float16:
            cross_attention_f16_kernel<<<grid, threads, shared_mem, stream>>>(
                static_cast<const __half*>(Q.data()),
                static_cast<const __half*>(K.data()),
                static_cast<const __half*>(V.data()),
                static_cast<__half*>(result.data()),
                n_heads, q_len, kv_len, head_dim, scale);
            break;
        default:
            throw std::runtime_error("cross_attention only supports float types");
    }

    sync_and_check("cross_attention kernel failed");
    return result;
}

// ============================================================================
// Conv2D operations
// ============================================================================

GPUArray im2col(const GPUArray& input,
                int K_h, int K_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                int dil_h, int dil_w) {
    // input: [N, C, H, W]
    // output: [N, C*K_h*K_w, H_out*W_out]

    if (input.ndim() != 4) {
        throw std::runtime_error("im2col expects 4D input [N, C, H, W]");
    }

    int N = static_cast<int>(input.shape()[0]);
    int C = static_cast<int>(input.shape()[1]);
    int H = static_cast<int>(input.shape()[2]);
    int W = static_cast<int>(input.shape()[3]);

    int H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) / stride_w + 1;

    GPUArray result({static_cast<size_t>(N),
                     static_cast<size_t>(C * K_h * K_w),
                     static_cast<size_t>(H_out * W_out)}, input.dtype());

    int total = N * C * K_h * K_w * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            im2col_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                N, C, H, W,
                K_h, K_w, pad_h, pad_w,
                stride_h, stride_w, dil_h, dil_w,
                H_out, W_out);
            break;
        default:
            throw std::runtime_error("im2col currently only supports float32");
    }

    sync_and_check("im2col kernel failed");
    return result;
}

GPUArray col2im(const GPUArray& input,
                int C, int H, int W,
                int K_h, int K_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                int dil_h, int dil_w) {
    // input: [N, C*K_h*K_w, H_in*W_in]
    // output: [N, C, H, W]

    if (input.ndim() != 3) {
        throw std::runtime_error("col2im expects 3D input [N, C*K_h*K_w, H_in*W_in]");
    }

    int N = static_cast<int>(input.shape()[0]);

    // Calculate input spatial dimensions from output
    int H_in = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) / stride_h + 1;
    int W_in = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) / stride_w + 1;

    GPUArray result({static_cast<size_t>(N),
                     static_cast<size_t>(C),
                     static_cast<size_t>(H),
                     static_cast<size_t>(W)}, input.dtype());

    // Zero initialize output for accumulation
    device_memset(result.data(), 0, result.nbytes());

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            col2im_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                N, C, H, W,
                K_h, K_w, pad_h, pad_w,
                stride_h, stride_w, dil_h, dil_w,
                H_in, W_in);
            break;
        default:
            throw std::runtime_error("col2im currently only supports float32");
    }

    sync_and_check("col2im kernel failed");
    return result;
}

GPUArray conv2d_1x1(const GPUArray& input, const GPUArray& weight, const GPUArray* bias) {
    // input: [N, C_in, H, W]
    // weight: [C_out, C_in]
    // bias: [C_out] or nullptr

    if (input.ndim() != 4) {
        throw std::runtime_error("conv2d_1x1 expects 4D input [N, C_in, H, W]");
    }
    if (weight.ndim() != 2) {
        throw std::runtime_error("conv2d_1x1 expects 2D weight [C_out, C_in]");
    }

    int N = static_cast<int>(input.shape()[0]);
    int C_in = static_cast<int>(input.shape()[1]);
    int H = static_cast<int>(input.shape()[2]);
    int W = static_cast<int>(input.shape()[3]);
    int C_out = static_cast<int>(weight.shape()[0]);

    if (weight.shape()[1] != static_cast<size_t>(C_in)) {
        throw std::runtime_error("conv2d_1x1: weight C_in mismatch");
    }

    GPUArray result({static_cast<size_t>(N),
                     static_cast<size_t>(C_out),
                     static_cast<size_t>(H),
                     static_cast<size_t>(W)}, input.dtype());

    int total = N * C_out * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    const float* bias_ptr = (bias != nullptr) ? static_cast<const float*>(bias->data()) : nullptr;

    switch (input.dtype()) {
        case DataType::Float32:
            conv2d_1x1_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(weight.data()),
                bias_ptr,
                static_cast<float*>(result.data()),
                N, C_in, C_out, H, W);
            break;
        default:
            throw std::runtime_error("conv2d_1x1 currently only supports float32");
    }

    sync_and_check("conv2d_1x1 kernel failed");
    return result;
}

GPUArray conv2d_3x3(const GPUArray& input, const GPUArray& weight, const GPUArray* bias,
                    int pad_h, int pad_w, int stride_h, int stride_w) {
    // input: [N, C_in, H, W]
    // weight: [C_out, C_in, 3, 3]
    // bias: [C_out] or nullptr

    if (input.ndim() != 4) {
        throw std::runtime_error("conv2d_3x3 expects 4D input [N, C_in, H, W]");
    }
    if (weight.ndim() != 4) {
        throw std::runtime_error("conv2d_3x3 expects 4D weight [C_out, C_in, 3, 3]");
    }

    int N = static_cast<int>(input.shape()[0]);
    int C_in = static_cast<int>(input.shape()[1]);
    int H = static_cast<int>(input.shape()[2]);
    int W = static_cast<int>(input.shape()[3]);
    int C_out = static_cast<int>(weight.shape()[0]);

    int H_out = (H + 2 * pad_h - 3) / stride_h + 1;
    int W_out = (W + 2 * pad_w - 3) / stride_w + 1;

    GPUArray result({static_cast<size_t>(N),
                     static_cast<size_t>(C_out),
                     static_cast<size_t>(H_out),
                     static_cast<size_t>(W_out)}, input.dtype());

    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    const float* bias_ptr = (bias != nullptr) ? static_cast<const float*>(bias->data()) : nullptr;

    switch (input.dtype()) {
        case DataType::Float32:
            conv2d_direct_3x3_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(weight.data()),
                bias_ptr,
                static_cast<float*>(result.data()),
                N, C_in, C_out, H, W,
                pad_h, pad_w, stride_h, stride_w,
                H_out, W_out);
            break;
        default:
            throw std::runtime_error("conv2d_3x3 currently only supports float32");
    }

    sync_and_check("conv2d_3x3 kernel failed");
    return result;
}

// ============================================================================
// FLUX-specific operations
// ============================================================================

GPUArray layer_norm_simple(const GPUArray& input, float eps) {
    // input: [B, N, D] or [total_rows, D]
    if (input.ndim() != 2 && input.ndim() != 3) {
        throw std::runtime_error("layer_norm_simple expects 2D or 3D input");
    }
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("layer_norm_simple currently only supports float32");
    }

    int total_rows, D;
    if (input.ndim() == 3) {
        int B = static_cast<int>(input.shape()[0]);
        int N = static_cast<int>(input.shape()[1]);
        D = static_cast<int>(input.shape()[2]);
        total_rows = B * N;
    } else {
        total_rows = static_cast<int>(input.shape()[0]);
        D = static_cast<int>(input.shape()[1]);
    }

    GPUArray result(input.shape(), input.dtype());

    int threads = 256;
    cudaStream_t stream = internal::get_capture_stream();

    nn::layer_norm_simple_f32_kernel<<<total_rows, threads, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(result.data()),
        total_rows, D, eps);

    sync_and_check("layer_norm_simple kernel failed");
    return result;
}

GPUArray modulate(const GPUArray& input, const GPUArray& scale, const GPUArray& shift) {
    // input: [B, N, D], scale/shift: [B, D]
    if (input.ndim() != 3) {
        throw std::runtime_error("modulate expects 3D input [B, N, D]");
    }
    if (scale.ndim() != 2 || shift.ndim() != 2) {
        throw std::runtime_error("modulate expects 2D scale and shift [B, D]");
    }
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("modulate currently only supports float32");
    }

    int B = static_cast<int>(input.shape()[0]);
    int N = static_cast<int>(input.shape()[1]);
    int D = static_cast<int>(input.shape()[2]);

    GPUArray result(input.shape(), input.dtype());

    int total = B * N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    nn::modulate_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(scale.data()),
        static_cast<const float*>(shift.data()),
        static_cast<float*>(result.data()),
        B, N, D);

    sync_and_check("modulate kernel failed");
    return result;
}

GPUArray gated_residual(const GPUArray& residual, const GPUArray& gate, const GPUArray& value) {
    // residual: [B, N, D], gate: [B, D], value: [B, N, D]
    if (residual.ndim() != 3 || value.ndim() != 3) {
        throw std::runtime_error("gated_residual expects 3D residual and value");
    }
    if (gate.ndim() != 2) {
        throw std::runtime_error("gated_residual expects 2D gate [B, D]");
    }
    if (residual.dtype() != DataType::Float32) {
        throw std::runtime_error("gated_residual currently only supports float32");
    }

    int B = static_cast<int>(residual.shape()[0]);
    int N = static_cast<int>(residual.shape()[1]);
    int D = static_cast<int>(residual.shape()[2]);

    GPUArray result(residual.shape(), residual.dtype());

    int total = B * N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    nn::gated_residual_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(residual.data()),
        static_cast<const float*>(gate.data()),
        static_cast<const float*>(value.data()),
        static_cast<float*>(result.data()),
        B, N, D);

    sync_and_check("gated_residual kernel failed");
    return result;
}

GPUArray scale_tensor(const GPUArray& input, float scale) {
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("scale_tensor currently only supports float32");
    }

    GPUArray result(input.shape(), input.dtype());

    int total = static_cast<int>(input.size());
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    nn::scale_tensor_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(result.data()),
        scale, total);

    sync_and_check("scale_tensor kernel failed");
    return result;
}

GPUArray concat_axis1(const GPUArray& a, const GPUArray& b) {
    // 3D: [B, N1, D] + [B, N2, D] -> [B, N1+N2, D]
    // 4D: [B, N1, H, D] + [B, N2, H, D] -> [B, N1+N2, H, D]
    if (a.ndim() != b.ndim()) {
        throw std::runtime_error("concat_axis1: a and b must have same ndim");
    }
    if (a.dtype() != DataType::Float32 || b.dtype() != DataType::Float32) {
        throw std::runtime_error("concat_axis1 currently only supports float32");
    }

    cudaStream_t stream = internal::get_capture_stream();

    if (a.ndim() == 3) {
        // 3D case: [B, N, D]
        int B = static_cast<int>(a.shape()[0]);
        int N1 = static_cast<int>(a.shape()[1]);
        int D = static_cast<int>(a.shape()[2]);
        int N2 = static_cast<int>(b.shape()[1]);

        if (b.shape()[0] != static_cast<size_t>(B) || b.shape()[2] != static_cast<size_t>(D)) {
            throw std::runtime_error("concat_axis1: shape mismatch (B or D)");
        }

        GPUArray result({static_cast<size_t>(B), static_cast<size_t>(N1 + N2), static_cast<size_t>(D)}, a.dtype());

        int total = B * (N1 + N2) * D;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        nn::concat_axis1_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(result.data()),
            B, N1, N2, D);

        sync_and_check("concat_axis1 (3D) kernel failed");
        return result;
    } else if (a.ndim() == 4) {
        // 4D case: [B, N, H, D]
        int B = static_cast<int>(a.shape()[0]);
        int N1 = static_cast<int>(a.shape()[1]);
        int H = static_cast<int>(a.shape()[2]);
        int D = static_cast<int>(a.shape()[3]);
        int N2 = static_cast<int>(b.shape()[1]);

        if (b.shape()[0] != static_cast<size_t>(B) ||
            b.shape()[2] != static_cast<size_t>(H) ||
            b.shape()[3] != static_cast<size_t>(D)) {
            throw std::runtime_error("concat_axis1: shape mismatch (B, H, or D)");
        }

        GPUArray result({static_cast<size_t>(B), static_cast<size_t>(N1 + N2),
                         static_cast<size_t>(H), static_cast<size_t>(D)}, a.dtype());

        int total = B * (N1 + N2) * H * D;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        nn::concat_axis1_4d_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(result.data()),
            B, N1, N2, H, D);

        sync_and_check("concat_axis1 (4D) kernel failed");
        return result;
    } else {
        throw std::runtime_error("concat_axis1 only supports 3D and 4D inputs");
    }
}

std::pair<GPUArray, GPUArray> split_axis1(const GPUArray& input, int split_size) {
    // 3D: [B, N, D] -> ([B, split_size, D], [B, N-split_size, D])
    // 4D: [B, N, H, D] -> ([B, split_size, H, D], [B, N-split_size, H, D])
    if (input.dtype() != DataType::Float32) {
        throw std::runtime_error("split_axis1 currently only supports float32");
    }

    cudaStream_t stream = internal::get_capture_stream();

    if (input.ndim() == 3) {
        // 3D case
        int B = static_cast<int>(input.shape()[0]);
        int N = static_cast<int>(input.shape()[1]);
        int D = static_cast<int>(input.shape()[2]);

        if (split_size > N) {
            throw std::runtime_error("split_axis1: split_size > N");
        }

        int N1 = split_size;
        int N2 = N - split_size;

        GPUArray first({static_cast<size_t>(B), static_cast<size_t>(N1), static_cast<size_t>(D)}, input.dtype());
        GPUArray second({static_cast<size_t>(B), static_cast<size_t>(N2), static_cast<size_t>(D)}, input.dtype());

        int threads = 256;

        // Copy first part
        int total1 = B * N1 * D;
        int blocks1 = (total1 + threads - 1) / threads;
        nn::copy_slice_axis1_f32_kernel<<<blocks1, threads, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(first.data()),
            B, N, N1, D, 0);

        // Copy second part
        int total2 = B * N2 * D;
        int blocks2 = (total2 + threads - 1) / threads;
        nn::copy_slice_axis1_f32_kernel<<<blocks2, threads, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(second.data()),
            B, N, N2, D, N1);

        sync_and_check("split_axis1 (3D) kernel failed");
        return std::make_pair(std::move(first), std::move(second));
    } else if (input.ndim() == 4) {
        // 4D case
        int B = static_cast<int>(input.shape()[0]);
        int N = static_cast<int>(input.shape()[1]);
        int H = static_cast<int>(input.shape()[2]);
        int D = static_cast<int>(input.shape()[3]);

        if (split_size > N) {
            throw std::runtime_error("split_axis1: split_size > N");
        }

        int N1 = split_size;
        int N2 = N - split_size;

        GPUArray first({static_cast<size_t>(B), static_cast<size_t>(N1),
                        static_cast<size_t>(H), static_cast<size_t>(D)}, input.dtype());
        GPUArray second({static_cast<size_t>(B), static_cast<size_t>(N2),
                         static_cast<size_t>(H), static_cast<size_t>(D)}, input.dtype());

        int threads = 256;

        // Copy first part
        int total1 = B * N1 * H * D;
        int blocks1 = (total1 + threads - 1) / threads;
        nn::copy_slice_axis1_4d_f32_kernel<<<blocks1, threads, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(first.data()),
            B, N, N1, H, D, 0);

        // Copy second part
        int total2 = B * N2 * H * D;
        int blocks2 = (total2 + threads - 1) / threads;
        nn::copy_slice_axis1_4d_f32_kernel<<<blocks2, threads, 0, stream>>>(
            static_cast<const float*>(input.data()),
            static_cast<float*>(second.data()),
            B, N, N2, H, D, N1);

        sync_and_check("split_axis1 (4D) kernel failed");
        return std::make_pair(std::move(first), std::move(second));
    } else {
        throw std::runtime_error("split_axis1 only supports 3D and 4D inputs");
    }
}

GPUArray apply_rope(const GPUArray& x, const GPUArray& cos_freq, const GPUArray& sin_freq) {
    // x: [B, N, H, D], cos/sin: [N, D]
    if (x.ndim() != 4) {
        throw std::runtime_error("apply_rope expects 4D input [B, N, H, D]");
    }
    if (cos_freq.ndim() != 2 || sin_freq.ndim() != 2) {
        throw std::runtime_error("apply_rope expects 2D cos/sin [N, D]");
    }
    if (x.dtype() != DataType::Float32) {
        throw std::runtime_error("apply_rope currently only supports float32");
    }

    int B = static_cast<int>(x.shape()[0]);
    int N = static_cast<int>(x.shape()[1]);
    int H = static_cast<int>(x.shape()[2]);
    int D = static_cast<int>(x.shape()[3]);

    GPUArray result(x.shape(), x.dtype());

    int total = B * N * H * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = internal::get_capture_stream();

    nn::apply_rope_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(cos_freq.data()),
        static_cast<const float*>(sin_freq.data()),
        static_cast<float*>(result.data()),
        B, N, H, D);

    sync_and_check("apply_rope kernel failed");
    return result;
}

}  // namespace ops
}  // namespace pygpukit
