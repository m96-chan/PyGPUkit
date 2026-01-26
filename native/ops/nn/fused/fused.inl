/**
 * Fused NN operations dispatch
 *
 * 1. Fused RMSNorm + Residual: y = rmsnorm(x + residual)
 * 2. Fused SwiGLU: y = silu(gate) * up
 * 3. Fused GeGLU: y = gelu(gate) * up
 */

#include "../fused_kernels.cuh"

namespace pygpukit {
namespace ops {

// ============================================================================
// Fused RMSNorm + Residual
// ============================================================================

// Internal dispatch helper
static void rmsnorm_residual_dispatch(
    const GPUArray& input,
    const GPUArray& residual,
    const GPUArray& gamma,
    GPUArray& output,
    float eps
) {
    // Shape: [batch, features]
    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    // One block per row
    const int block_size = 256;
    const int grid_size = static_cast<int>(batch_size);

    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::rmsnorm_residual_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(residual.data()),
                static_cast<const float*>(gamma.data()),
                static_cast<float*>(output.data()),
                batch_size, features, eps);
            break;
        case DataType::Float16:
            nn::rmsnorm_residual_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(residual.data()),
                static_cast<const __half*>(gamma.data()),
                static_cast<__half*>(output.data()),
                batch_size, features, eps);
            break;
        case DataType::BFloat16:
            nn::rmsnorm_residual_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(residual.data()),
                static_cast<const __nv_bfloat16*>(gamma.data()),
                static_cast<__nv_bfloat16*>(output.data()),
                batch_size, features, eps);
            break;
        default:
            throw std::runtime_error("rmsnorm_residual: unsupported dtype");
    }
}

GPUArray rmsnorm_residual(
    const GPUArray& input,
    const GPUArray& residual,
    const GPUArray& gamma,
    float eps
) {
    if (input.ndim() != 2) {
        throw std::runtime_error("rmsnorm_residual: input must be 2D [batch, features]");
    }
    if (input.shape() != residual.shape()) {
        throw std::runtime_error("rmsnorm_residual: input and residual shape mismatch");
    }
    if (gamma.ndim() != 1 || gamma.shape()[0] != input.shape()[1]) {
        throw std::runtime_error("rmsnorm_residual: gamma must be 1D with size == features");
    }
    if (input.dtype() != residual.dtype() || input.dtype() != gamma.dtype()) {
        throw std::runtime_error("rmsnorm_residual: all inputs must have same dtype");
    }
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("rmsnorm_residual: only float32/float16/bfloat16 supported");
    }

    GPUArray output(input.shape(), input.dtype());
    rmsnorm_residual_dispatch(input, residual, gamma, output, eps);
    sync_and_check("rmsnorm_residual kernel failed");
    return output;
}

// In-place version (for CUDA Graph capture)
void rmsnorm_residual(
    const GPUArray& input,
    const GPUArray& residual,
    const GPUArray& gamma,
    GPUArray& out,
    float eps
) {
    if (input.ndim() != 2) {
        throw std::runtime_error("rmsnorm_residual: input must be 2D [batch, features]");
    }
    if (input.shape() != residual.shape()) {
        throw std::runtime_error("rmsnorm_residual: input and residual shape mismatch");
    }
    if (gamma.ndim() != 1 || gamma.shape()[0] != input.shape()[1]) {
        throw std::runtime_error("rmsnorm_residual: gamma must be 1D with size == features");
    }
    if (input.dtype() != residual.dtype() || input.dtype() != gamma.dtype()) {
        throw std::runtime_error("rmsnorm_residual: all inputs must have same dtype");
    }
    if (out.shape() != input.shape() || out.dtype() != input.dtype()) {
        throw std::runtime_error("rmsnorm_residual: output shape/dtype mismatch");
    }

    rmsnorm_residual_dispatch(input, residual, gamma, out, eps);
    sync_and_check("rmsnorm_residual kernel failed");
}

// ============================================================================
// Fused SwiGLU
// ============================================================================

// Internal dispatch helper
static void swiglu_dispatch(
    const GPUArray& gate_proj,
    const GPUArray& up_proj,
    GPUArray& output
) {
    size_t n = gate_proj.size();

    cudaStream_t stream = internal::get_capture_stream();

    switch (gate_proj.dtype()) {
        case DataType::Float32: {
            const int block_size = 256;
            const int grid_size = (n + block_size - 1) / block_size;
            nn::swiglu_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(gate_proj.data()),
                static_cast<const float*>(up_proj.data()),
                static_cast<float*>(output.data()),
                n);
            break;
        }
        case DataType::Float16: {
            const int block_size = 256;
            const int grid_size = (n + block_size - 1) / block_size;
            nn::swiglu_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(gate_proj.data()),
                static_cast<const __half*>(up_proj.data()),
                static_cast<__half*>(output.data()),
                n);
            break;
        }
        case DataType::BFloat16: {
            // Use vectorized kernel for BF16 (processes 8 elements per thread)
            const int block_size = 256;
            size_t num_vec = (n + 7) / 8;  // Number of 8-element chunks
            const int grid_size = (num_vec + block_size - 1) / block_size;
            nn::swiglu_bf16_vec_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(gate_proj.data()),
                static_cast<const __nv_bfloat16*>(up_proj.data()),
                static_cast<__nv_bfloat16*>(output.data()),
                n);
            break;
        }
        default:
            throw std::runtime_error("swiglu: unsupported dtype");
    }
}

GPUArray swiglu(const GPUArray& gate_proj, const GPUArray& up_proj) {
    if (gate_proj.shape() != up_proj.shape()) {
        throw std::runtime_error("swiglu: gate_proj and up_proj shape mismatch");
    }
    if (gate_proj.dtype() != up_proj.dtype()) {
        throw std::runtime_error("swiglu: gate_proj and up_proj dtype mismatch");
    }
    if (gate_proj.dtype() != DataType::Float32 &&
        gate_proj.dtype() != DataType::Float16 &&
        gate_proj.dtype() != DataType::BFloat16) {
        throw std::runtime_error("swiglu: only float32/float16/bfloat16 supported");
    }

    GPUArray output(gate_proj.shape(), gate_proj.dtype());
    swiglu_dispatch(gate_proj, up_proj, output);
    sync_and_check("swiglu kernel failed");
    return output;
}

// In-place version (for CUDA Graph capture)
void swiglu(const GPUArray& gate_proj, const GPUArray& up_proj, GPUArray& out) {
    if (gate_proj.shape() != up_proj.shape()) {
        throw std::runtime_error("swiglu: gate_proj and up_proj shape mismatch");
    }
    if (gate_proj.dtype() != up_proj.dtype()) {
        throw std::runtime_error("swiglu: gate_proj and up_proj dtype mismatch");
    }
    if (out.shape() != gate_proj.shape() || out.dtype() != gate_proj.dtype()) {
        throw std::runtime_error("swiglu: output shape/dtype mismatch");
    }

    swiglu_dispatch(gate_proj, up_proj, out);
    sync_and_check("swiglu kernel failed");
}

// ============================================================================
// Fused GeGLU (GELU variant)
// ============================================================================

// Internal dispatch helper
static void geglu_dispatch(
    const GPUArray& gate_proj,
    const GPUArray& up_proj,
    GPUArray& output
) {
    size_t n = gate_proj.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (gate_proj.dtype()) {
        case DataType::Float32:
            nn::geglu_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(gate_proj.data()),
                static_cast<const float*>(up_proj.data()),
                static_cast<float*>(output.data()),
                n);
            break;
        case DataType::Float16:
            nn::geglu_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(gate_proj.data()),
                static_cast<const __half*>(up_proj.data()),
                static_cast<__half*>(output.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::geglu_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(gate_proj.data()),
                static_cast<const __nv_bfloat16*>(up_proj.data()),
                static_cast<__nv_bfloat16*>(output.data()),
                n);
            break;
        default:
            throw std::runtime_error("geglu: unsupported dtype");
    }
}

GPUArray geglu(const GPUArray& gate_proj, const GPUArray& up_proj) {
    if (gate_proj.shape() != up_proj.shape()) {
        throw std::runtime_error("geglu: gate_proj and up_proj shape mismatch");
    }
    if (gate_proj.dtype() != up_proj.dtype()) {
        throw std::runtime_error("geglu: gate_proj and up_proj dtype mismatch");
    }
    if (gate_proj.dtype() != DataType::Float32 &&
        gate_proj.dtype() != DataType::Float16 &&
        gate_proj.dtype() != DataType::BFloat16) {
        throw std::runtime_error("geglu: only float32/float16/bfloat16 supported");
    }

    GPUArray output(gate_proj.shape(), gate_proj.dtype());
    geglu_dispatch(gate_proj, up_proj, output);
    sync_and_check("geglu kernel failed");
    return output;
}

// In-place version (for CUDA Graph capture)
void geglu(const GPUArray& gate_proj, const GPUArray& up_proj, GPUArray& out) {
    if (gate_proj.shape() != up_proj.shape()) {
        throw std::runtime_error("geglu: gate_proj and up_proj shape mismatch");
    }
    if (gate_proj.dtype() != up_proj.dtype()) {
        throw std::runtime_error("geglu: gate_proj and up_proj dtype mismatch");
    }
    if (out.shape() != gate_proj.shape() || out.dtype() != gate_proj.dtype()) {
        throw std::runtime_error("geglu: output shape/dtype mismatch");
    }

    geglu_dispatch(gate_proj, up_proj, out);
    sync_and_check("geglu kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
