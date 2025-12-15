/**
 * Neural Network operations dispatch
 */
#include "nn_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"
#include <algorithm>

namespace pygpukit {
namespace ops {

using namespace nn;

// ============================================================================
// GELU Activation
// ============================================================================

GPUArray gelu(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("gelu only supports float types");
    }

    GPUArray result(input.shape(), input.dtype());
    size_t n = input.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (input.dtype()) {
        case DataType::Float32:
            gelu_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float64:
            gelu_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                n);
            break;
        case DataType::Float16:
            gelu_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            gelu_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }

    sync_and_check("gelu kernel failed");
    return result;
}

// ============================================================================
// Transpose
// ============================================================================

GPUArray transpose(const GPUArray& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("transpose expects 2D input [rows, cols]");
    }

    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    // Output shape is [cols, rows]
    GPUArray result({cols, rows}, input.dtype());

    // Use 32x32 tiles with 32x8 threads
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    switch (input.dtype()) {
        case DataType::Float32:
            transpose_f32_kernel<<<grid, block>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                rows, cols);
            break;
        case DataType::Float64:
            transpose_f64_kernel<<<grid, block>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                rows, cols);
            break;
        case DataType::Float16:
            transpose_f16_kernel<<<grid, block>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                rows, cols);
            break;
        case DataType::BFloat16:
            transpose_bf16_kernel<<<grid, block>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                rows, cols);
            break;
        default:
            throw std::runtime_error("transpose only supports float types");
    }

    sync_and_check("transpose kernel failed");
    return result;
}

// ============================================================================
// Bias Add
// ============================================================================

// In-place bias add: output[batch, features] += bias[features]
void bias_add_inplace(GPUArray& output, const GPUArray& bias) {
    if (output.ndim() != 2) {
        throw std::runtime_error("bias_add expects 2D output tensor [batch, features]");
    }
    if (bias.ndim() != 1) {
        throw std::runtime_error("bias_add expects 1D bias tensor [features]");
    }
    if (output.dtype() != bias.dtype()) {
        throw std::runtime_error("bias_add: dtype mismatch");
    }

    size_t batch_size = output.shape()[0];
    size_t features = output.shape()[1];

    if (bias.shape()[0] != features) {
        throw std::runtime_error("bias_add: bias size must match output features");
    }

    size_t n = batch_size * features;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (output.dtype()) {
        case DataType::Float32:
            bias_add_f32_kernel<<<grid_size, block_size>>>(
                static_cast<float*>(output.data()),
                static_cast<const float*>(bias.data()),
                batch_size, features);
            break;
        case DataType::Float64:
            bias_add_f64_kernel<<<grid_size, block_size>>>(
                static_cast<double*>(output.data()),
                static_cast<const double*>(bias.data()),
                batch_size, features);
            break;
        case DataType::Float16:
            bias_add_f16_kernel<<<grid_size, block_size>>>(
                static_cast<__half*>(output.data()),
                static_cast<const __half*>(bias.data()),
                batch_size, features);
            break;
        case DataType::BFloat16:
            bias_add_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<__nv_bfloat16*>(output.data()),
                static_cast<const __nv_bfloat16*>(bias.data()),
                batch_size, features);
            break;
        default:
            throw std::runtime_error("bias_add only supports float types");
    }

    sync_and_check("bias_add kernel failed");
}

// ============================================================================
// Linear Layer: y = xW^T + b
// ============================================================================

GPUArray linear(const GPUArray& input, const GPUArray& weight, const GPUArray* bias) {
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    // output: [batch, out_features]

    if (input.ndim() != 2) {
        throw std::runtime_error("linear expects 2D input [batch, in_features]");
    }
    if (weight.ndim() != 2) {
        throw std::runtime_error("linear expects 2D weight [out_features, in_features]");
    }
    if (input.dtype() != weight.dtype()) {
        throw std::runtime_error("linear: input and weight dtype mismatch");
    }

    size_t batch = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::runtime_error("linear: weight in_features must match input");
    }

    // Compute y = x @ W^T using matmul with transposed weight
    // For now, we'll transpose weight and use matmul
    // TODO: Add transpose operation or use cuBLAS GEMM directly

    // Create transposed weight [in_features, out_features]
    GPUArray weight_t({in_features, out_features}, weight.dtype());

    // Simple transpose kernel
    // For MVP, we'll just do matmul(input, weight.T)
    // This requires a transpose, which we'll implement inline

    // Launch transpose kernel (simple 2D transpose)
    const int block_dim = 16;
    dim3 block(block_dim, block_dim);
    dim3 grid((out_features + block_dim - 1) / block_dim,
              (in_features + block_dim - 1) / block_dim);

    // Inline transpose kernel launch
    auto transpose_f32 = [](const float* src, float* dst, int rows, int cols, dim3 grid, dim3 block) {
        // Simple element-wise transpose
        struct TransposeArgs { const float* src; float* dst; int rows; int cols; };
        // Use a lambda kernel via NVRTC would be ideal, but for now use a simple loop
        // This is temporary - proper transpose kernel should be in a separate file
    };

    // For MVP: use row-major matmul and handle transpose in a simple way
    // Actually, let's use the fact that (A @ B.T) = (B @ A.T).T for some cases
    // Or better: just implement it directly with cuBLAS-style GEMM semantics

    // Simplest approach for MVP: copy weight transposed element-by-element on host
    // This is slow but correct for small models like GPT-2

    // For now, compute output = input @ weight^T directly using our matmul
    // Our matmul does C = A @ B where A is MxK, B is KxN, C is MxN
    // We need: output = input @ weight^T
    // input: [batch, in_features] = [M, K]
    // weight: [out_features, in_features] = [N, K]
    // weight^T: [in_features, out_features] = [K, N]
    // output: [batch, out_features] = [M, N]

    // So we need to transpose weight first
    // For MVP, let's assume weight is stored as [out_features, in_features]
    // and we need [in_features, out_features]

    // Actually, the simplest MVP is to use a different matmul signature
    // that handles transposed B directly. For now, let's just do naive CPU transpose.

    // Even simpler: for MVP, assume weight is already in the right layout
    // or do the computation via multiple kernels

    // Let's do: output = matmul(input, weight_transposed)
    // where we transpose weight on GPU using a simple kernel

    // For GPT-2 small: in_features = 768, out_features = 768 or 3072
    // This is manageable

    // Create result first
    GPUArray result({batch, out_features}, input.dtype());

    // For MVP: use matmul with transposed semantics
    // We'll add a transposed matmul later, for now do element-wise transpose

    // Temporary: use internal matmul that can handle transpose
    // Our existing matmul assumes row-major A @ B
    // We need A @ B^T which is equivalent to (B @ A^T)^T

    // Simplest solution: call cuBLAS-style GEMM
    // For now, let's implement a simple transpose + matmul

    // Skip bias for now in basic implementation
    (void)bias;

    // For MVP, return a placeholder that works for small matrices
    // Real implementation would use optimized transpose + matmul

    // Actually, let's make this work by noting:
    // C[i,j] = sum_k A[i,k] * B[k,j]  (normal matmul)
    // We want: C[i,j] = sum_k A[i,k] * W[j,k]  (matmul with transposed W)
    // This is GEMM with transB = true

    // Our current matmul is C = A @ B (both row-major)
    // We need C = A @ B^T

    // Let's add this capability to our matmul

    throw std::runtime_error("linear: not yet implemented - use matmul + bias_add separately for MVP");
}

// ============================================================================
// LayerNorm
// ============================================================================

GPUArray layernorm(const GPUArray& input, const GPUArray& gamma, const GPUArray& beta, float eps) {
    // input: [batch, features]
    // gamma: [features]
    // beta: [features]

    if (input.ndim() != 2) {
        throw std::runtime_error("layernorm expects 2D input [batch, features]");
    }
    if (gamma.ndim() != 1 || beta.ndim() != 1) {
        throw std::runtime_error("layernorm expects 1D gamma and beta");
    }
    if (input.dtype() != gamma.dtype() || input.dtype() != beta.dtype()) {
        throw std::runtime_error("layernorm: dtype mismatch");
    }

    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    if (gamma.shape()[0] != features || beta.shape()[0] != features) {
        throw std::runtime_error("layernorm: gamma/beta size must match features");
    }

    GPUArray result(input.shape(), input.dtype());

    // One block per row, use enough threads to cover features
    int block_size = std::min(256, (int)((features + 31) / 32 * 32));
    block_size = std::max(32, block_size);

    switch (input.dtype()) {
        case DataType::Float32:
            layernorm_f32_kernel<<<batch_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(gamma.data()),
                static_cast<const float*>(beta.data()),
                static_cast<float*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::Float64:
            layernorm_f64_kernel<<<batch_size, block_size>>>(
                static_cast<const double*>(input.data()),
                static_cast<const double*>(gamma.data()),
                static_cast<const double*>(beta.data()),
                static_cast<double*>(result.data()),
                batch_size, features, (double)eps);
            break;
        case DataType::Float16:
            layernorm_f16_kernel<<<batch_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(gamma.data()),
                static_cast<const __half*>(beta.data()),
                static_cast<__half*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::BFloat16:
            layernorm_bf16_kernel<<<batch_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(gamma.data()),
                static_cast<const __nv_bfloat16*>(beta.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                batch_size, features, eps);
            break;
        default:
            throw std::runtime_error("layernorm only supports float types");
    }

    sync_and_check("layernorm kernel failed");
    return result;
}

} // namespace ops
} // namespace pygpukit
