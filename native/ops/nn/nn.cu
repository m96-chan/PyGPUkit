/**
 * Neural Network operations dispatch
 */
#include "nn_kernels.cuh"
#include "flash_attention.cuh"
#include "flash_decoding.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include <algorithm>
#include <cstdlib>

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

    // Use capture stream for CUDA Graph compatibility
    cudaStream_t stream = internal::get_capture_stream();

    switch (output.dtype()) {
        case DataType::Float32:
            bias_add_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(output.data()),
                static_cast<const float*>(bias.data()),
                batch_size, features);
            break;
        case DataType::Float64:
            bias_add_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(output.data()),
                static_cast<const double*>(bias.data()),
                batch_size, features);
            break;
        case DataType::Float16:
            bias_add_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(output.data()),
                static_cast<const __half*>(bias.data()),
                batch_size, features);
            break;
        case DataType::BFloat16:
            bias_add_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
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
// Softmax
// ============================================================================

GPUArray softmax(const GPUArray& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("softmax expects 2D input [batch, features]");
    }
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("softmax only supports float types");
    }

    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    GPUArray result(input.shape(), input.dtype());

    // One block per row
    int block_size = std::min(256, (int)((features + 31) / 32 * 32));
    block_size = std::max(32, block_size);

    switch (input.dtype()) {
        case DataType::Float32:
            nn::softmax_f32_kernel<<<batch_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                batch_size, features);
            break;
        case DataType::Float64:
            nn::softmax_f64_kernel<<<batch_size, block_size>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                batch_size, features);
            break;
        case DataType::Float16:
            nn::softmax_f16_kernel<<<batch_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                batch_size, features);
            break;
        case DataType::BFloat16:
            nn::softmax_bf16_kernel<<<batch_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                batch_size, features);
            break;
        default:
            break;
    }

    sync_and_check("softmax kernel failed");
    return result;
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

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// ============================================================================

// Internal helper for rmsnorm kernel dispatch
static void rmsnorm_dispatch(
    const GPUArray& input,
    const GPUArray& gamma,
    GPUArray& result,
    float eps
) {
    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    // One block per row, use enough threads to cover features
    int block_size = std::min(256, (int)((features + 31) / 32 * 32));
    block_size = std::max(32, block_size);

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::rmsnorm_f32_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(gamma.data()),
                static_cast<float*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::Float64:
            nn::rmsnorm_f64_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const double*>(input.data()),
                static_cast<const double*>(gamma.data()),
                static_cast<double*>(result.data()),
                batch_size, features, (double)eps);
            break;
        case DataType::Float16:
            nn::rmsnorm_f16_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(gamma.data()),
                static_cast<__half*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::BFloat16:
            nn::rmsnorm_bf16_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(gamma.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                batch_size, features, eps);
            break;
        default:
            throw std::runtime_error("rmsnorm only supports float types");
    }
}

GPUArray rmsnorm(const GPUArray& input, const GPUArray& gamma, float eps) {
    // input: [batch, features]
    // gamma: [features]

    if (input.ndim() != 2) {
        throw std::runtime_error("rmsnorm expects 2D input [batch, features]");
    }
    if (gamma.ndim() != 1) {
        throw std::runtime_error("rmsnorm expects 1D gamma");
    }
    if (input.dtype() != gamma.dtype()) {
        throw std::runtime_error("rmsnorm: dtype mismatch");
    }

    size_t features = input.shape()[1];

    if (gamma.shape()[0] != features) {
        throw std::runtime_error("rmsnorm: gamma size must match features");
    }

    GPUArray result(input.shape(), input.dtype());
    rmsnorm_dispatch(input, gamma, result, eps);
    sync_and_check("rmsnorm kernel failed");
    return result;
}

// In-place variant for CUDA Graph capture
void rmsnorm(const GPUArray& input, const GPUArray& gamma, GPUArray& out, float eps) {
    // input: [batch, features]
    // gamma: [features]
    // out: [batch, features]

    if (input.ndim() != 2) {
        throw std::runtime_error("rmsnorm expects 2D input [batch, features]");
    }
    if (gamma.ndim() != 1) {
        throw std::runtime_error("rmsnorm expects 1D gamma");
    }
    if (out.ndim() != 2) {
        throw std::runtime_error("rmsnorm expects 2D output");
    }
    if (input.dtype() != gamma.dtype() || input.dtype() != out.dtype()) {
        throw std::runtime_error("rmsnorm: dtype mismatch");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("rmsnorm: input and output shape mismatch");
    }

    size_t features = input.shape()[1];

    if (gamma.shape()[0] != features) {
        throw std::runtime_error("rmsnorm: gamma size must match features");
    }

    rmsnorm_dispatch(input, gamma, out, eps);
    sync_and_check("rmsnorm kernel failed");
}

// ============================================================================
// RoPE (Rotary Position Embedding) - In-place
// ============================================================================

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

// ============================================================================
// Split QKV Batch
// Splits fused QKV projection output [seq_len, q_dim + k_dim + v_dim]
// into separate Q, K, V tensors for batch decode
// ============================================================================

void split_qkv_batch(
    const GPUArray& qkv,
    GPUArray& q_out,
    GPUArray& k_out,
    GPUArray& v_out,
    int q_dim,
    int k_dim,
    int v_dim
) {
    if (qkv.ndim() != 2) {
        throw std::runtime_error("split_qkv_batch: qkv must be 2D [seq_len, total_dim]");
    }

    int seq_len = static_cast<int>(qkv.shape()[0]);
    int total_dim = q_dim + k_dim + v_dim;

    if (static_cast<int>(qkv.shape()[1]) != total_dim) {
        throw std::runtime_error("split_qkv_batch: qkv dim mismatch");
    }

    int total_elements = seq_len * total_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (qkv.dtype()) {
        case DataType::Float16:
            nn::split_qkv_batch_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(qkv.data()),
                static_cast<__half*>(q_out.data()),
                static_cast<__half*>(k_out.data()),
                static_cast<__half*>(v_out.data()),
                seq_len, q_dim, k_dim, v_dim);
            break;
        case DataType::Float32:
            nn::split_qkv_batch_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(qkv.data()),
                static_cast<float*>(q_out.data()),
                static_cast<float*>(k_out.data()),
                static_cast<float*>(v_out.data()),
                seq_len, q_dim, k_dim, v_dim);
            break;
        case DataType::BFloat16:
            nn::split_qkv_batch_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(qkv.data()),
                static_cast<__nv_bfloat16*>(q_out.data()),
                static_cast<__nv_bfloat16*>(k_out.data()),
                static_cast<__nv_bfloat16*>(v_out.data()),
                seq_len, q_dim, k_dim, v_dim);
            break;
        default:
            throw std::runtime_error("split_qkv_batch: unsupported dtype");
    }

    sync_and_check("split_qkv_batch kernel failed");
}

// ============================================================================
// SiLU (Swish) Activation: x * sigmoid(x)
// ============================================================================

// Internal dispatch helper with capture stream support
static void silu_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::silu_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float64:
            nn::silu_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::silu_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::silu_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray silu(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("silu only supports float types");
    }

    GPUArray result(input.shape(), input.dtype());
    silu_dispatch(input, result);
    sync_and_check("silu kernel failed");
    return result;
}

// SiLU with output buffer (for CUDA Graph capture)
void silu(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("silu only supports float types");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("silu: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("silu: shape mismatch between input and output");
    }

    silu_dispatch(input, out);
    sync_and_check("silu kernel failed");
}

// ============================================================================
// Sigmoid Activation: 1 / (1 + exp(-x))
// ============================================================================

static void sigmoid_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::sigmoid_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::sigmoid_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::sigmoid_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray sigmoid(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sigmoid only supports float types (f32, f16, bf16)");
    }

    GPUArray result(input.shape(), input.dtype());
    sigmoid_dispatch(input, result);
    sync_and_check("sigmoid kernel failed");
    return result;
}

void sigmoid(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sigmoid only supports float types (f32, f16, bf16)");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("sigmoid: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("sigmoid: shape mismatch between input and output");
    }

    sigmoid_dispatch(input, out);
    sync_and_check("sigmoid kernel failed");
}

// ============================================================================
// Tanh Activation
// ============================================================================

static void tanh_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::tanh_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::tanh_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::tanh_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray tanh(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("tanh only supports float types (f32, f16, bf16)");
    }

    GPUArray result(input.shape(), input.dtype());
    tanh_dispatch(input, result);
    sync_and_check("tanh kernel failed");
    return result;
}

void tanh(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("tanh only supports float types (f32, f16, bf16)");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("tanh: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("tanh: shape mismatch between input and output");
    }

    tanh_dispatch(input, out);
    sync_and_check("tanh kernel failed");
}

// ============================================================================
// Scaled Dot-Product Attention (SDPA) with Causal Mask
// ============================================================================

// Flash Attention mode:
// - "0" or "false": Always use standard SDPA
// - "1" or "true": Always use Flash Attention
// - "auto" or unset: Auto-select based on sequence length (>2048 uses Flash)
static int get_flash_attention_mode() {
    static int cached = -2;  // -2 = not checked, -1 = auto, 0 = off, 1 = on
    if (cached == -2) {
        const char* env = std::getenv("PYGPUKIT_FLASH_ATTENTION");
        if (env == nullptr || std::string(env) == "auto") {
            cached = -1;  // auto mode
        } else if (std::string(env) == "1" || std::string(env) == "true") {
            cached = 1;   // force on
        } else {
            cached = 0;   // force off
        }
    }
    return cached;
}

// Threshold for auto-selecting Flash Attention (sequence length)
constexpr int FLASH_ATTENTION_SEQ_THRESHOLD = 2048;

// Flash-Decoding workspace manager (lazy allocation, auto-expanding)
class FlashDecodingWorkspace {
public:
    static float* get(int n_heads, int head_dim, int kv_len) {
        static FlashDecodingWorkspace instance;
        size_t required = flash_decoding::flash_decoding_workspace_size(n_heads, head_dim, kv_len);
        if (required > instance.size_) {
            instance.resize(required);
        }
        return instance.buffer_;
    }

private:
    FlashDecodingWorkspace() : buffer_(nullptr), size_(0) {}

    ~FlashDecodingWorkspace() {
        if (buffer_) {
            device_free(buffer_);
        }
    }

    void resize(size_t new_size) {
        if (buffer_) {
            device_free(buffer_);
        }
        buffer_ = static_cast<float*>(device_malloc(new_size));
        size_ = new_size;
    }

    float* buffer_;
    size_t size_;
};

// Environment variable control for Flash-Decoding
// PYGPUKIT_FLASH_DECODING: 0=off, 1=on, -1=auto (default)
static int get_flash_decoding_mode() {
    static int cached = -999;
    if (cached == -999) {
        const char* env = std::getenv("PYGPUKIT_FLASH_DECODING");
        if (env) {
            cached = std::atoi(env);
        } else {
            cached = -1;  // Auto mode by default
        }
    }
    return cached;
}

// Internal helper for SDPA kernel dispatch
// context_len: if > 0, use this as kv_len (for fixed-length cache)
//              if <= 0, use K.shape()[1] as kv_len
static void sdpa_causal_dispatch(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& result, float scale, int context_len = 0
) {
    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];
    // kv_stride: actual K/V tensor size (for pointer calculations)
    int kv_stride = static_cast<int>(K.shape()[1]);
    // kv_len: number of KV positions to attend to (for masking)
    int kv_len = (context_len > 0) ? context_len : kv_stride;

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    // Causal offset for proper masking
    int causal_offset = kv_len - q_len;

    // Grid: one block per (head, query_position) pair
    dim3 grid(n_heads, q_len);
    int block_size = 128;  // Enough threads for reduction

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    // Flash-Decoding: Optimized for decode phase (q_len=1)
    // Parallelizes over KV sequence length for better GPU utilization
    int flash_decoding_mode = get_flash_decoding_mode();
    bool use_flash_decoding = false;
    if (q_len == 1 && head_dim <= 128) {
        if (flash_decoding_mode == 1) {
            // Force on
            use_flash_decoding = true;
        } else if (flash_decoding_mode == -1) {
            // Auto: use Flash-Decoding when it provides benefit
            // Crossover point is around kv_len=1024 (4 chunks with chunk_size=256)
            // Only enable for long contexts where parallelism benefit > kernel launch overhead
            use_flash_decoding = (kv_len >= 1024);
        }
    }

    if (use_flash_decoding) {
        // Flash-Decoding: chunk-parallel attention for decode phase
        float* workspace = FlashDecodingWorkspace::get(n_heads, head_dim, kv_len);

        switch (Q.dtype()) {
            case DataType::Float16:
                flash_decoding::flash_decoding_f16(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    workspace,
                    n_heads, head_dim, kv_len, kv_stride, stream
                );
                return;
            default:
                // Fall through to standard SDPA for unsupported dtypes
                break;
        }
    }

    // Determine whether to use Flash Attention
    // - Auto mode: use Flash for long sequences (>2048) where memory savings matter
    // - Force mode: respect user preference
    int flash_mode = get_flash_attention_mode();
    bool use_flash = false;
    if (flash_mode == 1) {
        // Force on
        use_flash = (head_dim <= 128);
    } else if (flash_mode == -1) {
        // Auto: use Flash for long sequences
        use_flash = (head_dim <= 128) && (kv_len > FLASH_ATTENTION_SEQ_THRESHOLD);
    }
    // flash_mode == 0: force off, use_flash stays false

    if (use_flash) {
        // Flash Attention 2: O(n) memory, tiled computation
        size_t shared_mem_size = nn::flash_attention_smem_size(head_dim);

        switch (Q.dtype()) {
            case DataType::Float32:
                nn::flash_attention_f32_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const float*>(Q.data()),
                    static_cast<const float*>(K.data()),
                    static_cast<const float*>(V.data()),
                    static_cast<float*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::Float16:
                nn::flash_attention_f16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::BFloat16:
                nn::flash_attention_bf16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            default:
                throw std::runtime_error("sdpa only supports Float32, Float16, BFloat16");
        }
    } else {
        // Standard SDPA: O(n²) memory for attention scores
        size_t shared_mem_size = kv_len * sizeof(float);

        switch (Q.dtype()) {
            case DataType::Float32:
                nn::sdpa_causal_f32_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const float*>(Q.data()),
                    static_cast<const float*>(K.data()),
                    static_cast<const float*>(V.data()),
                    static_cast<float*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::Float16:
                nn::sdpa_causal_f16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::BFloat16:
                nn::sdpa_causal_bf16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            default:
                throw std::runtime_error("sdpa only supports Float32, Float16, BFloat16");
        }
    }
}

GPUArray sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, float scale) {
    // Q: [n_heads, q_len, head_dim]
    // K: [n_heads, kv_len, head_dim]
    // V: [n_heads, kv_len, head_dim]
    // Output: [n_heads, q_len, head_dim]

    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }

    GPUArray result({(size_t)n_heads, (size_t)q_len, (size_t)head_dim}, Q.dtype());
    sdpa_causal_dispatch(Q, K, V, result, scale);
    sync_and_check("sdpa kernel failed");
    return result;
}

// SDPA with output buffer (for CUDA Graph capture)
void sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, GPUArray& out, float scale) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }
    if (out.shape()[0] != n_heads || out.shape()[1] != q_len || out.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: output shape mismatch");
    }

    sdpa_causal_dispatch(Q, K, V, out, scale);
    sync_and_check("sdpa kernel failed");
}

// SDPA with fixed-length KV cache support
// context_len: actual number of valid tokens in KV cache (K/V may have max_seq_len)
void sdpa_causal_fixed_cache(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& out, int context_len, float scale
) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }
    if (out.shape()[0] != n_heads || out.shape()[1] != q_len || out.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: output shape mismatch");
    }
    if (context_len <= 0 || context_len > static_cast<int>(K.shape()[1])) {
        throw std::runtime_error("sdpa: invalid context_len");
    }

    sdpa_causal_dispatch(Q, K, V, out, scale, context_len);
    sync_and_check("sdpa kernel failed");
}

// SDPA with fixed-length KV cache using pointer-based context_len (for CUDA Graph)
// context_len_buf: GPU buffer containing actual context_len (read at runtime)
// max_kv_len: Maximum KV length (for shared memory allocation during graph capture)
void sdpa_causal_fixed_cache_ptr(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& out, const GPUArray& context_len_buf, int max_kv_len, float scale
) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }
    if (context_len_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("sdpa: context_len_buf must be int32");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];
    int kv_stride = static_cast<int>(K.shape()[1]);

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }
    if (out.shape()[0] != n_heads || out.shape()[1] != q_len || out.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: output shape mismatch");
    }
    if (max_kv_len <= 0 || max_kv_len > kv_stride) {
        throw std::runtime_error("sdpa: invalid max_kv_len");
    }

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    // Grid: one block per (head, query_position) pair
    dim3 grid(n_heads, q_len);
    int block_size = 128;

    // Allocate shared memory for max_kv_len (allows dynamic context_len at runtime)
    size_t shared_mem_size = max_kv_len * sizeof(float);

    cudaStream_t stream = internal::get_capture_stream();

    switch (Q.dtype()) {
        case DataType::Float32:
            nn::sdpa_causal_f32_kernel_ptr<<<grid, block_size, shared_mem_size, stream>>>(
                static_cast<const float*>(Q.data()),
                static_cast<const float*>(K.data()),
                static_cast<const float*>(V.data()),
                static_cast<float*>(out.data()),
                static_cast<const int*>(context_len_buf.data()),
                n_heads, q_len, kv_stride, head_dim, scale);
            break;
        case DataType::Float16:
            nn::sdpa_causal_f16_kernel_ptr<<<grid, block_size, shared_mem_size, stream>>>(
                static_cast<const __half*>(Q.data()),
                static_cast<const __half*>(K.data()),
                static_cast<const __half*>(V.data()),
                static_cast<__half*>(out.data()),
                static_cast<const int*>(context_len_buf.data()),
                n_heads, q_len, kv_stride, head_dim, scale);
            break;
        case DataType::BFloat16:
            nn::sdpa_causal_bf16_kernel_ptr<<<grid, block_size, shared_mem_size, stream>>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const int*>(context_len_buf.data()),
                n_heads, q_len, kv_stride, head_dim, scale);
            break;
        default:
            throw std::runtime_error("sdpa: unsupported dtype");
    }

    sync_and_check("sdpa_causal_fixed_cache_ptr kernel failed");
}

// ============================================================================
// Tensor Manipulation Operations
// ============================================================================

// Concat two tensors along axis 0
// a: [dim0_a, ...], b: [dim0_b, ...] -> output: [dim0_a + dim0_b, ...]
GPUArray concat_axis0(const GPUArray& a, const GPUArray& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("concat: dtype mismatch");
    }
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float16 &&
        a.dtype() != DataType::BFloat16 && a.dtype() != DataType::UInt8) {
        throw std::runtime_error("concat: only float32/float16/bfloat16/uint8 supported");
    }
    if (a.ndim() < 1 || b.ndim() < 1 || a.ndim() != b.ndim()) {
        throw std::runtime_error("concat: dimension mismatch");
    }

    // Check that all dimensions except axis 0 match
    for (size_t i = 1; i < a.ndim(); i++) {
        if (a.shape()[i] != b.shape()[i]) {
            throw std::runtime_error("concat: shape mismatch on non-concat axis");
        }
    }

    // Compute output shape
    std::vector<size_t> out_shape = a.shape();
    out_shape[0] = a.shape()[0] + b.shape()[0];

    GPUArray result(out_shape, a.dtype());

    // Compute stride (elements per "row" along axis 0)
    size_t stride = 1;
    for (size_t i = 1; i < a.ndim(); i++) {
        stride *= a.shape()[i];
    }

    size_t total = result.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            nn::concat_axis0_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        case DataType::Float16:
            nn::concat_axis0_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        case DataType::BFloat16:
            nn::concat_axis0_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        case DataType::UInt8:
            nn::concat_axis0_u8_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(a.data()),
                static_cast<const uint8_t*>(b.data()),
                static_cast<uint8_t*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        default:
            break;
    }

    sync_and_check("concat_axis0 kernel failed");
    return result;
}

// Repeat interleave along axis 1 (for GQA expansion)
// input: [dim0, dim1, dim2] -> output: [dim0, dim1 * repeats, dim2]
GPUArray repeat_interleave_axis1(const GPUArray& input, size_t repeats) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("repeat_interleave: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("repeat_interleave: expects 3D tensor [dim0, dim1, dim2]");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];

    std::vector<size_t> out_shape = {dim0, dim1 * repeats, dim2};
    GPUArray result(out_shape, input.dtype());

    size_t total = result.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    switch (input.dtype()) {
        case DataType::Float32:
            nn::repeat_interleave_axis1_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                dim0, dim1, dim2, repeats);
            break;
        case DataType::Float16:
            nn::repeat_interleave_axis1_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                dim0, dim1, dim2, repeats);
            break;
        case DataType::BFloat16:
            nn::repeat_interleave_axis1_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                dim0, dim1, dim2, repeats);
            break;
        default:
            break;
    }

    sync_and_check("repeat_interleave_axis1 kernel failed");
    return result;
}

// Internal helper for transpose_3d_021 kernel dispatch
static void transpose_3d_021_dispatch(
    const GPUArray& input,
    GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_021_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                dim0, dim1, dim2);
            break;
        case DataType::Float16:
            nn::transpose_021_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                dim0, dim1, dim2);
            break;
        case DataType::BFloat16:
            nn::transpose_021_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                dim0, dim1, dim2);
            break;
        default:
            throw std::runtime_error("transpose_3d_021: unsupported dtype");
    }
}

// Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]
GPUArray transpose_3d_021(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_021: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("transpose_3d_021: expects 3D tensor");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];

    // Output shape: [dim1, dim0, dim2]
    std::vector<size_t> out_shape = {dim1, dim0, dim2};
    GPUArray result(out_shape, input.dtype());

    transpose_3d_021_dispatch(input, result, dim0, dim1, dim2);
    sync_and_check("transpose_3d_021 kernel failed");
    return result;
}

// Transpose 3D tensor with output buffer (for CUDA Graph capture)
void transpose_3d_021(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_021: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("transpose_3d_021: expects 3D tensor");
    }
    if (out.ndim() != 3) {
        throw std::runtime_error("transpose_3d_021: output expects 3D tensor");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_3d_021: dtype mismatch");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];

    // Verify output shape: [dim1, dim0, dim2]
    if (out.shape()[0] != dim1 || out.shape()[1] != dim0 || out.shape()[2] != dim2) {
        throw std::runtime_error("transpose_3d_021: output shape mismatch, expected [" +
            std::to_string(dim1) + ", " + std::to_string(dim0) + ", " + std::to_string(dim2) + "]");
    }

    transpose_3d_021_dispatch(input, out, dim0, dim1, dim2);
    sync_and_check("transpose_3d_021 kernel failed");
}

// Internal helper for transpose_4d_0213 kernel dispatch
static void transpose_4d_0213_dispatch(
    const GPUArray& input,
    GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_0213_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                dim0, dim1, dim2, dim3);
            break;
        case DataType::Float16:
            nn::transpose_0213_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                dim0, dim1, dim2, dim3);
            break;
        case DataType::BFloat16:
            nn::transpose_0213_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                dim0, dim1, dim2, dim3);
            break;
        default:
            throw std::runtime_error("transpose_4d_0213: unsupported dtype");
    }
}

// Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
GPUArray transpose_4d_0213(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0213: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0213: expects 4D tensor");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];
    size_t dim3 = input.shape()[3];

    // Output shape: [dim0, dim2, dim1, dim3]
    std::vector<size_t> out_shape = {dim0, dim2, dim1, dim3};
    GPUArray result(out_shape, input.dtype());

    transpose_4d_0213_dispatch(input, result, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0213 kernel failed");
    return result;
}

// Transpose 4D tensor with output buffer (for CUDA Graph capture)
void transpose_4d_0213(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0213: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0213: expects 4D tensor");
    }
    if (out.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0213: output expects 4D tensor");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_4d_0213: dtype mismatch");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];
    size_t dim3 = input.shape()[3];

    // Verify output shape: [dim0, dim2, dim1, dim3]
    if (out.shape()[0] != dim0 || out.shape()[1] != dim2 ||
        out.shape()[2] != dim1 || out.shape()[3] != dim3) {
        throw std::runtime_error("transpose_4d_0213: output shape mismatch, expected [" +
            std::to_string(dim0) + ", " + std::to_string(dim2) + ", " +
            std::to_string(dim1) + ", " + std::to_string(dim3) + "]");
    }

    transpose_4d_0213_dispatch(input, out, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0213 kernel failed");
}

// ============================================================================
// 3D Transpose: [d0, d1, d2] -> [d0, d2, d1] (swaps last two axes)
// ============================================================================

// Internal helper for transpose_3d_012 kernel dispatch
static void transpose_3d_012_dispatch(
    const GPUArray& input,
    GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_012_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                dim0, dim1, dim2);
            break;
        case DataType::Float16:
            nn::transpose_012_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                dim0, dim1, dim2);
            break;
        case DataType::BFloat16:
            nn::transpose_012_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                dim0, dim1, dim2);
            break;
        default:
            throw std::runtime_error("transpose_3d_012: unsupported dtype");
    }
}

// Transpose 3D tensor: [d0, d1, d2] -> [d0, d2, d1]
GPUArray transpose_3d_012(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_012: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("transpose_3d_012: expects 3D tensor");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];

    // Output shape: [dim0, dim2, dim1]
    std::vector<size_t> out_shape = {dim0, dim2, dim1};
    GPUArray result(out_shape, input.dtype());

    transpose_3d_012_dispatch(input, result, dim0, dim1, dim2);
    sync_and_check("transpose_3d_012 kernel failed");
    return result;
}

// Transpose 3D tensor with output buffer (for CUDA Graph capture)
void transpose_3d_012(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_012: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("transpose_3d_012: expects 3D tensor");
    }
    if (out.ndim() != 3) {
        throw std::runtime_error("transpose_3d_012: output expects 3D tensor");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_3d_012: dtype mismatch");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];

    // Verify output shape: [dim0, dim2, dim1]
    if (out.shape()[0] != dim0 || out.shape()[1] != dim2 || out.shape()[2] != dim1) {
        throw std::runtime_error("transpose_3d_012: output shape mismatch, expected [" +
            std::to_string(dim0) + ", " + std::to_string(dim2) + ", " + std::to_string(dim1) + "]");
    }

    transpose_3d_012_dispatch(input, out, dim0, dim1, dim2);
    sync_and_check("transpose_3d_012 kernel failed");
}

// ============================================================================
// 4D Transpose: [d0, d1, d2, d3] -> [d0, d1, d3, d2] (swaps last two axes)
// ============================================================================

// Internal helper for transpose_4d_0132 kernel dispatch
static void transpose_4d_0132_dispatch(
    const GPUArray& input,
    GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_0132_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                dim0, dim1, dim2, dim3);
            break;
        case DataType::Float16:
            nn::transpose_0132_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                dim0, dim1, dim2, dim3);
            break;
        case DataType::BFloat16:
            nn::transpose_0132_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                dim0, dim1, dim2, dim3);
            break;
        default:
            throw std::runtime_error("transpose_4d_0132: unsupported dtype");
    }
}

// Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d1, d3, d2]
GPUArray transpose_4d_0132(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0132: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0132: expects 4D tensor");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];
    size_t dim3 = input.shape()[3];

    // Output shape: [dim0, dim1, dim3, dim2]
    std::vector<size_t> out_shape = {dim0, dim1, dim3, dim2};
    GPUArray result(out_shape, input.dtype());

    transpose_4d_0132_dispatch(input, result, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0132 kernel failed");
    return result;
}

// Transpose 4D tensor with output buffer (for CUDA Graph capture)
void transpose_4d_0132(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0132: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0132: expects 4D tensor");
    }
    if (out.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0132: output expects 4D tensor");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_4d_0132: dtype mismatch");
    }

    size_t dim0 = input.shape()[0];
    size_t dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2];
    size_t dim3 = input.shape()[3];

    // Verify output shape: [dim0, dim1, dim3, dim2]
    if (out.shape()[0] != dim0 || out.shape()[1] != dim1 ||
        out.shape()[2] != dim3 || out.shape()[3] != dim2) {
        throw std::runtime_error("transpose_4d_0132: output shape mismatch, expected [" +
            std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
            std::to_string(dim3) + ", " + std::to_string(dim2) + "]");
    }

    transpose_4d_0132_dispatch(input, out, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0132 kernel failed");
}

// Internal helper for reshape_copy kernel dispatch
static void reshape_copy_dispatch(
    const GPUArray& input,
    GPUArray& result,
    size_t total_size
) {
    const int block_size = 256;
    const int grid_size = (total_size + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::copy_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                total_size);
            break;
        case DataType::Float16:
            nn::copy_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                total_size);
            break;
        case DataType::BFloat16:
            nn::copy_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                total_size);
            break;
        default:
            throw std::runtime_error("reshape_copy: unsupported dtype");
    }
}

// Reshape with copy (creates contiguous tensor with new shape)
GPUArray reshape_copy(const GPUArray& input, const std::vector<size_t>& new_shape) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("reshape_copy: only float32/float16/bfloat16 supported");
    }

    // Verify total size matches
    size_t input_size = input.size();
    size_t output_size = 1;
    for (size_t dim : new_shape) {
        output_size *= dim;
    }

    if (input_size != output_size) {
        throw std::runtime_error("reshape_copy: total size mismatch");
    }

    GPUArray result(new_shape, input.dtype());

    reshape_copy_dispatch(input, result, input_size);
    sync_and_check("reshape_copy kernel failed");
    return result;
}

// Reshape with copy into output buffer (for CUDA Graph capture)
void reshape_copy(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("reshape_copy: only float32/float16/bfloat16 supported");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("reshape_copy: dtype mismatch");
    }

    // Verify total size matches
    size_t input_size = input.size();
    size_t output_size = out.size();

    if (input_size != output_size) {
        throw std::runtime_error("reshape_copy: total size mismatch (" +
            std::to_string(input_size) + " vs " + std::to_string(output_size) + ")");
    }

    reshape_copy_dispatch(input, out, input_size);
    sync_and_check("reshape_copy kernel failed");
}

// ============================================================================
// Fixed-Length KV Cache Operations (CUDA Graph Support)
// ============================================================================

void kv_cache_update(
    const GPUArray& new_kv,
    GPUArray& cache,
    int position
) {
    // new_kv: [1, num_kv_heads, head_dim]
    // cache: [max_seq_len, num_kv_heads, head_dim]
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_update: expected 3D tensors");
    }
    if (new_kv.shape()[0] != 1) {
        throw std::runtime_error("kv_cache_update: new_kv should have seq_len=1");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_update: dtype mismatch");
    }
    if (new_kv.shape()[1] != cache.shape()[1] || new_kv.shape()[2] != cache.shape()[2]) {
        throw std::runtime_error("kv_cache_update: shape mismatch (num_kv_heads, head_dim)");
    }

    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int total_elements = num_kv_heads * head_dim;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_update_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_kv_heads, head_dim, position);
            break;
        case DataType::BFloat16:
            nn::kv_cache_update_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_kv_heads, head_dim, position);
            break;
        case DataType::Float32:
            nn::kv_cache_update_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_kv_heads, head_dim, position);
            break;
        default:
            throw std::runtime_error("kv_cache_update: unsupported dtype");
    }

    sync_and_check("kv_cache_update kernel failed");
}

void kv_cache_prefill(
    const GPUArray& new_kv,
    GPUArray& cache,
    int start_pos
) {
    // new_kv: [seq_len, num_kv_heads, head_dim]
    // cache: [max_seq_len, num_kv_heads, head_dim]
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_prefill: expected 3D tensors");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_prefill: dtype mismatch");
    }
    if (new_kv.shape()[1] != cache.shape()[1] || new_kv.shape()[2] != cache.shape()[2]) {
        throw std::runtime_error("kv_cache_prefill: shape mismatch (num_kv_heads, head_dim)");
    }

    int seq_len = static_cast<int>(new_kv.shape()[0]);
    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int total_elements = seq_len * num_kv_heads * head_dim;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_prefill_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_kv_heads, head_dim, start_pos, seq_len);
            break;
        case DataType::BFloat16:
            nn::kv_cache_prefill_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_kv_heads, head_dim, start_pos, seq_len);
            break;
        case DataType::Float32:
            nn::kv_cache_prefill_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_kv_heads, head_dim, start_pos, seq_len);
            break;
        default:
            throw std::runtime_error("kv_cache_prefill: unsupported dtype");
    }

    sync_and_check("kv_cache_prefill kernel failed");
}

// GQA-expanded KV cache update
// new_kv: [1, num_kv_heads, head_dim]
// cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)
void kv_cache_update_gqa(
    const GPUArray& new_kv,
    GPUArray& cache,
    int num_heads,
    int position
) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_update_gqa: expected 3D tensors");
    }
    if (new_kv.shape()[0] != 1) {
        throw std::runtime_error("kv_cache_update_gqa: new_kv should have seq_len=1");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_update_gqa: dtype mismatch");
    }
    if (static_cast<int>(cache.shape()[0]) != num_heads) {
        throw std::runtime_error("kv_cache_update_gqa: cache shape[0] should equal num_heads");
    }

    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int max_seq_len = static_cast<int>(cache.shape()[1]);
    int total_elements = num_heads * head_dim;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_update_gqa_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, position);
            break;
        case DataType::BFloat16:
            nn::kv_cache_update_gqa_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, position);
            break;
        case DataType::Float32:
            nn::kv_cache_update_gqa_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, position);
            break;
        default:
            throw std::runtime_error("kv_cache_update_gqa: unsupported dtype");
    }

    sync_and_check("kv_cache_update_gqa kernel failed");
}

// GQA-expanded KV cache update with GPU position pointer (for CUDA Graph replay)
void kv_cache_update_gqa_ptr(
    const GPUArray& new_kv,
    GPUArray& cache,
    int num_heads,
    const GPUArray& position_buf
) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: expected 3D tensors");
    }
    if (new_kv.shape()[0] != 1) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: new_kv should have seq_len=1");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: dtype mismatch");
    }
    if (static_cast<int>(cache.shape()[0]) != num_heads) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: cache shape[0] should equal num_heads");
    }
    if (position_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: position_buf must be int32");
    }

    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int max_seq_len = static_cast<int>(cache.shape()[1]);
    int total_elements = num_heads * head_dim;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_update_gqa_f16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len,
                static_cast<const int*>(position_buf.data()));
            break;
        case DataType::BFloat16:
            nn::kv_cache_update_gqa_bf16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len,
                static_cast<const int*>(position_buf.data()));
            break;
        case DataType::Float32:
            nn::kv_cache_update_gqa_f32_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len,
                static_cast<const int*>(position_buf.data()));
            break;
        default:
            throw std::runtime_error("kv_cache_update_gqa_ptr: unsupported dtype");
    }

    sync_and_check("kv_cache_update_gqa_ptr kernel failed");
}

// GQA-expanded KV cache prefill
// new_kv: [seq_len, num_kv_heads, head_dim]
// cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)
void kv_cache_prefill_gqa(
    const GPUArray& new_kv,
    GPUArray& cache,
    int num_heads,
    int start_pos
) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_prefill_gqa: expected 3D tensors");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_prefill_gqa: dtype mismatch");
    }
    if (static_cast<int>(cache.shape()[0]) != num_heads) {
        throw std::runtime_error("kv_cache_prefill_gqa: cache shape[0] should equal num_heads");
    }

    int seq_len = static_cast<int>(new_kv.shape()[0]);
    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int max_seq_len = static_cast<int>(cache.shape()[1]);
    int total_elements = seq_len * num_heads * head_dim;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_prefill_gqa_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, start_pos, seq_len);
            break;
        case DataType::BFloat16:
            nn::kv_cache_prefill_gqa_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, start_pos, seq_len);
            break;
        case DataType::Float32:
            nn::kv_cache_prefill_gqa_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, start_pos, seq_len);
            break;
        default:
            throw std::runtime_error("kv_cache_prefill_gqa: unsupported dtype");
    }

    sync_and_check("kv_cache_prefill_gqa kernel failed");
}

// Embedding lookup - copy row from embedding matrix to output buffer
void embedding_lookup(
    const GPUArray& embed_matrix,
    GPUArray& out,
    int token_id
) {
    // embed_matrix: [vocab_size, hidden_size]
    // out: [1, hidden_size] or [hidden_size]
    if (embed_matrix.ndim() != 2) {
        throw std::runtime_error("embedding_lookup: embed_matrix must be 2D");
    }
    if (embed_matrix.dtype() != out.dtype()) {
        throw std::runtime_error("embedding_lookup: dtype mismatch");
    }

    int hidden_size = static_cast<int>(embed_matrix.shape()[1]);

    const int block_size = 256;
    const int grid_size = (hidden_size + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (embed_matrix.dtype()) {
        case DataType::Float16:
            nn::embedding_lookup_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(embed_matrix.data()),
                static_cast<__half*>(out.data()),
                hidden_size, token_id);
            break;
        case DataType::BFloat16:
            nn::embedding_lookup_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embed_matrix.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                hidden_size, token_id);
            break;
        case DataType::Float32:
            nn::embedding_lookup_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(embed_matrix.data()),
                static_cast<float*>(out.data()),
                hidden_size, token_id);
            break;
        default:
            throw std::runtime_error("embedding_lookup: unsupported dtype");
    }

    sync_and_check("embedding_lookup kernel failed");
}

// Embedding lookup with GPU index pointer (for CUDA Graph replay)
void embedding_lookup_ptr(
    const GPUArray& embed_matrix,
    GPUArray& out,
    const GPUArray& token_id_buf
) {
    if (embed_matrix.ndim() != 2) {
        throw std::runtime_error("embedding_lookup_ptr: embed_matrix must be 2D");
    }
    if (embed_matrix.dtype() != out.dtype()) {
        throw std::runtime_error("embedding_lookup_ptr: dtype mismatch");
    }
    if (token_id_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("embedding_lookup_ptr: token_id_buf must be int32");
    }

    int hidden_size = static_cast<int>(embed_matrix.shape()[1]);

    const int block_size = 256;
    const int grid_size = (hidden_size + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (embed_matrix.dtype()) {
        case DataType::Float16:
            nn::embedding_lookup_f16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(embed_matrix.data()),
                static_cast<__half*>(out.data()),
                hidden_size,
                static_cast<const int*>(token_id_buf.data()));
            break;
        case DataType::BFloat16:
            nn::embedding_lookup_bf16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embed_matrix.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                hidden_size,
                static_cast<const int*>(token_id_buf.data()));
            break;
        case DataType::Float32:
            nn::embedding_lookup_f32_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(embed_matrix.data()),
                static_cast<float*>(out.data()),
                hidden_size,
                static_cast<const int*>(token_id_buf.data()));
            break;
        default:
            throw std::runtime_error("embedding_lookup_ptr: unsupported dtype");
    }

    sync_and_check("embedding_lookup_ptr kernel failed");
}

// Batch embedding lookup from GPU token ID array (for batch CUDA Graph)
void embedding_lookup_batch(
    const GPUArray& embed_matrix, GPUArray& out,
    const GPUArray& token_ids_buf, int batch_size
) {
    if (embed_matrix.ndim() != 2) {
        throw std::runtime_error("embedding_lookup_batch: embed_matrix must be 2D");
    }
    if (embed_matrix.dtype() != out.dtype()) {
        throw std::runtime_error("embedding_lookup_batch: dtype mismatch");
    }
    if (token_ids_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("embedding_lookup_batch: token_ids_buf must be int32");
    }

    int hidden_size = static_cast<int>(embed_matrix.shape()[1]);
    int total_elements = batch_size * hidden_size;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (embed_matrix.dtype()) {
        case DataType::Float16:
            nn::embedding_lookup_batch_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(embed_matrix.data()),
                static_cast<__half*>(out.data()),
                static_cast<const int*>(token_ids_buf.data()),
                batch_size, hidden_size);
            break;
        case DataType::BFloat16:
            nn::embedding_lookup_batch_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embed_matrix.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const int*>(token_ids_buf.data()),
                batch_size, hidden_size);
            break;
        case DataType::Float32:
            nn::embedding_lookup_batch_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(embed_matrix.data()),
                static_cast<float*>(out.data()),
                static_cast<const int*>(token_ids_buf.data()),
                batch_size, hidden_size);
            break;
        default:
            throw std::runtime_error("embedding_lookup_batch: unsupported dtype");
    }

    sync_and_check("embedding_lookup_batch kernel failed");
}

// Slice consecutive rows from table using GPU-stored start position
void slice_rows_range_ptr(
    const GPUArray& table,
    GPUArray& out,
    const GPUArray& start_pos_buf,
    int count
) {
    if (table.ndim() != 2) {
        throw std::runtime_error("slice_rows_range_ptr: table must be 2D");
    }
    if (table.dtype() != out.dtype()) {
        throw std::runtime_error("slice_rows_range_ptr: dtype mismatch");
    }
    if (start_pos_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("slice_rows_range_ptr: start_pos_buf must be int32");
    }

    int row_dim = static_cast<int>(table.shape()[1]);
    int total_elements = count * row_dim;

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (table.dtype()) {
        case DataType::Float16:
            nn::slice_rows_range_ptr_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(table.data()),
                static_cast<__half*>(out.data()),
                static_cast<const int*>(start_pos_buf.data()),
                count, row_dim);
            break;
        case DataType::BFloat16:
            nn::slice_rows_range_ptr_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(table.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const int*>(start_pos_buf.data()),
                count, row_dim);
            break;
        case DataType::Float32:
            nn::slice_rows_range_ptr_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(table.data()),
                static_cast<float*>(out.data()),
                static_cast<const int*>(start_pos_buf.data()),
                count, row_dim);
            break;
        default:
            throw std::runtime_error("slice_rows_range_ptr: unsupported dtype");
    }

    sync_and_check("slice_rows_range_ptr kernel failed");
}

// In-place addition: a += b
void add_inplace(GPUArray& a, const GPUArray& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("add_inplace: dtype mismatch");
    }
    size_t n = a.size();
    if (n != b.size()) {
        throw std::runtime_error("add_inplace: size mismatch");
    }

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (a.dtype()) {
        case DataType::Float16:
            nn::add_inplace_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(a.data()),
                static_cast<const __half*>(b.data()), n);
            break;
        case DataType::BFloat16:
            nn::add_inplace_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()), n);
            break;
        case DataType::Float32:
            nn::add_inplace_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(a.data()),
                static_cast<const float*>(b.data()), n);
            break;
        case DataType::Float64:
            nn::add_inplace_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(a.data()),
                static_cast<const double*>(b.data()), n);
            break;
        default:
            throw std::runtime_error("add_inplace: unsupported dtype");
    }

    sync_and_check("add_inplace kernel failed");
}

// In-place multiplication: a *= b
void mul_inplace(GPUArray& a, const GPUArray& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("mul_inplace: dtype mismatch");
    }
    size_t n = a.size();
    if (n != b.size()) {
        throw std::runtime_error("mul_inplace: size mismatch");
    }

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (a.dtype()) {
        case DataType::Float16:
            nn::mul_inplace_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(a.data()),
                static_cast<const __half*>(b.data()), n);
            break;
        case DataType::BFloat16:
            nn::mul_inplace_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()), n);
            break;
        case DataType::Float32:
            nn::mul_inplace_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(a.data()),
                static_cast<const float*>(b.data()), n);
            break;
        case DataType::Float64:
            nn::mul_inplace_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(a.data()),
                static_cast<const double*>(b.data()), n);
            break;
        default:
            throw std::runtime_error("mul_inplace: unsupported dtype");
    }

    sync_and_check("mul_inplace kernel failed");
}

// GPU-to-GPU copy
void copy_to(const GPUArray& src, GPUArray& dst) {
    if (src.dtype() != dst.dtype()) {
        throw std::runtime_error("copy_to: dtype mismatch");
    }
    size_t n = src.size();
    if (n != dst.size()) {
        throw std::runtime_error("copy_to: size mismatch");
    }

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (src.dtype()) {
        case DataType::Float16:
            nn::copy_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(src.data()),
                static_cast<__half*>(dst.data()), n);
            break;
        case DataType::BFloat16:
            nn::copy_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(src.data()),
                static_cast<__nv_bfloat16*>(dst.data()), n);
            break;
        case DataType::Float32:
            nn::copy_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(src.data()),
                static_cast<float*>(dst.data()), n);
            break;
        case DataType::Int32:
            nn::copy_i32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const int*>(src.data()),
                static_cast<int*>(dst.data()), n);
            break;
        default:
            throw std::runtime_error("copy_to: unsupported dtype");
    }

    sync_and_check("copy_to kernel failed");
}

// ============================================================================
// Dtype Cast Operations
// ============================================================================

GPUArray cast_f32_to_bf16(const GPUArray& src) {
    if (src.dtype() != DataType::Float32) {
        throw std::runtime_error("cast_f32_to_bf16: input must be float32");
    }

    GPUArray dst(src.shape(), DataType::BFloat16);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f32_to_bf16_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(src.data()),
        static_cast<__nv_bfloat16*>(dst.data()),
        n);

    sync_and_check("cast_f32_to_bf16 kernel failed");
    return dst;
}

void cast_f32_to_bf16(const GPUArray& src, GPUArray& dst) {
    if (src.dtype() != DataType::Float32) {
        throw std::runtime_error("cast_f32_to_bf16: input must be float32");
    }
    if (dst.dtype() != DataType::BFloat16) {
        throw std::runtime_error("cast_f32_to_bf16: output must be bfloat16");
    }
    if (src.size() != dst.size()) {
        throw std::runtime_error("cast_f32_to_bf16: size mismatch");
    }

    size_t n = src.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f32_to_bf16_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(src.data()),
        static_cast<__nv_bfloat16*>(dst.data()),
        n);

    sync_and_check("cast_f32_to_bf16 kernel failed");
}

GPUArray cast_f32_to_f16(const GPUArray& src) {
    if (src.dtype() != DataType::Float32) {
        throw std::runtime_error("cast_f32_to_f16: input must be float32");
    }

    GPUArray dst(src.shape(), DataType::Float16);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f32_to_f16_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(src.data()),
        static_cast<__half*>(dst.data()),
        n);

    sync_and_check("cast_f32_to_f16 kernel failed");
    return dst;
}

GPUArray cast_bf16_to_f32(const GPUArray& src) {
    if (src.dtype() != DataType::BFloat16) {
        throw std::runtime_error("cast_bf16_to_f32: input must be bfloat16");
    }

    GPUArray dst(src.shape(), DataType::Float32);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_bf16_to_f32_kernel<<<grid_size, block_size>>>(
        static_cast<const __nv_bfloat16*>(src.data()),
        static_cast<float*>(dst.data()),
        n);

    sync_and_check("cast_bf16_to_f32 kernel failed");
    return dst;
}

GPUArray cast_f16_to_f32(const GPUArray& src) {
    if (src.dtype() != DataType::Float16) {
        throw std::runtime_error("cast_f16_to_f32: input must be float16");
    }

    GPUArray dst(src.shape(), DataType::Float32);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f16_to_f32_kernel<<<grid_size, block_size>>>(
        static_cast<const __half*>(src.data()),
        static_cast<float*>(dst.data()),
        n);

    sync_and_check("cast_f16_to_f32 kernel failed");
    return dst;
}

} // namespace ops
} // namespace pygpukit
