/**
 * Neural Network operations dispatch
 */
#include "nn_kernels.cuh"
#include "flash_attention.cuh"
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

    switch (q.dtype()) {
        case DataType::Float32:
            nn::rope_f32_kernel<<<grid_size, block_size>>>(
                static_cast<float*>(q.data()),
                static_cast<float*>(k.data()),
                static_cast<const float*>(cos.data()),
                static_cast<const float*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        case DataType::Float16:
            nn::rope_f16_kernel<<<grid_size, block_size>>>(
                static_cast<__half*>(q.data()),
                static_cast<__half*>(k.data()),
                static_cast<const __half*>(cos.data()),
                static_cast<const __half*>(sin.data()),
                seq_len, n_heads_q, n_heads_k, head_dim);
            break;
        case DataType::BFloat16:
            nn::rope_bf16_kernel<<<grid_size, block_size>>>(
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
// Scaled Dot-Product Attention (SDPA) with Causal Mask
// ============================================================================

// Check if Flash Attention is enabled via environment variable
static bool is_flash_attention_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("PYGPUKIT_FLASH_ATTENTION");
        cached = (env != nullptr && (std::string(env) == "1" || std::string(env) == "true"));
    }
    return cached != 0;
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
    int kv_len = (context_len > 0) ? context_len : static_cast<int>(K.shape()[1]);

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

    // Use Flash Attention if enabled and head_dim is reasonable
    bool use_flash = is_flash_attention_enabled() && head_dim <= 128;

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
                    n_heads, q_len, kv_len, head_dim, scale, causal_offset);
                break;
            case DataType::Float16:
                nn::flash_attention_f16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    n_heads, q_len, kv_len, head_dim, scale, causal_offset);
                break;
            case DataType::BFloat16:
                nn::flash_attention_bf16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    n_heads, q_len, kv_len, head_dim, scale, causal_offset);
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
                    n_heads, q_len, kv_len, head_dim, scale, causal_offset);
                break;
            case DataType::Float16:
                nn::sdpa_causal_f16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    n_heads, q_len, kv_len, head_dim, scale, causal_offset);
                break;
            case DataType::BFloat16:
                nn::sdpa_causal_bf16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    n_heads, q_len, kv_len, head_dim, scale, causal_offset);
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
        a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("concat: only float32/float16/bfloat16 supported");
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

    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_021_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                dim0, dim1, dim2);
            break;
        case DataType::Float16:
            nn::transpose_021_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                dim0, dim1, dim2);
            break;
        case DataType::BFloat16:
            nn::transpose_021_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                dim0, dim1, dim2);
            break;
        default:
            break;
    }

    sync_and_check("transpose_3d_021 kernel failed");
    return result;
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

    const int block_size = 256;
    const int grid_size = (input_size + block_size - 1) / block_size;

    switch (input.dtype()) {
        case DataType::Float32:
            nn::copy_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                input_size);
            break;
        case DataType::Float16:
            nn::copy_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                input_size);
            break;
        case DataType::BFloat16:
            nn::copy_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                input_size);
            break;
        default:
            break;
    }

    sync_and_check("reshape_copy kernel failed");
    return result;
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

} // namespace ops
} // namespace pygpukit
