/**
 * Fused matmul operations (CUTLASS epilogue fusion)
 */
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include "../common/error.cuh"
#include "../ops.cuh"  // For transpose(), gelu(), bias_add_inplace()

#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// CUTLASS BiasGELU fused operations (extern declarations from matmul_cutlass.cu)
extern "C" {
    cudaError_t cutlass_gemm_tf32_bias_gelu(const float* A, const float* B, const float* bias, float* D, int M, int N, int K, cudaStream_t stream);
    cudaError_t cutlass_gemm_fp16_bias_gelu(const __half* A, const __half* B, const __half* bias, __half* D, int M, int N, int K, cudaStream_t stream);
    cudaError_t cutlass_gemm_bf16_bias_gelu(const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* bias, __nv_bfloat16* D, int M, int N, int K, cudaStream_t stream);
    bool cutlass_is_compatible(int M, int N, int K);
    bool cutlass_is_sm_supported();
}

namespace pygpukit {
namespace ops {

// Forward declarations for fallback path
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c);

/**
 * Fused linear + bias + GELU activation.
 *
 * Computes: output = GELU(input @ weight^T + bias)
 *
 * Uses CUTLASS epilogue fusion when available (SM >= 86, dimensions divisible by 16).
 * Falls back to native matmul + bias_add + gelu when CUTLASS is not available.
 *
 * @param input  Input tensor [batch, in_features]
 * @param weight Weight matrix [out_features, in_features]
 * @param bias   Bias vector [out_features]
 * @return Output tensor [batch, out_features]
 */
GPUArray linear_bias_gelu(const GPUArray& input, const GPUArray& weight, const GPUArray& bias) {
    // Validate shapes: input [batch, in_features], weight [out_features, in_features], bias [out_features]
    if (input.ndim() != 2) {
        throw std::runtime_error("linear_bias_gelu: input must be 2D [batch, in_features]");
    }
    if (weight.ndim() != 2) {
        throw std::runtime_error("linear_bias_gelu: weight must be 2D [out_features, in_features]");
    }
    if (bias.ndim() != 1) {
        throw std::runtime_error("linear_bias_gelu: bias must be 1D [out_features]");
    }

    size_t batch = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::runtime_error("linear_bias_gelu: weight.shape[1] must match input.shape[1]");
    }
    if (bias.shape()[0] != out_features) {
        throw std::runtime_error("linear_bias_gelu: bias.shape[0] must match weight.shape[0]");
    }

    // Validate dtypes
    if (input.dtype() != weight.dtype() || input.dtype() != bias.dtype()) {
        throw std::runtime_error("linear_bias_gelu: all inputs must have the same dtype");
    }

    // Check if CUTLASS fused kernel can be used
    // Requirements: dimensions must be multiples of 16 AND SM >= 86
    bool use_cutlass = cutlass_is_compatible(batch, out_features, in_features) && cutlass_is_sm_supported();

    // Also check if CUTLASS is disabled via environment variable
    const char* no_cutlass_env = std::getenv("PYGPUKIT_NO_CUTLASS");
    if (no_cutlass_env && (no_cutlass_env[0] == '1' || no_cutlass_env[0] == 'y' || no_cutlass_env[0] == 'Y')) {
        use_cutlass = false;
    }

    // Transpose weight for both paths (needed for input @ weight^T)
    GPUArray weight_T = transpose(weight);  // [in_features, out_features]

    // Allocate output
    GPUArray output({batch, out_features}, input.dtype());

    if (use_cutlass) {
        // CUTLASS fused BiasGELU kernel path
        cudaError_t err = cudaSuccess;
        cudaStream_t stream = internal::get_capture_stream();

        switch (input.dtype()) {
            case DataType::Float32:
                err = cutlass_gemm_tf32_bias_gelu(
                    static_cast<const float*>(input.data()),
                    static_cast<const float*>(weight_T.data()),
                    static_cast<const float*>(bias.data()),
                    static_cast<float*>(output.data()),
                    batch, out_features, in_features, stream);
                break;
            case DataType::Float16:
                err = cutlass_gemm_fp16_bias_gelu(
                    static_cast<const __half*>(input.data()),
                    static_cast<const __half*>(weight_T.data()),
                    static_cast<const __half*>(bias.data()),
                    static_cast<__half*>(output.data()),
                    batch, out_features, in_features, stream);
                break;
            case DataType::BFloat16:
                err = cutlass_gemm_bf16_bias_gelu(
                    static_cast<const __nv_bfloat16*>(input.data()),
                    static_cast<const __nv_bfloat16*>(weight_T.data()),
                    static_cast<const __nv_bfloat16*>(bias.data()),
                    static_cast<__nv_bfloat16*>(output.data()),
                    batch, out_features, in_features, stream);
                break;
            default:
                throw std::runtime_error("linear_bias_gelu only supports float32, float16, and bfloat16");
        }

        // If CUTLASS fails (e.g., not compiled in), fall back to native path
        if (err == cudaSuccess) {
            sync_and_check("linear_bias_gelu CUTLASS kernel failed");
            return output;
        }
        // Fall through to native path if CUTLASS returns error
    }

    // Native fallback path: matmul + bias_add_inplace + gelu
    // This works for any dimensions and when CUTLASS is not available
    matmul(input, weight_T, output);
    bias_add_inplace(output, bias);
    output = gelu(output);

    return output;
}

} // namespace ops
} // namespace pygpukit
