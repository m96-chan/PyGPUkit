/**
 * CUTLASS GEMM instantiation for PyGPUkit
 *
 * This file instantiates CUTLASS templates for SM 86.
 * Separated from main matmul.cu to isolate template compilation.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if PYGPUKIT_HAS_CUTLASS

#include "gemm/bf16/bf16/sm80/bf16_cutlass.cuh"

namespace pygpukit {
namespace ops {

// ============================================================================
// Explicit C-linkage wrappers for CUTLASS GEMM
// These can be called from the main matmul dispatch
// ============================================================================

extern "C" {

cudaError_t cutlass_gemm_tf32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cutlass_gemm::gemm_tf32(A, B, C, M, N, K, 1.0f, 0.0f, stream);
}

cudaError_t cutlass_gemm_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cutlass_gemm::gemm_fp16(A, B, C, M, N, K, 1.0f, 0.0f, stream);
}

cudaError_t cutlass_gemm_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cutlass_gemm::gemm_bf16(A, B, C, M, N, K, 1.0f, 0.0f, stream);
}

bool cutlass_is_compatible(int M, int N, int K) {
    return cutlass_gemm::is_cutlass_compatible(M, N, K);
}

bool cutlass_is_sm_supported() {
    return cutlass_gemm::is_sm_supported();
}

// ============================================================================
// BiasGELU fused operations
// ============================================================================

cudaError_t cutlass_gemm_tf32_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cutlass_gemm::gemm_tf32_bias_gelu(A, B, bias, D, M, N, K, stream);
}

cudaError_t cutlass_gemm_fp16_bias_gelu(
    const __half* A,
    const __half* B,
    const __half* bias,
    __half* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cutlass_gemm::gemm_fp16_bias_gelu(A, B, bias, D, M, N, K, stream);
}

cudaError_t cutlass_gemm_bf16_bias_gelu(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    const __nv_bfloat16* bias,
    __nv_bfloat16* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cutlass_gemm::gemm_bf16_bias_gelu(A, B, bias, D, M, N, K, stream);
}

}  // extern "C"

}  // namespace ops
}  // namespace pygpukit

#else  // !PYGPUKIT_HAS_CUTLASS

// Stub implementations when CUTLASS is not available
extern "C" {

cudaError_t cutlass_gemm_tf32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t cutlass_gemm_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t cutlass_gemm_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

bool cutlass_is_compatible(int M, int N, int K) {
    return false;
}

bool cutlass_is_sm_supported() {
    return false;
}

cudaError_t cutlass_gemm_tf32_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t cutlass_gemm_fp16_bias_gelu(
    const __half* A,
    const __half* B,
    const __half* bias,
    __half* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t cutlass_gemm_bf16_bias_gelu(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    const __nv_bfloat16* bias,
    __nv_bfloat16* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

}  // extern "C"

#endif  // PYGPUKIT_HAS_CUTLASS
