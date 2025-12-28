/**
 * cuBLASLt GEMM wrapper for PyGPUkit
 *
 * This header provides GEMM functions using dynamically-loaded cuBLASLt.
 * No CUDA Toolkit required at runtime - only the GPU driver.
 *
 * cuBLASLt provides:
 * - Better performance for small matrices
 * - More flexible algorithm selection
 * - Better integration with CUDA Graphs
 */

#pragma once

#include "../../jit/cublaslt_loader.hpp"

namespace pygpukit {
namespace ops {
namespace cublaslt_gemm {

// Re-export convenience functions from dynamic loader

// FP16 GEMM using cuBLASLt: C = A @ B
// A: [M, K], B: [K, N], C: [M, N] (all row-major)
inline cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    return cublaslt::gemm_fp16(A, B, C, M, N, K, stream);
}

// FP32 GEMM using cuBLASLt
inline cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    return cublaslt::gemm_fp32(A, B, C, M, N, K, stream);
}

// BF16 GEMM using cuBLASLt
inline cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    return cublaslt::gemm_bf16(A, B, C, M, N, K, stream);
}

// Check if cuBLASLt is available
inline bool is_available() {
    return cublaslt::is_available();
}

}  // namespace cublaslt_gemm
}  // namespace ops
}  // namespace pygpukit
