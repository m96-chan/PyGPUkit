/**
 * cuBLAS GEMM wrapper for PyGPUkit
 *
 * Uses cuBLAS for efficient matmul, especially for small batch sizes (M=1).
 * cuBLAS is column-major, so we use the identity:
 *   C = A @ B (row-major) == C^T = B^T @ A^T (column-major)
 *
 * This means we call cuBLAS with swapped arguments:
 *   cublas*gemm(N, M, K, B, A, C) instead of (M, N, K, A, B, C)
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <mutex>

namespace pygpukit {
namespace ops {
namespace cublas_gemm {

// Singleton cuBLAS handle manager
class CublasHandle {
public:
    static cublasHandle_t get() {
        static CublasHandle instance;
        return instance.handle_;
    }

private:
    CublasHandle() {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }

    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t handle_ = nullptr;
};

// FP16 GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N] (all row-major)
inline cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cublasHandle_t handle = CublasHandle::get();

    if (stream) {
        cublasSetStream(handle, stream);
    }

    // cuBLAS uses column-major, so we compute C^T = B^T @ A^T
    // This is equivalent to swapping A<->B and M<->N
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasStatus_t status = cublasHgemm(
        handle,
        CUBLAS_OP_N,  // B is not transposed (as B^T in col-major = B in row-major)
        CUBLAS_OP_N,  // A is not transposed (as A^T in col-major = A in row-major)
        N,            // Number of rows of C^T (= cols of C)
        M,            // Number of cols of C^T (= rows of C)
        K,            // Inner dimension
        &alpha,
        B, N,         // B: [K, N] row-major, ldb = N
        A, K,         // A: [M, K] row-major, lda = K
        &beta,
        C, N          // C: [M, N] row-major, ldc = N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// FP32 GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N] (all row-major)
inline cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cublasHandle_t handle = CublasHandle::get();

    if (stream) {
        cublasSetStream(handle, stream);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// BF16 GEMM using cuBLAS GemmEx (requires compute capability >= 8.0)
inline cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cublasHandle_t handle = CublasHandle::get();

    if (stream) {
        cublasSetStream(handle, stream);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Use GemmEx for BF16
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

}  // namespace cublas_gemm
}  // namespace ops
}  // namespace pygpukit
