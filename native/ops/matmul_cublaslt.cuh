/**
 * cuBLASLt GEMM wrapper for PyGPUkit
 *
 * cuBLASLt is the new lightweight cuBLAS API that provides:
 * - Better performance for small matrices
 * - More flexible algorithm selection
 * - Better integration with CUDA Graphs (potentially)
 */

#pragma once

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace pygpukit {
namespace ops {
namespace cublaslt_gemm {

// Singleton cuBLASLt handle manager
class CublasLtHandle {
public:
    static cublasLtHandle_t get() {
        static CublasLtHandle instance;
        return instance.handle_;
    }

private:
    CublasLtHandle() {
        cublasStatus_t status = cublasLtCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLASLt handle");
        }
    }

    ~CublasLtHandle() {
        if (handle_) {
            cublasLtDestroy(handle_);
        }
    }

    CublasLtHandle(const CublasLtHandle&) = delete;
    CublasLtHandle& operator=(const CublasLtHandle&) = delete;

    cublasLtHandle_t handle_ = nullptr;
};

// FP16 GEMM using cuBLASLt: C = A @ B
// A: [M, K], B: [K, N], C: [M, N] (all row-major)
inline cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cublasLtHandle_t handle = CublasLtHandle::get();

    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasStatus_t status;

    // Create matmul descriptor (for row-major, we swap and use transposed logic)
    // C = A @ B (row-major) == C^T = B^T @ A^T (column-major)
    status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    // Set transpose operations (none for our swapped layout)
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Create matrix layouts (swapped for row-major to column-major conversion)
    // B: [K, N] row-major -> treated as [N, K] col-major (B^T)
    status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, N, K, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    // A: [M, K] row-major -> treated as [K, M] col-major (A^T)
    status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    // C: [M, N] row-major -> treated as [N, M] col-major (C^T)
    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    {
        // Perform matmul
        __half alpha = __float2half(1.0f);
        __half beta = __float2half(0.0f);

        status = cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            B, Bdesc,  // Swapped
            A, Adesc,  // Swapped
            &beta,
            C, Cdesc,
            C, Cdesc,
            nullptr,   // heuristic result (use default)
            nullptr,   // workspace
            0,         // workspace size
            stream
        );
    }

cleanup:
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// FP32 GEMM using cuBLASLt
inline cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cublasLtHandle_t handle = CublasLtHandle::get();

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasStatus_t status;

    status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    {
        float alpha = 1.0f;
        float beta = 0.0f;

        status = cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            B, Bdesc,
            A, Adesc,
            &beta,
            C, Cdesc,
            C, Cdesc,
            nullptr, nullptr, 0, stream
        );
    }

cleanup:
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// BF16 GEMM using cuBLASLt
inline cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cublasLtHandle_t handle = CublasLtHandle::get();

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasStatus_t status;

    status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, N, K, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    {
        float alpha = 1.0f;
        float beta = 0.0f;

        status = cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            B, Bdesc,
            A, Adesc,
            &beta,
            C, Cdesc,
            C, Cdesc,
            nullptr, nullptr, 0, stream
        );
    }

cleanup:
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

}  // namespace cublaslt_gemm
}  // namespace ops
}  // namespace pygpukit
