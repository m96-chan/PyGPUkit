/**
 * Batched matrix multiplication operations
 *
 * Uses cuBLAS sgemm_strided_batched for high-performance batched GEMM.
 * Falls back to loop-based GPU matmul if cuBLAS is unavailable.
 */
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include "../common/error.cuh"
#include "../../jit/cublas_loader.hpp"

#include <stdexcept>
#include <cstdio>

namespace pygpukit {
namespace ops {

/**
 * Batched strided matrix multiplication (FP32).
 *
 * Computes C[i] = A[i] @ B[i] for i in 0..batch_count-1.
 * Each matrix is accessed via strided offsets from the base pointer.
 *
 * Row-major to column-major conversion:
 * - cuBLAS is column-major, our tensors are row-major
 * - For row-major: C = A @ B
 * - We compute: C^T = B^T @ A^T (which gives us C in row-major)
 *
 * @param A Input matrix A, shape [batch_count, M, K] (row-major)
 * @param B Input matrix B, shape [batch_count, K, N] (row-major)
 * @param C Output matrix C, shape [batch_count, M, N] (row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 * @param batch_count Number of batches
 * @param strideA Stride between A matrices (in elements)
 * @param strideB Stride between B matrices (in elements)
 * @param strideC Stride between C matrices (in elements)
 */
void batched_matmul_fp32(const GPUArray& A, const GPUArray& B, GPUArray& C,
                         int M, int N, int K, int batch_count,
                         int64_t strideA, int64_t strideB, int64_t strideC) {
    // Validate inputs
    if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || C.dtype() != DataType::Float32) {
        throw std::runtime_error("batched_matmul_fp32: all inputs must be float32");
    }

    // Get cuBLAS handle
    if (!cublas::is_available()) {
        throw std::runtime_error("batched_matmul_fp32: cuBLAS not available");
    }

    cublas::cublasHandle_t handle = cublas::get_handle();
    if (!handle) {
        throw std::runtime_error("batched_matmul_fp32: failed to get cuBLAS handle");
    }

    // Set stream for cuBLAS operations
    cudaStream_t stream = internal::get_capture_stream();
    cublas::cublasStatus_t set_status = cublas::set_stream(handle, stream);
    if (set_status != cublas::CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("batched_matmul_fp32: failed to set cuBLAS stream");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Row-major to column-major conversion:
    // For row-major C[M,N] = A[M,K] @ B[K,N]
    // We compute: C^T[N,M] = B^T[N,K] @ A^T[K,M]
    // cuBLAS: C = op(A) @ op(B)
    // With CUBLAS_OP_N (no transpose), cuBLAS interprets row-major as column-major transpose
    // So: C[N,M] = B[N,K] @ A[K,M] (treating row-major as column-major)
    // Result is C^T in column-major = C in row-major

    const float* A_ptr = static_cast<const float*>(A.data());
    const float* B_ptr = static_cast<const float*>(B.data());
    float* C_ptr = static_cast<float*>(C.data());

    // cuBLAS sgemm_strided_batched expects:
    // - m, n, k: dimensions of the output matrix (m rows, n cols)
    // - For C = A @ B in row-major, we call with swapped A/B and transposed dims
    //   C^T[N,M] = B^T[N,K] @ A^T[K,M]
    //   So cuBLAS m=N, n=M, k=K, with B as first matrix, A as second

    cublas::cublasStatus_t status = cublas::sgemm_strided_batched(
        handle,
        cublas::CUBLAS_OP_N,  // op on B (no transpose - B^T is already what we want)
        cublas::CUBLAS_OP_N,  // op on A (no transpose - A^T is already what we want)
        N,            // m = number of rows of C (in column-major) = N
        M,            // n = number of cols of C (in column-major) = M
        K,            // k = inner dimension
        &alpha,
        B_ptr,        // B comes first (we're computing B^T @ A^T)
        N,            // ldb = leading dimension of B = N (row-major K x N means N stride)
        strideB,
        A_ptr,        // A comes second
        K,            // lda = leading dimension of A = K (row-major M x K means K stride)
        strideA,
        &beta,
        C_ptr,
        N,            // ldc = leading dimension of C = N (row-major M x N means N stride)
        strideC,
        batch_count
    );

    if (status != cublas::CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[batched_matmul_fp32] cuBLAS sgemm_strided_batched failed: %d\n",
                static_cast<int>(status));
        throw std::runtime_error("batched_matmul_fp32: cuBLAS sgemm_strided_batched failed");
    }

    sync_and_check("batched_matmul_fp32 kernel failed");
}

} // namespace ops
} // namespace pygpukit
