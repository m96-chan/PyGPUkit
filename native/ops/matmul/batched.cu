/**
 * Batched matrix multiplication operations
 *
 * Currently a placeholder - batched GEMM requires CUTLASS implementation.
 */
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include "../common/error.cuh"

#include <stdexcept>

namespace pygpukit {
namespace ops {

/**
 * Batched strided matrix multiplication (FP32).
 *
 * Computes C[i] = A[i] @ B[i] for i in 0..batch_count-1.
 * Each matrix is accessed via strided offsets from the base pointer.
 *
 * @param A Input matrix A, shape [batch_count * strideA]
 * @param B Input matrix B, shape [batch_count * strideB]
 * @param C Output matrix C, shape [batch_count * strideC]
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

    // TODO: Implement batched GEMM with CUTLASS or cuBLASLt
    // For now, this is a placeholder that throws
    (void)M; (void)N; (void)K;
    (void)batch_count;
    (void)strideA; (void)strideB; (void)strideC;
    throw std::runtime_error("batched_matmul_fp32: not yet implemented");
}

} // namespace ops
} // namespace pygpukit
