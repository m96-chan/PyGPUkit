/**
 * FP16/BF16 Matrix Multiplication
 *
 * Uses FP32 accumulation for numerical stability
 * Supports:
 * - FP16 input -> FP16 output (FP32 accumulation)
 * - BF16 input -> BF16 output (FP32 accumulation)
 *
 * Note: WMMA/TensorCore optimization can be added later.
 * Current implementation uses simple kernels for correctness.
 */

#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {
namespace fp16_bf16_matmul {

// Simple FP16 GEMM using FP32 accumulation
// C = A @ B where A is (M, K), B is (K, N), C is (M, N)
__global__ void sgemm_f16_simple_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

// Simple BF16 GEMM using FP32 accumulation
// C = A @ B where A is (M, K), B is (K, N), C is (M, N)
__global__ void sgemm_bf16_simple_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
        }
        C[row * N + col] = __float2bfloat16_rn(sum);
    }
}

// Launch FP16 matmul
inline cudaError_t launch_sgemm_f16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    sgemm_f16_simple_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

// Launch BF16 matmul
inline cudaError_t launch_sgemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    sgemm_bf16_simple_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace fp16_bf16_matmul
} // namespace ops
} // namespace pygpukit
