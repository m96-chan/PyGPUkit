/**
 * Ampere-Optimized FP32 GEMM Kernel for RTX 3090 Ti
 *
 * Target: 22-30 TFLOPS (62-85% of 35.6 TFLOPS theoretical)
 *
 * Key optimizations based on CUTLASS/cuBLAS patterns:
 * - cp.async with 4-stage software pipeline
 * - BK=16 with 4 stages for proper latency hiding
 * - Single __syncthreads() per K iteration
 * - Warp-contiguous memory access patterns
 * - 128-byte cache line aligned loads
 * - Proper wait_group(STAGES-2) placement AFTER load issue
 *
 * Architecture: SM 8.6 (Ampere, RTX 3090 Ti)
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../../../../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {
namespace ampere {

// ============================================================================
// Configuration Constants - Tuned for RTX 3090 Ti
// ============================================================================

// CTA tile dimensions - ROW-MAJOR A with float4 cp.async
constexpr int BM = 128;           // Tile rows per block
constexpr int BN = 128;           // Tile cols per block
constexpr int BK = 16;            // Tile depth - 16 for good balance

// Thread tile dimensions
constexpr int TM = 8;             // Rows per thread
constexpr int TN = 8;             // Cols per thread

// Block dimensions: (BN/TN, BM/TM) = (16, 16) = 256 threads
constexpr int BLOCK_DIM_X = BN / TN;  // 16
constexpr int BLOCK_DIM_Y = BM / TM;  // 16
constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;  // 256

// Pipeline stages - 4 stages for latency hiding
// wait_group(STAGES-2) = wait_group(2) allows 2 groups in flight
constexpr int STAGES = 4;

// ============================================================================
// Shared memory layout for ROW-MAJOR A storage (BK=16)
// ============================================================================
// A is stored ROW-MAJOR: Am[stage][m][k] where:
//   - m = 0..127 (BM rows)
//   - k = 0..15 (BK columns)
//   - Stride = BK + PAD for each row
//
// B is stored ROW-MAJOR: Bs[stage][k][n] where:
//   - k = 0..15 (BK rows)
//   - n = 0..127 (BN columns)
//   - Stride = BN + PAD for each row

constexpr int SMEM_PAD_A = 4;     // stride=20 for row-major A (BK=16)
constexpr int SMEM_PAD_B = 8;     // stride=136 for B

// Shared memory strides
constexpr int A_SMEM_STRIDE = BK + SMEM_PAD_A;    // 20 (row-major A: m rows, k cols)
constexpr int B_SMEM_STRIDE = BN + SMEM_PAD_B;    // 136

// Shared memory sizes per stage
// A: BM rows x stride = 128 x 20 = 2560 floats per stage
// B: BK rows x stride = 16 x 136 = 2176 floats per stage
constexpr int A_STAGE_SIZE = BM * A_SMEM_STRIDE;  // 128 * 20 = 2560 floats
constexpr int B_STAGE_SIZE = BK * B_SMEM_STRIDE;  // 16 * 136 = 2176 floats

// Total shared memory: 4 stages * (2560 + 2176) * 4 bytes = 75,776 bytes = 74 KB
// Fits within RTX 3090 Ti's 100KB limit!

// Configuration for smaller BK (4-stage variant)
constexpr int BK_SMALL = 16;
constexpr int STAGES_4 = 4;
constexpr int A_STAGE_SIZE_SMALL = BK_SMALL * A_SMEM_STRIDE;  // 16 * 136 = 2176
constexpr int B_STAGE_SIZE_SMALL = BK_SMALL * B_SMEM_STRIDE;  // 16 * 136 = 2176

// ============================================================================
// Helper Functions for cp.async
// ============================================================================

// Convert generic pointer to shared memory address for PTX
__device__ __forceinline__ unsigned int cvta_to_shared(const void* ptr) {
    unsigned int smem_addr;
    asm volatile(
        "{ .reg .u64 smem_ptr64;\n"
        "  cvta.to.shared.u64 smem_ptr64, %1;\n"
        "  cvt.u32.u64 %0, smem_ptr64; }\n"
        : "=r"(smem_addr) : "l"(ptr)
    );
    return smem_addr;
}

// cp.async 4-byte copy (single float) - cache at all levels (.ca)
// Note: .cg only supports 16 bytes, .ca supports 4, 8, 16 bytes
__device__ __forceinline__ void cp_async_cg_4(void* dst, const void* src) {
    unsigned int dst_smem = cvta_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(dst_smem), "l"(src)
    );
}

// cp.async 16-byte copy (float4) - cache global (.cg) for better throughput
__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    unsigned int dst_smem = cvta_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_smem), "l"(src)
    );
}

// Commit current async copy group
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait for async copy groups - N = max groups still in flight
__device__ __forceinline__ void cp_async_wait_group(int N) {
    // Note: N must be a compile-time constant in real usage
    // Using template specialization for common cases
    if (N == 0) {
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else if (N == 1) {
        asm volatile("cp.async.wait_group 1;\n" ::);
    } else if (N == 2) {
        asm volatile("cp.async.wait_group 2;\n" ::);
    } else if (N == 3) {
        asm volatile("cp.async.wait_group 3;\n" ::);
    }
}

// ============================================================================
// Launch Function Declaration
// ============================================================================

cudaError_t launch_sgemm_ampere(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

}  // namespace ampere
}  // namespace ops
}  // namespace pygpukit
