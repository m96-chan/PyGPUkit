/**
 * CUTLASS-inspired GEMV Kernel for M=1 (LLM Decode Path)
 *
 * Purpose: Replace cuBLASLt GEMV with CUTLASS-based implementation
 *
 * Design decisions:
 * 1. M=1 is memory-bound, not compute-bound
 * 2. TensorCore is inefficient for M=1 (MMA tiles are wasted)
 * 3. Scalar FMA with vectorized loads is optimal
 * 4. A[1,K] is small, broadcasts via L1/L2 cache
 * 5. B[K,N] row-major: adjacent threads read adjacent addresses (coalesced)
 *
 * Target architectures:
 * - SM86 (RTX 30xx): Primary target
 * - SM89 (RTX 40xx): Supported
 * - SM90 (H100): Supported
 * - SM120 (RTX 5090): BF16 fallback
 *
 * Future extensions:
 * - Batched GEMV for continuous batching
 * - FP8 for SM90/SM120 when available
 * - Fused bias/scale epilogue
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Configuration - Per-size tuning
// ============================================================================

// Default configuration (medium sizes: K=2048-8192, N=1024-8192)
struct GemvConfig {
    static constexpr int BLOCK_SIZE = 256;  // 8 warps
    static constexpr int TILE_N = 256;
    static constexpr int UNROLL_K = 8;
    static constexpr int MIN_N = 128;
};

// Small K configuration (K < 2048)
// - Smaller unroll to reduce register pressure
// - Good for embedding lookups, small hidden sizes
struct GemvConfigSmallK {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int TILE_N = 256;
    static constexpr int UNROLL_K = 4;      // Less unrolling for small K
    static constexpr int MIN_N = 128;
};

// Large K configuration (K > 8192)
// - Larger unroll for more ILP
// - Trades registers for throughput
struct GemvConfigLargeK {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int TILE_N = 256;
    static constexpr int UNROLL_K = 16;     // More unrolling for large K
    static constexpr int MIN_N = 128;
};

// Small N configuration (N < 1024)
// - Smaller tile to avoid wasted threads
// - Better for narrow outputs
struct GemvConfigSmallN {
    static constexpr int BLOCK_SIZE = 128;  // 4 warps
    static constexpr int TILE_N = 128;
    static constexpr int UNROLL_K = 8;
    static constexpr int MIN_N = 64;
};

// Large matrices (K > 8192 AND N > 8192)
// - Maximum unrolling
// - Optimized for LLM MLP layers (8192x28672 etc)
struct GemvConfigLarge {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int TILE_N = 256;
    static constexpr int UNROLL_K = 16;
    static constexpr int MIN_N = 128;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Convert BF16 to FP32 with cache hint
__device__ __forceinline__ float ldg_bf16_to_f32(const __nv_bfloat16* ptr) {
    return __bfloat162float(__ldg(ptr));
}

// Convert FP16 to FP32 with cache hint
__device__ __forceinline__ float ldg_fp16_to_f32(const __half* ptr) {
    return __half2float(__ldg(ptr));
}

// Vectorized load: Load 2 BF16 values as bfloat162
__device__ __forceinline__ __nv_bfloat162 ldg_bf16x2(const __nv_bfloat16* ptr) {
    return __ldg(reinterpret_cast<const __nv_bfloat162*>(ptr));
}

// Vectorized load: Load 4 BF16 values as 2x bfloat162
__device__ __forceinline__ void ldg_bf16x4(const __nv_bfloat16* ptr,
                                            __nv_bfloat162& v01, __nv_bfloat162& v23) {
    const __nv_bfloat162* ptr2 = reinterpret_cast<const __nv_bfloat162*>(ptr);
    v01 = __ldg(ptr2);
    v23 = __ldg(ptr2 + 1);
}

// ============================================================================
// BF16 GEMV Kernel
// ============================================================================

/**
 * GEMV kernel for BF16: C[1,N] = alpha * A[1,K] @ B[K,N] + beta * C[1,N]
 *
 * Memory layout (all row-major):
 * - A: [1, K] contiguous, small, broadcasts well
 * - B: [K, N] row-major, B[k,n] at address k*N+n
 * - C: [1, N] contiguous output
 *
 * Thread mapping:
 * - Each thread handles one output element C[global_n]
 * - All threads in block iterate over K together
 * - Coalesced access: threads 0-255 read B[k, block_start:block_start+256]
 *
 * Optimization techniques:
 * 1. __ldg() for read-only cache (B access)
 * 2. A broadcast via L1/L2 (all threads read same A[k])
 * 3. FMA accumulation in FP32 for precision
 * 4. K-loop unrolling (UNROLL_K=8) for ILP
 * 5. Predicated loads for K remainder handling
 * 6. Vectorized BF16x2 loads for A (reduces memory transactions)
 */
template<typename Config = GemvConfig>
__global__ void gemv_bf16_kernel(
    __nv_bfloat16 const* __restrict__ A,  // [1, K]
    __nv_bfloat16 const* __restrict__ B,  // [K, N]
    __nv_bfloat16* __restrict__ C,        // [1, N]
    int K,
    int N,
    float alpha,
    float beta
) {
    // Thread/block indexing
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    // Bounds check for partial blocks at the end
    if (global_n >= N) return;

    // Accumulator in FP32 for numerical precision
    // cuBLASLt also uses FP32 accumulation for BF16
    float acc = 0.0f;

    // Base pointer for this thread's column of B
    // B[k, global_n] = B[k * N + global_n]
    const __nv_bfloat16* B_col = B + global_n;

    // Main K loop with UNROLL_K unrolling
    // Rationale: Hides memory latency, increases ILP
    int k = 0;
    constexpr int UNROLL = Config::UNROLL_K;

    // Template-based unrolling: UNROLL_K can be 4, 8, or 16
    for (; k + UNROLL <= K; k += UNROLL) {
        // UNROLL_K=4: Load 2 bfloat162 (4 values)
        // UNROLL_K=8: Load 4 bfloat162 (8 values)
        // UNROLL_K=16: Load 8 bfloat162 (16 values)

        if constexpr (UNROLL == 4) {
            __nv_bfloat162 a01 = ldg_bf16x2(A + k + 0);
            __nv_bfloat162 a23 = ldg_bf16x2(A + k + 2);
            float a0 = __low2float(a01);
            float a1 = __high2float(a01);
            float a2 = __low2float(a23);
            float a3 = __high2float(a23);
            float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
            float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
            float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
            float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
        } else if constexpr (UNROLL == 8) {
            __nv_bfloat162 a01 = ldg_bf16x2(A + k + 0);
            __nv_bfloat162 a23 = ldg_bf16x2(A + k + 2);
            __nv_bfloat162 a45 = ldg_bf16x2(A + k + 4);
            __nv_bfloat162 a67 = ldg_bf16x2(A + k + 6);
            float a0 = __low2float(a01);
            float a1 = __high2float(a01);
            float a2 = __low2float(a23);
            float a3 = __high2float(a23);
            float a4 = __low2float(a45);
            float a5 = __high2float(a45);
            float a6 = __low2float(a67);
            float a7 = __high2float(a67);
            float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
            float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
            float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
            float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
            float b4 = ldg_bf16_to_f32(B_col + (k + 4) * N);
            float b5 = ldg_bf16_to_f32(B_col + (k + 5) * N);
            float b6 = ldg_bf16_to_f32(B_col + (k + 6) * N);
            float b7 = ldg_bf16_to_f32(B_col + (k + 7) * N);
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
            acc = fmaf(a4, b4, acc);
            acc = fmaf(a5, b5, acc);
            acc = fmaf(a6, b6, acc);
            acc = fmaf(a7, b7, acc);
        } else if constexpr (UNROLL == 16) {
            __nv_bfloat162 a01 = ldg_bf16x2(A + k + 0);
            __nv_bfloat162 a23 = ldg_bf16x2(A + k + 2);
            __nv_bfloat162 a45 = ldg_bf16x2(A + k + 4);
            __nv_bfloat162 a67 = ldg_bf16x2(A + k + 6);
            __nv_bfloat162 a89 = ldg_bf16x2(A + k + 8);
            __nv_bfloat162 aAB = ldg_bf16x2(A + k + 10);
            __nv_bfloat162 aCD = ldg_bf16x2(A + k + 12);
            __nv_bfloat162 aEF = ldg_bf16x2(A + k + 14);
            float a0 = __low2float(a01);
            float a1 = __high2float(a01);
            float a2 = __low2float(a23);
            float a3 = __high2float(a23);
            float a4 = __low2float(a45);
            float a5 = __high2float(a45);
            float a6 = __low2float(a67);
            float a7 = __high2float(a67);
            float a8 = __low2float(a89);
            float a9 = __high2float(a89);
            float aA = __low2float(aAB);
            float aB = __high2float(aAB);
            float aC = __low2float(aCD);
            float aD = __high2float(aCD);
            float aE = __low2float(aEF);
            float aF = __high2float(aEF);
            float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
            float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
            float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
            float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
            float b4 = ldg_bf16_to_f32(B_col + (k + 4) * N);
            float b5 = ldg_bf16_to_f32(B_col + (k + 5) * N);
            float b6 = ldg_bf16_to_f32(B_col + (k + 6) * N);
            float b7 = ldg_bf16_to_f32(B_col + (k + 7) * N);
            float b8 = ldg_bf16_to_f32(B_col + (k + 8) * N);
            float b9 = ldg_bf16_to_f32(B_col + (k + 9) * N);
            float bA = ldg_bf16_to_f32(B_col + (k + 10) * N);
            float bB = ldg_bf16_to_f32(B_col + (k + 11) * N);
            float bC = ldg_bf16_to_f32(B_col + (k + 12) * N);
            float bD = ldg_bf16_to_f32(B_col + (k + 13) * N);
            float bE = ldg_bf16_to_f32(B_col + (k + 14) * N);
            float bF = ldg_bf16_to_f32(B_col + (k + 15) * N);
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
            acc = fmaf(a4, b4, acc);
            acc = fmaf(a5, b5, acc);
            acc = fmaf(a6, b6, acc);
            acc = fmaf(a7, b7, acc);
            acc = fmaf(a8, b8, acc);
            acc = fmaf(a9, b9, acc);
            acc = fmaf(aA, bA, acc);
            acc = fmaf(aB, bB, acc);
            acc = fmaf(aC, bC, acc);
            acc = fmaf(aD, bD, acc);
            acc = fmaf(aE, bE, acc);
            acc = fmaf(aF, bF, acc);
        }
    }

    // Handle K remainder (when K is not divisible by UNROLL_K)
    for (; k < K; ++k) {
        float a = __bfloat162float(A[k]);
        float b = ldg_bf16_to_f32(B_col + k * N);
        acc = fmaf(a, b, acc);
    }

    // Epilogue: Apply alpha/beta scaling
    // Matches cuBLASLt behavior: D = alpha * A @ B + beta * C
    if (beta != 0.0f) {
        float c_old = __bfloat162float(C[global_n]);
        acc = fmaf(alpha, acc, beta * c_old);
    } else {
        acc *= alpha;
    }

    // Store result
    C[global_n] = __float2bfloat16(acc);
}

// ============================================================================
// FP16 GEMV Kernel
// ============================================================================

/**
 * GEMV kernel for FP16: C[1,N] = alpha * A[1,K] @ B[K,N] + beta * C[1,N]
 * Same design as BF16, using FP16 intrinsics
 */
template<typename Config = GemvConfig>
__global__ void gemv_fp16_kernel(
    __half const* __restrict__ A,
    __half const* __restrict__ B,
    __half* __restrict__ C,
    int K,
    int N,
    float alpha,
    float beta
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    float acc = 0.0f;
    const __half* B_col = B + global_n;

    int k = 0;
    constexpr int UNROLL = Config::UNROLL_K;

    for (; k + UNROLL <= K; k += UNROLL) {
        float a0 = __half2float(A[k + 0]);
        float a1 = __half2float(A[k + 1]);
        float a2 = __half2float(A[k + 2]);
        float a3 = __half2float(A[k + 3]);
        float a4 = __half2float(A[k + 4]);
        float a5 = __half2float(A[k + 5]);
        float a6 = __half2float(A[k + 6]);
        float a7 = __half2float(A[k + 7]);

        float b0 = ldg_fp16_to_f32(B_col + (k + 0) * N);
        float b1 = ldg_fp16_to_f32(B_col + (k + 1) * N);
        float b2 = ldg_fp16_to_f32(B_col + (k + 2) * N);
        float b3 = ldg_fp16_to_f32(B_col + (k + 3) * N);
        float b4 = ldg_fp16_to_f32(B_col + (k + 4) * N);
        float b5 = ldg_fp16_to_f32(B_col + (k + 5) * N);
        float b6 = ldg_fp16_to_f32(B_col + (k + 6) * N);
        float b7 = ldg_fp16_to_f32(B_col + (k + 7) * N);

        acc = fmaf(a0, b0, acc);
        acc = fmaf(a1, b1, acc);
        acc = fmaf(a2, b2, acc);
        acc = fmaf(a3, b3, acc);
        acc = fmaf(a4, b4, acc);
        acc = fmaf(a5, b5, acc);
        acc = fmaf(a6, b6, acc);
        acc = fmaf(a7, b7, acc);
    }

    for (; k < K; ++k) {
        float a = __half2float(A[k]);
        float b = ldg_fp16_to_f32(B_col + k * N);
        acc = fmaf(a, b, acc);
    }

    if (beta != 0.0f) {
        float c_old = __half2float(C[global_n]);
        acc = fmaf(alpha, acc, beta * c_old);
    } else {
        acc *= alpha;
    }

    C[global_n] = __float2half(acc);
}

// ============================================================================
// TF32 GEMV Kernel (FP32 input, TF32-style accumulation)
// ============================================================================

/**
 * GEMV kernel for FP32: C[1,N] = alpha * A[1,K] @ B[K,N] + beta * C[1,N]
 * Uses FP32 accumulation (no TensorCore at M=1)
 */
template<typename Config = GemvConfig>
__global__ void gemv_fp32_kernel(
    float const* __restrict__ A,
    float const* __restrict__ B,
    float* __restrict__ C,
    int K,
    int N,
    float alpha,
    float beta
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    float acc = 0.0f;
    const float* B_col = B + global_n;

    int k = 0;
    constexpr int UNROLL = Config::UNROLL_K;

    for (; k + UNROLL <= K; k += UNROLL) {
        float a0 = A[k + 0];
        float a1 = A[k + 1];
        float a2 = A[k + 2];
        float a3 = A[k + 3];
        float a4 = A[k + 4];
        float a5 = A[k + 5];
        float a6 = A[k + 6];
        float a7 = A[k + 7];

        float b0 = __ldg(B_col + (k + 0) * N);
        float b1 = __ldg(B_col + (k + 1) * N);
        float b2 = __ldg(B_col + (k + 2) * N);
        float b3 = __ldg(B_col + (k + 3) * N);
        float b4 = __ldg(B_col + (k + 4) * N);
        float b5 = __ldg(B_col + (k + 5) * N);
        float b6 = __ldg(B_col + (k + 6) * N);
        float b7 = __ldg(B_col + (k + 7) * N);

        acc = fmaf(a0, b0, acc);
        acc = fmaf(a1, b1, acc);
        acc = fmaf(a2, b2, acc);
        acc = fmaf(a3, b3, acc);
        acc = fmaf(a4, b4, acc);
        acc = fmaf(a5, b5, acc);
        acc = fmaf(a6, b6, acc);
        acc = fmaf(a7, b7, acc);
    }

    for (; k < K; ++k) {
        float a = A[k];
        float b = __ldg(B_col + k * N);
        acc = fmaf(a, b, acc);
    }

    if (beta != 0.0f) {
        acc = fmaf(alpha, acc, beta * C[global_n]);
    } else {
        acc *= alpha;
    }

    C[global_n] = acc;
}

// ============================================================================
// Batched GEMV Kernels (for continuous batching)
// ============================================================================

/**
 * Batched GEMV: C[batch,1,N] = A[batch,1,K] @ B[K,N]
 * B is shared across batches (weight matrix)
 * A is different per batch (activations)
 *
 * Grid: (ceil(N/TILE_N), batch_count)
 * Each block handles one (batch, tile_n) pair
 */
template<typename Config = GemvConfig>
__global__ void gemv_bf16_batched_kernel(
    __nv_bfloat16 const* __restrict__ A,  // [batch, K]
    __nv_bfloat16 const* __restrict__ B,  // [K, N] shared
    __nv_bfloat16* __restrict__ C,        // [batch, N]
    int K,
    int N,
    int batch_count,
    float alpha,
    float beta
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int batch_idx = blockIdx.y;
    const int global_n = block_n + tid;

    if (global_n >= N || batch_idx >= batch_count) return;

    // Batch-specific A and C pointers
    const __nv_bfloat16* A_batch = A + batch_idx * K;
    __nv_bfloat16* C_batch = C + batch_idx * N;

    float acc = 0.0f;
    const __nv_bfloat16* B_col = B + global_n;

    int k = 0;
    constexpr int UNROLL = Config::UNROLL_K;

    // Template-based unrolling: UNROLL_K can be 4, 8, or 16
    for (; k + UNROLL <= K; k += UNROLL) {
        if constexpr (UNROLL == 4) {
            __nv_bfloat162 a01 = ldg_bf16x2(A_batch + k + 0);
            __nv_bfloat162 a23 = ldg_bf16x2(A_batch + k + 2);
            float a0 = __low2float(a01);
            float a1 = __high2float(a01);
            float a2 = __low2float(a23);
            float a3 = __high2float(a23);
            float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
            float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
            float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
            float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
        } else if constexpr (UNROLL == 8) {
            __nv_bfloat162 a01 = ldg_bf16x2(A_batch + k + 0);
            __nv_bfloat162 a23 = ldg_bf16x2(A_batch + k + 2);
            __nv_bfloat162 a45 = ldg_bf16x2(A_batch + k + 4);
            __nv_bfloat162 a67 = ldg_bf16x2(A_batch + k + 6);
            float a0 = __low2float(a01);
            float a1 = __high2float(a01);
            float a2 = __low2float(a23);
            float a3 = __high2float(a23);
            float a4 = __low2float(a45);
            float a5 = __high2float(a45);
            float a6 = __low2float(a67);
            float a7 = __high2float(a67);
            float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
            float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
            float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
            float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
            float b4 = ldg_bf16_to_f32(B_col + (k + 4) * N);
            float b5 = ldg_bf16_to_f32(B_col + (k + 5) * N);
            float b6 = ldg_bf16_to_f32(B_col + (k + 6) * N);
            float b7 = ldg_bf16_to_f32(B_col + (k + 7) * N);
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
            acc = fmaf(a4, b4, acc);
            acc = fmaf(a5, b5, acc);
            acc = fmaf(a6, b6, acc);
            acc = fmaf(a7, b7, acc);
        } else if constexpr (UNROLL == 16) {
            __nv_bfloat162 a01 = ldg_bf16x2(A_batch + k + 0);
            __nv_bfloat162 a23 = ldg_bf16x2(A_batch + k + 2);
            __nv_bfloat162 a45 = ldg_bf16x2(A_batch + k + 4);
            __nv_bfloat162 a67 = ldg_bf16x2(A_batch + k + 6);
            __nv_bfloat162 a89 = ldg_bf16x2(A_batch + k + 8);
            __nv_bfloat162 aAB = ldg_bf16x2(A_batch + k + 10);
            __nv_bfloat162 aCD = ldg_bf16x2(A_batch + k + 12);
            __nv_bfloat162 aEF = ldg_bf16x2(A_batch + k + 14);
            float a0 = __low2float(a01);
            float a1 = __high2float(a01);
            float a2 = __low2float(a23);
            float a3 = __high2float(a23);
            float a4 = __low2float(a45);
            float a5 = __high2float(a45);
            float a6 = __low2float(a67);
            float a7 = __high2float(a67);
            float a8 = __low2float(a89);
            float a9 = __high2float(a89);
            float aA = __low2float(aAB);
            float aB = __high2float(aAB);
            float aC = __low2float(aCD);
            float aD = __high2float(aCD);
            float aE = __low2float(aEF);
            float aF = __high2float(aEF);
            float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
            float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
            float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
            float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
            float b4 = ldg_bf16_to_f32(B_col + (k + 4) * N);
            float b5 = ldg_bf16_to_f32(B_col + (k + 5) * N);
            float b6 = ldg_bf16_to_f32(B_col + (k + 6) * N);
            float b7 = ldg_bf16_to_f32(B_col + (k + 7) * N);
            float b8 = ldg_bf16_to_f32(B_col + (k + 8) * N);
            float b9 = ldg_bf16_to_f32(B_col + (k + 9) * N);
            float bA = ldg_bf16_to_f32(B_col + (k + 10) * N);
            float bB = ldg_bf16_to_f32(B_col + (k + 11) * N);
            float bC = ldg_bf16_to_f32(B_col + (k + 12) * N);
            float bD = ldg_bf16_to_f32(B_col + (k + 13) * N);
            float bE = ldg_bf16_to_f32(B_col + (k + 14) * N);
            float bF = ldg_bf16_to_f32(B_col + (k + 15) * N);
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
            acc = fmaf(a4, b4, acc);
            acc = fmaf(a5, b5, acc);
            acc = fmaf(a6, b6, acc);
            acc = fmaf(a7, b7, acc);
            acc = fmaf(a8, b8, acc);
            acc = fmaf(a9, b9, acc);
            acc = fmaf(aA, bA, acc);
            acc = fmaf(aB, bB, acc);
            acc = fmaf(aC, bC, acc);
            acc = fmaf(aD, bD, acc);
            acc = fmaf(aE, bE, acc);
            acc = fmaf(aF, bF, acc);
        }
    }

    for (; k < K; ++k) {
        float a = __bfloat162float(A_batch[k]);
        float b = ldg_bf16_to_f32(B_col + k * N);
        acc = fmaf(a, b, acc);
    }

    if (beta != 0.0f) {
        float c_old = __bfloat162float(C_batch[global_n]);
        acc = fmaf(alpha, acc, beta * c_old);
    } else {
        acc *= alpha;
    }

    C_batch[global_n] = __float2bfloat16(acc);
}

// ============================================================================
// Launch Functions
// ============================================================================

/**
 * Launch BF16 GEMV with per-size configuration selection
 *
 * Configuration selection logic:
 * - Small N (< 1024): Use smaller block/tile (GemvConfigSmallN)
 * - Small K (< 2048): Use smaller unroll (GemvConfigSmallK)
 * - Large K (> 8192) AND Large N (> 8192): Maximum unroll (GemvConfigLarge)
 * - Large K (> 8192): Larger unroll (GemvConfigLargeK)
 * - Default: Balanced configuration (GemvConfig)
 */
inline cudaError_t launch_gemv_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int K,
    int N,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Per-size configuration dispatch
    if (N < 1024) {
        // Small N: use smaller block to avoid wasted threads
        using Config = GemvConfigSmallN;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);
        gemv_bf16_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, alpha, beta);
    } else if (K > 8192 && N > 8192) {
        // Large matrices: maximum unrolling
        using Config = GemvConfigLarge;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);
        gemv_bf16_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, alpha, beta);
    } else if (K > 8192) {
        // Large K: more unrolling for ILP
        using Config = GemvConfigLargeK;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);
        gemv_bf16_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, alpha, beta);
    } else if (K < 2048) {
        // Small K: less unrolling
        using Config = GemvConfigSmallK;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);
        gemv_bf16_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, alpha, beta);
    } else {
        // Default: balanced configuration
        using Config = GemvConfig;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);
        gemv_bf16_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, alpha, beta);
    }

    return cudaGetLastError();
}

/**
 * Launch FP16 GEMV
 */
inline cudaError_t launch_gemv_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    int K,
    int N,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    using Config = GemvConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);

    gemv_fp16_kernel<Config><<<grid, block, 0, stream>>>(
        A, B, C, K, N, alpha, beta
    );

    return cudaGetLastError();
}

/**
 * Launch FP32 GEMV
 */
inline cudaError_t launch_gemv_fp32(
    const float* A,
    const float* B,
    float* C,
    int K,
    int N,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    using Config = GemvConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);

    gemv_fp32_kernel<Config><<<grid, block, 0, stream>>>(
        A, B, C, K, N, alpha, beta
    );

    return cudaGetLastError();
}

/**
 * Launch batched BF16 GEMV with per-size configuration selection
 */
inline cudaError_t launch_gemv_bf16_batched(
    const __nv_bfloat16* A,  // [batch, K]
    const __nv_bfloat16* B,  // [K, N]
    __nv_bfloat16* C,        // [batch, N]
    int K,
    int N,
    int batch_count,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Per-size configuration dispatch (same logic as non-batched)
    if (N < 1024) {
        using Config = GemvConfigSmallN;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);
        gemv_bf16_batched_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, batch_count, alpha, beta);
    } else if (K > 8192 && N > 8192) {
        using Config = GemvConfigLarge;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);
        gemv_bf16_batched_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, batch_count, alpha, beta);
    } else if (K > 8192) {
        using Config = GemvConfigLargeK;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);
        gemv_bf16_batched_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, batch_count, alpha, beta);
    } else if (K < 2048) {
        using Config = GemvConfigSmallK;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);
        gemv_bf16_batched_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, batch_count, alpha, beta);
    } else {
        using Config = GemvConfig;
        dim3 block(Config::BLOCK_SIZE);
        dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);
        gemv_bf16_batched_kernel<Config><<<grid, block, 0, stream>>>(
            A, B, C, K, N, batch_count, alpha, beta);
    }

    return cudaGetLastError();
}

// ============================================================================
// Dispatch Function (M=1 detection)
// ============================================================================

/**
 * GEMM/GEMV dispatcher
 *
 * Selects GEMV kernel when M=1, otherwise falls through to GEMM
 * Returns true if GEMV was dispatched, false if GEMM should be used
 */
inline bool dispatch_gemv_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // GEMV dispatch conditions:
    // 1. M == 1 (single row)
    // 2. N >= MIN_N (avoid overhead for tiny outputs)
    if (M == 1 && N >= GemvConfig::MIN_N) {
        launch_gemv_bf16(A, B, C, K, N, alpha, beta, stream);
        return true;
    }
    return false;
}

inline bool dispatch_gemv_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    if (M == 1 && N >= GemvConfig::MIN_N) {
        launch_gemv_fp16(A, B, C, K, N, alpha, beta, stream);
        return true;
    }
    return false;
}

inline bool dispatch_gemv_fp32(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    if (M == 1 && N >= GemvConfig::MIN_N) {
        launch_gemv_fp32(A, B, C, K, N, alpha, beta, stream);
        return true;
    }
    return false;
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
