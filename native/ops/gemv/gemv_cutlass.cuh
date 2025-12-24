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
// Configuration
// ============================================================================

// GEMV kernel configuration
// Tuned for memory bandwidth maximization
struct GemvConfig {
    // Block size: 256 threads = 8 warps
    // Rationale: Good occupancy on SM86+ (up to 16 blocks/SM)
    static constexpr int BLOCK_SIZE = 256;

    // Tile N: Each block processes 256 output elements
    // Rationale: Matches BLOCK_SIZE for simple thread-to-output mapping
    static constexpr int TILE_N = 256;

    // K unroll factor: Process 8 K values per iteration
    // Rationale: Hide memory latency, utilize instruction-level parallelism
    static constexpr int UNROLL_K = 8;

    // Minimum N for GEMV dispatch (below this, GEMM might be faster)
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

    for (; k + UNROLL <= K; k += UNROLL) {
        // Vectorized load: 8 BF16 values using 4x BF16x2 loads
        // This reduces memory transactions for A (broadcast)
        __nv_bfloat162 a01 = ldg_bf16x2(A + k + 0);
        __nv_bfloat162 a23 = ldg_bf16x2(A + k + 2);
        __nv_bfloat162 a45 = ldg_bf16x2(A + k + 4);
        __nv_bfloat162 a67 = ldg_bf16x2(A + k + 6);

        // Extract individual floats from bfloat162
        float a0 = __low2float(a01);
        float a1 = __high2float(a01);
        float a2 = __low2float(a23);
        float a3 = __high2float(a23);
        float a4 = __low2float(a45);
        float a5 = __high2float(a45);
        float a6 = __low2float(a67);
        float a7 = __high2float(a67);

        // Load UNROLL_K values of B (coalesced across threads)
        // Using __ldg() for read-only cache optimization
        // Note: Adjacent threads access adjacent memory locations at each k
        //       Thread tid reads B[k*N + block_n + tid], which is coalesced
        float b0 = ldg_bf16_to_f32(B_col + (k + 0) * N);
        float b1 = ldg_bf16_to_f32(B_col + (k + 1) * N);
        float b2 = ldg_bf16_to_f32(B_col + (k + 2) * N);
        float b3 = ldg_bf16_to_f32(B_col + (k + 3) * N);
        float b4 = ldg_bf16_to_f32(B_col + (k + 4) * N);
        float b5 = ldg_bf16_to_f32(B_col + (k + 5) * N);
        float b6 = ldg_bf16_to_f32(B_col + (k + 6) * N);
        float b7 = ldg_bf16_to_f32(B_col + (k + 7) * N);

        // FMA accumulation
        // Using fmaf for precision and potential hardware fusion
        acc = fmaf(a0, b0, acc);
        acc = fmaf(a1, b1, acc);
        acc = fmaf(a2, b2, acc);
        acc = fmaf(a3, b3, acc);
        acc = fmaf(a4, b4, acc);
        acc = fmaf(a5, b5, acc);
        acc = fmaf(a6, b6, acc);
        acc = fmaf(a7, b7, acc);
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

    for (; k + UNROLL <= K; k += UNROLL) {
        // Vectorized load for A (broadcast)
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
 * Launch BF16 GEMV
 *
 * CTA/Warp configuration rationale:
 * - Block size 256 = 8 warps
 * - SM86: max 1536 threads/SM = 6 blocks/SM at 256 threads
 * - SM89: max 1536 threads/SM = 6 blocks/SM at 256 threads
 * - SM90: max 2048 threads/SM = 8 blocks/SM at 256 threads
 * - Good occupancy across all target SMs
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
    using Config = GemvConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);

    gemv_bf16_kernel<Config><<<grid, block, 0, stream>>>(
        A, B, C, K, N, alpha, beta
    );

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
 * Launch batched BF16 GEMV
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
    using Config = GemvConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);

    gemv_bf16_batched_kernel<Config><<<grid, block, 0, stream>>>(
        A, B, C, K, N, batch_count, alpha, beta
    );

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
