/**
 * Optimized BF16 GEMV Kernel (SM120) - B[N,K] Layout
 *
 * Design: Same as FP8 GEMV for maximum speed
 * - B[N, K] row-major (each row = one output's weights)
 * - Warp-level reduction over K (32 threads per output)
 * - Shared memory for A broadcast
 * - 128-bit vectorized loads (4 BF16 = 8 bytes)
 *
 * Target: Match FP8 GEMV speed (~10-20us) with BF16 precision (~0.6% error)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Configuration
// ============================================================================

struct GemvBF16OptConfig {
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;  // 256 threads
    static constexpr int WARP_SIZE = 32;
    static constexpr int VEC_SIZE = 4;  // Load 4 BF16 = 8 bytes
};

// ============================================================================
// BF16 Optimized GEMV Kernel
// ============================================================================

/**
 * BF16 GEMV with warp-level reduction (B[N,K] layout)
 *
 * Each warp handles ONE output element (N dimension)
 * 32 threads in warp cooperatively reduce over K dimension
 *
 * Memory layout:
 * - A: [K] BF16 activation vector
 * - B: [N, K] BF16 weight matrix (row-major, row = output)
 * - C: [N] BF16 output vector
 *
 * @param A     [K] BF16 activation
 * @param B_nk  [N, K] BF16 weights (row-major)
 * @param C     [N] BF16 output
 * @param K     Inner dimension
 * @param N     Output dimension
 */
template<typename Config = GemvBF16OptConfig>
__global__ void gemv_bf16_opt_kernel(
    __nv_bfloat16 const* __restrict__ A,
    __nv_bfloat16 const* __restrict__ B_nk,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    // Shared memory for A (BF16)
    extern __shared__ __nv_bfloat16 smem_A[];

    // Cooperative load of A into shared memory using 64-bit loads
    const int K_aligned4 = K & ~3;
    for (int k = threadIdx.x * 4; k < K_aligned4; k += Config::BLOCK_SIZE * 4) {
        // Load 4 BF16 = 8 bytes = uint64_t
        uint64_t data = *reinterpret_cast<const uint64_t*>(&A[k]);
        *reinterpret_cast<uint64_t*>(&smem_A[k]) = data;
    }
    // Handle remainder
    for (int k = K_aligned4 + threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // B row pointer for this output
    const __nv_bfloat16* B_row = B_nk + global_n * K;

    // FP32 accumulator for precision
    float acc = 0.0f;

    // Main loop: 64-bit loads (4 BF16 values)
    const int K_aligned_loop = K & ~(Config::WARP_SIZE * 4 - 1);

    for (int k_base = lane_id * 4; k_base < K_aligned_loop; k_base += Config::WARP_SIZE * 4) {
        // Load 4 BF16 from A (shared memory)
        uint64_t a4_raw = *reinterpret_cast<const uint64_t*>(&smem_A[k_base]);
        __nv_bfloat16* a4 = reinterpret_cast<__nv_bfloat16*>(&a4_raw);

        // Load 4 BF16 from B (global with cache hint)
        uint64_t b4_raw = __ldg(reinterpret_cast<const uint64_t*>(&B_row[k_base]));
        __nv_bfloat16* b4 = reinterpret_cast<__nv_bfloat16*>(&b4_raw);

        // FMA for 4 elements
        acc = fmaf(__bfloat162float(a4[0]), __bfloat162float(b4[0]), acc);
        acc = fmaf(__bfloat162float(a4[1]), __bfloat162float(b4[1]), acc);
        acc = fmaf(__bfloat162float(a4[2]), __bfloat162float(b4[2]), acc);
        acc = fmaf(__bfloat162float(a4[3]), __bfloat162float(b4[3]), acc);
    }

    // Handle remainder with scalar
    for (int k = K_aligned_loop + lane_id; k < K; k += Config::WARP_SIZE) {
        float a = __bfloat162float(smem_A[k]);
        float b = __bfloat162float(__ldg(&B_row[k]));
        acc = fmaf(a, b, acc);
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        C[global_n] = __float2bfloat16(acc);
    }
}

/**
 * Vectorized variant: 128-bit loads (8 BF16 = 16 bytes)
 * Better for very large K dimensions.
 */
template<typename Config = GemvBF16OptConfig>
__global__ void gemv_bf16_opt_vec8_kernel(
    __nv_bfloat16 const* __restrict__ A,
    __nv_bfloat16 const* __restrict__ B_nk,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    // Shared memory for A
    extern __shared__ __nv_bfloat16 smem_A[];

    // Cooperative load of A using 128-bit loads
    const int K_aligned8 = K & ~7;
    for (int k = threadIdx.x * 8; k < K_aligned8; k += Config::BLOCK_SIZE * 8) {
        uint4 data = *reinterpret_cast<const uint4*>(&A[k]);
        *reinterpret_cast<uint4*>(&smem_A[k]) = data;
    }
    // Remainder with 64-bit
    const int K_aligned4 = K & ~3;
    for (int k = K_aligned8 + threadIdx.x * 4; k < K_aligned4; k += Config::BLOCK_SIZE * 4) {
        uint64_t data = *reinterpret_cast<const uint64_t*>(&A[k]);
        *reinterpret_cast<uint64_t*>(&smem_A[k]) = data;
    }
    // Scalar remainder
    for (int k = K_aligned4 + threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    const __nv_bfloat16* B_row = B_nk + global_n * K;

    // 4 independent accumulators to hide FMA latency
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Main loop: 128-bit loads (8 BF16 values)
    const int K_aligned_loop = K & ~(Config::WARP_SIZE * 8 - 1);

    for (int k_base = lane_id * 8; k_base < K_aligned_loop; k_base += Config::WARP_SIZE * 8) {
        // Load 8 BF16 from A (shared memory)
        uint4 a8_raw = *reinterpret_cast<const uint4*>(&smem_A[k_base]);
        __nv_bfloat16* a8 = reinterpret_cast<__nv_bfloat16*>(&a8_raw);

        // Load 8 BF16 from B (global with cache hint)
        uint4 b8_raw;
        b8_raw.x = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base]));
        b8_raw.y = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base + 2]));
        b8_raw.z = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base + 4]));
        b8_raw.w = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base + 6]));
        __nv_bfloat16* b8 = reinterpret_cast<__nv_bfloat16*>(&b8_raw);

        // FMA to 4 accumulators (2 elements each)
        acc0 = fmaf(__bfloat162float(a8[0]), __bfloat162float(b8[0]), acc0);
        acc0 = fmaf(__bfloat162float(a8[1]), __bfloat162float(b8[1]), acc0);
        acc1 = fmaf(__bfloat162float(a8[2]), __bfloat162float(b8[2]), acc1);
        acc1 = fmaf(__bfloat162float(a8[3]), __bfloat162float(b8[3]), acc1);
        acc2 = fmaf(__bfloat162float(a8[4]), __bfloat162float(b8[4]), acc2);
        acc2 = fmaf(__bfloat162float(a8[5]), __bfloat162float(b8[5]), acc2);
        acc3 = fmaf(__bfloat162float(a8[6]), __bfloat162float(b8[6]), acc3);
        acc3 = fmaf(__bfloat162float(a8[7]), __bfloat162float(b8[7]), acc3);
    }

    // Handle remainder with 64-bit loads
    for (int k_base = K_aligned_loop + lane_id * 4; k_base < K_aligned4; k_base += Config::WARP_SIZE * 4) {
        uint64_t a4_raw = *reinterpret_cast<const uint64_t*>(&smem_A[k_base]);
        uint64_t b4_raw = __ldg(reinterpret_cast<const uint64_t*>(&B_row[k_base]));
        __nv_bfloat16* a4 = reinterpret_cast<__nv_bfloat16*>(&a4_raw);
        __nv_bfloat16* b4 = reinterpret_cast<__nv_bfloat16*>(&b4_raw);
        acc0 = fmaf(__bfloat162float(a4[0]), __bfloat162float(b4[0]), acc0);
        acc0 = fmaf(__bfloat162float(a4[1]), __bfloat162float(b4[1]), acc0);
        acc0 = fmaf(__bfloat162float(a4[2]), __bfloat162float(b4[2]), acc0);
        acc0 = fmaf(__bfloat162float(a4[3]), __bfloat162float(b4[3]), acc0);
    }

    // Scalar remainder
    for (int k = K_aligned4 + lane_id; k < K; k += Config::WARP_SIZE) {
        float a = __bfloat162float(smem_A[k]);
        float b = __bfloat162float(__ldg(&B_row[k]));
        acc0 = fmaf(a, b, acc0);
    }

    // Combine accumulators
    float sum = acc0 + acc1 + acc2 + acc3;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        C[global_n] = __float2bfloat16(sum);
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

inline cudaError_t launch_gemv_bf16_opt(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B_nk,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream = nullptr
) {
    using Config = GemvBF16OptConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK);

    // Shared memory for A (BF16 = 2 bytes)
    size_t smem_size = K * sizeof(__nv_bfloat16);

    // Use vec8 kernel for large K
    if (K >= 4096) {
        gemv_bf16_opt_vec8_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, C, K, N
        );
    } else {
        gemv_bf16_opt_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, C, K, N
        );
    }

    return cudaGetLastError();
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
