/**
 * Accurate FP8/FP8 GEMV Kernel (SM120) - Issue #123
 *
 * A[K] (FP8) x B[N,K] (FP8) -> C[N] (BF16)
 *
 * Key accuracy improvement over fast version:
 * - Smaller scale blocks: 32 elements instead of 128
 *
 * This captures local dynamic range better, reducing quantization error.
 * Target: <0.5% relative error (vs ~1-2% in fast version with per-block quant)
 * Trade-off: ~1.5-2x slower due to more scale factor loads
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Accurate Configuration - Only difference: SCALE_BLOCK_SIZE = 32
// ============================================================================

struct GemvFP8AccurateConfig {
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;  // 256 threads
    static constexpr int WARP_SIZE = 32;
    static constexpr int SCALE_BLOCK_SIZE = 32;  // Smaller blocks for accuracy (was 128)
};

// ============================================================================
// FP8 E4M3 to float conversion (inline)
// ============================================================================

__device__ __forceinline__ float fp8_e4m3_to_float_acc(uint8_t val) {
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = val;
    return float(fp8_val);
}

// ============================================================================
// Accurate FP8 GEMV Kernel - Same structure as fast, different SCALE_BLOCK_SIZE
// ============================================================================

/**
 * Optimized accurate kernel using same structure as fast version.
 * Key optimizations from fast version:
 * - 128-bit vector loads (16 FP8 values at once via uint4)
 * - __ldg() for cached global memory reads
 * - 4 independent accumulators to hide FMA latency
 *
 * Accuracy improvement:
 * - SCALE_BLOCK_SIZE = 32 (vs 128 in fast version)
 */
template<typename Config = GemvFP8AccurateConfig>
__global__ void gemv_fp8_accurate_kernel(
    uint8_t const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    float const* __restrict__ scale_A,
    float const* __restrict__ scale_B,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    // Shared memory for A (FP8)
    extern __shared__ uint8_t smem_A[];

    // Cooperative load of A into shared memory using 128-bit loads
    const int K_aligned16 = K & ~15;
    for (int k = threadIdx.x * 16; k < K_aligned16; k += Config::BLOCK_SIZE * 16) {
        uint4 data = *reinterpret_cast<const uint4*>(&A[k]);
        *reinterpret_cast<uint4*>(&smem_A[k]) = data;
    }
    // Handle remainder with 64-bit
    const int K_rem_start = K_aligned16;
    const int K_aligned8 = K & ~7;
    for (int k = K_rem_start + threadIdx.x * 8; k < K_aligned8; k += Config::BLOCK_SIZE * 8) {
        *reinterpret_cast<uint64_t*>(&smem_A[k]) =
            *reinterpret_cast<const uint64_t*>(&A[k]);
    }
    // Scalar remainder
    for (int k = K_aligned8 + threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // Scale dimensions (smaller blocks = more scales)
    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;

    // B row pointer with __ldg for caching
    const uint8_t* B_row = B_nk + global_n * K;

    // 4 independent accumulators to hide FMA latency
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Main loop: 128-bit loads (16 FP8 values)
    // Each lane handles 16 elements per iteration, processes 4 at a time to 4 accumulators
    const int K_aligned_loop = K & ~(Config::WARP_SIZE * 16 - 1);

    for (int k_base = lane_id * 16; k_base < K_aligned_loop; k_base += Config::WARP_SIZE * 16) {
        // Load scale factors for this position
        const int scale_k = k_base / Config::SCALE_BLOCK_SIZE;
        float sA = __ldg(&scale_A[scale_k]);
        float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
        float combined_scale = sA * sB;

        // Load 16 FP8 values from A (shared memory) and B (global with __ldg)
        uint4 a16 = *reinterpret_cast<const uint4*>(&smem_A[k_base]);
        uint4 b16;
        b16.x = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base]));
        b16.y = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base + 4]));
        b16.z = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base + 8]));
        b16.w = __ldg(reinterpret_cast<const uint32_t*>(&B_row[k_base + 12]));

        // Process 4 values to each accumulator
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.x >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.x >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float_acc(a_val) * combined_scale;
            float b = fp8_e4m3_to_float_acc(b_val);
            acc0 = fmaf(a, b, acc0);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.y >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.y >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float_acc(a_val) * combined_scale;
            float b = fp8_e4m3_to_float_acc(b_val);
            acc1 = fmaf(a, b, acc1);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.z >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.z >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float_acc(a_val) * combined_scale;
            float b = fp8_e4m3_to_float_acc(b_val);
            acc2 = fmaf(a, b, acc2);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.w >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.w >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float_acc(a_val) * combined_scale;
            float b = fp8_e4m3_to_float_acc(b_val);
            acc3 = fmaf(a, b, acc3);
        }
    }

    // Handle remainder with 64-bit loads
    for (int k_base = K_aligned_loop + lane_id * 8; k_base < K_aligned8; k_base += Config::WARP_SIZE * 8) {
        const int scale_k = k_base / Config::SCALE_BLOCK_SIZE;
        float sA = __ldg(&scale_A[scale_k]);
        float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
        float combined_scale = sA * sB;

        uint64_t a8 = *reinterpret_cast<const uint64_t*>(&smem_A[k_base]);
        uint64_t b8 = __ldg(reinterpret_cast<const uint64_t*>(&B_row[k_base]));

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t a_val = (a8 >> (i * 8)) & 0xFF;
            uint8_t b_val = (b8 >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float_acc(a_val) * combined_scale;
            float b = fp8_e4m3_to_float_acc(b_val);
            acc0 = fmaf(a, b, acc0);
        }
    }

    // Scalar remainder
    for (int k = K_aligned8 + lane_id; k < K; k += Config::WARP_SIZE) {
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;
        float sA = __ldg(&scale_A[scale_k]);
        float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
        float a = fp8_e4m3_to_float_acc(smem_A[k]) * sA;
        float b = fp8_e4m3_to_float_acc(B_row[k]) * sB;
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
// Launch Function Declarations
// ============================================================================

cudaError_t launch_gemv_fp8_accurate(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
