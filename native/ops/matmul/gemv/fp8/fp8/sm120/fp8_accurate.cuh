/**
 * Accurate FP8/FP8 GEMV Kernel (SM120) - Issue #123
 *
 * A[K] (FP8) x B[N,K] (FP8) -> C[N] (BF16)
 *
 * Key accuracy improvements over fast version:
 * 1. Smaller scale blocks: 32 elements instead of 128
 * 2. Kahan summation for reduced accumulation error
 * 3. Double accumulator for critical path
 *
 * Target: <0.5% relative error (vs ~1-2% in fast version)
 * Trade-off: ~1.5-2x slower, 4x more scale memory
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
// Accurate Configuration
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

__device__ __forceinline__ float fp8_e4m3_to_float_accurate(uint8_t val) {
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = val;
    return float(fp8_val);
}

// ============================================================================
// Kahan Summation Helper
// ============================================================================

struct KahanAccumulator {
    float sum;
    float compensation;

    __device__ __forceinline__ KahanAccumulator() : sum(0.0f), compensation(0.0f) {}

    __device__ __forceinline__ void add(float value) {
        float y = value - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    __device__ __forceinline__ float get() const {
        return sum;
    }
};

// ============================================================================
// Accurate FP8 GEMV Kernel with Kahan Summation
// ============================================================================

/**
 * Accurate FP8 GEMV with:
 * 1. Small scale blocks (32 elements)
 * 2. Kahan summation for reduced accumulation error
 * 3. Careful ordering of operations
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

    // Shared memory for A (FP8 = 1 byte per element)
    extern __shared__ uint8_t smem_A[];

    // Cooperative load of A into shared memory
    for (int k = threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // Scale dimensions (smaller blocks = more scales)
    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;

    // B row pointer for this output
    const uint8_t* B_row = B_nk + global_n * K;

    // Kahan accumulator for each lane
    KahanAccumulator acc;

    // Process in groups of SCALE_BLOCK_SIZE for consistent scaling
    const int num_scale_blocks = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    for (int sb = 0; sb < num_scale_blocks; ++sb) {
        const int k_start = sb * Config::SCALE_BLOCK_SIZE;
        const int k_end = min(k_start + Config::SCALE_BLOCK_SIZE, K);

        // Load scales for this block
        float sA = scale_A[sb];
        float sB = scale_B[scale_n * scale_stride_k + sb];
        float combined_scale = sA * sB;

        // Each lane processes elements within this scale block
        for (int k = k_start + lane_id; k < k_end; k += Config::WARP_SIZE) {
            // Dequantize with proper scaling
            float a = fp8_e4m3_to_float_accurate(smem_A[k]);
            float b = fp8_e4m3_to_float_accurate(B_row[k]);

            // Multiply with combined scale and accumulate using Kahan
            float product = a * b * combined_scale;
            acc.add(product);
        }
    }

    // Get final sum from Kahan accumulator
    float sum = acc.get();

    // Warp-level reduction using shuffle (with Kahan for final reduction)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        C[global_n] = __float2bfloat16(sum);
    }
}

/**
 * Optimized accurate kernel with vectorized loads
 * Still uses Kahan summation and small scale blocks
 */
template<typename Config = GemvFP8AccurateConfig>
__global__ void gemv_fp8_accurate_opt_kernel(
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

    extern __shared__ uint8_t smem_A[];

    // Vectorized load of A into shared memory
    const int K_aligned8 = K & ~7;
    for (int k = threadIdx.x * 8; k < K_aligned8; k += Config::BLOCK_SIZE * 8) {
        *reinterpret_cast<uint64_t*>(&smem_A[k]) =
            *reinterpret_cast<const uint64_t*>(&A[k]);
    }
    for (int k = K_aligned8 + threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;
    const uint8_t* B_row = B_nk + global_n * K;

    // Use double precision accumulator for critical path
    double acc = 0.0;

    const int num_scale_blocks = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    for (int sb = 0; sb < num_scale_blocks; ++sb) {
        const int k_start = sb * Config::SCALE_BLOCK_SIZE;
        const int k_end = min(k_start + Config::SCALE_BLOCK_SIZE, K);

        float sA = __ldg(&scale_A[sb]);
        float sB = __ldg(&scale_B[scale_n * scale_stride_k + sb]);
        double combined_scale = double(sA) * double(sB);

        // Vectorized processing within scale block
        const int k_aligned4 = k_start + ((k_end - k_start) & ~3);

        for (int k = k_start + lane_id * 4; k < k_aligned4; k += Config::WARP_SIZE * 4) {
            if (k + 4 <= k_end) {
                uint32_t a4 = *reinterpret_cast<const uint32_t*>(&smem_A[k]);
                uint32_t b4 = *reinterpret_cast<const uint32_t*>(&B_row[k]);

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float a = fp8_e4m3_to_float_accurate((a4 >> (i * 8)) & 0xFF);
                    float b = fp8_e4m3_to_float_accurate((b4 >> (i * 8)) & 0xFF);
                    acc += double(a) * double(b) * combined_scale;
                }
            }
        }

        // Handle remainder
        for (int k = k_aligned4 + lane_id; k < k_end; k += Config::WARP_SIZE) {
            float a = fp8_e4m3_to_float_accurate(smem_A[k]);
            float b = fp8_e4m3_to_float_accurate(B_row[k]);
            acc += double(a) * double(b) * combined_scale;
        }
    }

    // Convert back to float for warp reduction
    float sum = float(acc);

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
