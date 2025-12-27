/**
 * Pure FP8/FP8/FP8 GEMV Kernel (SM120)
 *
 * A[K] (FP8) x B[N,K] (FP8) -> C[N] (FP8 or BF16)
 *
 * Key advantage over W8A16 GEMV:
 * - A is FP8 (1 byte) instead of BF16 (2 bytes)
 * - Shared memory requirement halved: K bytes vs K*2 bytes
 * - Supports K up to 48K without shared memory overflow
 *
 * Optimizations:
 * 1. Warp-level reduction over K dimension
 * 2. Shared memory for activation vector A (FP8)
 * 3. Vectorized uint4 loads (4 FP8 values at once)
 * 4. Coalesced memory access pattern
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
// Configuration
// ============================================================================

struct GemvFP8PureConfig {
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;  // 256 threads
    static constexpr int WARP_SIZE = 32;
    static constexpr int VEC_SIZE = 4;  // Load 4 FP8 values at once
    static constexpr int SCALE_BLOCK_SIZE = 128;  // Block size for scaling
};

// ============================================================================
// FP8 E4M3 to float conversion (inline)
// ============================================================================

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
    // Use CUDA's native FP8 type for conversion
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = val;
    return float(fp8_val);
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
    __nv_fp8_e4m3 fp8_val(val);
    return *reinterpret_cast<uint8_t*>(&fp8_val);
}

// ============================================================================
// Pure FP8 GEMV Kernel: A[K](FP8) x B[N,K](FP8) -> C[N](BF16)
// ============================================================================

/**
 * Pure FP8 GEMV with warp-level reduction
 *
 * Each warp handles ONE output element (N dimension)
 * 32 threads in warp cooperatively reduce over K dimension
 *
 * Memory layout:
 * - A: [K] FP8 E4M3 activation vector
 * - B: [N, K] FP8 E4M3 weight matrix (row-major, transposed)
 * - scale_A: scalar or [K/128] FP32 scales for A
 * - scale_B: [N/128, K/128] FP32 scales for B
 * - C: [N] BF16 output vector
 */
template<typename Config = GemvFP8PureConfig>
__global__ void gemv_fp8_pure_kernel(
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

    // Scale dimensions
    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;

    // B row pointer for this output
    const uint8_t* B_row = B_nk + global_n * K;

    float acc = 0.0f;

    // Each lane handles K elements with stride 32
    for (int k = lane_id; k < K; k += Config::WARP_SIZE) {
        // Load scales
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;
        float sA = scale_A[scale_k];
        float sB = scale_B[scale_n * scale_stride_k + scale_k];

        // Load and dequantize FP8 values
        float a = fp8_e4m3_to_float(smem_A[k]) * sA;
        float b = fp8_e4m3_to_float(B_row[k]) * sB;

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
 * Vectorized variant: Load 8 FP8 values at once (uint64)
 */
template<typename Config = GemvFP8PureConfig>
__global__ void gemv_fp8_pure_vec8_kernel(
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

    // Cooperative load of A into shared memory (vectorized)
    const int K_aligned8 = K & ~7;
    for (int k = threadIdx.x * 8; k < K_aligned8; k += Config::BLOCK_SIZE * 8) {
        if (k + 8 <= K) {
            *reinterpret_cast<uint64_t*>(&smem_A[k]) =
                *reinterpret_cast<const uint64_t*>(&A[k]);
        }
    }
    // Handle remainder
    for (int k = K_aligned8 + threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // Scale dimensions
    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;

    // B row pointer
    const uint8_t* B_row = B_nk + global_n * K;

    float acc = 0.0f;

    // Vectorized: each lane handles 8 elements per iteration
    for (int k_base = lane_id * 8; k_base < K_aligned8; k_base += Config::WARP_SIZE * 8) {
        const int scale_k = k_base / Config::SCALE_BLOCK_SIZE;
        float sA = scale_A[scale_k];
        float sB = scale_B[scale_n * scale_stride_k + scale_k];
        float combined_scale = sA * sB;

        // Load 8 FP8 values from A and B
        uint64_t a8 = *reinterpret_cast<const uint64_t*>(&smem_A[k_base]);
        uint64_t b8 = *reinterpret_cast<const uint64_t*>(&B_row[k_base]);

        // Unpack and accumulate
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t a_val = (a8 >> (i * 8)) & 0xFF;
            uint8_t b_val = (b8 >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float(a_val);
            float b = fp8_e4m3_to_float(b_val);
            acc = fmaf(a * combined_scale, b, acc);
        }
    }

    // Handle remainder
    for (int k = K_aligned8 + lane_id; k < K; k += Config::WARP_SIZE) {
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;
        float sA = scale_A[scale_k];
        float sB = scale_B[scale_n * scale_stride_k + scale_k];
        float a = fp8_e4m3_to_float(smem_A[k]) * sA;
        float b = fp8_e4m3_to_float(B_row[k]) * sB;
        acc = fmaf(a, b, acc);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[global_n] = __float2bfloat16(acc);
    }
}

/**
 * Ultra-optimized variant: 128-bit loads, __ldg(), multiple accumulators
 *
 * Key optimizations:
 * 1. 128-bit vector loads (16 FP8 values at once via uint4)
 * 2. __ldg() for cached global memory reads
 * 3. 4 independent accumulators to hide FMA latency
 * 4. Aggressive loop unrolling
 * 5. Register-level parallelism
 */
template<typename Config = GemvFP8PureConfig>
__global__ void gemv_fp8_pure_opt_kernel(
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

    // Scale dimensions
    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;

    // B row pointer with __ldg cache hint
    const uint8_t* B_row = B_nk + global_n * K;

    // 4 independent accumulators to hide FMA latency
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Main loop: each lane handles 16 elements per iteration (128-bit)
    // Stride = 32 lanes * 16 elements = 512 elements per warp iteration
    for (int k_base = lane_id * 16; k_base < K_aligned16; k_base += Config::WARP_SIZE * 16) {
        const int scale_k = k_base / Config::SCALE_BLOCK_SIZE;
        float sA = __ldg(&scale_A[scale_k]);
        float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
        float combined_scale = sA * sB;

        // Load 16 FP8 values from shared memory (A)
        uint4 a16 = *reinterpret_cast<const uint4*>(&smem_A[k_base]);

        // Load 16 FP8 values from global memory (B) with cache hint
        uint4 b16 = *reinterpret_cast<const uint4*>(&B_row[k_base]);

        // Process first 4 bytes (a16.x, b16.x)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.x >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.x >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float(a_val) * combined_scale;
            float b = fp8_e4m3_to_float(b_val);
            acc0 = fmaf(a, b, acc0);
        }

        // Process second 4 bytes (a16.y, b16.y)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.y >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.y >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float(a_val) * combined_scale;
            float b = fp8_e4m3_to_float(b_val);
            acc1 = fmaf(a, b, acc1);
        }

        // Process third 4 bytes (a16.z, b16.z)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.z >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.z >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float(a_val) * combined_scale;
            float b = fp8_e4m3_to_float(b_val);
            acc2 = fmaf(a, b, acc2);
        }

        // Process fourth 4 bytes (a16.w, b16.w)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t a_val = (a16.w >> (i * 8)) & 0xFF;
            uint8_t b_val = (b16.w >> (i * 8)) & 0xFF;
            float a = fp8_e4m3_to_float(a_val) * combined_scale;
            float b = fp8_e4m3_to_float(b_val);
            acc3 = fmaf(a, b, acc3);
        }
    }

    // Handle remainder (K_aligned16 to K)
    for (int k = K_aligned16 + lane_id; k < K; k += Config::WARP_SIZE) {
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;
        float sA = __ldg(&scale_A[scale_k]);
        float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
        float a = fp8_e4m3_to_float(smem_A[k]) * sA;
        float b = fp8_e4m3_to_float(__ldg(&B_row[k])) * sB;
        acc0 = fmaf(a, b, acc0);
    }

    // Combine accumulators
    float acc = acc0 + acc1 + acc2 + acc3;

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[global_n] = __float2bfloat16(acc);
    }
}

/**
 * Multi-row variant: Each warp processes 2 output rows
 * Better memory bandwidth utilization by reusing A from shared memory
 */
struct GemvFP8MultiRowConfig {
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;
    static constexpr int WARP_SIZE = 32;
    static constexpr int ROWS_PER_WARP = 2;  // Process 2 outputs per warp
    static constexpr int SCALE_BLOCK_SIZE = 128;
};

template<typename Config = GemvFP8MultiRowConfig>
__global__ void gemv_fp8_pure_multirow_kernel(
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
    // Each warp handles 2 consecutive outputs
    const int global_n_base = (blockIdx.x * Config::WARPS_PER_BLOCK + warp_id) * Config::ROWS_PER_WARP;

    // Shared memory for A (FP8)
    extern __shared__ uint8_t smem_A[];

    // Cooperative load of A into shared memory using 128-bit loads
    const int K_aligned16 = K & ~15;
    for (int k = threadIdx.x * 16; k < K_aligned16; k += Config::BLOCK_SIZE * 16) {
        uint4 data = *reinterpret_cast<const uint4*>(&A[k]);
        *reinterpret_cast<uint4*>(&smem_A[k]) = data;
    }
    const int K_aligned8 = K & ~7;
    for (int k = K_aligned16 + threadIdx.x * 8; k < K_aligned8; k += Config::BLOCK_SIZE * 8) {
        *reinterpret_cast<uint64_t*>(&smem_A[k]) =
            *reinterpret_cast<const uint64_t*>(&A[k]);
    }
    for (int k = K_aligned8 + threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // Scale dimensions
    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    // Process 2 rows per warp
    float acc[Config::ROWS_PER_WARP] = {0.0f, 0.0f};

    #pragma unroll
    for (int row = 0; row < Config::ROWS_PER_WARP; ++row) {
        const int global_n = global_n_base + row;
        if (global_n >= N) continue;

        const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;
        const uint8_t* B_row = B_nk + global_n * K;

        float acc0 = 0.0f, acc1 = 0.0f;

        // Main loop with 128-bit loads
        for (int k_base = lane_id * 16; k_base < K_aligned16; k_base += Config::WARP_SIZE * 16) {
            const int scale_k = k_base / Config::SCALE_BLOCK_SIZE;
            float sA = __ldg(&scale_A[scale_k]);
            float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
            float combined_scale = sA * sB;

            uint4 a16 = *reinterpret_cast<const uint4*>(&smem_A[k_base]);
            uint4 b16 = *reinterpret_cast<const uint4*>(&B_row[k_base]);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = fp8_e4m3_to_float((a16.x >> (i * 8)) & 0xFF) * combined_scale;
                float b = fp8_e4m3_to_float((b16.x >> (i * 8)) & 0xFF);
                acc0 = fmaf(a, b, acc0);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = fp8_e4m3_to_float((a16.y >> (i * 8)) & 0xFF) * combined_scale;
                float b = fp8_e4m3_to_float((b16.y >> (i * 8)) & 0xFF);
                acc0 = fmaf(a, b, acc0);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = fp8_e4m3_to_float((a16.z >> (i * 8)) & 0xFF) * combined_scale;
                float b = fp8_e4m3_to_float((b16.z >> (i * 8)) & 0xFF);
                acc1 = fmaf(a, b, acc1);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = fp8_e4m3_to_float((a16.w >> (i * 8)) & 0xFF) * combined_scale;
                float b = fp8_e4m3_to_float((b16.w >> (i * 8)) & 0xFF);
                acc1 = fmaf(a, b, acc1);
            }
        }

        // Remainder
        for (int k = K_aligned16 + lane_id; k < K; k += Config::WARP_SIZE) {
            const int scale_k = k / Config::SCALE_BLOCK_SIZE;
            float sA = __ldg(&scale_A[scale_k]);
            float sB = __ldg(&scale_B[scale_n * scale_stride_k + scale_k]);
            float a = fp8_e4m3_to_float(smem_A[k]) * sA;
            float b = fp8_e4m3_to_float(__ldg(&B_row[k])) * sB;
            acc0 = fmaf(a, b, acc0);
        }

        acc[row] = acc0 + acc1;
    }

    // Warp-level reduction for both rows
    #pragma unroll
    for (int row = 0; row < Config::ROWS_PER_WARP; ++row) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);
        }
    }

    // Lane 0 writes results
    if (lane_id == 0) {
        #pragma unroll
        for (int row = 0; row < Config::ROWS_PER_WARP; ++row) {
            const int global_n = global_n_base + row;
            if (global_n < N) {
                C[global_n] = __float2bfloat16(acc[row]);
            }
        }
    }
}

/**
 * FP8 output variant: A[K](FP8) x B[N,K](FP8) -> C[N](FP8)
 */
template<typename Config = GemvFP8PureConfig>
__global__ void gemv_fp8_pure_fp8out_kernel(
    uint8_t const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    float const* __restrict__ scale_A,
    float const* __restrict__ scale_B,
    uint8_t* __restrict__ C,
    float scale_C,
    int K,
    int N
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    extern __shared__ uint8_t smem_A[];

    // Cooperative load
    for (int k = threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    const int scale_stride_k = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;
    const int scale_n = global_n / Config::SCALE_BLOCK_SIZE;
    const uint8_t* B_row = B_nk + global_n * K;

    float acc = 0.0f;

    for (int k = lane_id; k < K; k += Config::WARP_SIZE) {
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;
        float sA = scale_A[scale_k];
        float sB = scale_B[scale_n * scale_stride_k + scale_k];
        float a = fp8_e4m3_to_float(smem_A[k]) * sA;
        float b = fp8_e4m3_to_float(B_row[k]) * sB;
        acc = fmaf(a, b, acc);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        // Quantize output to FP8
        C[global_n] = float_to_fp8_e4m3(acc / scale_C);
    }
}

// ============================================================================
// Launch Function Declarations
// ============================================================================

cudaError_t launch_gemv_fp8_pure(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

cudaError_t launch_gemv_fp8_pure_fp8out(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    uint8_t* C,
    float scale_C,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
