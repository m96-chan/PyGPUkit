/**
 * Optimized FP8 GEMV Kernel
 *
 * Optimizations:
 * 1. Warp-level reduction over K dimension (32 threads per output)
 * 2. Shared memory for activation vector A
 * 3. Vectorized uint4 loads (4 FP8 values at once)
 * 4. Coalesced memory access pattern
 *
 * Layout: B[N, K] (row-major, each row is one output's weights)
 * This enables coalesced loads when threads read consecutive K values.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Include fp8.cuh for FP8_E4M3_LUT definition
#include "fp8.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Configuration
// ============================================================================

struct GemvFP8OptConfig {
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;  // 256 threads
    static constexpr int WARP_SIZE = 32;
    static constexpr int VEC_SIZE = 4;  // Load 4 FP8 values at once
    static constexpr int BLOCK_QUANT_SIZE = 128;
};

// ============================================================================
// Optimized Kernel: Warp-level reduction
// ============================================================================

/**
 * Optimized FP8 GEMV with warp-level reduction
 *
 * Each warp handles ONE output element (N dimension)
 * 32 threads in warp cooperatively reduce over K dimension
 *
 * Memory layout:
 * - A: [K] activation vector (BF16)
 * - B: [N, K] transposed weight matrix (FP8), row-major
 * - B_scale: [N/128, K/128] block-wise scales (BF16)
 * - C: [N] output vector (BF16)
 *
 * @param A       [K] BF16 activation vector
 * @param B_nk    [N, K] FP8 weights (transposed, row = output)
 * @param B_scale [N/128, K/128] BF16 scales
 * @param C       [N] BF16 output
 * @param K       Inner dimension
 * @param N       Output dimension
 */
template<typename Config = GemvFP8OptConfig>
__global__ void gemv_fp8_warp_reduce_kernel(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    __nv_bfloat16 const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    // Shared memory for A (sized dynamically)
    extern __shared__ __nv_bfloat16 smem_A[];

    // Cooperative load of A into shared memory
    for (int k = threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // Scale dimensions
    const int scale_stride_k = (K + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;
    const int scale_n = global_n / Config::BLOCK_QUANT_SIZE;

    // B row pointer for this output
    const uint8_t* B_row = B_nk + global_n * K;

    float acc = 0.0f;

    // Each lane handles K elements with stride 32
    // lane 0: k=0,32,64,...
    // lane 1: k=1,33,65,...
    // etc.
    for (int k = lane_id; k < K; k += Config::WARP_SIZE) {
        // Load scale (changes every 128 elements)
        const int scale_k = k / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_n * scale_stride_k + scale_k]);

        // Load activation from shared memory
        float a = __bfloat162float(smem_A[k]);

        // Load FP8 weight (coalesced: consecutive lanes read consecutive addresses)
        uint8_t b_fp8 = B_row[k];
        float b = FP8_E4M3_LUT[b_fp8] * scale;

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
 * Vectorized variant: Load 4 FP8 values at once
 *
 * Better for large K dimensions.
 * Requires K to be aligned to 4.
 */
template<typename Config = GemvFP8OptConfig>
__global__ void gemv_fp8_warp_reduce_vec4_kernel(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    __nv_bfloat16 const* __restrict__ B_scale,
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

    // Cooperative load of A into shared memory
    for (int k = threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // Scale dimensions
    const int scale_stride_k = (K + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;
    const int scale_n = global_n / Config::BLOCK_QUANT_SIZE;

    // B row pointer for this output
    const uint8_t* B_row = B_nk + global_n * K;

    float acc = 0.0f;

    // Vectorized: each lane handles 4 elements per iteration
    // Total K elements processed per iteration: 32 lanes * 4 = 128
    const int K_aligned = K & ~3;  // Round down to multiple of 4

    for (int k_base = lane_id * 4; k_base < K_aligned; k_base += Config::WARP_SIZE * 4) {
        // Load scale
        const int scale_k = k_base / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_n * scale_stride_k + scale_k]);

        // Vectorized load of 4 FP8 values
        uint32_t b4 = *reinterpret_cast<const uint32_t*>(B_row + k_base);
        uint8_t b0 = (b4 >> 0) & 0xFF;
        uint8_t b1 = (b4 >> 8) & 0xFF;
        uint8_t b2 = (b4 >> 16) & 0xFF;
        uint8_t b3 = (b4 >> 24) & 0xFF;

        // Load 4 activations
        float a0 = __bfloat162float(smem_A[k_base + 0]);
        float a1 = __bfloat162float(smem_A[k_base + 1]);
        float a2 = __bfloat162float(smem_A[k_base + 2]);
        float a3 = __bfloat162float(smem_A[k_base + 3]);

        // Dequantize and accumulate
        acc = fmaf(a0, FP8_E4M3_LUT[b0] * scale, acc);
        acc = fmaf(a1, FP8_E4M3_LUT[b1] * scale, acc);
        acc = fmaf(a2, FP8_E4M3_LUT[b2] * scale, acc);
        acc = fmaf(a3, FP8_E4M3_LUT[b3] * scale, acc);
    }

    // Handle remainder (K not divisible by 4)
    for (int k = K_aligned + lane_id; k < K; k += Config::WARP_SIZE) {
        const int scale_k = k / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_n * scale_stride_k + scale_k]);
        float a = __bfloat162float(smem_A[k]);
        float b = FP8_E4M3_LUT[B_row[k]] * scale;
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
 * Batched optimized GEMV
 *
 * C[batch, N] = A[batch, K] @ B[N, K]^T
 */
template<typename Config = GemvFP8OptConfig>
__global__ void gemv_fp8_warp_reduce_batched_kernel(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    __nv_bfloat16 const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N,
    int batch_count
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;
    const int batch_idx = blockIdx.y;

    if (global_n >= N || batch_idx >= batch_count) return;

    // Pointers for this batch
    const __nv_bfloat16* A_batch = A + batch_idx * K;
    __nv_bfloat16* C_batch = C + batch_idx * N;

    // Shared memory for A (per batch in block)
    extern __shared__ __nv_bfloat16 smem_A[];

    // Cooperative load of A
    for (int k = threadIdx.x; k < K; k += Config::BLOCK_SIZE) {
        smem_A[k] = A_batch[k];
    }
    __syncthreads();

    // Scale dimensions
    const int scale_stride_k = (K + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;
    const int scale_n = global_n / Config::BLOCK_QUANT_SIZE;

    const uint8_t* B_row = B_nk + global_n * K;

    float acc = 0.0f;
    const int K_aligned = K & ~3;

    for (int k_base = lane_id * 4; k_base < K_aligned; k_base += Config::WARP_SIZE * 4) {
        const int scale_k = k_base / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_n * scale_stride_k + scale_k]);

        uint32_t b4 = *reinterpret_cast<const uint32_t*>(B_row + k_base);
        uint8_t b0 = (b4 >> 0) & 0xFF;
        uint8_t b1 = (b4 >> 8) & 0xFF;
        uint8_t b2 = (b4 >> 16) & 0xFF;
        uint8_t b3 = (b4 >> 24) & 0xFF;

        float a0 = __bfloat162float(smem_A[k_base + 0]);
        float a1 = __bfloat162float(smem_A[k_base + 1]);
        float a2 = __bfloat162float(smem_A[k_base + 2]);
        float a3 = __bfloat162float(smem_A[k_base + 3]);

        acc = fmaf(a0, FP8_E4M3_LUT[b0] * scale, acc);
        acc = fmaf(a1, FP8_E4M3_LUT[b1] * scale, acc);
        acc = fmaf(a2, FP8_E4M3_LUT[b2] * scale, acc);
        acc = fmaf(a3, FP8_E4M3_LUT[b3] * scale, acc);
    }

    for (int k = K_aligned + lane_id; k < K; k += Config::WARP_SIZE) {
        const int scale_k = k / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_n * scale_stride_k + scale_k]);
        float a = __bfloat162float(smem_A[k]);
        float b = FP8_E4M3_LUT[B_row[k]] * scale;
        acc = fmaf(a, b, acc);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C_batch[global_n] = __float2bfloat16(acc);
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_fp8_opt(
    const __nv_bfloat16* A,
    const uint8_t* B_nk,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

cudaError_t launch_gemv_fp8_opt_batched(
    const __nv_bfloat16* A,
    const uint8_t* B_nk,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    int batch_count,
    cudaStream_t stream = nullptr
);

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
