/**
 * Int4 GEMV Kernel (SM120)
 *
 * For M=1 decode in LLM inference with Int4 quantization.
 * Uses warp-level reduction over K dimension.
 *
 * Int4 packed: 2 signed 4-bit values per byte, low nibble first.
 * Sign extension: values in range [-8, 7]
 *
 * Layout:
 * - A: [K/2] packed Int4 (RowMajor activation vector)
 * - B: [N, K/2] packed Int4 (weights, row-major)
 * - C: [N] Int32 output
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Configuration
// ============================================================================

struct GemvInt4Config {
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;  // 256 threads
    static constexpr int WARP_SIZE = 32;
    static constexpr int VEC_SIZE = 4;  // Load 4 bytes (8 Int4 values) at once
};

// ============================================================================
// Helper: Unpack Int4 to Int8 with sign extension
// ============================================================================

__device__ __forceinline__ int8_t unpack_int4_low(uint8_t packed) {
    int8_t val = static_cast<int8_t>(packed << 4) >> 4;  // Sign extend low nibble
    return val;
}

__device__ __forceinline__ int8_t unpack_int4_high(uint8_t packed) {
    int8_t val = static_cast<int8_t>(packed) >> 4;  // Sign extend high nibble
    return val;
}

// ============================================================================
// Int4 x Int4 GEMV with warp-level reduction
// ============================================================================

/**
 * Int4 GEMV with warp-level reduction
 *
 * Each warp handles ONE output element (N dimension)
 * 32 threads in warp cooperatively reduce over K dimension
 *
 * @param A         [K/2] packed Int4 activation vector
 * @param B_nk      [N, K/2] packed Int4 weights
 * @param C         [N] Int32 output
 * @param K         Unpacked K dimension (must be even)
 * @param N         Output dimension
 * @param scale_A   Scale for A (applied to result)
 * @param scale_B   Scale for B (applied to result)
 */
template<typename Config = GemvInt4Config>
__global__ void gemv_int4_warp_reduce_kernel(
    uint8_t const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    int32_t* __restrict__ C,
    int K,
    int N,
    float scale_A,
    float scale_B
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    const int K_packed = K / 2;  // Bytes in packed dimension

    // Shared memory for A (packed)
    extern __shared__ uint8_t smem_A[];

    // Cooperative load of A into shared memory
    for (int k = threadIdx.x; k < K_packed; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // B row pointer for this output
    const uint8_t* B_row = B_nk + global_n * K_packed;

    int32_t acc = 0;

    // Each lane handles packed bytes with stride 32
    // Each byte contains 2 Int4 values
    for (int kp = lane_id; kp < K_packed; kp += Config::WARP_SIZE) {
        // Load packed bytes
        uint8_t a_packed = smem_A[kp];
        uint8_t b_packed = B_row[kp];

        // Unpack to Int8
        int8_t a0 = unpack_int4_low(a_packed);
        int8_t a1 = unpack_int4_high(a_packed);
        int8_t b0 = unpack_int4_low(b_packed);
        int8_t b1 = unpack_int4_high(b_packed);

        // Accumulate as Int32
        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0);
        acc += static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        // Apply scales and round
        float result = static_cast<float>(acc) * scale_A * scale_B;
        C[global_n] = static_cast<int32_t>(roundf(result));
    }
}

/**
 * Vectorized variant: Load 4 packed bytes (8 Int4 values) at once
 */
template<typename Config = GemvInt4Config>
__global__ void gemv_int4_warp_reduce_vec4_kernel(
    uint8_t const* __restrict__ A,
    uint8_t const* __restrict__ B_nk,
    int32_t* __restrict__ C,
    int K,
    int N,
    float scale_A,
    float scale_B
) {
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    const int K_packed = K / 2;

    // Shared memory for A (packed)
    extern __shared__ uint8_t smem_A[];

    // Cooperative load of A into shared memory
    for (int k = threadIdx.x; k < K_packed; k += Config::BLOCK_SIZE) {
        smem_A[k] = A[k];
    }
    __syncthreads();

    // B row pointer for this output
    const uint8_t* B_row = B_nk + global_n * K_packed;

    int32_t acc = 0;

    // Vectorized: each lane handles 4 packed bytes (8 Int4 values) per iteration
    const int K_packed_aligned = K_packed & ~3;  // Round down to multiple of 4

    for (int kp_base = lane_id * 4; kp_base < K_packed_aligned; kp_base += Config::WARP_SIZE * 4) {
        // Vectorized load of 4 packed bytes
        uint32_t a4 = *reinterpret_cast<const uint32_t*>(smem_A + kp_base);
        uint32_t b4 = *reinterpret_cast<const uint32_t*>(B_row + kp_base);

        // Process 4 bytes (8 Int4 pairs)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t a_packed = (a4 >> (i * 8)) & 0xFF;
            uint8_t b_packed = (b4 >> (i * 8)) & 0xFF;

            int8_t a0 = unpack_int4_low(a_packed);
            int8_t a1 = unpack_int4_high(a_packed);
            int8_t b0 = unpack_int4_low(b_packed);
            int8_t b1 = unpack_int4_high(b_packed);

            acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0);
            acc += static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
        }
    }

    // Handle remainder
    for (int kp = K_packed_aligned + lane_id; kp < K_packed; kp += Config::WARP_SIZE) {
        uint8_t a_packed = smem_A[kp];
        uint8_t b_packed = B_row[kp];

        int8_t a0 = unpack_int4_low(a_packed);
        int8_t a1 = unpack_int4_high(a_packed);
        int8_t b0 = unpack_int4_low(b_packed);
        int8_t b1 = unpack_int4_high(b_packed);

        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0);
        acc += static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        float result = static_cast<float>(acc) * scale_A * scale_B;
        C[global_n] = static_cast<int32_t>(roundf(result));
    }
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
