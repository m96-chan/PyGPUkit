/**
 * Pure NVF4/NVF4/NVF4 GEMV Kernel (SM120)
 *
 * A[K] (NVF4) x B[N,K] (NVF4) -> C[N] (BF16)
 *
 * Key advantage over W4A16 GEMV:
 * - A is NVF4 (0.5 bytes) instead of BF16 (2 bytes)
 * - Shared memory requirement: K/2 bytes vs K*2 bytes (4x reduction!)
 * - Supports K up to 96K without shared memory overflow
 *
 * Memory layout (ROW-MAJOR B for coalesced access):
 * - A_data: [K/2] packed NVF4 (2 values per byte)
 * - A_scale: [K/32] UE4M3 scale factors
 * - B_data: [N, K/2] packed NVF4 (row-major, contiguous K for each N)
 * - B_scale: [N, K/32] UE4M3 scale factors (row-major)
 * - C: [N] BF16 output
 *
 * Use quantize_bf16_to_nvf4_rowmajor() to create B in this layout.
 *
 * Optimizations:
 * 1. Warp-level reduction over K dimension
 * 2. Shared memory for A (NVF4 packed)
 * 3. LUT-based dequantization (constant memory)
 * 4. Vectorized loads (uint64 = 16 NVF4 values)
 * 5. Multiple accumulators
 * 6. Row-major B layout for coalesced memory access
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace gemv_nvf4_pure {

// ============================================================================
// NVF4 Dequantization (from existing implementation)
// ============================================================================

// NVF4 E2M1 lookup table (4-bit -> float)
__device__ __constant__ float NVF4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,     // 0-7: positive
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // 8-15: negative
};

// UE4M3 scale factor lookup table
__device__ __constant__ float UE4M3_SCALE_LUT[256] = {
    // exp=0-15 (128 entries)
    0.0078125f, 0.0087890625f, 0.009765625f, 0.0107421875f, 0.01171875f, 0.0126953125f, 0.013671875f, 0.0146484375f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 480.0f,
    // Mirror for bit 7 set (128-255)
    0.0078125f, 0.0087890625f, 0.009765625f, 0.0107421875f, 0.01171875f, 0.0126953125f, 0.013671875f, 0.0146484375f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 480.0f,
};

__device__ __forceinline__ float decode_ue4m3_scale(uint8_t ue4m3) {
    return UE4M3_SCALE_LUT[ue4m3];
}

// Dequantize single NVF4 value
__device__ __forceinline__ float dequant_nvf4(uint8_t nvf4_val) {
    return NVF4_LUT[nvf4_val & 0x0F];
}

// ============================================================================
// Configuration
// ============================================================================

struct GemvNvf4PureConfig {
    static constexpr int BLOCK_SIZE = 256;      // Threads per block
    static constexpr int TILE_N = 256;          // Output elements per block (1 thread = 1 output)
    static constexpr int SCALE_BLOCK_SIZE = 32; // NVF4 uses 32-element blocks
};

// ============================================================================
// Pure NVF4 GEMV Kernel: A[K](NVF4) x B[K,N](NVF4) -> C[N](BF16)
// ============================================================================

/**
 * Pure NVF4 GEMV with 1 thread = 1 output pattern (like W4A16)
 *
 * Each thread handles ONE output element, loops over all K
 * Uses pre-scaled LUT in registers for efficient dequantization
 *
 * Memory layout (ROW-MAJOR for B - contiguous K for coalesced access):
 * - A_data: [K/2] packed NVF4 (2 values per byte)
 * - A_scale: [K/32] UE4M3 scale factors
 * - B_data: [N, K/2] packed NVF4 (row-major: contiguous K for each N)
 * - B_scale: [N, K/32] UE4M3 scale factors (row-major)
 * - C: [N] BF16 output vector
 */
template<typename Config = GemvNvf4PureConfig>
__global__ void gemv_nvf4_pure_kernel(
    uint8_t const* __restrict__ A_data,
    uint8_t const* __restrict__ A_scale,
    uint8_t const* __restrict__ B_data,
    uint8_t const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    const int K_packed = K / 2;
    const int K_scale_blocks = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    // B row pointers (row-major: contiguous K for each N)
    const uint8_t* B_row = B_data + global_n * K_packed;
    const uint8_t* B_scale_row = B_scale + global_n * K_scale_blocks;

    float acc = 0.0f;

    // Process in scale blocks (32 elements = 16 packed bytes per block)
    for (int sb = 0; sb < K_scale_blocks; ++sb) {
        // Load scale factors for this block
        float sA = decode_ue4m3_scale(A_scale[sb]);
        float sB = decode_ue4m3_scale(__ldg(&B_scale_row[sb]));
        float combined_scale = sA * sB;

        // Pre-compute scaled LUT in registers (16 values)
        // NVF4 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (positive)
        //              0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 (negative)
        float lut[16];
        lut[0]  = 0.0f;
        lut[1]  = 0.5f * combined_scale;
        lut[2]  = 1.0f * combined_scale;
        lut[3]  = 1.5f * combined_scale;
        lut[4]  = 2.0f * combined_scale;
        lut[5]  = 3.0f * combined_scale;
        lut[6]  = 4.0f * combined_scale;
        lut[7]  = 6.0f * combined_scale;
        lut[8]  = 0.0f;
        lut[9]  = -0.5f * combined_scale;
        lut[10] = -1.0f * combined_scale;
        lut[11] = -1.5f * combined_scale;
        lut[12] = -2.0f * combined_scale;
        lut[13] = -3.0f * combined_scale;
        lut[14] = -4.0f * combined_scale;
        lut[15] = -6.0f * combined_scale;

        int k_start = sb * Config::SCALE_BLOCK_SIZE;
        int k_end = min(k_start + Config::SCALE_BLOCK_SIZE, K);
        int k_packed_start = k_start / 2;
        int k_packed_end = k_end / 2;

        // Process pairs (2 NVF4 values per byte)
        #pragma unroll 4
        for (int kp = k_packed_start; kp < k_packed_end; ++kp) {
            // Load packed bytes
            uint8_t a_packed = A_data[kp];
            uint8_t b_packed = __ldg(&B_row[kp]);

            // Dequantize using pre-scaled LUT (product of dequantized values)
            // Result = (a_raw * sA) * (b_raw * sB) = a_raw * b_raw * combined_scale
            float a0 = NVF4_LUT[a_packed & 0x0F];
            float a1 = NVF4_LUT[(a_packed >> 4) & 0x0F];
            float b0 = lut[b_packed & 0x0F];
            float b1 = lut[(b_packed >> 4) & 0x0F];

            // Accumulate: a * (b * combined_scale) = a * b * sA * sB
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
        }
    }

    C[global_n] = __float2bfloat16(acc);
}

/**
 * Optimized variant with full unrolling per scale block (like W4A16)
 *
 * 1 thread = 1 output, pre-scaled LUT in registers
 * Unrolled inner loop for better instruction scheduling
 */
template<typename Config = GemvNvf4PureConfig>
__global__ void gemv_nvf4_pure_opt_kernel(
    uint8_t const* __restrict__ A_data,
    uint8_t const* __restrict__ A_scale,
    uint8_t const* __restrict__ B_data,
    uint8_t const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    const int K_packed = K / 2;
    const int num_scale_blocks = K / Config::SCALE_BLOCK_SIZE;
    const int K_remainder = K % Config::SCALE_BLOCK_SIZE;
    const int K_scale_blocks = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    // B row pointers (row-major: contiguous K for each N)
    const uint8_t* B_row = B_data + global_n * K_packed;
    const uint8_t* B_scale_row = B_scale + global_n * K_scale_blocks;

    float acc = 0.0f;

    // Main loop: process complete scale blocks with full unroll
    for (int sb = 0; sb < num_scale_blocks; ++sb) {
        // Load scale factors for this block
        float sA = decode_ue4m3_scale(A_scale[sb]);
        float sB = decode_ue4m3_scale(__ldg(&B_scale_row[sb]));
        float combined_scale = sA * sB;

        // Pre-compute scaled LUT in registers
        float lut0  = 0.0f;
        float lut1  = 0.5f * combined_scale;
        float lut2  = 1.0f * combined_scale;
        float lut3  = 1.5f * combined_scale;
        float lut4  = 2.0f * combined_scale;
        float lut5  = 3.0f * combined_scale;
        float lut6  = 4.0f * combined_scale;
        float lut7  = 6.0f * combined_scale;
        float lut8  = 0.0f;
        float lut9  = -0.5f * combined_scale;
        float lut10 = -1.0f * combined_scale;
        float lut11 = -1.5f * combined_scale;
        float lut12 = -2.0f * combined_scale;
        float lut13 = -3.0f * combined_scale;
        float lut14 = -4.0f * combined_scale;
        float lut15 = -6.0f * combined_scale;

        float lut[16] = {lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7,
                         lut8, lut9, lut10, lut11, lut12, lut13, lut14, lut15};

        int k_packed_base = sb * (Config::SCALE_BLOCK_SIZE / 2);

        // Process 32 elements (16 packed bytes) with full unroll
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            // Load 4 packed bytes from A and B
            uint8_t a0 = A_data[k_packed_base + i + 0];
            uint8_t a1 = A_data[k_packed_base + i + 1];
            uint8_t a2 = A_data[k_packed_base + i + 2];
            uint8_t a3 = A_data[k_packed_base + i + 3];

            uint8_t b0 = __ldg(&B_row[k_packed_base + i + 0]);
            uint8_t b1 = __ldg(&B_row[k_packed_base + i + 1]);
            uint8_t b2 = __ldg(&B_row[k_packed_base + i + 2]);
            uint8_t b3 = __ldg(&B_row[k_packed_base + i + 3]);

            // Dequantize A from constant LUT, B from pre-scaled register LUT
            float da0_0 = NVF4_LUT[a0 & 0x0F];
            float da0_1 = NVF4_LUT[(a0 >> 4) & 0x0F];
            float da1_0 = NVF4_LUT[a1 & 0x0F];
            float da1_1 = NVF4_LUT[(a1 >> 4) & 0x0F];
            float da2_0 = NVF4_LUT[a2 & 0x0F];
            float da2_1 = NVF4_LUT[(a2 >> 4) & 0x0F];
            float da3_0 = NVF4_LUT[a3 & 0x0F];
            float da3_1 = NVF4_LUT[(a3 >> 4) & 0x0F];

            float db0_0 = lut[b0 & 0x0F];
            float db0_1 = lut[(b0 >> 4) & 0x0F];
            float db1_0 = lut[b1 & 0x0F];
            float db1_1 = lut[(b1 >> 4) & 0x0F];
            float db2_0 = lut[b2 & 0x0F];
            float db2_1 = lut[(b2 >> 4) & 0x0F];
            float db3_0 = lut[b3 & 0x0F];
            float db3_1 = lut[(b3 >> 4) & 0x0F];

            // Accumulate
            acc = fmaf(da0_0, db0_0, acc);
            acc = fmaf(da0_1, db0_1, acc);
            acc = fmaf(da1_0, db1_0, acc);
            acc = fmaf(da1_1, db1_1, acc);
            acc = fmaf(da2_0, db2_0, acc);
            acc = fmaf(da2_1, db2_1, acc);
            acc = fmaf(da3_0, db3_0, acc);
            acc = fmaf(da3_1, db3_1, acc);
        }
    }

    // Handle remainder (if K is not multiple of SCALE_BLOCK_SIZE)
    if (K_remainder > 0) {
        int sb = num_scale_blocks;
        float sA = decode_ue4m3_scale(A_scale[sb]);
        float sB = decode_ue4m3_scale(__ldg(&B_scale_row[sb]));
        float combined_scale = sA * sB;

        float lut[16];
        lut[0]  = 0.0f;
        lut[1]  = 0.5f * combined_scale;
        lut[2]  = 1.0f * combined_scale;
        lut[3]  = 1.5f * combined_scale;
        lut[4]  = 2.0f * combined_scale;
        lut[5]  = 3.0f * combined_scale;
        lut[6]  = 4.0f * combined_scale;
        lut[7]  = 6.0f * combined_scale;
        lut[8]  = 0.0f;
        lut[9]  = -0.5f * combined_scale;
        lut[10] = -1.0f * combined_scale;
        lut[11] = -1.5f * combined_scale;
        lut[12] = -2.0f * combined_scale;
        lut[13] = -3.0f * combined_scale;
        lut[14] = -4.0f * combined_scale;
        lut[15] = -6.0f * combined_scale;

        int k_packed_base = sb * (Config::SCALE_BLOCK_SIZE / 2);
        int k_packed_end = K_packed;

        for (int kp = k_packed_base; kp < k_packed_end; ++kp) {
            uint8_t a_packed = A_data[kp];
            uint8_t b_packed = __ldg(&B_row[kp]);

            float a0 = NVF4_LUT[a_packed & 0x0F];
            float a1 = NVF4_LUT[(a_packed >> 4) & 0x0F];
            float b0 = lut[b_packed & 0x0F];
            float b1 = lut[(b_packed >> 4) & 0x0F];

            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
        }
    }

    C[global_n] = __float2bfloat16(acc);
}

// ============================================================================
// Launch Function Declarations
// ============================================================================

cudaError_t launch_gemv_nvf4_pure(
    const uint8_t* A_data,
    const uint8_t* A_scale,
    const uint8_t* B_data,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

}  // namespace gemv_nvf4_pure
}  // namespace ops
}  // namespace pygpukit
