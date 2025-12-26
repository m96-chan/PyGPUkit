/**
 * NVF4 GEMV Kernel for SM120 (Blackwell GeForce) with BF16 I/O
 *
 * Purpose: Memory-efficient GEMV for LLM inference decode path
 *
 * Data flow:
 *   A[1,K] (BF16) x B[K,N] (NVF4 + scale) -> C[1,N] (BF16)
 *
 * NVF4 (float_e2m1_t) format:
 * - 4-bit per element (2 elements per byte)
 * - Values: 0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6
 * - Block scaling: 32 elements share one scale factor (float_ue4m3_t)
 *
 * Memory layout:
 * - B_data: [K, N/2] packed NVF4 (column-major for coalesced access)
 * - B_scale: [K/32, N] scale factors (one per 32-element block along K)
 *
 * Advantages over BF16 GEMV:
 * - 4x less memory bandwidth for weights
 * - Better cache utilization
 * - Ideal for memory-bound M=1 decode
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace gemv_nvf4 {

// ============================================================================
// NVF4 Dequantization
// ============================================================================

// NVF4 E2M1 lookup table (4-bit -> float)
// Index 0-7: positive values, 8-15: negative values
__device__ __constant__ float NVF4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,   // 0-7: positive
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // 8-15: negative (sign bit)
};

// Dequantize NVF4 value using lookup table
__device__ __forceinline__ float dequant_nvf4(uint8_t nvf4_val) {
    return NVF4_LUT[nvf4_val & 0x0F];
}

// Dequantize packed byte (2 NVF4 values) and apply scale
__device__ __forceinline__ void dequant_nvf4x2(
    uint8_t packed,
    float scale,
    float& out0,
    float& out1
) {
    out0 = NVF4_LUT[packed & 0x0F] * scale;
    out1 = NVF4_LUT[(packed >> 4) & 0x0F] * scale;
}

// UE4M3 scale factor lookup table (256 entries for direct byte indexing)
// UE4M3: 4-bit unsigned exponent (bits 3-6), 3-bit mantissa (bits 0-2)
// Value = (1 + mantissa/8) * 2^(exponent - 7)
// Note: bit 7 is unused, so entries 128-255 mirror 0-127
__device__ __constant__ float UE4M3_SCALE_LUT[256] = {
    // exp=0: 2^(-7) = 0.0078125
    0.0078125f, 0.0087890625f, 0.009765625f, 0.0107421875f, 0.01171875f, 0.0126953125f, 0.013671875f, 0.0146484375f,
    // exp=1: 2^(-6) = 0.015625
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    // exp=2: 2^(-5) = 0.03125
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    // exp=3: 2^(-4) = 0.0625
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    // exp=4: 2^(-3) = 0.125
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    // exp=5: 2^(-2) = 0.25
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    // exp=6: 2^(-1) = 0.5
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    // exp=7: 2^0 = 1.0
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    // exp=8: 2^1 = 2.0
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    // exp=9: 2^2 = 4.0
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    // exp=10: 2^3 = 8.0
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    // exp=11: 2^4 = 16.0
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    // exp=12: 2^5 = 32.0
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    // exp=13: 2^6 = 64.0
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    // exp=14: 2^7 = 128.0
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    // exp=15: 2^8 = 256.0
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

// Fast UE4M3 scale decode using LUT (single memory access)
__device__ __forceinline__ float decode_ue4m3_scale(uint8_t ue4m3) {
    return UE4M3_SCALE_LUT[ue4m3];
}

// ============================================================================
// Configuration
// ============================================================================

struct GemvNvf4Config {
    static constexpr int BLOCK_SIZE = 256;  // Threads per block
    static constexpr int TILE_N = 256;      // Output elements per block
    static constexpr int UNROLL_K = 8;      // K-loop unrolling (must be multiple of 2)
    static constexpr int SCALE_BLOCK = 32;  // Elements per scale factor
};

// ============================================================================
// NVF4 GEMV Kernel
// ============================================================================

/**
 * GEMV kernel: C[1,N] = A[1,K] @ B[K,N] where B is NVF4 quantized
 *
 * Memory layout:
 * - A: [K] BF16 contiguous (input vector)
 * - B_data: [K/2, N] packed NVF4 (2 elements per byte, row-major)
 *   B_data[k/2, n] contains B[k, n] (low nibble) and B[k+1, n] (high nibble)
 * - B_scale: [K/32, N] UE4M3 scale factors
 * - C: [N] BF16 output
 */
template<typename Config = GemvNvf4Config>
__global__ void gemv_nvf4_bf16_kernel(
    __nv_bfloat16 const* __restrict__ A,      // [K] BF16
    uint8_t const* __restrict__ B_data,        // [K/2, N] packed NVF4
    uint8_t const* __restrict__ B_scale,       // [K/32, N] UE4M3 scales
    __nv_bfloat16* __restrict__ C,             // [N] BF16 output
    int K,
    int N,
    float alpha
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    float acc = 0.0f;

    // Base pointers for this thread's column
    const uint8_t* B_col = B_data + global_n;    // B_data[0, global_n]
    const uint8_t* S_col = B_scale + global_n;   // B_scale[0, global_n]

    const int K_packed = K / 2;  // Packed dimension
    const int num_scale_blocks = (K + Config::SCALE_BLOCK - 1) / Config::SCALE_BLOCK;

    // Process in scale blocks (32 elements = 16 packed bytes per block)
    for (int sb = 0; sb < num_scale_blocks; ++sb) {
        // Load scale factor for this block
        float scale = decode_ue4m3_scale(__ldg(S_col + sb * N));

        int k_start = sb * Config::SCALE_BLOCK;
        int k_end = min(k_start + Config::SCALE_BLOCK, K);

        // Process pairs (2 NVF4 values per byte)
        for (int k = k_start; k < k_end; k += 2) {
            int k_packed = k / 2;

            // Load packed NVF4 byte
            uint8_t packed = __ldg(B_col + k_packed * N);

            // Dequantize
            float b0, b1;
            dequant_nvf4x2(packed, scale, b0, b1);

            // Load A values
            float a0 = __bfloat162float(A[k]);
            float a1 = (k + 1 < K) ? __bfloat162float(A[k + 1]) : 0.0f;

            // Accumulate
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
        }
    }

    // Apply alpha and store
    C[global_n] = __float2bfloat16(alpha * acc);
}

/**
 * Optimized kernel with register-cached scaled LUT
 *
 * Key optimization:
 * - Pre-compute scaled LUT values once per scale block (16 regs)
 * - Eliminates per-value multiply by scale
 * - Unrolled inner loop for ILP
 */
template<typename Config = GemvNvf4Config>
__global__ void gemv_nvf4_bf16_kernel_unrolled(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_data,
    uint8_t const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N,
    float alpha
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    float acc = 0.0f;

    const uint8_t* B_col = B_data + global_n;
    const uint8_t* S_col = B_scale + global_n;

    const int num_scale_blocks = K / Config::SCALE_BLOCK;
    const int K_remainder = K % Config::SCALE_BLOCK;

    // Main loop: process complete scale blocks
    for (int sb = 0; sb < num_scale_blocks; ++sb) {
        int k_base = sb * Config::SCALE_BLOCK;

        // Load and decode scale factor
        float scale = decode_ue4m3_scale(__ldg(S_col + sb * N));

        // Pre-compute scaled LUT in registers (16 values)
        // This eliminates 32 multiplies per scale block (saves 16 net)
        float lut0  = 0.0f;                // NVF4_LUT[0] * scale
        float lut1  = 0.5f * scale;        // NVF4_LUT[1] * scale
        float lut2  = 1.0f * scale;        // NVF4_LUT[2] * scale
        float lut3  = 1.5f * scale;        // NVF4_LUT[3] * scale
        float lut4  = 2.0f * scale;        // NVF4_LUT[4] * scale
        float lut5  = 3.0f * scale;        // NVF4_LUT[5] * scale
        float lut6  = 4.0f * scale;        // NVF4_LUT[6] * scale
        float lut7  = 6.0f * scale;        // NVF4_LUT[7] * scale
        float lut8  = 0.0f;                // NVF4_LUT[8] * scale (neg zero)
        float lut9  = -0.5f * scale;       // NVF4_LUT[9] * scale
        float lut10 = -1.0f * scale;       // NVF4_LUT[10] * scale
        float lut11 = -1.5f * scale;       // NVF4_LUT[11] * scale
        float lut12 = -2.0f * scale;       // NVF4_LUT[12] * scale
        float lut13 = -3.0f * scale;       // NVF4_LUT[13] * scale
        float lut14 = -4.0f * scale;       // NVF4_LUT[14] * scale
        float lut15 = -6.0f * scale;       // NVF4_LUT[15] * scale

        // Pack into array for indexed access
        float scaled_lut[16] = {
            lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7,
            lut8, lut9, lut10, lut11, lut12, lut13, lut14, lut15
        };

        int k_packed_base = k_base / 2;

        // Process 32 elements (16 packed bytes) with full unroll
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            // Load 4 packed bytes
            uint8_t p0 = __ldg(B_col + (k_packed_base + i + 0) * N);
            uint8_t p1 = __ldg(B_col + (k_packed_base + i + 1) * N);
            uint8_t p2 = __ldg(B_col + (k_packed_base + i + 2) * N);
            uint8_t p3 = __ldg(B_col + (k_packed_base + i + 3) * N);

            // Dequantize using pre-scaled LUT (no per-value multiply)
            float b0 = scaled_lut[p0 & 0x0F];
            float b1 = scaled_lut[(p0 >> 4) & 0x0F];
            float b2 = scaled_lut[p1 & 0x0F];
            float b3 = scaled_lut[(p1 >> 4) & 0x0F];
            float b4 = scaled_lut[p2 & 0x0F];
            float b5 = scaled_lut[(p2 >> 4) & 0x0F];
            float b6 = scaled_lut[p3 & 0x0F];
            float b7 = scaled_lut[(p3 >> 4) & 0x0F];

            // Load A values (L1 cache should hit well)
            int a_idx = k_base + i * 2;
            float a0 = __bfloat162float(A[a_idx + 0]);
            float a1 = __bfloat162float(A[a_idx + 1]);
            float a2 = __bfloat162float(A[a_idx + 2]);
            float a3 = __bfloat162float(A[a_idx + 3]);
            float a4 = __bfloat162float(A[a_idx + 4]);
            float a5 = __bfloat162float(A[a_idx + 5]);
            float a6 = __bfloat162float(A[a_idx + 6]);
            float a7 = __bfloat162float(A[a_idx + 7]);

            // Accumulate with FMA
            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
            acc = fmaf(a2, b2, acc);
            acc = fmaf(a3, b3, acc);
            acc = fmaf(a4, b4, acc);
            acc = fmaf(a5, b5, acc);
            acc = fmaf(a6, b6, acc);
            acc = fmaf(a7, b7, acc);
        }
    }

    // Handle remainder (if K is not multiple of SCALE_BLOCK)
    if (K_remainder > 0) {
        int sb = num_scale_blocks;
        int k_base = sb * Config::SCALE_BLOCK;

        float scale = decode_ue4m3_scale(__ldg(S_col + sb * N));

        for (int k = 0; k < K_remainder; k += 2) {
            int k_packed = (k_base + k) / 2;
            uint8_t packed = __ldg(B_col + k_packed * N);

            float b0 = NVF4_LUT[packed & 0x0F] * scale;
            float b1 = NVF4_LUT[(packed >> 4) & 0x0F] * scale;

            float a0 = __bfloat162float(A[k_base + k]);
            float a1 = (k + 1 < K_remainder) ? __bfloat162float(A[k_base + k + 1]) : 0.0f;

            acc = fmaf(a0, b0, acc);
            acc = fmaf(a1, b1, acc);
        }
    }

    C[global_n] = __float2bfloat16(alpha * acc);
}

/**
 * Optimized kernel with 2 outputs per thread
 *
 * Key optimization:
 * - Each thread computes 2 output columns
 * - A vector loads shared between both columns
 * - Higher arithmetic intensity, better ILP
 */
template<int COLS_PER_THREAD = 2, typename Config = GemvNvf4Config>
__global__ void gemv_nvf4_bf16_kernel_multi(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_data,
    uint8_t const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N,
    float alpha
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N * COLS_PER_THREAD;
    const int global_n0 = block_n + tid;
    const int global_n1 = global_n0 + Config::TILE_N;

    const bool valid0 = (global_n0 < N);
    const bool valid1 = (global_n1 < N);

    if (!valid0 && !valid1) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const uint8_t* B_col0 = B_data + global_n0;
    const uint8_t* B_col1 = B_data + global_n1;
    const uint8_t* S_col0 = B_scale + global_n0;
    const uint8_t* S_col1 = B_scale + global_n1;

    const int num_scale_blocks = K / Config::SCALE_BLOCK;

    // Main loop: process complete scale blocks
    for (int sb = 0; sb < num_scale_blocks; ++sb) {
        int k_base = sb * Config::SCALE_BLOCK;

        // Load scales for both columns
        float scale0 = valid0 ? decode_ue4m3_scale(__ldg(S_col0 + sb * N)) : 0.0f;
        float scale1 = valid1 ? decode_ue4m3_scale(__ldg(S_col1 + sb * N)) : 0.0f;

        int k_packed_base = k_base / 2;

        // Process 32 elements (16 packed bytes) with full unroll
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            // Load A values once (shared between both columns)
            int a_idx = k_base + i * 2;
            float a0 = __bfloat162float(A[a_idx + 0]);
            float a1 = __bfloat162float(A[a_idx + 1]);
            float a2 = __bfloat162float(A[a_idx + 2]);
            float a3 = __bfloat162float(A[a_idx + 3]);
            float a4 = __bfloat162float(A[a_idx + 4]);
            float a5 = __bfloat162float(A[a_idx + 5]);
            float a6 = __bfloat162float(A[a_idx + 6]);
            float a7 = __bfloat162float(A[a_idx + 7]);

            // Process column 0
            if (valid0) {
                uint8_t p0 = __ldg(B_col0 + (k_packed_base + i + 0) * N);
                uint8_t p1 = __ldg(B_col0 + (k_packed_base + i + 1) * N);
                uint8_t p2 = __ldg(B_col0 + (k_packed_base + i + 2) * N);
                uint8_t p3 = __ldg(B_col0 + (k_packed_base + i + 3) * N);

                acc0 = fmaf(a0, NVF4_LUT[p0 & 0x0F] * scale0, acc0);
                acc0 = fmaf(a1, NVF4_LUT[(p0 >> 4) & 0x0F] * scale0, acc0);
                acc0 = fmaf(a2, NVF4_LUT[p1 & 0x0F] * scale0, acc0);
                acc0 = fmaf(a3, NVF4_LUT[(p1 >> 4) & 0x0F] * scale0, acc0);
                acc0 = fmaf(a4, NVF4_LUT[p2 & 0x0F] * scale0, acc0);
                acc0 = fmaf(a5, NVF4_LUT[(p2 >> 4) & 0x0F] * scale0, acc0);
                acc0 = fmaf(a6, NVF4_LUT[p3 & 0x0F] * scale0, acc0);
                acc0 = fmaf(a7, NVF4_LUT[(p3 >> 4) & 0x0F] * scale0, acc0);
            }

            // Process column 1
            if (valid1) {
                uint8_t p0 = __ldg(B_col1 + (k_packed_base + i + 0) * N);
                uint8_t p1 = __ldg(B_col1 + (k_packed_base + i + 1) * N);
                uint8_t p2 = __ldg(B_col1 + (k_packed_base + i + 2) * N);
                uint8_t p3 = __ldg(B_col1 + (k_packed_base + i + 3) * N);

                acc1 = fmaf(a0, NVF4_LUT[p0 & 0x0F] * scale1, acc1);
                acc1 = fmaf(a1, NVF4_LUT[(p0 >> 4) & 0x0F] * scale1, acc1);
                acc1 = fmaf(a2, NVF4_LUT[p1 & 0x0F] * scale1, acc1);
                acc1 = fmaf(a3, NVF4_LUT[(p1 >> 4) & 0x0F] * scale1, acc1);
                acc1 = fmaf(a4, NVF4_LUT[p2 & 0x0F] * scale1, acc1);
                acc1 = fmaf(a5, NVF4_LUT[(p2 >> 4) & 0x0F] * scale1, acc1);
                acc1 = fmaf(a6, NVF4_LUT[p3 & 0x0F] * scale1, acc1);
                acc1 = fmaf(a7, NVF4_LUT[(p3 >> 4) & 0x0F] * scale1, acc1);
            }
        }
    }

    // Store results
    if (valid0) C[global_n0] = __float2bfloat16(alpha * acc0);
    if (valid1) C[global_n1] = __float2bfloat16(alpha * acc1);
}

// ============================================================================
// Launch Functions
// ============================================================================

/**
 * Launch NVF4 GEMV
 *
 * @param A       Input vector [K] BF16
 * @param B_data  Weight matrix [K/2, N] packed NVF4
 * @param B_scale Scale factors [K/32, N] UE4M3
 * @param C       Output vector [N] BF16
 * @param K       Inner dimension
 * @param N       Output dimension
 * @param alpha   Scaling factor (default 1.0)
 * @param stream  CUDA stream
 */
inline cudaError_t launch_gemv_nvf4_bf16(
    const __nv_bfloat16* A,
    const uint8_t* B_data,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    float alpha = 1.0f,
    cudaStream_t stream = nullptr
) {
    using Config = GemvNvf4Config;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);

    // Use unrolled kernel for aligned K
    if (K % Config::SCALE_BLOCK == 0 && K >= Config::SCALE_BLOCK) {
        gemv_nvf4_bf16_kernel_unrolled<Config><<<grid, block, 0, stream>>>(
            A, B_data, B_scale, C, K, N, alpha
        );
    } else {
        gemv_nvf4_bf16_kernel<Config><<<grid, block, 0, stream>>>(
            A, B_data, B_scale, C, K, N, alpha
        );
    }

    return cudaGetLastError();
}

// ============================================================================
// Quantization Kernel (BF16 -> NVF4)
// ============================================================================

/**
 * Quantize BF16 matrix to NVF4 with block scaling
 *
 * Input:  B[K, N] BF16 row-major
 * Output: B_data[K/2, N] packed NVF4
 *         B_scale[K/32, N] UE4M3 scale factors
 */
__global__ void quantize_bf16_to_nvf4_kernel(
    __nv_bfloat16 const* __restrict__ input,  // [K, N] row-major
    uint8_t* __restrict__ output_data,         // [K/2, N] packed NVF4
    uint8_t* __restrict__ output_scale,        // [K/32, N] scale factors
    int K,
    int N
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_block = blockIdx.y;

    if (n >= N) return;

    const int SCALE_BLOCK = 32;
    const int k_start = scale_block * SCALE_BLOCK;
    const int k_end = min(k_start + SCALE_BLOCK, K);

    // Find max absolute value in block
    float max_abs = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        float val = fabsf(__bfloat162float(input[k * N + n]));
        max_abs = fmaxf(max_abs, val);
    }

    // Compute scale factor (target range: [-6, 6] for NVF4)
    const float NVF4_MAX = 6.0f;
    float scale = (max_abs > 1e-8f) ? (max_abs / NVF4_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Encode scale as UE4M3
    // UE4M3: value = (1 + mantissa/8) * 2^(exponent - 7)
    // We need to find exp and mant such that scale ~= (1 + mant/8) * 2^(exp-7)

    // First, find exponent by getting floor(log2(scale)) and shift to [1,2) range
    int exp_raw = 0;
    float normalized = scale;

    if (normalized >= 2.0f) {
        while (normalized >= 2.0f && exp_raw < 8) {
            normalized *= 0.5f;
            exp_raw++;
        }
    } else if (normalized < 1.0f && normalized > 1e-8f) {
        while (normalized < 1.0f && exp_raw > -7) {
            normalized *= 2.0f;
            exp_raw--;
        }
    }

    // Now normalized is in [1.0, 2.0), compute mantissa
    // mantissa = (normalized - 1) * 8, rounded to nearest integer
    int mant = __float2int_rn((normalized - 1.0f) * 8.0f);
    mant = max(0, min(7, mant));

    // Compute biased exponent
    int exp_biased = exp_raw + 7;
    exp_biased = max(0, min(15, exp_biased));

    uint8_t scale_encoded = ((exp_biased & 0xF) << 3) | (mant & 0x7);
    output_scale[scale_block * N + n] = scale_encoded;

    // Recompute actual encoded scale for accurate quantization
    float encoded_scale = (1.0f + mant / 8.0f) * ldexpf(1.0f, exp_biased - 7);
    inv_scale = 1.0f / encoded_scale;

    // Quantize values to NVF4
    for (int k = k_start; k < k_end; k += 2) {
        float v0 = __bfloat162float(input[k * N + n]) * inv_scale;
        float v1 = (k + 1 < k_end) ? __bfloat162float(input[(k + 1) * N + n]) * inv_scale : 0.0f;

        // Quantize to NVF4 (nearest value in lookup table)
        auto quantize_nvf4 = [](float val) -> uint8_t {
            uint8_t sign = (val < 0) ? 0x8 : 0x0;
            val = fabsf(val);
            if (val < 0.25f) return sign | 0;       // 0
            if (val < 0.75f) return sign | 1;       // 0.5
            if (val < 1.25f) return sign | 2;       // 1.0
            if (val < 1.75f) return sign | 3;       // 1.5
            if (val < 2.5f)  return sign | 4;       // 2.0
            if (val < 3.5f)  return sign | 5;       // 3.0
            if (val < 5.0f)  return sign | 6;       // 4.0
            return sign | 7;                         // 6.0
        };

        uint8_t q0 = quantize_nvf4(v0);
        uint8_t q1 = quantize_nvf4(v1);

        // Pack: low nibble = first element, high nibble = second
        int k_packed = k / 2;
        output_data[k_packed * N + n] = (q1 << 4) | (q0 & 0x0F);
    }
}

/**
 * Launch quantization kernel
 */
inline cudaError_t quantize_bf16_to_nvf4(
    const __nv_bfloat16* input,
    uint8_t* output_data,
    uint8_t* output_scale,
    int K,
    int N,
    cudaStream_t stream = nullptr
) {
    const int SCALE_BLOCK = 32;
    int num_scale_blocks = (K + SCALE_BLOCK - 1) / SCALE_BLOCK;

    dim3 block(256);
    dim3 grid((N + 255) / 256, num_scale_blocks);

    quantize_bf16_to_nvf4_kernel<<<grid, block, 0, stream>>>(
        input, output_data, output_scale, K, N
    );

    return cudaGetLastError();
}

// ============================================================================
// High-Level API
// ============================================================================

/**
 * Check if NVF4 GEMV is available (SM120+)
 */
inline bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major == 12);  // SM120/SM121
}

}  // namespace gemv_nvf4
}  // namespace ops
}  // namespace pygpukit
