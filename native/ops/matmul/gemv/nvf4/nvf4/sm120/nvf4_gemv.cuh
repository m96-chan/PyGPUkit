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
    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;  // 256 threads
    static constexpr int WARP_SIZE = 32;
    static constexpr int SCALE_BLOCK_SIZE = 32;  // NVF4 uses 32-element blocks
};

// ============================================================================
// Pure NVF4 GEMV Kernel: A[K](NVF4) x B[K,N](NVF4) -> C[N](BF16)
// ============================================================================

/**
 * Pure NVF4 GEMV with warp-level reduction
 *
 * Each warp handles ONE output element (N dimension)
 * 32 threads in warp cooperatively reduce over K dimension
 *
 * Memory layout (ROW-MAJOR for B - contiguous K for coalesced access):
 * - A_data: [K/2] packed NVF4 (2 values per byte)
 * - A_scale: [K/32] UE4M3 scale factors
 * - B_data: [N, K/2] packed NVF4 (row-major: contiguous K for each N)
 * - B_scale: [N, K/32] UE4M3 scale factors (row-major)
 * - C: [N] BF16 output vector
 *
 * This layout enables coalesced memory access when reading B.
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
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    // Shared memory layout:
    // [0, K/2): A_data packed NVF4
    // [K/2, K/2 + K/32): A_scale UE4M3
    extern __shared__ uint8_t smem[];
    uint8_t* smem_A_data = smem;
    uint8_t* smem_A_scale = smem + (K / 2);

    const int K_packed = K / 2;
    const int K_scale_blocks = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    // Cooperative load of A_data into shared memory
    for (int i = threadIdx.x; i < K_packed; i += Config::BLOCK_SIZE) {
        smem_A_data[i] = A_data[i];
    }
    // Cooperative load of A_scale
    for (int i = threadIdx.x; i < K_scale_blocks; i += Config::BLOCK_SIZE) {
        smem_A_scale[i] = A_scale[i];
    }
    __syncthreads();

    // B_data is [N, K/2] row-major: element at (n, k_packed) is at B_data[n * K_packed + k_packed]
    // B_scale is [N, K/32] row-major: element at (n, scale_k) is at B_scale[n * K_scale_blocks + scale_k]
    const uint8_t* B_row = B_data + global_n * K_packed;
    const uint8_t* S_row = B_scale + global_n * K_scale_blocks;

    float acc = 0.0f;

    // Each lane handles elements with stride 32
    // Process 2 values per byte (packed NVF4)
    for (int k = lane_id * 2; k < K; k += Config::WARP_SIZE * 2) {
        const int packed_idx = k / 2;
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;

        // Load scales
        float sA = decode_ue4m3_scale(smem_A_scale[scale_k]);
        float sB = decode_ue4m3_scale(__ldg(&S_row[scale_k]));

        // Load packed bytes (row-major for B - contiguous access)
        uint8_t a_packed = smem_A_data[packed_idx];
        uint8_t b_packed = __ldg(&B_row[packed_idx]);

        // Dequantize and accumulate (2 values per byte)
        float a0 = dequant_nvf4(a_packed & 0x0F) * sA;
        float a1 = dequant_nvf4((a_packed >> 4) & 0x0F) * sA;
        float b0 = dequant_nvf4(b_packed & 0x0F) * sB;
        float b1 = dequant_nvf4((b_packed >> 4) & 0x0F) * sB;

        acc = fmaf(a0, b0, acc);
        acc = fmaf(a1, b1, acc);
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
 * Optimized variant: 64-bit loads (16 NVF4 values at once)
 *
 * Memory layout (ROW-MAJOR for B):
 * - B_data: [N, K/2] row-major
 * - B_scale: [N, K/32] row-major
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
    const int warp_id = threadIdx.x / Config::WARP_SIZE;
    const int lane_id = threadIdx.x % Config::WARP_SIZE;
    const int global_n = blockIdx.x * Config::WARPS_PER_BLOCK + warp_id;

    if (global_n >= N) return;

    // Shared memory
    extern __shared__ uint8_t smem[];
    uint8_t* smem_A_data = smem;
    uint8_t* smem_A_scale = smem + (K / 2);

    const int K_packed = K / 2;
    const int K_scale_blocks = (K + Config::SCALE_BLOCK_SIZE - 1) / Config::SCALE_BLOCK_SIZE;

    // Vectorized load of A_data (64-bit = 8 bytes = 16 NVF4 values)
    const int K_packed_aligned8 = K_packed & ~7;
    for (int i = threadIdx.x * 8; i < K_packed_aligned8; i += Config::BLOCK_SIZE * 8) {
        *reinterpret_cast<uint64_t*>(&smem_A_data[i]) =
            *reinterpret_cast<const uint64_t*>(&A_data[i]);
    }
    for (int i = K_packed_aligned8 + threadIdx.x; i < K_packed; i += Config::BLOCK_SIZE) {
        smem_A_data[i] = A_data[i];
    }
    // Load A_scale
    for (int i = threadIdx.x; i < K_scale_blocks; i += Config::BLOCK_SIZE) {
        smem_A_scale[i] = A_scale[i];
    }
    __syncthreads();

    // B row pointers (row-major: contiguous K for each N)
    const uint8_t* B_row = B_data + global_n * K_packed;
    const uint8_t* S_row = B_scale + global_n * K_scale_blocks;

    // 4 independent accumulators
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Main loop: each lane handles 16 NVF4 values (8 bytes) per iteration
    for (int k_base = lane_id * 16; k_base < (K & ~15); k_base += Config::WARP_SIZE * 16) {
        const int packed_base = k_base / 2;
        const int scale_k = k_base / Config::SCALE_BLOCK_SIZE;

        float sA = decode_ue4m3_scale(smem_A_scale[scale_k]);
        float sB = decode_ue4m3_scale(__ldg(&S_row[scale_k]));
        float combined_scale = sA * sB;

        // Load 8 packed bytes (16 NVF4 values) - contiguous access!
        uint64_t a8 = *reinterpret_cast<const uint64_t*>(&smem_A_data[packed_base]);
        uint64_t b8 = *reinterpret_cast<const uint64_t*>(&B_row[packed_base]);

        // Unpack and accumulate (4 accumulators for 16 values)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint8_t a_byte = (a8 >> (i * 8)) & 0xFF;
            uint8_t b_byte = (b8 >> (i * 8)) & 0xFF;
            float a0 = dequant_nvf4(a_byte & 0x0F) * combined_scale;
            float a1 = dequant_nvf4((a_byte >> 4) & 0x0F) * combined_scale;
            float b0 = dequant_nvf4(b_byte & 0x0F);
            float b1 = dequant_nvf4((b_byte >> 4) & 0x0F);
            acc0 = fmaf(a0, b0, acc0);
            acc0 = fmaf(a1, b1, acc0);
        }
        #pragma unroll
        for (int i = 2; i < 4; ++i) {
            uint8_t a_byte = (a8 >> (i * 8)) & 0xFF;
            uint8_t b_byte = (b8 >> (i * 8)) & 0xFF;
            float a0 = dequant_nvf4(a_byte & 0x0F) * combined_scale;
            float a1 = dequant_nvf4((a_byte >> 4) & 0x0F) * combined_scale;
            float b0 = dequant_nvf4(b_byte & 0x0F);
            float b1 = dequant_nvf4((b_byte >> 4) & 0x0F);
            acc1 = fmaf(a0, b0, acc1);
            acc1 = fmaf(a1, b1, acc1);
        }
        #pragma unroll
        for (int i = 4; i < 6; ++i) {
            uint8_t a_byte = (a8 >> (i * 8)) & 0xFF;
            uint8_t b_byte = (b8 >> (i * 8)) & 0xFF;
            float a0 = dequant_nvf4(a_byte & 0x0F) * combined_scale;
            float a1 = dequant_nvf4((a_byte >> 4) & 0x0F) * combined_scale;
            float b0 = dequant_nvf4(b_byte & 0x0F);
            float b1 = dequant_nvf4((b_byte >> 4) & 0x0F);
            acc2 = fmaf(a0, b0, acc2);
            acc2 = fmaf(a1, b1, acc2);
        }
        #pragma unroll
        for (int i = 6; i < 8; ++i) {
            uint8_t a_byte = (a8 >> (i * 8)) & 0xFF;
            uint8_t b_byte = (b8 >> (i * 8)) & 0xFF;
            float a0 = dequant_nvf4(a_byte & 0x0F) * combined_scale;
            float a1 = dequant_nvf4((a_byte >> 4) & 0x0F) * combined_scale;
            float b0 = dequant_nvf4(b_byte & 0x0F);
            float b1 = dequant_nvf4((b_byte >> 4) & 0x0F);
            acc3 = fmaf(a0, b0, acc3);
            acc3 = fmaf(a1, b1, acc3);
        }
    }

    // Handle remainder
    const int K_aligned16 = K & ~15;
    for (int k = K_aligned16 + lane_id * 2; k < K; k += Config::WARP_SIZE * 2) {
        const int packed_idx = k / 2;
        const int scale_k = k / Config::SCALE_BLOCK_SIZE;

        float sA = decode_ue4m3_scale(smem_A_scale[scale_k]);
        float sB = decode_ue4m3_scale(__ldg(&S_row[scale_k]));

        uint8_t a_packed = smem_A_data[packed_idx];
        uint8_t b_packed = B_row[packed_idx];

        float a0 = dequant_nvf4(a_packed & 0x0F) * sA;
        float a1 = dequant_nvf4((a_packed >> 4) & 0x0F) * sA;
        float b0 = dequant_nvf4(b_packed & 0x0F) * sB;
        float b1 = dequant_nvf4((b_packed >> 4) & 0x0F) * sB;

        acc0 = fmaf(a0, b0, acc0);
        acc0 = fmaf(a1, b1, acc0);
    }

    // Combine accumulators
    float acc = acc0 + acc1 + acc2 + acc3;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[global_n] = __float2bfloat16(acc);
    }
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
