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
// Launch Function Declarations
// ============================================================================

cudaError_t launch_gemv_nvf4_bf16(
    const __nv_bfloat16* A,
    const uint8_t* B_data,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    float alpha = 1.0f,
    cudaStream_t stream = nullptr
);

cudaError_t quantize_bf16_to_nvf4(
    const __nv_bfloat16* input,
    uint8_t* output_data,
    uint8_t* output_scale,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

// Row-major version for pure NVF4/NVF4 GEMV (coalesced memory access)
// Output: [N, K/2] data, [N, K/32] scale (row-major)
cudaError_t quantize_bf16_to_nvf4_rowmajor(
    const __nv_bfloat16* input,
    uint8_t* output_data,
    uint8_t* output_scale,
    int K,
    int N,
    cudaStream_t stream = nullptr
);

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
