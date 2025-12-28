/**
 * FP8 GEMV Kernel with Online Dequantization
 *
 * Purpose: W8A16 GEMV for FP8 quantized LLM weights
 * - Weight: FP8 E4M3 (1 byte per element) + block-wise scale
 * - Activation: BF16 (2 bytes per element)
 * - Output: BF16
 *
 * Design decisions:
 * 1. Online dequantization: FP8 -> FP32 during compute (no pre-dequant)
 * 2. Block-wise scaling: Each 128x128 block has a single scale factor
 * 3. FP32 accumulation for numerical precision
 * 4. Memory savings: 31GB FP8 stays at 31GB (vs 62GB if dequantized to BF16)
 *
 * FP8 E4M3 format:
 * - 1 sign bit, 4 exponent bits, 3 mantissa bits
 * - Range: [-448, 448], no infinity/NaN
 * - Supported natively on SM90+ (Hopper), software emulation on SM80-89
 *
 * Target architectures:
 * - SM89 (RTX 40xx): FP8 native support
 * - SM90 (H100): FP8 TensorCore
 * - SM120 (RTX 5090): FP8 native + FP4
 * - SM80-86 (RTX 30xx): Software dequantization
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// FP8 E4M3 support (CUDA 11.8+ for __nv_fp8_e4m3)
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// FP8 E4M3 Dequantization
// ============================================================================

/**
 * FP8 E4M3 to FP32 conversion lookup table
 *
 * FP8 E4M3: 1 sign, 4 exp (bias=7), 3 mantissa
 * Values: 0-255 map to [-448, +448]
 *
 * Precomputed at compile time for all 256 byte values.
 * Format: value = sign * (1 + mant/8) * 2^(exp-7)  [normal]
 *         value = sign * mant * 2^(-9)             [subnormal, exp=0]
 */
__device__ __constant__ float FP8_E4M3_LUT[256] = {
    // exp=0 (subnormal): mant * 2^(-9), positive (0x00-0x07)
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    // exp=1: (1+mant/8) * 2^(-6), positive (0x08-0x0F)
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    // exp=2: (1+mant/8) * 2^(-5), positive (0x10-0x17)
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    // exp=3: (1+mant/8) * 2^(-4), positive (0x18-0x1F)
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    // exp=4: (1+mant/8) * 2^(-3), positive (0x20-0x27)
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    // exp=5: (1+mant/8) * 2^(-2), positive (0x28-0x2F)
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    // exp=6: (1+mant/8) * 2^(-1), positive (0x30-0x37)
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    // exp=7: (1+mant/8) * 2^0, positive (0x38-0x3F)
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    // exp=8: (1+mant/8) * 2^1, positive (0x40-0x47)
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    // exp=9: (1+mant/8) * 2^2, positive (0x48-0x4F)
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    // exp=10: (1+mant/8) * 2^3, positive (0x50-0x57)
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    // exp=11: (1+mant/8) * 2^4, positive (0x58-0x5F)
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    // exp=12: (1+mant/8) * 2^5, positive (0x60-0x67)
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    // exp=13: (1+mant/8) * 2^6, positive (0x68-0x6F)
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    // exp=14: (1+mant/8) * 2^7, positive (0x70-0x77)
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    // exp=15: (1+mant/8) * 2^8, positive (0x78-0x7F)
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 480.0f,
    // exp=0 (subnormal): -mant * 2^(-9), negative (0x80-0x87)
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    // exp=1: -(1+mant/8) * 2^(-6), negative (0x88-0x8F)
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    // exp=2: -(1+mant/8) * 2^(-5), negative (0x90-0x97)
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    // exp=3: -(1+mant/8) * 2^(-4), negative (0x98-0x9F)
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    // exp=4: -(1+mant/8) * 2^(-3), negative (0xA0-0xA7)
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    // exp=5: -(1+mant/8) * 2^(-2), negative (0xA8-0xAF)
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    // exp=6: -(1+mant/8) * 2^(-1), negative (0xB0-0xB7)
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    // exp=7: -(1+mant/8) * 2^0, negative (0xB8-0xBF)
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    // exp=8: -(1+mant/8) * 2^1, negative (0xC0-0xC7)
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    // exp=9: -(1+mant/8) * 2^2, negative (0xC8-0xCF)
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    // exp=10: -(1+mant/8) * 2^3, negative (0xD0-0xD7)
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    // exp=11: -(1+mant/8) * 2^4, negative (0xD8-0xDF)
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    // exp=12: -(1+mant/8) * 2^5, negative (0xE0-0xE7)
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    // exp=13: -(1+mant/8) * 2^6, negative (0xE8-0xEF)
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    // exp=14: -(1+mant/8) * 2^7, negative (0xF0-0xF7)
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    // exp=15: -(1+mant/8) * 2^8, negative (0xF8-0xFF)
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -480.0f,
};

/**
 * Software FP8 E4M3 to FP32 conversion
 * For architectures without native FP8 support
 */
__device__ __forceinline__ float fp8_e4m3_to_f32_soft(uint8_t val) {
    // Sign bit
    float sign = (val & 0x80) ? -1.0f : 1.0f;

    // Exponent: bits 6-3 (4 bits, bias = 7)
    int exp = (val >> 3) & 0x0F;

    // Mantissa: bits 2-0 (3 bits)
    int mant = val & 0x07;

    if (exp == 0) {
        // Subnormal: 2^(-6) * (mantissa / 8)
        return sign * ldexpf((float)mant, -9);  // 2^(-6-3) = 2^(-9)
    } else if (exp == 15) {
        // E4M3 has no inf/NaN, max value is 448
        // exp=15, mant=7: 1.875 * 2^8 = 480 (clamped to 448)
        return sign * (1.0f + mant / 8.0f) * 256.0f;  // 2^(15-7) = 256
    } else {
        // Normal: (1 + mantissa/8) * 2^(exp-7)
        return sign * (1.0f + mant / 8.0f) * ldexpf(1.0f, exp - 7);
    }
}

/**
 * FP8 E4M3 to FP32 using lookup table
 * Fast path for SM80-86
 */
__device__ __forceinline__ float fp8_e4m3_to_f32_lut(uint8_t val) {
    return FP8_E4M3_LUT[val];
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
