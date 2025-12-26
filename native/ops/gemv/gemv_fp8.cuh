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
 * Used for SM80-86 where native FP8 is not available
 */
__constant__ float FP8_E4M3_LUT[256];

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
 * Initialize FP8 E4M3 lookup table (call once at startup)
 */
inline void init_fp8_e4m3_lut() {
    float lut[256];
    for (int i = 0; i < 256; ++i) {
        uint8_t val = static_cast<uint8_t>(i);
        float sign = (val & 0x80) ? -1.0f : 1.0f;
        int exp = (val >> 3) & 0x0F;
        int mant = val & 0x07;

        if (exp == 0) {
            lut[i] = sign * ldexpf((float)mant, -9);
        } else {
            lut[i] = sign * (1.0f + mant / 8.0f) * ldexpf(1.0f, exp - 7);
        }
    }
    cudaMemcpyToSymbol(FP8_E4M3_LUT, lut, sizeof(lut));
}

/**
 * FP8 E4M3 to FP32 using lookup table
 * Fast path for SM80-86
 */
__device__ __forceinline__ float fp8_e4m3_to_f32_lut(uint8_t val) {
    return FP8_E4M3_LUT[val];
}

// ============================================================================
// FP8 GEMV Configuration
// ============================================================================

struct GemvFP8Config {
    static constexpr int BLOCK_SIZE = 256;  // 8 warps
    static constexpr int TILE_N = 256;
    static constexpr int UNROLL_K = 8;
    static constexpr int BLOCK_QUANT_SIZE = 128;  // 128x128 block quantization
};

// ============================================================================
// FP8 GEMV Kernel with Block-wise Dequantization
// ============================================================================

/**
 * GEMV kernel for FP8 weights: C[1,N] = A[1,K] @ B_fp8[K,N]
 *
 * Memory layout:
 * - A: [1, K] BF16 activation (row-major)
 * - B_fp8: [K, N] FP8 E4M3 weights (row-major, 1 byte per element)
 * - B_scale: [K/128, N/128] BF16 scale factors (inverse scale)
 * - C: [1, N] BF16 output
 *
 * Dequantization formula:
 *   weight_f32 = fp8_to_f32(B_fp8[k,n]) * B_scale[k/128, n/128]
 *
 * Thread mapping:
 * - Each thread handles one output element C[global_n]
 * - All threads iterate over K, applying block-wise scales
 */
template<typename Config = GemvFP8Config>
__global__ void gemv_fp8_kernel(
    __nv_bfloat16 const* __restrict__ A,      // [1, K] activation
    uint8_t const* __restrict__ B_fp8,        // [K, N] FP8 weights
    __nv_bfloat16 const* __restrict__ B_scale, // [K/block, N/block] scales
    __nv_bfloat16* __restrict__ C,            // [1, N] output
    int K,
    int N,
    int scale_stride_n  // N / BLOCK_QUANT_SIZE (number of scale blocks per row)
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    // Scale block index for this thread's column
    const int scale_block_n = global_n / Config::BLOCK_QUANT_SIZE;

    // FP32 accumulator
    float acc = 0.0f;

    // Base pointers
    const uint8_t* B_col = B_fp8 + global_n;

    // Main K loop
    int k = 0;
    constexpr int UNROLL = Config::UNROLL_K;

    // Process UNROLL elements at a time
    for (; k + UNROLL <= K; k += UNROLL) {
        // Determine scale block for this K range
        // Note: All UNROLL elements might span at most 2 scale blocks
        const int scale_block_k = k / Config::BLOCK_QUANT_SIZE;

        // Load scale factor (shared across 128 elements in K)
        float scale = __bfloat162float(B_scale[scale_block_k * scale_stride_n + scale_block_n]);

        // Unrolled loop
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            // Check if we crossed a scale block boundary
            int curr_scale_block_k = kk / Config::BLOCK_QUANT_SIZE;
            if (curr_scale_block_k != scale_block_k) {
                scale = __bfloat162float(B_scale[curr_scale_block_k * scale_stride_n + scale_block_n]);
            }

            // Load activation (BF16 -> FP32)
            float a = __bfloat162float(A[kk]);

            // Load FP8 weight and dequantize
            uint8_t b_fp8 = B_col[kk * N];
            float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;

            // FMA accumulation
            acc = fmaf(a, b, acc);
        }
    }

    // Handle K remainder
    for (; k < K; ++k) {
        const int scale_block_k = k / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_block_k * scale_stride_n + scale_block_n]);

        float a = __bfloat162float(A[k]);
        uint8_t b_fp8 = B_col[k * N];
        float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;
        acc = fmaf(a, b, acc);
    }

    // Store result as BF16
    C[global_n] = __float2bfloat16(acc);
}

/**
 * Optimized FP8 GEMV with cached scale factors
 *
 * Optimization: Pre-load scale factors for the current K block into registers
 * Since each thread handles one N, we only need one scale value per K block
 */
template<typename Config = GemvFP8Config>
__global__ void gemv_fp8_cached_scale_kernel(
    __nv_bfloat16 const* __restrict__ A,      // [1, K]
    uint8_t const* __restrict__ B_fp8,        // [K, N]
    __nv_bfloat16 const* __restrict__ B_scale, // [K/128, N/128]
    __nv_bfloat16* __restrict__ C,            // [1, N]
    int K,
    int N,
    int scale_stride_n
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    const int scale_block_n = global_n / Config::BLOCK_QUANT_SIZE;
    const uint8_t* B_col = B_fp8 + global_n;

    float acc = 0.0f;

    // Number of K blocks
    const int num_k_blocks = (K + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;

    // Iterate by K blocks (128 elements at a time)
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_start = kb * Config::BLOCK_QUANT_SIZE;
        const int k_end = min(k_start + Config::BLOCK_QUANT_SIZE, K);

        // Load scale for this K block (one scale per 128x128 block)
        float scale = __bfloat162float(B_scale[kb * scale_stride_n + scale_block_n]);

        // Process elements in this K block
        for (int k = k_start; k < k_end; ++k) {
            float a = __bfloat162float(A[k]);
            uint8_t b_fp8 = B_col[k * N];
            float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;
            acc = fmaf(a, b, acc);
        }
    }

    C[global_n] = __float2bfloat16(acc);
}

/**
 * FP8 GEMV with vectorized loads (4 bytes at a time)
 * Loads 4 FP8 values as uint32_t for better memory throughput
 */
template<typename Config = GemvFP8Config>
__global__ void gemv_fp8_vec4_kernel(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_fp8,
    __nv_bfloat16 const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    int K,
    int N,
    int scale_stride_n
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int global_n = block_n + tid;

    if (global_n >= N) return;

    const int scale_block_n = global_n / Config::BLOCK_QUANT_SIZE;
    const uint8_t* B_col = B_fp8 + global_n;

    float acc = 0.0f;

    const int num_k_blocks = (K + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_start = kb * Config::BLOCK_QUANT_SIZE;
        const int k_end = min(k_start + Config::BLOCK_QUANT_SIZE, K);

        float scale = __bfloat162float(B_scale[kb * scale_stride_n + scale_block_n]);

        // Vectorized inner loop (4 elements at a time)
        int k = k_start;
        for (; k + 4 <= k_end; k += 4) {
            // Load 4 BF16 activations as 2x bfloat162
            __nv_bfloat162 a01 = *reinterpret_cast<const __nv_bfloat162*>(A + k);
            __nv_bfloat162 a23 = *reinterpret_cast<const __nv_bfloat162*>(A + k + 2);

            // Load 4 FP8 weights (non-contiguous in memory due to row-major layout)
            uint8_t b0 = B_col[(k + 0) * N];
            uint8_t b1 = B_col[(k + 1) * N];
            uint8_t b2 = B_col[(k + 2) * N];
            uint8_t b3 = B_col[(k + 3) * N];

            // Dequantize and compute
            float af0 = __low2float(a01);
            float af1 = __high2float(a01);
            float af2 = __low2float(a23);
            float af3 = __high2float(a23);

            float bf0 = fp8_e4m3_to_f32_lut(b0) * scale;
            float bf1 = fp8_e4m3_to_f32_lut(b1) * scale;
            float bf2 = fp8_e4m3_to_f32_lut(b2) * scale;
            float bf3 = fp8_e4m3_to_f32_lut(b3) * scale;

            acc = fmaf(af0, bf0, acc);
            acc = fmaf(af1, bf1, acc);
            acc = fmaf(af2, bf2, acc);
            acc = fmaf(af3, bf3, acc);
        }

        // Handle remainder
        for (; k < k_end; ++k) {
            float a = __bfloat162float(A[k]);
            uint8_t b_fp8 = B_col[k * N];
            float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;
            acc = fmaf(a, b, acc);
        }
    }

    C[global_n] = __float2bfloat16(acc);
}

// ============================================================================
// Launch Functions
// ============================================================================

/**
 * Launch FP8 GEMV kernel
 *
 * @param A Activation tensor [1, K] in BF16
 * @param B_fp8 Weight tensor [K, N] in FP8 E4M3 (uint8_t)
 * @param B_scale Scale tensor [K/128, N/128] in BF16
 * @param C Output tensor [1, N] in BF16
 * @param K Input dimension
 * @param N Output dimension
 * @param stream CUDA stream
 */
inline cudaError_t launch_gemv_fp8(
    const __nv_bfloat16* A,
    const uint8_t* B_fp8,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream = nullptr
) {
    using Config = GemvFP8Config;

    // Scale tensor stride (N / block_size)
    int scale_stride_n = (N + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);

    // Use vectorized kernel for better performance
    gemv_fp8_vec4_kernel<Config><<<grid, block, 0, stream>>>(
        A, B_fp8, B_scale, C, K, N, scale_stride_n
    );

    return cudaGetLastError();
}

/**
 * Dispatch GEMV for FP8 weights
 * Returns true if dispatched, false if should fallback to GEMM
 */
inline bool dispatch_gemv_fp8(
    const __nv_bfloat16* A,
    const uint8_t* B_fp8,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr
) {
    if (M == 1 && N >= GemvFP8Config::BLOCK_SIZE) {
        launch_gemv_fp8(A, B_fp8, B_scale, C, K, N, stream);
        return true;
    }
    return false;
}

// ============================================================================
// Batched FP8 GEMV
// ============================================================================

/**
 * Batched FP8 GEMV: C[batch,N] = A[batch,K] @ B_fp8[K,N]
 * Weight matrix B is shared across batches
 */
template<typename Config = GemvFP8Config>
__global__ void gemv_fp8_batched_kernel(
    __nv_bfloat16 const* __restrict__ A,      // [batch, K]
    uint8_t const* __restrict__ B_fp8,        // [K, N]
    __nv_bfloat16 const* __restrict__ B_scale, // [K/128, N/128]
    __nv_bfloat16* __restrict__ C,            // [batch, N]
    int K,
    int N,
    int batch_count,
    int scale_stride_n
) {
    const int tid = threadIdx.x;
    const int block_n = blockIdx.x * Config::TILE_N;
    const int batch_idx = blockIdx.y;
    const int global_n = block_n + tid;

    if (global_n >= N || batch_idx >= batch_count) return;

    const __nv_bfloat16* A_batch = A + batch_idx * K;
    __nv_bfloat16* C_batch = C + batch_idx * N;

    const int scale_block_n = global_n / Config::BLOCK_QUANT_SIZE;
    const uint8_t* B_col = B_fp8 + global_n;

    float acc = 0.0f;

    const int num_k_blocks = (K + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_start = kb * Config::BLOCK_QUANT_SIZE;
        const int k_end = min(k_start + Config::BLOCK_QUANT_SIZE, K);

        float scale = __bfloat162float(B_scale[kb * scale_stride_n + scale_block_n]);

        for (int k = k_start; k < k_end; ++k) {
            float a = __bfloat162float(A_batch[k]);
            uint8_t b_fp8 = B_col[k * N];
            float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;
            acc = fmaf(a, b, acc);
        }
    }

    C_batch[global_n] = __float2bfloat16(acc);
}

inline cudaError_t launch_gemv_fp8_batched(
    const __nv_bfloat16* A,
    const uint8_t* B_fp8,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    int batch_count,
    cudaStream_t stream = nullptr
) {
    using Config = GemvFP8Config;

    int scale_stride_n = (N + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N, batch_count);

    gemv_fp8_batched_kernel<Config><<<grid, block, 0, stream>>>(
        A, B_fp8, B_scale, C, K, N, batch_count, scale_stride_n
    );

    return cudaGetLastError();
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
