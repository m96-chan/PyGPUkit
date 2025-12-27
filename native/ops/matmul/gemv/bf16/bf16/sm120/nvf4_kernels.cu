/**
 * NVF4 GEMV Kernel Implementations
 */

#include "nvf4.cuh"

namespace pygpukit {
namespace ops {
namespace gemv_nvf4 {

// ============================================================================
// NVF4 GEMV Kernels
// ============================================================================

/**
 * GEMV kernel: C[1,N] = A[1,K] @ B[K,N] where B is NVF4 quantized
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
        float lut0  = 0.0f;
        float lut1  = 0.5f * scale;
        float lut2  = 1.0f * scale;
        float lut3  = 1.5f * scale;
        float lut4  = 2.0f * scale;
        float lut5  = 3.0f * scale;
        float lut6  = 4.0f * scale;
        float lut7  = 6.0f * scale;
        float lut8  = 0.0f;
        float lut9  = -0.5f * scale;
        float lut10 = -1.0f * scale;
        float lut11 = -1.5f * scale;
        float lut12 = -2.0f * scale;
        float lut13 = -3.0f * scale;
        float lut14 = -4.0f * scale;
        float lut15 = -6.0f * scale;

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

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_nvf4_bf16(
    const __nv_bfloat16* A,
    const uint8_t* B_data,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    float alpha,
    cudaStream_t stream
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
// Quantization Kernel
// ============================================================================

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

cudaError_t quantize_bf16_to_nvf4(
    const __nv_bfloat16* input,
    uint8_t* output_data,
    uint8_t* output_scale,
    int K,
    int N,
    cudaStream_t stream
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
// Row-Major Quantization Kernel (for pure NVF4/NVF4 GEMV)
// ============================================================================

/**
 * Row-major quantization kernel
 * Input:  [K, N] BF16 row-major
 * Output: [N, K/2] packed NVF4 row-major (contiguous K for each N)
 *         [N, K/32] scale factors row-major
 */
__global__ void quantize_bf16_to_nvf4_rowmajor_kernel(
    __nv_bfloat16 const* __restrict__ input,  // [K, N] row-major
    uint8_t* __restrict__ output_data,         // [N, K/2] row-major
    uint8_t* __restrict__ output_scale,        // [N, K/32] row-major
    int K,
    int N
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_block = blockIdx.y;

    if (n >= N) return;

    const int SCALE_BLOCK = 32;
    const int k_start = scale_block * SCALE_BLOCK;
    const int k_end = min(k_start + SCALE_BLOCK, K);
    const int K_packed = K / 2;
    const int K_scale = (K + SCALE_BLOCK - 1) / SCALE_BLOCK;

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

    // Compute mantissa
    int mant = __float2int_rn((normalized - 1.0f) * 8.0f);
    mant = max(0, min(7, mant));

    // Compute biased exponent
    int exp_biased = exp_raw + 7;
    exp_biased = max(0, min(15, exp_biased));

    uint8_t scale_encoded = ((exp_biased & 0xF) << 3) | (mant & 0x7);
    // Row-major: [N, K/32] -> index = n * K_scale + scale_block
    output_scale[n * K_scale + scale_block] = scale_encoded;

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
        // Row-major: [N, K/2] -> index = n * K_packed + k_packed
        output_data[n * K_packed + k_packed] = (q1 << 4) | (q0 & 0x0F);
    }
}

cudaError_t quantize_bf16_to_nvf4_rowmajor(
    const __nv_bfloat16* input,
    uint8_t* output_data,
    uint8_t* output_scale,
    int K,
    int N,
    cudaStream_t stream
) {
    const int SCALE_BLOCK = 32;
    int num_scale_blocks = (K + SCALE_BLOCK - 1) / SCALE_BLOCK;

    dim3 block(256);
    dim3 grid((N + 255) / 256, num_scale_blocks);

    quantize_bf16_to_nvf4_rowmajor_kernel<<<grid, block, 0, stream>>>(
        input, output_data, output_scale, K, N
    );

    return cudaGetLastError();
}

}  // namespace gemv_nvf4
}  // namespace ops
}  // namespace pygpukit
