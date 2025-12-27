/**
 * FP8 GEMV Kernel Implementations
 */

#include "fp8.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// FP8 GEMV Kernels
// ============================================================================

template<typename Config = GemvFP8Config>
__global__ void gemv_fp8_kernel(
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
    float acc = 0.0f;
    const uint8_t* B_col = B_fp8 + global_n;

    int k = 0;
    constexpr int UNROLL = Config::UNROLL_K;

    for (; k + UNROLL <= K; k += UNROLL) {
        const int scale_block_k = k / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_block_k * scale_stride_n + scale_block_n]);

        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            int curr_scale_block_k = kk / Config::BLOCK_QUANT_SIZE;
            if (curr_scale_block_k != scale_block_k) {
                scale = __bfloat162float(B_scale[curr_scale_block_k * scale_stride_n + scale_block_n]);
            }

            float a = __bfloat162float(A[kk]);
            uint8_t b_fp8 = B_col[kk * N];
            float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;
            acc = fmaf(a, b, acc);
        }
    }

    for (; k < K; ++k) {
        const int scale_block_k = k / Config::BLOCK_QUANT_SIZE;
        float scale = __bfloat162float(B_scale[scale_block_k * scale_stride_n + scale_block_n]);

        float a = __bfloat162float(A[k]);
        uint8_t b_fp8 = B_col[k * N];
        float b = fp8_e4m3_to_f32_lut(b_fp8) * scale;
        acc = fmaf(a, b, acc);
    }

    C[global_n] = __float2bfloat16(acc);
}

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

template<typename Config = GemvFP8Config>
__global__ void gemv_fp8_batched_kernel(
    __nv_bfloat16 const* __restrict__ A,
    uint8_t const* __restrict__ B_fp8,
    __nv_bfloat16 const* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
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

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_fp8(
    const __nv_bfloat16* A,
    const uint8_t* B_fp8,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    using Config = GemvFP8Config;

    int scale_stride_n = (N + Config::BLOCK_QUANT_SIZE - 1) / Config::BLOCK_QUANT_SIZE;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);

    gemv_fp8_vec4_kernel<Config><<<grid, block, 0, stream>>>(
        A, B_fp8, B_scale, C, K, N, scale_stride_n
    );

    return cudaGetLastError();
}

bool dispatch_gemv_fp8(
    const __nv_bfloat16* A,
    const uint8_t* B_fp8,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    if (M == 1 && N >= GemvFP8Config::BLOCK_SIZE) {
        launch_gemv_fp8(A, B_fp8, B_scale, C, K, N, stream);
        return true;
    }
    return false;
}

cudaError_t launch_gemv_fp8_batched(
    const __nv_bfloat16* A,
    const uint8_t* B_fp8,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    int batch_count,
    cudaStream_t stream
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
