/**
 * INT8/INT4 Quantization Kernels for PyGPUkit
 *
 * Weight-only quantization: INT8 weights + FP16 activations -> FP16 output
 * Dequantization happens on-the-fly during matmul for memory efficiency.
 *
 * Supported formats:
 * - Per-column INT8: W_int8[out_features, in_features] + scale[out_features]
 * - Per-group INT8: W_int8[out_features, in_features] + scale[out_features, num_groups]
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace quantize {

// ============================================================================
// INT8 Dequantization Kernels (Per-Row Scaling)
// ============================================================================

/**
 * Dequantize INT8 to FP16: output = input_int8 * scale
 * Per-row scaling (one scale per row/output channel)
 */
__global__ void dequantize_int8_to_f16_kernel(
    const int8_t* __restrict__ input,   // [rows, cols]
    const __half* __restrict__ scale,   // [rows] - per-row scale
    __half* __restrict__ output,        // [rows, cols]
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int row = idx / cols;
        float val = static_cast<float>(input[idx]) * __half2float(scale[row]);
        output[idx] = __float2half(val);
    }
}

/**
 * Dequantize INT8 to FP32: output = input_int8 * scale
 * Per-row scaling
 */
__global__ void dequantize_int8_to_f32_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int row = idx / cols;
        output[idx] = static_cast<float>(input[idx]) * scale[row];
    }
}

// ============================================================================
// Quantized Linear (INT8 weight × FP16 activation → FP16 output)
// ============================================================================

/**
 * INT8 Weight × FP16 Activation -> FP16 Output
 *
 * Performs: output = activation @ (weight_int8 * scale).T
 *
 * Parameters:
 *   activation: [M, K] FP16 input
 *   weight_int8: [N, K] INT8 quantized weight (row-major, transposed for matmul)
 *   scale: [N] FP16 per-output-channel scale
 *   output: [M, N] FP16 result
 *
 * Dequantization happens on-the-fly: no intermediate FP16 weight storage needed.
 */
__global__ void linear_int8_f16_kernel(
    const __half* __restrict__ activation,  // [M, K]
    const int8_t* __restrict__ weight,      // [N, K] (weight for output channel n is weight[n*K:(n+1)*K])
    const __half* __restrict__ scale,       // [N]
    __half* __restrict__ output,            // [M, N]
    int M,  // batch size
    int N,  // out_features
    int K   // in_features
) {
    // Each thread computes one output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row >= M || col >= N) return;

    // Accumulate in FP32 for precision
    float acc = 0.0f;

    // Get scale for this output channel
    float s = __half2float(scale[col]);

    // Dot product: activation[row, :] @ weight[col, :]
    for (int k = 0; k < K; k++) {
        float a = __half2float(activation[row * K + k]);
        float w = static_cast<float>(weight[col * K + k]) * s;
        acc += a * w;
    }

    output[row * N + col] = __float2half(acc);
}

/**
 * Optimized INT8 Linear with shared memory tiling
 *
 * Uses shared memory to reduce global memory accesses.
 * Tile size: TILE_M x TILE_N with TILE_K reduction.
 */
constexpr int Q_TILE_M = 16;
constexpr int Q_TILE_N = 16;
constexpr int Q_TILE_K = 32;

__global__ void linear_int8_f16_tiled_kernel(
    const __half* __restrict__ activation,  // [M, K]
    const int8_t* __restrict__ weight,      // [N, K]
    const __half* __restrict__ scale,       // [N]
    __half* __restrict__ output,            // [M, N]
    int M,
    int N,
    int K
) {
    // Block position
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Thread position within block
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Global position
    int row = block_row * Q_TILE_M + thread_row;
    int col = block_col * Q_TILE_N + thread_col;

    // Shared memory for tiles
    __shared__ float As[Q_TILE_M][Q_TILE_K];
    __shared__ float Ws[Q_TILE_N][Q_TILE_K];

    // Get scale for this output channel
    float s = (col < N) ? __half2float(scale[col]) : 0.0f;

    // Accumulator
    float acc = 0.0f;

    // Loop over K dimension in tiles
    int num_tiles = (K + Q_TILE_K - 1) / Q_TILE_K;

    for (int tile = 0; tile < num_tiles; tile++) {
        int k_start = tile * Q_TILE_K;

        // Load activation tile (each thread loads multiple elements)
        // Thread (ty, tx) loads element (ty, tx) and potentially more
        for (int k_offset = thread_col; k_offset < Q_TILE_K; k_offset += Q_TILE_N) {
            int k = k_start + k_offset;
            if (row < M && k < K) {
                As[thread_row][k_offset] = __half2float(activation[row * K + k]);
            } else {
                As[thread_row][k_offset] = 0.0f;
            }
        }

        // Load weight tile (dequantize on load)
        for (int k_offset = thread_row; k_offset < Q_TILE_K; k_offset += Q_TILE_M) {
            int k = k_start + k_offset;
            if (col < N && k < K) {
                Ws[thread_col][k_offset] = static_cast<float>(weight[col * K + k]) * s;
            } else {
                Ws[thread_col][k_offset] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int kk = 0; kk < Q_TILE_K; kk++) {
            acc += As[thread_row][kk] * Ws[thread_col][kk];
        }

        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        output[row * N + col] = __float2half(acc);
    }
}

// ============================================================================
// Quantization Utility Kernels (Per-Row for Linear Layers)
// ============================================================================

/**
 * Quantize FP16 to INT8 with per-row scaling
 *
 * For weight [N, K] (N=out_features, K=in_features):
 * Each row (output channel) gets its own scale factor.
 *
 * weight_int8[row, col] = round(weight_fp16[row, col] / scale[row] * 127)
 */
__global__ void quantize_f16_to_int8_kernel(
    const __half* __restrict__ input,   // [rows, cols]
    int8_t* __restrict__ output,        // [rows, cols]
    __half* __restrict__ scale,         // [rows] - per-row scale
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Step 1: Find max absolute value in this row (using all threads in block)
    extern __shared__ float smem[];
    float* row_max = smem;

    float thread_max = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = fabsf(__half2float(input[row * cols + col]));
        thread_max = fmaxf(thread_max, val);
    }

    // Reduce within block
    row_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            row_max[threadIdx.x] = fmaxf(row_max[threadIdx.x], row_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_val = row_max[0];
    float row_scale = max_val / 127.0f;

    // Avoid division by zero
    if (row_scale < 1e-10f) row_scale = 1e-10f;

    // Step 2: Quantize
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = __half2float(input[row * cols + col]);
        int quantized = __float2int_rn(val / row_scale);
        // Clamp to INT8 range
        quantized = max(-128, min(127, quantized));
        output[row * cols + col] = static_cast<int8_t>(quantized);
    }

    // Write scale
    if (threadIdx.x == 0) {
        scale[row] = __float2half(row_scale);
    }
}

/**
 * Quantize FP32 to INT8 with per-row scaling
 */
__global__ void quantize_f32_to_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* row_max = smem;

    float thread_max = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = fabsf(input[row * cols + col]);
        thread_max = fmaxf(thread_max, val);
    }

    row_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            row_max[threadIdx.x] = fmaxf(row_max[threadIdx.x], row_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_val = row_max[0];
    float row_scale = max_val / 127.0f;
    if (row_scale < 1e-10f) row_scale = 1e-10f;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = input[row * cols + col];
        int quantized = __float2int_rn(val / row_scale);
        quantized = max(-128, min(127, quantized));
        output[row * cols + col] = static_cast<int8_t>(quantized);
    }

    if (threadIdx.x == 0) {
        scale[row] = row_scale;
    }
}

} // namespace quantize
} // namespace ops
} // namespace pygpukit
