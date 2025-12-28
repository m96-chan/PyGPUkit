/**
 * Memory operation kernels
 *
 * Provides: Transpose, Concat, RepeatInterleave, Copy operations
 * Extracted from nn_kernels.cuh for better organization
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// Matrix Transpose
// ============================================================================

// Transpose kernel using shared memory for coalesced access
// Input: [rows, cols], Output: [cols, rows]
// Uses 32x32 tile with padding to avoid bank conflicts

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_f32_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Transpose indices for output
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // swapped
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // swapped

    // Write transposed tile (coalesced write)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_f64_kernel(const double* __restrict__ input,
                                      double* __restrict__ output,
                                      int rows, int cols) {
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_f16_kernel(const __half* __restrict__ input,
                                      __half* __restrict__ output,
                                      int rows, int cols) {
    __shared__ __half tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                       __nv_bfloat16* __restrict__ output,
                                       int rows, int cols) {
    __shared__ __nv_bfloat16 tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Tensor Manipulation Operations
// ============================================================================

// Concat two tensors along axis 0
// src1: [dim0_1, dim1, dim2], src2: [dim0_2, dim1, dim2]
// dst: [dim0_1 + dim0_2, dim1, dim2]
__global__ void concat_axis0_f32_kernel(
    const float* __restrict__ src1,
    const float* __restrict__ src2,
    float* __restrict__ dst,
    size_t dim0_1,      // First tensor's dim0
    size_t dim0_2,      // Second tensor's dim0
    size_t stride       // dim1 * dim2 (elements per row)
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// FP16 concat along axis 0
__global__ void concat_axis0_f16_kernel(
    const __half* __restrict__ src1,
    const __half* __restrict__ src2,
    __half* __restrict__ dst,
    size_t dim0_1,
    size_t dim0_2,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// BF16 concat along axis 0
__global__ void concat_axis0_bf16_kernel(
    const __nv_bfloat16* __restrict__ src1,
    const __nv_bfloat16* __restrict__ src2,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0_1,
    size_t dim0_2,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// UInt8 concat along axis 0 (for FP8 weights)
__global__ void concat_axis0_u8_kernel(
    const uint8_t* __restrict__ src1,
    const uint8_t* __restrict__ src2,
    uint8_t* __restrict__ dst,
    size_t dim0_1,
    size_t dim0_2,
    size_t stride
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_src1 = dim0_1 * stride;
    size_t total = (dim0_1 + dim0_2) * stride;

    if (idx < total) {
        if (idx < total_src1) {
            dst[idx] = src1[idx];
        } else {
            dst[idx] = src2[idx - total_src1];
        }
    }
}

// Repeat tensor along axis 1 (for GQA expansion)
// src: [dim0, dim1, dim2] -> dst: [dim0, dim1 * repeats, dim2]
// Each element in dim1 is repeated 'repeats' times
__global__ void repeat_interleave_axis1_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t repeats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * repeats * dim2;

    if (idx < total) {
        // Compute output coordinates
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1_out = remaining % (dim1 * repeats);
        size_t d0 = remaining / (dim1 * repeats);

        // Map output d1 to input d1 (integer division gives the source index)
        size_t d1_in = d1_out / repeats;

        // Compute source index
        size_t src_idx = d0 * dim1 * dim2 + d1_in * dim2 + d2;
        dst[idx] = src[src_idx];
    }
}

// FP16 repeat interleave along axis 1
__global__ void repeat_interleave_axis1_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t repeats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * repeats * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1_out = remaining % (dim1 * repeats);
        size_t d0 = remaining / (dim1 * repeats);
        size_t d1_in = d1_out / repeats;
        size_t src_idx = d0 * dim1 * dim2 + d1_in * dim2 + d2;
        dst[idx] = src[src_idx];
    }
}

// BF16 repeat interleave along axis 1
__global__ void repeat_interleave_axis1_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t repeats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * repeats * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1_out = remaining % (dim1 * repeats);
        size_t d0 = remaining / (dim1 * repeats);
        size_t d1_in = d1_out / repeats;
        size_t src_idx = d0 * dim1 * dim2 + d1_in * dim2 + d2;
        dst[idx] = src[src_idx];
    }
}

// Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]
// Swaps axes 0 and 1
__global__ void transpose_021_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        // Compute source coordinates [d0, d1, d2]
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        // Compute destination index [d1, d0, d2]
        size_t dst_idx = d1 * dim0 * dim2 + d0 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// Transpose 3D FP16: [d0, d1, d2] -> [d1, d0, d2]
__global__ void transpose_021_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d1 * dim0 * dim2 + d0 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// Transpose 3D BF16: [d0, d1, d2] -> [d1, d0, d2]
__global__ void transpose_021_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d1 * dim0 * dim2 + d0 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// ============================================================================
// 3D Transpose: [d0, d1, d2] -> [d0, d2, d1]
// Swaps last two axes (common in attention)
// ============================================================================

__global__ void transpose_012_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        // Compute source coordinates [d0, d1, d2]
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        // Compute destination index [d0, d2, d1]
        size_t dst_idx = d0 * dim2 * dim1 + d2 * dim1 + d1;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_012_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d0 * dim2 * dim1 + d2 * dim1 + d1;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_012_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2;

    if (idx < total) {
        size_t d2 = idx % dim2;
        size_t remaining = idx / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d0 * dim2 * dim1 + d2 * dim1 + d1;
        dst[dst_idx] = src[idx];
    }
}

// ============================================================================
// 4D Transpose: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
// Swaps axes 1 and 2 (common in attention: batch, seq, heads, dim -> batch, heads, seq, dim)
// ============================================================================

__global__ void transpose_0213_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2 * dim3;

    if (idx < total) {
        // Compute source coordinates [d0, d1, d2, d3]
        size_t d3 = idx % dim3;
        size_t remaining = idx / dim3;
        size_t d2 = remaining % dim2;
        remaining = remaining / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        // Compute destination index [d0, d2, d1, d3]
        size_t dst_idx = d0 * (dim2 * dim1 * dim3) + d2 * (dim1 * dim3) + d1 * dim3 + d3;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_0213_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2 * dim3;

    if (idx < total) {
        size_t d3 = idx % dim3;
        size_t remaining = idx / dim3;
        size_t d2 = remaining % dim2;
        remaining = remaining / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d0 * (dim2 * dim1 * dim3) + d2 * (dim1 * dim3) + d1 * dim3 + d3;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_0213_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2 * dim3;

    if (idx < total) {
        size_t d3 = idx % dim3;
        size_t remaining = idx / dim3;
        size_t d2 = remaining % dim2;
        remaining = remaining / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d0 * (dim2 * dim1 * dim3) + d2 * (dim1 * dim3) + d1 * dim3 + d3;
        dst[dst_idx] = src[idx];
    }
}

// ============================================================================
// 4D Transpose: [d0, d1, d2, d3] -> [d0, d1, d3, d2]
// Swaps last two axes (for K^T in attention)
// ============================================================================

__global__ void transpose_0132_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2 * dim3;

    if (idx < total) {
        // Compute source coordinates [d0, d1, d2, d3]
        size_t d3 = idx % dim3;
        size_t remaining = idx / dim3;
        size_t d2 = remaining % dim2;
        remaining = remaining / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        // Compute destination index [d0, d1, d3, d2]
        size_t dst_idx = d0 * (dim1 * dim3 * dim2) + d1 * (dim3 * dim2) + d3 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_0132_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2 * dim3;

    if (idx < total) {
        size_t d3 = idx % dim3;
        size_t remaining = idx / dim3;
        size_t d2 = remaining % dim2;
        remaining = remaining / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d0 * (dim1 * dim3 * dim2) + d1 * (dim3 * dim2) + d3 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_0132_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim0 * dim1 * dim2 * dim3;

    if (idx < total) {
        size_t d3 = idx % dim3;
        size_t remaining = idx / dim3;
        size_t d2 = remaining % dim2;
        remaining = remaining / dim2;
        size_t d1 = remaining % dim1;
        size_t d0 = remaining / dim1;

        size_t dst_idx = d0 * (dim1 * dim3 * dim2) + d1 * (dim3 * dim2) + d3 * dim2 + d2;
        dst[dst_idx] = src[idx];
    }
}

// Reshape with copy (ensures contiguous output)
// Simply copies data - reshape is handled by changing shape metadata
__global__ void copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// FP16 copy kernel
__global__ void copy_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// BF16 copy kernel
__global__ void copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// INT32 copy kernel (for position buffers in CUDA Graph)
__global__ void copy_i32_kernel(
    const int* __restrict__ src,
    int* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Arange - generate sequence [start, start+step, start+2*step, ...]
// ============================================================================

__global__ void arange_f32_kernel(float* output, float start, float step, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = start + static_cast<float>(idx) * step;
    }
}

__global__ void arange_i32_kernel(int* output, int start, int step, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = start + static_cast<int>(idx) * step;
    }
}

__global__ void arange_i64_kernel(int64_t* output, int64_t start, int64_t step, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = start + static_cast<int64_t>(idx) * step;
    }
}

// ============================================================================
// Scatter Add - indexed accumulation: output[indices[i]] += src[i]
// ============================================================================

__global__ void scatter_add_f32_kernel(
    float* __restrict__ output,
    const int64_t* __restrict__ indices,
    const float* __restrict__ src,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&output[indices[idx]], src[idx]);
    }
}

__global__ void scatter_add_f16_kernel(
    __half* __restrict__ output,
    const int64_t* __restrict__ indices,
    const __half* __restrict__ src,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP16 atomicAdd requires sm_70+
        atomicAdd(&output[indices[idx]], src[idx]);
    }
}

__global__ void scatter_add_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const int64_t* __restrict__ indices,
    const __nv_bfloat16* __restrict__ src,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16 atomicAdd requires sm_80+
        atomicAdd(&output[indices[idx]], src[idx]);
    }
}

} // namespace nn
} // namespace ops
} // namespace pygpukit
