/**
 * Embedding Lookup Kernels
 *
 * Provides: embedding lookup operations for CUDA Graph execution
 * - Single token lookup (with constant and GPU pointer variants)
 * - Batch token lookup
 * - Row slicing for RoPE position embeddings
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// Embedding Lookup (for CUDA Graph - no CPU→GPU transfer)
// ============================================================================
// Copy embedding from GPU matrix to output buffer
// embed_matrix: [vocab_size, hidden_size]
// out: [1, hidden_size]
// token_id: which row to copy

__global__ void embedding_lookup_f16_kernel(
    const __half* __restrict__ embed_matrix,
    __half* __restrict__ out,
    int hidden_size,
    int token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        out[idx] = embed_matrix[token_id * hidden_size + idx];
    }
}

__global__ void embedding_lookup_bf16_kernel(
    const __nv_bfloat16* __restrict__ embed_matrix,
    __nv_bfloat16* __restrict__ out,
    int hidden_size,
    int token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        out[idx] = embed_matrix[token_id * hidden_size + idx];
    }
}

__global__ void embedding_lookup_f32_kernel(
    const float* __restrict__ embed_matrix,
    float* __restrict__ out,
    int hidden_size,
    int token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        out[idx] = embed_matrix[token_id * hidden_size + idx];
    }
}

// =============================================================================
// Embedding Lookup with GPU index pointer (for CUDA Graph replay)
// =============================================================================

__global__ void embedding_lookup_f16_kernel_ptr(
    const __half* __restrict__ embed_matrix,
    __half* __restrict__ out,
    int hidden_size,
    const int* __restrict__ token_id_ptr
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int token_id = *reinterpret_cast<volatile const int*>(token_id_ptr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        out[idx] = embed_matrix[token_id * hidden_size + idx];
    }
}

__global__ void embedding_lookup_bf16_kernel_ptr(
    const __nv_bfloat16* __restrict__ embed_matrix,
    __nv_bfloat16* __restrict__ out,
    int hidden_size,
    const int* __restrict__ token_id_ptr
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int token_id = *reinterpret_cast<volatile const int*>(token_id_ptr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        out[idx] = embed_matrix[token_id * hidden_size + idx];
    }
}

__global__ void embedding_lookup_f32_kernel_ptr(
    const float* __restrict__ embed_matrix,
    float* __restrict__ out,
    int hidden_size,
    const int* __restrict__ token_id_ptr
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int token_id = *reinterpret_cast<volatile const int*>(token_id_ptr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        out[idx] = embed_matrix[token_id * hidden_size + idx];
    }
}

// =============================================================================
// Batch Embedding Lookup with GPU index array (for batch CUDA Graph)
// =============================================================================
// Looks up multiple tokens at once from a GPU buffer of token IDs
// out[i, :] = embed_matrix[token_ids[i], :]

__global__ void embedding_lookup_batch_f16_kernel(
    const __half* __restrict__ embed_matrix,
    __half* __restrict__ out,
    const int* __restrict__ token_ids,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_size;
    if (idx >= total_elements) return;

    int row = idx / hidden_size;
    int col = idx % hidden_size;
    int token_id = token_ids[row];
    out[idx] = embed_matrix[token_id * hidden_size + col];
}

__global__ void embedding_lookup_batch_bf16_kernel(
    const __nv_bfloat16* __restrict__ embed_matrix,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ token_ids,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_size;
    if (idx >= total_elements) return;

    int row = idx / hidden_size;
    int col = idx % hidden_size;
    int token_id = token_ids[row];
    out[idx] = embed_matrix[token_id * hidden_size + col];
}

__global__ void embedding_lookup_batch_f32_kernel(
    const float* __restrict__ embed_matrix,
    float* __restrict__ out,
    const int* __restrict__ token_ids,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_size;
    if (idx >= total_elements) return;

    int row = idx / hidden_size;
    int col = idx % hidden_size;
    int token_id = token_ids[row];
    out[idx] = embed_matrix[token_id * hidden_size + col];
}

// =============================================================================
// Slice Rows Range from GPU Pointer (for batch CUDA Graph - zero allocation)
// =============================================================================
// Copies `count` consecutive rows starting from start_position (read from GPU buffer)
// out[i, :] = table[start_pos + i, :]
// Used for RoPE lookup in batch decode graphs where positions are consecutive

__global__ void slice_rows_range_ptr_f16_kernel(
    const __half* __restrict__ table,
    __half* __restrict__ out,
    const int* __restrict__ start_pos_ptr,
    int count,
    int row_dim
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int start_pos = *reinterpret_cast<volatile const int*>(start_pos_ptr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = count * row_dim;
    if (idx >= total_elements) return;

    int row = idx / row_dim;
    int col = idx % row_dim;
    int src_row = start_pos + row;
    out[idx] = table[src_row * row_dim + col];
}

__global__ void slice_rows_range_ptr_bf16_kernel(
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ start_pos_ptr,
    int count,
    int row_dim
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int start_pos = *reinterpret_cast<volatile const int*>(start_pos_ptr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = count * row_dim;
    if (idx >= total_elements) return;

    int row = idx / row_dim;
    int col = idx % row_dim;
    int src_row = start_pos + row;
    out[idx] = table[src_row * row_dim + col];
}

__global__ void slice_rows_range_ptr_f32_kernel(
    const float* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ start_pos_ptr,
    int count,
    int row_dim
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int start_pos = *reinterpret_cast<volatile const int*>(start_pos_ptr);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = count * row_dim;
    if (idx >= total_elements) return;

    int row = idx / row_dim;
    int col = idx % row_dim;
    int src_row = start_pos + row;
    out[idx] = table[src_row * row_dim + col];
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
