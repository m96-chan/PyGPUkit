/**
 * KV Cache Update Kernels for LLM Inference
 *
 * Provides fixed-length KV cache update kernels optimized for CUDA Graph execution.
 * Supports both MHA (Multi-Head Attention) and GQA (Grouped Query Attention) layouts.
 *
 * Cache Layouts:
 * - Standard: [max_seq_len, num_kv_heads, head_dim]
 * - GQA-expanded: [num_heads, max_seq_len, head_dim] (transposed + expanded)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// KV Cache Update Kernel (Fixed-Length KV Cache for CUDA Graph)
// ============================================================================

// Copy new K/V values to position in fixed-length cache
// new_kv: [1, num_kv_heads, head_dim] - single token K or V
// cache: [max_seq_len, num_kv_heads, head_dim] - pre-allocated cache
// position: where to write in cache (0-indexed)
template <typename T>
__global__ void kv_cache_update_kernel(
    const T* __restrict__ new_kv,
    T* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    // Total elements per position: num_kv_heads * head_dim
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // new_kv is [1, num_kv_heads, head_dim], so offset is just idx
        // cache is [max_seq_len, num_kv_heads, head_dim]
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// FP16 version
__global__ void kv_cache_update_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// BF16 version
__global__ void kv_cache_update_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// FP32 version
__global__ void kv_cache_update_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int position
) {
    int total_elements = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int cache_offset = position * total_elements + idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// Prefill version: Copy multiple tokens from prefill K/V to cache
// new_kv: [seq_len, num_kv_heads, head_dim]
// cache: [max_seq_len, num_kv_heads, head_dim]
// start_pos: where to start writing in cache
// seq_len: number of tokens to copy
__global__ void kv_cache_prefill_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len
) {
    int elements_per_pos = num_kv_heads * head_dim;
    int total_elements = seq_len * elements_per_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int seq_pos = idx / elements_per_pos;
        int elem_idx = idx % elements_per_pos;
        int cache_offset = (start_pos + seq_pos) * elements_per_pos + elem_idx;
        cache[cache_offset] = new_kv[idx];
    }
}

__global__ void kv_cache_prefill_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len
) {
    int elements_per_pos = num_kv_heads * head_dim;
    int total_elements = seq_len * elements_per_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int seq_pos = idx / elements_per_pos;
        int elem_idx = idx % elements_per_pos;
        int cache_offset = (start_pos + seq_pos) * elements_per_pos + elem_idx;
        cache[cache_offset] = new_kv[idx];
    }
}

__global__ void kv_cache_prefill_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len
) {
    int elements_per_pos = num_kv_heads * head_dim;
    int total_elements = seq_len * elements_per_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int seq_pos = idx / elements_per_pos;
        int elem_idx = idx % elements_per_pos;
        int cache_offset = (start_pos + seq_pos) * elements_per_pos + elem_idx;
        cache[cache_offset] = new_kv[idx];
    }
}

// ============================================================================
// GQA-expanded KV Cache Update (for CUDA Graph optimization)
// ============================================================================
// These kernels write to a transposed, GQA-expanded cache layout:
// Input: new_kv [1, num_kv_heads, head_dim] or [seq_len, num_kv_heads, head_dim]
// Cache: [num_heads, max_seq_len, head_dim] (transposed and expanded)
// This eliminates per-step transpose and GQA expansion overhead.

// Single token update with GQA expansion
// new_kv: [1, num_kv_heads, head_dim]
// cache: [num_heads, max_seq_len, head_dim]
__global__ void kv_cache_update_gqa_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    // Total output elements: num_heads * head_dim
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;

        // GQA: find source kv_head
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;

        // Source: new_kv[0, kv_head, d] = new_kv[kv_head * head_dim + d]
        int src_offset = kv_head * head_dim + d;

        // Dest: cache[head, position, d] = cache[head * max_seq_len * head_dim + position * head_dim + d]
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;

        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_update_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_update_gqa_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

// =============================================================================
// KV Cache Update with GPU position pointer (for CUDA Graph replay)
// =============================================================================

__global__ void kv_cache_update_gqa_f16_kernel_ptr(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const int* __restrict__ position_ptr
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int position = *reinterpret_cast<volatile const int*>(position_ptr);
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_update_gqa_bf16_kernel_ptr(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const int* __restrict__ position_ptr
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int position = *reinterpret_cast<volatile const int*>(position_ptr);
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_update_gqa_f32_kernel_ptr(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const int* __restrict__ position_ptr
) {
    // Use volatile read to ensure fresh value during CUDA Graph replay
    int position = *reinterpret_cast<volatile const int*>(position_ptr);
    int total_elements = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int head = idx / head_dim;
        int d = idx % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + position * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

// Prefill with GQA expansion
// new_kv: [seq_len, num_kv_heads, head_dim]
// cache: [num_heads, max_seq_len, head_dim]
__global__ void kv_cache_prefill_gqa_f16_kernel(
    const __half* __restrict__ new_kv,
    __half* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int seq_len
) {
    // Total output elements: seq_len * num_heads * head_dim
    int total_elements = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int elements_per_seq = num_heads * head_dim;
        int seq_pos = idx / elements_per_seq;
        int remaining = idx % elements_per_seq;
        int head = remaining / head_dim;
        int d = remaining % head_dim;

        // GQA: find source kv_head
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;

        // Source: new_kv[seq_pos, kv_head, d]
        int src_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + d;

        // Dest: cache[head, start_pos + seq_pos, d]
        int dst_offset = head * max_seq_len * head_dim + (start_pos + seq_pos) * head_dim + d;

        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_prefill_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ new_kv,
    __nv_bfloat16* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int seq_len
) {
    int total_elements = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int elements_per_seq = num_heads * head_dim;
        int seq_pos = idx / elements_per_seq;
        int remaining = idx % elements_per_seq;
        int head = remaining / head_dim;
        int d = remaining % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + (start_pos + seq_pos) * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

__global__ void kv_cache_prefill_gqa_f32_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int seq_len
) {
    int total_elements = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int elements_per_seq = num_heads * head_dim;
        int seq_pos = idx / elements_per_seq;
        int remaining = idx % elements_per_seq;
        int head = remaining / head_dim;
        int d = remaining % head_dim;
        int num_kv_groups = num_heads / num_kv_heads;
        int kv_head = head / num_kv_groups;
        int src_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + d;
        int dst_offset = head * max_seq_len * head_dim + (start_pos + seq_pos) * head_dim + d;
        cache[dst_offset] = new_kv[src_offset];
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
