// Copyright (c) 2024 PyGPUkit Authors
// SPDX-License-Identifier: MIT
//
// MoE token permutation kernels
// Routes tokens to experts and builds dispatch tables

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace moe {

// =============================================================================
// Count tokens per expert (histogram)
// =============================================================================

__global__ void count_tokens_per_expert_kernel(
    const int32_t* __restrict__ expert_indices,  // [num_tokens, k]
    int32_t* __restrict__ expert_counts,         // [num_experts]
    int num_tokens,
    int num_experts,
    int k
) {
    // Use atomicAdd to count
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * k;

    if (idx < total) {
        int expert_id = expert_indices[idx];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_counts[expert_id], 1);
        }
    }
}

// =============================================================================
// Compute expert offsets (exclusive prefix sum)
// Simple single-block implementation for small num_experts
// =============================================================================

__global__ void compute_expert_offsets_kernel(
    const int32_t* __restrict__ expert_counts,  // [num_experts]
    int32_t* __restrict__ expert_offsets,       // [num_experts + 1]
    int num_experts
) {
    // Single thread exclusive scan (num_experts is small, typically 8-64)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t offset = 0;
        for (int i = 0; i < num_experts; ++i) {
            expert_offsets[i] = offset;
            offset += expert_counts[i];
        }
        expert_offsets[num_experts] = offset;  // Total count
    }
}

// =============================================================================
// Build permutation indices
// Maps each (token, expert_slot) to position in sorted order
// =============================================================================

__global__ void build_permute_indices_kernel(
    const int32_t* __restrict__ expert_indices,   // [num_tokens, k]
    const int32_t* __restrict__ expert_offsets,   // [num_experts + 1]
    int32_t* __restrict__ permute_indices,        // [num_tokens * k]
    int32_t* __restrict__ expert_write_offsets,   // [num_experts] - atomic counters
    int num_tokens,
    int num_experts,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * k;

    if (idx < total) {
        int expert_id = expert_indices[idx];
        if (expert_id >= 0 && expert_id < num_experts) {
            // Atomically get write position within this expert's segment
            int write_pos = atomicAdd(&expert_write_offsets[expert_id], 1);
            int base_offset = expert_offsets[expert_id];
            permute_indices[base_offset + write_pos] = idx;
        }
    }
}

// =============================================================================
// Gather hidden states for experts
// Reorders hidden states according to permutation
// =============================================================================

template <typename T>
__global__ void gather_hidden_states_kernel(
    const T* __restrict__ hidden,           // [num_tokens, hidden_size]
    const int32_t* __restrict__ permute_indices,  // [num_tokens * k]
    T* __restrict__ gathered,               // [num_tokens * k, hidden_size]
    int num_tokens,
    int hidden_size,
    int k
) {
    int out_idx = blockIdx.x;  // Output token index
    int total_out = num_tokens * k;

    if (out_idx >= total_out) return;

    // Get original token index (permute_indices stores token_idx * k + slot)
    int perm_idx = permute_indices[out_idx];
    int token_idx = perm_idx / k;

    // Copy hidden state
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        gathered[out_idx * hidden_size + h] = hidden[token_idx * hidden_size + h];
    }
}

// Vectorized gather for better memory bandwidth (float4)
__global__ void gather_hidden_states_f32_vec4_kernel(
    const float* __restrict__ hidden,
    const int32_t* __restrict__ permute_indices,
    float* __restrict__ gathered,
    int num_tokens,
    int hidden_size,
    int k
) {
    int out_idx = blockIdx.x;
    int total_out = num_tokens * k;

    if (out_idx >= total_out) return;

    int perm_idx = permute_indices[out_idx];
    int token_idx = perm_idx / k;

    const float4* src = reinterpret_cast<const float4*>(hidden + token_idx * hidden_size);
    float4* dst = reinterpret_cast<float4*>(gathered + out_idx * hidden_size);
    int vec_size = hidden_size / 4;

    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        dst[i] = src[i];
    }

    // Handle remainder
    int remainder_start = vec_size * 4;
    for (int i = remainder_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        gathered[out_idx * hidden_size + i] = hidden[token_idx * hidden_size + i];
    }
}

// BF16 vectorized gather (bfloat162)
__global__ void gather_hidden_states_bf16_vec2_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int32_t* __restrict__ permute_indices,
    __nv_bfloat16* __restrict__ gathered,
    int num_tokens,
    int hidden_size,
    int k
) {
    int out_idx = blockIdx.x;
    int total_out = num_tokens * k;

    if (out_idx >= total_out) return;

    int perm_idx = permute_indices[out_idx];
    int token_idx = perm_idx / k;

    const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
        hidden + token_idx * hidden_size);
    __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
        gathered + out_idx * hidden_size);
    int vec_size = hidden_size / 2;

    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        dst[i] = src[i];
    }

    // Handle odd hidden_size
    if (hidden_size % 2 != 0 && threadIdx.x == 0) {
        gathered[out_idx * hidden_size + hidden_size - 1] =
            hidden[token_idx * hidden_size + hidden_size - 1];
    }
}

// =============================================================================
// Scatter expert outputs back to original order (unpermute)
// =============================================================================

template <typename T>
__global__ void scatter_expert_outputs_kernel(
    const T* __restrict__ expert_out,       // [num_tokens * k, hidden_size]
    const T* __restrict__ router_weights,   // [num_tokens, k]
    const int32_t* __restrict__ permute_indices,  // [num_tokens * k]
    T* __restrict__ output,                 // [num_tokens, hidden_size]
    int num_tokens,
    int hidden_size,
    int k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // For each output position, accumulate weighted expert outputs
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;

        for (int slot = 0; slot < k; ++slot) {
            int flat_idx = token_idx * k + slot;
            // Find where this (token, slot) was placed in permuted order
            // We need reverse lookup - scan permute_indices
            // TODO: Optimize with reverse permutation array
        }

        output[token_idx * hidden_size + h] = T(sum);
    }
}

// Simpler scatter using reverse permutation (pre-computed)
template <typename T>
__global__ void scatter_with_reverse_perm_kernel(
    const T* __restrict__ expert_out,           // [num_tokens * k, hidden_size]
    const T* __restrict__ router_weights,       // [num_tokens, k]
    const int32_t* __restrict__ reverse_perm,   // [num_tokens * k] -> position in expert_out
    T* __restrict__ output,                     // [num_tokens, hidden_size]
    int num_tokens,
    int hidden_size,
    int k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;

        for (int slot = 0; slot < k; ++slot) {
            int flat_idx = token_idx * k + slot;
            int expert_out_idx = reverse_perm[flat_idx];
            float weight = float(router_weights[flat_idx]);
            sum += weight * float(expert_out[expert_out_idx * hidden_size + h]);
        }

        output[token_idx * hidden_size + h] = T(sum);
    }
}

// Build reverse permutation
__global__ void build_reverse_perm_kernel(
    const int32_t* __restrict__ permute_indices,   // [num_tokens * k]
    int32_t* __restrict__ reverse_perm,            // [num_tokens * k]
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int orig_idx = permute_indices[idx];
        reverse_perm[orig_idx] = idx;
    }
}

}  // namespace moe
}  // namespace pygpukit
