// Copyright (c) 2024 PyGPUkit Authors
// SPDX-License-Identifier: MIT
//
// Mixture of Experts (MoE) core kernels
// Includes router, dispatch, and combine operations

#pragma once

#include "topk_kernels.cuh"
#include "permute_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace moe {

// =============================================================================
// MoE Forward Pass Components
// =============================================================================

// Structure to hold MoE dispatch info
struct MoEDispatchInfo {
    int32_t* expert_indices;    // [num_tokens, k] - selected expert IDs
    int32_t* expert_counts;     // [num_experts] - tokens per expert
    int32_t* expert_offsets;    // [num_experts + 1] - cumulative offsets
    int32_t* permute_indices;   // [num_tokens * k] - reorder mapping
    int32_t* reverse_perm;      // [num_tokens * k] - inverse mapping
    void* router_weights;       // [num_tokens, k] - softmax weights
    int num_tokens;
    int num_experts;
    int k;
};

// =============================================================================
// Expert FFN kernels (SwiGLU variant for Mixtral)
// For small models, loop over experts is acceptable
// For large models, use grouped GEMM
// =============================================================================

// Simple per-expert SwiGLU: gate(x) * up(x), then down
// This is the naive implementation - each expert processed separately
template <typename T>
__global__ void expert_swiglu_kernel(
    const T* __restrict__ input,       // [batch_size, hidden_size]
    const T* __restrict__ gate_weight, // [intermediate_size, hidden_size]
    const T* __restrict__ up_weight,   // [intermediate_size, hidden_size]
    const T* __restrict__ down_weight, // [hidden_size, intermediate_size]
    T* __restrict__ output,            // [batch_size, hidden_size]
    int batch_size,
    int hidden_size,
    int intermediate_size
) {
    // Each block handles one token
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size) return;

    extern __shared__ char smem[];
    float* gate_act = reinterpret_cast<float*>(smem);
    float* up_act = gate_act + intermediate_size;

    const T* x = input + token_idx * hidden_size;
    T* y = output + token_idx * hidden_size;

    // Step 1: Compute gate and up projections
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;

        for (int j = 0; j < hidden_size; ++j) {
            float xj = float(x[j]);
            gate_sum += xj * float(gate_weight[i * hidden_size + j]);
            up_sum += xj * float(up_weight[i * hidden_size + j]);
        }

        // SiLU activation on gate: x * sigmoid(x)
        float silu = gate_sum / (1.0f + expf(-gate_sum));
        gate_act[i] = silu * up_sum;
    }

    __syncthreads();

    // Step 2: Down projection
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; ++j) {
            sum += gate_act[j] * float(down_weight[i * intermediate_size + j]);
        }
        y[i] = T(sum);
    }
}

// =============================================================================
// Fused router (Linear + TopK + Softmax)
// =============================================================================

template <typename T>
__global__ void moe_router_kernel(
    const T* __restrict__ hidden,        // [num_tokens, hidden_size]
    const T* __restrict__ gate_weight,   // [num_experts, hidden_size]
    T* __restrict__ router_weights,      // [num_tokens, k]
    int32_t* __restrict__ expert_indices,// [num_tokens, k]
    int num_tokens,
    int hidden_size,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    extern __shared__ float logits[];
    const T* x = hidden + token_idx * hidden_size;

    // Step 1: Compute logits for all experts
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            sum += float(x[h]) * float(gate_weight[e * hidden_size + h]);
        }
        logits[e] = sum;
    }
    __syncthreads();

    // Step 2: Top-K selection (single thread for simplicity)
    if (threadIdx.x == 0) {
        float local_logits[64];
        for (int i = 0; i < num_experts; ++i) {
            local_logits[i] = logits[i];
        }

        float selected_logits[8];
        int selected_indices[8];

        for (int j = 0; j < k; ++j) {
            float max_val = -1e30f;
            int max_idx = 0;
            for (int i = 0; i < num_experts; ++i) {
                if (local_logits[i] > max_val) {
                    max_val = local_logits[i];
                    max_idx = i;
                }
            }
            selected_logits[j] = max_val;
            selected_indices[j] = max_idx;
            local_logits[max_idx] = -1e30f;
        }

        // Step 3: Softmax over selected
        float max_val = selected_logits[0];
        for (int j = 1; j < k; ++j) {
            max_val = fmaxf(max_val, selected_logits[j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            selected_logits[j] = expf(selected_logits[j] - max_val);
            sum += selected_logits[j];
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < k; ++j) {
            router_weights[token_idx * k + j] = T(selected_logits[j] * inv_sum);
            expert_indices[token_idx * k + j] = selected_indices[j];
        }
    }
}

// =============================================================================
// Combined scatter-add for expert outputs
// =============================================================================

template <typename T>
__global__ void moe_combine_outputs_kernel(
    const T* __restrict__ expert_outputs,   // [total_expert_tokens, hidden_size]
    const T* __restrict__ router_weights,   // [num_tokens, k]
    const int32_t* __restrict__ token_indices,  // [total_expert_tokens] - original token idx
    const int32_t* __restrict__ slot_indices,   // [total_expert_tokens] - which k slot
    T* __restrict__ output,                 // [num_tokens, hidden_size]
    int num_tokens,
    int hidden_size,
    int k,
    int total_expert_tokens
) {
    // Each block handles one expert output
    int expert_token_idx = blockIdx.x;
    if (expert_token_idx >= total_expert_tokens) return;

    int token_idx = token_indices[expert_token_idx];
    int slot_idx = slot_indices[expert_token_idx];
    float weight = float(router_weights[token_idx * k + slot_idx]);

    const T* expert_out = expert_outputs + expert_token_idx * hidden_size;
    T* token_out = output + token_idx * hidden_size;

    // Atomic add (for concurrent writes from multiple experts)
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float val = weight * float(expert_out[h]);
        atomicAdd(&token_out[h], val);
    }
}

// Non-atomic version when we know order (use reverse permutation)
template <typename T>
__global__ void moe_combine_outputs_ordered_kernel(
    const T* __restrict__ expert_outputs,   // [num_tokens * k, hidden_size]
    const T* __restrict__ router_weights,   // [num_tokens, k]
    const int32_t* __restrict__ reverse_perm,   // [num_tokens * k]
    T* __restrict__ output,                 // [num_tokens, hidden_size]
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
            int expert_out_pos = reverse_perm[flat_idx];
            float weight = float(router_weights[flat_idx]);
            sum += weight * float(expert_outputs[expert_out_pos * hidden_size + h]);
        }

        output[token_idx * hidden_size + h] = T(sum);
    }
}

// =============================================================================
// Utility: Expand expert_offsets to row_expert_ids
// Used for grouped GEMM v2 API
// =============================================================================

__global__ void expand_expert_offsets_kernel(
    const int32_t* __restrict__ expert_offsets,  // [num_experts + 1]
    int32_t* __restrict__ row_expert_ids,        // [M_total]
    int num_experts,
    int M_total
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M_total) return;

    // Binary search to find which expert this row belongs to
    int low = 0, high = num_experts;
    while (low < high) {
        int mid = (low + high) / 2;
        if (expert_offsets[mid + 1] <= row) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    row_expert_ids[row] = low;
}

}  // namespace moe
}  // namespace pygpukit
