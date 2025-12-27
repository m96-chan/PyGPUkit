// Copyright (c) 2024 PyGPUkit Authors
// SPDX-License-Identifier: MIT
//
// Top-K selection kernels for MoE routing
// Optimized for small num_experts (8-64)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace moe {

// =============================================================================
// Top-K selection for MoE routing
// Input:  logits [num_tokens, num_experts]
// Output: values [num_tokens, k], indices [num_tokens, k]
// =============================================================================

// Simple insertion sort for small K (K <= 8)
// Each thread handles one token
template <typename T, int MAX_EXPERTS = 64>
__global__ void topk_with_indices_kernel(
    const T* __restrict__ logits,      // [num_tokens, num_experts]
    T* __restrict__ values,            // [num_tokens, k]
    int32_t* __restrict__ indices,     // [num_tokens, k]
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const T* token_logits = logits + token_idx * num_experts;
    T* token_values = values + token_idx * k;
    int32_t* token_indices = indices + token_idx * k;

    // Load all expert logits into registers (for small num_experts)
    T local_logits[MAX_EXPERTS];
    for (int i = 0; i < num_experts; ++i) {
        local_logits[i] = token_logits[i];
    }

    // Simple selection: find top-k by repeated max finding
    // For small k (2-8) and small num_experts (8-64), this is efficient
    for (int j = 0; j < k; ++j) {
        T max_val = T(-1e9f);
        int max_idx = 0;

        for (int i = 0; i < num_experts; ++i) {
            if (float(local_logits[i]) > float(max_val)) {
                max_val = local_logits[i];
                max_idx = i;
            }
        }

        token_values[j] = max_val;
        token_indices[j] = max_idx;

        // Mark as used
        local_logits[max_idx] = T(-1e10f);
    }
}

// FP32 specialization
__global__ void topk_with_indices_f32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ values,
    int32_t* __restrict__ indices,
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const float* token_logits = logits + token_idx * num_experts;
    float* token_values = values + token_idx * k;
    int32_t* token_indices = indices + token_idx * k;

    // For Mixtral: num_experts=8, k=2
    // Load into registers
    float local_logits[64];  // Max 64 experts
    for (int i = 0; i < num_experts; ++i) {
        local_logits[i] = token_logits[i];
    }

    // Find top-k
    for (int j = 0; j < k; ++j) {
        float max_val = -1e30f;
        int max_idx = 0;

        #pragma unroll 8
        for (int i = 0; i < num_experts; ++i) {
            if (local_logits[i] > max_val) {
                max_val = local_logits[i];
                max_idx = i;
            }
        }

        token_values[j] = max_val;
        token_indices[j] = max_idx;
        local_logits[max_idx] = -1e30f;
    }
}

// BF16 specialization with FP32 accumulation
__global__ void topk_with_indices_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    __nv_bfloat16* __restrict__ values,
    int32_t* __restrict__ indices,
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const __nv_bfloat16* token_logits = logits + token_idx * num_experts;
    __nv_bfloat16* token_values = values + token_idx * k;
    int32_t* token_indices = indices + token_idx * k;

    // Load and convert to FP32 for comparison
    float local_logits[64];
    for (int i = 0; i < num_experts; ++i) {
        local_logits[i] = __bfloat162float(token_logits[i]);
    }

    for (int j = 0; j < k; ++j) {
        float max_val = -1e30f;
        int max_idx = 0;

        for (int i = 0; i < num_experts; ++i) {
            if (local_logits[i] > max_val) {
                max_val = local_logits[i];
                max_idx = i;
            }
        }

        token_values[j] = __float2bfloat16(max_val);
        token_indices[j] = max_idx;
        local_logits[max_idx] = -1e30f;
    }
}

// FP16 specialization
__global__ void topk_with_indices_f16_kernel(
    const __half* __restrict__ logits,
    __half* __restrict__ values,
    int32_t* __restrict__ indices,
    int num_tokens,
    int num_experts,
    int k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const __half* token_logits = logits + token_idx * num_experts;
    __half* token_values = values + token_idx * k;
    int32_t* token_indices = indices + token_idx * k;

    // Load and convert to FP32 for comparison
    float local_logits[64];
    for (int i = 0; i < num_experts; ++i) {
        local_logits[i] = __half2float(token_logits[i]);
    }

    for (int j = 0; j < k; ++j) {
        float max_val = -1e30f;
        int max_idx = 0;

        for (int i = 0; i < num_experts; ++i) {
            if (local_logits[i] > max_val) {
                max_val = local_logits[i];
                max_idx = i;
            }
        }

        token_values[j] = __float2half(max_val);
        token_indices[j] = max_idx;
        local_logits[max_idx] = -1e30f;
    }
}

// =============================================================================
// Softmax over selected experts (for router weights)
// =============================================================================

__global__ void softmax_topk_f32_kernel(
    float* __restrict__ values,  // [num_tokens, k] - in-place
    int num_tokens,
    int k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    float* token_values = values + token_idx * k;

    // Find max for numerical stability
    float max_val = token_values[0];
    for (int i = 1; i < k; ++i) {
        max_val = fmaxf(max_val, token_values[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        token_values[i] = expf(token_values[i] - max_val);
        sum += token_values[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < k; ++i) {
        token_values[i] *= inv_sum;
    }
}

__global__ void softmax_topk_bf16_kernel(
    __nv_bfloat16* __restrict__ values,
    int num_tokens,
    int k
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    __nv_bfloat16* token_values = values + token_idx * k;

    // Load to FP32
    float local[8];  // Max k=8
    for (int i = 0; i < k; ++i) {
        local[i] = __bfloat162float(token_values[i]);
    }

    // Find max
    float max_val = local[0];
    for (int i = 1; i < k; ++i) {
        max_val = fmaxf(max_val, local[i]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        local[i] = expf(local[i] - max_val);
        sum += local[i];
    }

    // Normalize and store
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < k; ++i) {
        token_values[i] = __float2bfloat16(local[i] * inv_sum);
    }
}

}  // namespace moe
}  // namespace pygpukit
