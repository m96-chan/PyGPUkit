// Copyright (c) 2024 PyGPUkit Authors
// SPDX-License-Identifier: MIT
//
// MoE operations dispatch

#include "moe_kernels.cuh"
#include <cuda_runtime.h>

namespace pygpukit {
namespace moe {

// =============================================================================
// Host-side dispatch functions
// =============================================================================

void topk_with_indices_f32(
    const float* logits,
    float* values,
    int32_t* indices,
    int num_tokens,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    topk_with_indices_f32_kernel<<<blocks, threads, 0, stream>>>(
        logits, values, indices, num_tokens, num_experts, k
    );
}

void topk_with_indices_bf16(
    const __nv_bfloat16* logits,
    __nv_bfloat16* values,
    int32_t* indices,
    int num_tokens,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    topk_with_indices_bf16_kernel<<<blocks, threads, 0, stream>>>(
        logits, values, indices, num_tokens, num_experts, k
    );
}

void topk_with_indices_f16(
    const __half* logits,
    __half* values,
    int32_t* indices,
    int num_tokens,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    topk_with_indices_f16_kernel<<<blocks, threads, 0, stream>>>(
        logits, values, indices, num_tokens, num_experts, k
    );
}

void softmax_topk_f32(
    float* values,
    int num_tokens,
    int k,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    softmax_topk_f32_kernel<<<blocks, threads, 0, stream>>>(
        values, num_tokens, k
    );
}

void softmax_topk_bf16(
    __nv_bfloat16* values,
    int num_tokens,
    int k,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    softmax_topk_bf16_kernel<<<blocks, threads, 0, stream>>>(
        values, num_tokens, k
    );
}

// =============================================================================
// MoE Permutation functions
// =============================================================================

void moe_compute_permutation(
    const int32_t* expert_indices,  // [num_tokens, k]
    int32_t* expert_counts,         // [num_experts]
    int32_t* expert_offsets,        // [num_experts + 1]
    int32_t* permute_indices,       // [num_tokens * k]
    int32_t* reverse_perm,          // [num_tokens * k]
    int num_tokens,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    int total = num_tokens * k;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Zero expert counts
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream);

    // Step 1: Count tokens per expert
    count_tokens_per_expert_kernel<<<blocks, threads, 0, stream>>>(
        expert_indices, expert_counts, num_tokens, num_experts, k
    );

    // Step 2: Compute offsets (exclusive scan)
    compute_expert_offsets_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, num_experts
    );

    // Allocate temporary write counters (same as offsets initially)
    int32_t* write_offsets;
    cudaMallocAsync(&write_offsets, num_experts * sizeof(int32_t), stream);
    cudaMemsetAsync(write_offsets, 0, num_experts * sizeof(int32_t), stream);

    // Step 3: Build permute indices
    build_permute_indices_kernel<<<blocks, threads, 0, stream>>>(
        expert_indices, expert_offsets, permute_indices, write_offsets,
        num_tokens, num_experts, k
    );

    // Step 4: Build reverse permutation
    build_reverse_perm_kernel<<<blocks, threads, 0, stream>>>(
        permute_indices, reverse_perm, total
    );

    cudaFreeAsync(write_offsets, stream);
}

// =============================================================================
// Gather/Scatter operations
// =============================================================================

void moe_gather_f32(
    const float* hidden,
    const int32_t* permute_indices,
    float* gathered,
    int num_tokens,
    int hidden_size,
    int k,
    cudaStream_t stream
) {
    int total = num_tokens * k;
    int threads = 256;
    gather_hidden_states_f32_vec4_kernel<<<total, threads, 0, stream>>>(
        hidden, permute_indices, gathered, num_tokens, hidden_size, k
    );
}

void moe_gather_bf16(
    const __nv_bfloat16* hidden,
    const int32_t* permute_indices,
    __nv_bfloat16* gathered,
    int num_tokens,
    int hidden_size,
    int k,
    cudaStream_t stream
) {
    int total = num_tokens * k;
    int threads = 256;
    gather_hidden_states_bf16_vec2_kernel<<<total, threads, 0, stream>>>(
        hidden, permute_indices, gathered, num_tokens, hidden_size, k
    );
}

void moe_scatter_f32(
    const float* expert_outputs,
    const float* router_weights,
    const int32_t* reverse_perm,
    float* output,
    int num_tokens,
    int hidden_size,
    int k,
    cudaStream_t stream
) {
    // Zero output first
    cudaMemsetAsync(output, 0, num_tokens * hidden_size * sizeof(float), stream);

    int threads = 256;
    moe_combine_outputs_ordered_kernel<float><<<num_tokens, threads, 0, stream>>>(
        expert_outputs, router_weights, reverse_perm, output,
        num_tokens, hidden_size, k
    );
}

void moe_scatter_bf16(
    const __nv_bfloat16* expert_outputs,
    const __nv_bfloat16* router_weights,
    const int32_t* reverse_perm,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_size,
    int k,
    cudaStream_t stream
) {
    cudaMemsetAsync(output, 0, num_tokens * hidden_size * sizeof(__nv_bfloat16), stream);

    int threads = 256;
    moe_combine_outputs_ordered_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        expert_outputs, router_weights, reverse_perm, output,
        num_tokens, hidden_size, k
    );
}

// =============================================================================
// Fused router (gate linear + topk + softmax)
// =============================================================================

void moe_router_f32(
    const float* hidden,
    const float* gate_weight,
    float* router_weights,
    int32_t* expert_indices,
    int num_tokens,
    int hidden_size,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    int smem_size = num_experts * sizeof(float);
    moe_router_kernel<float><<<num_tokens, 256, smem_size, stream>>>(
        hidden, gate_weight, router_weights, expert_indices,
        num_tokens, hidden_size, num_experts, k
    );
}

void moe_router_bf16(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* gate_weight,
    __nv_bfloat16* router_weights,
    int32_t* expert_indices,
    int num_tokens,
    int hidden_size,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    int smem_size = num_experts * sizeof(float);
    moe_router_kernel<__nv_bfloat16><<<num_tokens, 256, smem_size, stream>>>(
        hidden, gate_weight, router_weights, expert_indices,
        num_tokens, hidden_size, num_experts, k
    );
}

void expand_expert_offsets(
    const int32_t* expert_offsets,
    int32_t* row_expert_ids,
    int num_experts,
    int M_total,
    cudaStream_t stream
) {
    if (M_total == 0) return;
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (M_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    expand_expert_offsets_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        expert_offsets, row_expert_ids, num_experts, M_total
    );
}

}  // namespace moe
}  // namespace pygpukit
