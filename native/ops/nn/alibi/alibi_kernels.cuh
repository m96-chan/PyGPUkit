/**
 * ALiBi (Attention with Linear Biases) kernels
 *
 * Adds linear bias to attention scores based on query-key distance.
 * Paper: "Train Short, Test Long" (Press et al., 2022)
 *
 * Formula: attention_scores[i, j] = Q[i] @ K[j]^T - m * |i - j|
 * Where m is a head-specific slope: m_h = 2^(-8 * h / num_heads)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// ALiBi Init Slopes - Compute head-specific slopes
// ============================================================================

__global__ void alibi_init_slopes_kernel(
    float* __restrict__ slopes,
    int num_heads
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < num_heads) {
        // m_h = 2^(-8 * (h+1) / num_heads)
        // Note: h is 0-indexed, so we use (h+1) to match paper convention
        float exponent = -8.0f * (float)(h + 1) / (float)num_heads;
        slopes[h] = powf(2.0f, exponent);
    }
}

// ============================================================================
// ALiBi Compute Bias - Create bias matrix for attention
// ============================================================================

__global__ void alibi_compute_bias_f32_kernel(
    float* __restrict__ bias,
    const float* __restrict__ slopes,
    int seq_len,
    int num_heads
) {
    // bias: [num_heads, seq_len, seq_len]
    // For causal attention, we only compute lower triangular

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * seq_len;

    if (idx < total) {
        int h = idx / (seq_len * seq_len);
        int i = (idx / seq_len) % seq_len;  // query position
        int j = idx % seq_len;               // key position

        float slope = slopes[h];
        // ALiBi bias: -slope * |i - j|
        // For causal: only j <= i is used, so distance is (i - j)
        int distance = i - j;
        if (distance >= 0) {
            bias[idx] = -slope * (float)distance;
        } else {
            // For non-causal or positions j > i, set to large negative (masked)
            bias[idx] = -1e9f;
        }
    }
}

// Causal-only version (more efficient)
__global__ void alibi_compute_bias_causal_f32_kernel(
    float* __restrict__ bias,
    const float* __restrict__ slopes,
    int seq_len,
    int num_heads
) {
    // bias: [num_heads, seq_len, seq_len] but we only compute lower triangular
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * seq_len;

    if (idx < total) {
        int h = idx / (seq_len * seq_len);
        int i = (idx / seq_len) % seq_len;
        int j = idx % seq_len;

        if (j <= i) {
            float slope = slopes[h];
            bias[idx] = -slope * (float)(i - j);
        } else {
            bias[idx] = -1e9f;  // Causal mask
        }
    }
}

// ============================================================================
// ALiBi Add Bias - Add bias to attention scores in-place
// ============================================================================

__global__ void alibi_add_bias_f32_kernel(
    float* __restrict__ scores,
    const float* __restrict__ slopes,
    int batch_size,
    int num_heads,
    int q_len,
    int kv_len,
    int start_pos
) {
    // scores: [batch, num_heads, q_len, kv_len]
    // For each (i, j) in scores, add -slope * |start_pos + i - j|

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * q_len * kv_len;

    if (idx < total) {
        int h = (idx / (q_len * kv_len)) % num_heads;
        int i = (idx / kv_len) % q_len;   // query position (relative)
        int j = idx % kv_len;              // key position

        int q_pos = start_pos + i;  // absolute query position
        int distance = q_pos - j;

        float slope = slopes[h];
        // Only apply for causal (j <= q_pos)
        if (distance >= 0) {
            scores[idx] += -slope * (float)distance;
        }
        // Note: causal masking should be applied separately
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
