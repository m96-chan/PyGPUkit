/**
 * Extended RoPE kernels for context length extension
 *
 * Implements:
 * - NTK-aware interpolation
 * - YaRN (Yet another RoPE extensioN)
 *
 * These methods allow models to handle sequences longer than training context.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// NTK-aware RoPE - Scale base frequency for context extension
// ============================================================================

__global__ void rope_init_ntk_aware_f32_kernel(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    int max_seq_len,
    int head_dim,
    float base,
    float scale
) {
    // NTK-aware: base' = base * scale^(dim / (dim - 2))
    // This preserves high-frequency components better than linear interpolation

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * head_dim;

    if (idx < total) {
        int pos = idx / head_dim;
        int dim = idx % head_dim;

        // NTK scaling factor for base
        float ntk_factor = powf(scale, (float)head_dim / ((float)head_dim - 2.0f));
        float scaled_base = base * ntk_factor;

        // Compute frequency with scaled base
        float freq = 1.0f / powf(scaled_base, (float)(dim / 2 * 2) / (float)head_dim);
        float angle = (float)pos * freq;

        if (dim % 2 == 0) {
            cos_table[idx] = cosf(angle);
            sin_table[idx] = sinf(angle);
        } else {
            // For odd dimensions, use the same angle as even (paired)
            float even_freq = 1.0f / powf(scaled_base, (float)((dim - 1) / 2 * 2) / (float)head_dim);
            float even_angle = (float)pos * even_freq;
            cos_table[idx] = cosf(even_angle);
            sin_table[idx] = sinf(even_angle);
        }
    }
}

// ============================================================================
// YaRN RoPE - Dimension-wise interpolation with attention scaling
// ============================================================================

// YaRN uses different interpolation for different frequency bands:
// - Low frequency (local attention): no interpolation
// - High frequency: full interpolation
// - Mid frequency: gradual transition

__device__ __forceinline__ float yarn_get_mscale(float scale, float mscale_factor) {
    // Attention scaling factor to compensate for interpolation
    // mscale = 0.1 * ln(scale) + 1.0 (default)
    if (mscale_factor <= 0.0f) {
        return 1.0f;  // No scaling
    }
    return mscale_factor * logf(scale) + 1.0f;
}

__device__ __forceinline__ float yarn_find_correction_dim(
    int dim,
    int head_dim,
    float base,
    int max_position_embeddings
) {
    // Find the correction dimension for YaRN
    // Based on wavelength analysis
    return (float)dim * logf((float)max_position_embeddings / (2.0f * (float)M_PI * (float)dim)) /
           (2.0f * logf(base));
}

__device__ __forceinline__ float yarn_find_correction_range(
    float low_rot,
    float high_rot,
    int dim,
    float base,
    int max_position_embeddings
) {
    // Linear ramp between correction ranges
    float low = floorf(yarn_find_correction_dim(dim, dim, base, max_position_embeddings) * low_rot);
    float high = ceilf(yarn_find_correction_dim(dim, dim, base, max_position_embeddings) * high_rot);
    return fmaxf(low, 0.0f);
}

__global__ void rope_init_yarn_f32_kernel(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    int max_seq_len,
    int head_dim,
    float base,
    float scale,
    int original_max_len,
    float beta_fast,
    float beta_slow,
    float mscale
) {
    // YaRN: dimension-wise interpolation
    // Low freq dims: no scaling (preserve local attention)
    // High freq dims: full scaling
    // Mid freq dims: linear ramp

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * head_dim;

    if (idx < total) {
        int pos = idx / head_dim;
        int dim = idx % head_dim;
        int dim_pair = dim / 2 * 2;  // Even dimension in pair

        // Compute frequency bounds for interpolation
        float low_freq_wavelen = (float)original_max_len / beta_fast;
        float high_freq_wavelen = (float)original_max_len / beta_slow;

        // Current wavelength for this dimension
        float wavelen = 2.0f * (float)M_PI * powf(base, (float)dim_pair / (float)head_dim);

        float freq_factor;
        if (wavelen >= high_freq_wavelen) {
            // High frequency: no interpolation
            freq_factor = 1.0f;
        } else if (wavelen <= low_freq_wavelen) {
            // Low frequency: full interpolation
            freq_factor = 1.0f / scale;
        } else {
            // Mid frequency: linear interpolation
            float smooth = (wavelen - low_freq_wavelen) / (high_freq_wavelen - low_freq_wavelen);
            freq_factor = (1.0f - smooth) / scale + smooth;
        }

        // Compute angle with interpolated frequency
        float inv_freq = 1.0f / powf(base, (float)dim_pair / (float)head_dim);
        float scaled_freq = inv_freq * freq_factor;
        float angle = (float)pos * scaled_freq;

        // Apply mscale (attention scaling)
        float attention_scale = yarn_get_mscale(scale, mscale);

        if (dim % 2 == 0) {
            cos_table[idx] = cosf(angle) * attention_scale;
            sin_table[idx] = sinf(angle) * attention_scale;
        } else {
            cos_table[idx] = cosf(angle) * attention_scale;
            sin_table[idx] = sinf(angle) * attention_scale;
        }
    }
}

// ============================================================================
// Linear Position Interpolation (PI) - Simple baseline
// ============================================================================

__global__ void rope_init_linear_interpolation_f32_kernel(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    int max_seq_len,
    int head_dim,
    float base,
    float scale
) {
    // Linear interpolation: pos' = pos / scale
    // Simple but degrades quality at high scales

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * head_dim;

    if (idx < total) {
        int pos = idx / head_dim;
        int dim = idx % head_dim;
        int dim_pair = dim / 2 * 2;

        // Scale position instead of frequency
        float scaled_pos = (float)pos / scale;
        float freq = 1.0f / powf(base, (float)dim_pair / (float)head_dim);
        float angle = scaled_pos * freq;

        if (dim % 2 == 0) {
            cos_table[idx] = cosf(angle);
            sin_table[idx] = sinf(angle);
        } else {
            cos_table[idx] = cosf(angle);
            sin_table[idx] = sinf(angle);
        }
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
