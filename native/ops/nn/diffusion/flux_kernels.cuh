/**
 * FLUX-specific GPU kernels for efficient transformer operations
 *
 * These kernels eliminate H2D/D2H transfers by keeping all data on GPU.
 * Issue #187: Performance optimization for FLUX.1 transformer
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
// Layer Normalization (no learnable parameters)
// ============================================================================

// LayerNorm kernel - normalizes over last dimension without gamma/beta
// Input shape: [B, N, D]
__global__ void layer_norm_simple_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int N, int D,
    float eps
) {
    int row = blockIdx.x;
    int batch_idx = row / N;

    if (batch_idx >= B) return;

    const float* row_input = input + row * D;
    float* row_output = output + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += row_input[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float mean;
    if (threadIdx.x == 0) mean = sum / D;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) shared_sum[warp_id] = var_sum;
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(var_sum / D + eps);
    __syncthreads();

    // Normalize (no scale/shift)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = row_input[i];
        row_output[i] = (x - mean) * inv_std;
    }
}

// ============================================================================
// Modulate: y = x * (1 + scale) + shift
// ============================================================================

// Modulate kernel for AdaLN-style modulation
// Input: [B, N, D], Scale/Shift: [B, D]
__global__ void modulate_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    float* __restrict__ output,
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;

    if (idx >= total) return;

    int batch_idx = idx / (N * D);
    int feat_idx = idx % D;

    float x = input[idx];
    float s = scale[batch_idx * D + feat_idx];
    float sh = shift[batch_idx * D + feat_idx];

    output[idx] = x * (1.0f + s) + sh;
}

// ============================================================================
// Gated Residual: y = residual + gate * value
// ============================================================================

// Gated residual kernel
// Residual: [B, N, D], Gate: [B, D], Value: [B, N, D]
__global__ void gated_residual_f32_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ gate,
    const float* __restrict__ value,
    float* __restrict__ output,
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;

    if (idx >= total) return;

    int batch_idx = idx / (N * D);
    int feat_idx = idx % D;

    float res = residual[idx];
    float g = gate[batch_idx * D + feat_idx];
    float val = value[idx];

    output[idx] = res + g * val;
}

// In-place version
__global__ void gated_residual_inplace_f32_kernel(
    float* __restrict__ residual,
    const float* __restrict__ gate,
    const float* __restrict__ value,
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;

    if (idx >= total) return;

    int batch_idx = idx / (N * D);
    int feat_idx = idx % D;

    float g = gate[batch_idx * D + feat_idx];
    float val = value[idx];

    residual[idx] += g * val;
}

// ============================================================================
// Scale: y = x * scalar
// ============================================================================

__global__ void scale_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx] * scale;
}

// ============================================================================
// Concatenate along axis 1: [B, N1, D] + [B, N2, D] -> [B, N1+N2, D]
// ============================================================================

__global__ void concat_axis1_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int B, int N1, int N2, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * (N1 + N2) * D;

    if (idx >= total) return;

    int batch_idx = idx / ((N1 + N2) * D);
    int seq_feat = idx % ((N1 + N2) * D);
    int seq_idx = seq_feat / D;
    int feat_idx = seq_feat % D;

    if (seq_idx < N1) {
        // From tensor a
        output[idx] = a[batch_idx * N1 * D + seq_idx * D + feat_idx];
    } else {
        // From tensor b
        int seq_in_b = seq_idx - N1;
        output[idx] = b[batch_idx * N2 * D + seq_in_b * D + feat_idx];
    }
}

// ============================================================================
// Split along axis 1: [B, N1+N2, D] -> [B, N1, D], [B, N2, D]
// ============================================================================

__global__ void split_axis1_first_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int N_total, int N_first, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N_first * D;

    if (idx >= total) return;

    int batch_idx = idx / (N_first * D);
    int seq_feat = idx % (N_first * D);
    int seq_idx = seq_feat / D;
    int feat_idx = seq_feat % D;

    output[idx] = input[batch_idx * N_total * D + seq_idx * D + feat_idx];
}

__global__ void split_axis1_second_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int N_total, int N_first, int D
) {
    int N_second = N_total - N_first;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N_second * D;

    if (idx >= total) return;

    int batch_idx = idx / (N_second * D);
    int seq_feat = idx % (N_second * D);
    int seq_idx = seq_feat / D;
    int feat_idx = seq_feat % D;

    int input_seq_idx = N_first + seq_idx;
    output[idx] = input[batch_idx * N_total * D + input_seq_idx * D + feat_idx];
}

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

// Apply RoPE to Q or K
// x: [B, N, H, D], cos/sin: [N, D]
// Rotation: x_rot[..., 0::2] = -x[..., 1::2], x_rot[..., 1::2] = x[..., 0::2]
// Result: x * cos + x_rot * sin
__global__ void apply_rope_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cos_freq,
    const float* __restrict__ sin_freq,
    float* __restrict__ output,
    int B, int N, int H, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * H * D;

    if (idx >= total) return;

    // Compute indices
    int batch_idx = idx / (N * H * D);
    int remainder = idx % (N * H * D);
    int seq_idx = remainder / (H * D);
    int head_feat = remainder % (H * D);
    int head_idx = head_feat / D;
    int feat_idx = head_feat % D;

    // Get cos/sin for this position and feature
    float c = cos_freq[seq_idx * D + feat_idx];
    float s = sin_freq[seq_idx * D + feat_idx];

    // Get current value
    float x_val = x[idx];

    // Get paired value for rotation
    float x_pair;
    if (feat_idx % 2 == 0) {
        // Even index: pair with next (odd)
        x_pair = -x[idx + 1];
    } else {
        // Odd index: pair with previous (even)
        x_pair = x[idx - 1];
    }

    output[idx] = x_val * c + x_pair * s;
}

// ============================================================================
// Fused LayerNorm + Modulate
// ============================================================================

// Fused: y = LayerNorm(x) * (1 + scale) + shift
__global__ void layer_norm_modulate_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    float* __restrict__ output,
    int B, int N, int D,
    float eps
) {
    int row = blockIdx.x;
    int batch_idx = row / N;

    if (batch_idx >= B) return;

    const float* row_input = input + row * D;
    const float* row_scale = scale + batch_idx * D;
    const float* row_shift = shift + batch_idx * D;
    float* row_output = output + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += row_input[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float mean;
    if (threadIdx.x == 0) mean = sum / D;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) shared_sum[warp_id] = var_sum;
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(var_sum / D + eps);
    __syncthreads();

    // Normalize and apply modulation
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = row_input[i];
        float normalized = (x - mean) * inv_std;
        float s = row_scale[i];
        float sh = row_shift[i];
        row_output[i] = normalized * (1.0f + s) + sh;
    }
}

// ============================================================================
// Add with broadcasting: [B, N, D] + [B, D] -> [B, N, D]
// ============================================================================

__global__ void add_broadcast_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;

    if (idx >= total) return;

    int batch_idx = idx / (N * D);
    int feat_idx = idx % D;

    output[idx] = x[idx] + bias[batch_idx * D + feat_idx];
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
