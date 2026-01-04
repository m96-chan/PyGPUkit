/**
 * FLUX-specific CUDA kernels for diffusion models (Issue #187)
 *
 * Provides GPU kernels for FLUX operations:
 * - LayerNorm without learnable parameters
 * - Modulate: y = x * (1 + scale) + shift
 * - Gated residual: y = residual + gate * value
 * - RoPE (Rotary Position Embedding)
 * - Concat/Split along axis 1 and 2
 * - Slice chunk for modulation parameters
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace pygpukit {
namespace ops {
namespace nn {

// =============================================================================
// LayerNorm Simple (no learnable parameters)
// =============================================================================

__global__ void layer_norm_simple_f32_kernel(
    const float* __restrict__ input,   // [B, N, D]
    float* __restrict__ output,        // [B, N, D]
    int total_rows,                    // B * N
    int D,
    float eps
) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    const float* row_in = input + row * D;
    float* row_out = output + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += row_in[i];
    }

    // Warp reduction for sum
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
        float diff = row_in[i] - mean;
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

    __shared__ float rstd;
    if (threadIdx.x == 0) {
        float var = var_sum / D;
        rstd = rsqrtf(var + eps);
    }
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * rstd;
    }
}

// =============================================================================
// Modulate: y = x * (1 + scale) + shift
// =============================================================================

__global__ void modulate_f32_kernel(
    const float* __restrict__ input,   // [B, N, D]
    const float* __restrict__ scale,   // [B, D]
    const float* __restrict__ shift,   // [B, D]
    float* __restrict__ output,        // [B, N, D]
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;

    int d = idx % D;
    int n = (idx / D) % N;
    int b = idx / (N * D);

    int scale_idx = b * D + d;
    output[idx] = input[idx] * (1.0f + scale[scale_idx]) + shift[scale_idx];
}

// =============================================================================
// Gated Residual: y = residual + gate * value
// =============================================================================

__global__ void gated_residual_f32_kernel(
    const float* __restrict__ residual,  // [B, N, D]
    const float* __restrict__ gate,      // [B, D]
    const float* __restrict__ value,     // [B, N, D]
    float* __restrict__ output,          // [B, N, D]
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;

    int d = idx % D;
    int b = idx / (N * D);

    int gate_idx = b * D + d;
    output[idx] = residual[idx] + gate[gate_idx] * value[idx];
}

__global__ void gated_residual_inplace_f32_kernel(
    float* __restrict__ residual,        // [B, N, D]
    const float* __restrict__ gate,      // [B, D]
    const float* __restrict__ value,     // [B, N, D]
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;

    int d = idx % D;
    int b = idx / (N * D);

    int gate_idx = b * D + d;
    residual[idx] = residual[idx] + gate[gate_idx] * value[idx];
}

// =============================================================================
// Scale tensor: y = x * scale
// =============================================================================

__global__ void scale_tensor_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] * scale;
}

// =============================================================================
// Concat axis 1: [B, N1, D] + [B, N2, D] -> [B, N1+N2, D]
// =============================================================================

__global__ void concat_axis1_f32_kernel(
    const float* __restrict__ a,     // [B, N1, D]
    const float* __restrict__ b,     // [B, N2, D]
    float* __restrict__ output,      // [B, N1+N2, D]
    int B, int N1, int N2, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_out = N1 + N2;
    int total = B * N_out * D;
    if (idx >= total) return;

    int d = idx % D;
    int n = (idx / D) % N_out;
    int batch = idx / (N_out * D);

    if (n < N1) {
        output[idx] = a[batch * N1 * D + n * D + d];
    } else {
        output[idx] = b[batch * N2 * D + (n - N1) * D + d];
    }
}

// =============================================================================
// Split axis 1: [B, N, D] -> [B, split_size, D], [B, N-split_size, D]
// Uses two separate copy kernels
// =============================================================================

__global__ void copy_slice_axis1_f32_kernel(
    const float* __restrict__ input,  // [B, N, D]
    float* __restrict__ output,       // [B, N_out, D]
    int B, int N_in, int N_out, int D,
    int start_n                       // Starting index along axis 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N_out * D;
    if (idx >= total) return;

    int d = idx % D;
    int n_out = (idx / D) % N_out;
    int batch = idx / (N_out * D);

    int n_in = start_n + n_out;
    output[idx] = input[batch * N_in * D + n_in * D + d];
}

// =============================================================================
// Apply RoPE: rotary position embedding
// x: [B, N, H, D] with cos/sin: [N, D]
// =============================================================================

__global__ void apply_rope_f32_kernel(
    const float* __restrict__ x,        // [B, N, H, D]
    const float* __restrict__ cos_freq, // [N, D]
    const float* __restrict__ sin_freq, // [N, D]
    float* __restrict__ output,         // [B, N, H, D]
    int B, int N, int H, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int h = (idx / D) % H;
    int n = (idx / (D * H)) % N;
    // int b = idx / (N * H * D);

    int freq_idx = n * D + d;

    // RoPE rotation: split into pairs
    // For even d: x[d] * cos - x[d+1] * sin
    // For odd d:  x[d-1] * sin + x[d] * cos
    float cos_val = cos_freq[freq_idx];
    float sin_val = sin_freq[freq_idx];

    if (d % 2 == 0) {
        // Even index: need to read x[d+1]
        float x0 = x[idx];
        float x1 = (d + 1 < D) ? x[idx + 1] : 0.0f;
        output[idx] = x0 * cos_val - x1 * sin_val;
    } else {
        // Odd index: need to read x[d-1]
        float x0 = x[idx - 1];
        float x1 = x[idx];
        output[idx] = x0 * sin_val + x1 * cos_val;
    }
}

// =============================================================================
// Fused LayerNorm + Modulate
// =============================================================================

__global__ void layer_norm_modulate_f32_kernel(
    const float* __restrict__ input,   // [B, N, D]
    const float* __restrict__ scale,   // [B, D]
    const float* __restrict__ shift,   // [B, D]
    float* __restrict__ output,        // [B, N, D]
    int B, int N, int D,
    float eps
) {
    int row = blockIdx.x;
    int total_rows = B * N;
    if (row >= total_rows) return;

    int batch_idx = row / N;
    const float* row_in = input + row * D;
    const float* row_scale = scale + batch_idx * D;
    const float* row_shift = shift + batch_idx * D;
    float* row_out = output + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += row_in[i];
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
        float diff = row_in[i] - mean;
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

    __shared__ float rstd;
    if (threadIdx.x == 0) {
        float var = var_sum / D;
        rstd = rsqrtf(var + eps);
    }
    __syncthreads();

    // Normalize and modulate
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float normalized = (row_in[i] - mean) * rstd;
        row_out[i] = normalized * (1.0f + row_scale[i]) + row_shift[i];
    }
}

// =============================================================================
// Add with broadcasting: [B, N, D] + [B, D] -> [B, N, D]
// =============================================================================

__global__ void add_broadcast_f32_kernel(
    const float* __restrict__ x,     // [B, N, D]
    const float* __restrict__ bias,  // [B, D]
    float* __restrict__ output,      // [B, N, D]
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;

    int d = idx % D;
    int b = idx / (N * D);

    int bias_idx = b * D + d;
    output[idx] = x[idx] + bias[bias_idx];
}

// =============================================================================
// Concat axis 2: [B, N, D1] + [B, N, D2] -> [B, N, D1+D2]
// =============================================================================

__global__ void concat_axis2_f32_kernel(
    const float* __restrict__ a,     // [B, N, D1]
    const float* __restrict__ b,     // [B, N, D2]
    float* __restrict__ output,      // [B, N, D1+D2]
    int B, int N, int D1, int D2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D_out = D1 + D2;
    int total = B * N * D_out;
    if (idx >= total) return;

    int d = idx % D_out;
    int n = (idx / D_out) % N;
    int batch = idx / (N * D_out);

    if (d < D1) {
        output[idx] = a[batch * N * D1 + n * D1 + d];
    } else {
        output[idx] = b[batch * N * D2 + n * D2 + (d - D1)];
    }
}

// =============================================================================
// Slice chunk: [B, num_parts * D] -> [B, D] at chunk_idx
// =============================================================================

__global__ void slice_chunk_f32_kernel(
    const float* __restrict__ input,  // [B, num_parts * D]
    float* __restrict__ output,       // [B, D]
    int B, int D, int num_parts, int chunk_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;
    if (idx >= total) return;

    int d = idx % D;
    int b = idx / D;

    int full_D = num_parts * D;
    int offset = chunk_idx * D;
    output[idx] = input[b * full_D + offset + d];
}

// =============================================================================
// Concat axis 1 (4D): [B, N1, H, D] + [B, N2, H, D] -> [B, N1+N2, H, D]
// For attention tensors: [batch, seq, heads, head_dim]
// =============================================================================

__global__ void concat_axis1_4d_f32_kernel(
    const float* __restrict__ a,     // [B, N1, H, D]
    const float* __restrict__ b,     // [B, N2, H, D]
    float* __restrict__ output,      // [B, N1+N2, H, D]
    int B, int N1, int N2, int H, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_out = N1 + N2;
    int HD = H * D;
    int total = B * N_out * HD;
    if (idx >= total) return;

    int d = idx % D;
    int h = (idx / D) % H;
    int n = (idx / HD) % N_out;
    int batch = idx / (N_out * HD);

    if (n < N1) {
        // From tensor a
        output[idx] = a[batch * N1 * HD + n * HD + h * D + d];
    } else {
        // From tensor b
        output[idx] = b[batch * N2 * HD + (n - N1) * HD + h * D + d];
    }
}

// =============================================================================
// Split axis 1 (4D): [B, N, H, D] -> [B, N_out, H, D]
// Copy a slice from position start_n
// =============================================================================

__global__ void copy_slice_axis1_4d_f32_kernel(
    const float* __restrict__ input,  // [B, N, H, D]
    float* __restrict__ output,       // [B, N_out, H, D]
    int B, int N_in, int N_out, int H, int D,
    int start_n                       // Starting index along axis 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HD = H * D;
    int total = B * N_out * HD;
    if (idx >= total) return;

    int d = idx % D;
    int h = (idx / D) % H;
    int n_out = (idx / HD) % N_out;
    int batch = idx / (N_out * HD);

    int n_in = start_n + n_out;
    output[idx] = input[batch * N_in * HD + n_in * HD + h * D + d];
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
