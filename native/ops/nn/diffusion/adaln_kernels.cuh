/**
 * Adaptive Layer Normalization (AdaLN) kernels for diffusion models
 *
 * AdaLN: y = (x - mean) / sqrt(var + eps) * (1 + scale) + shift
 * AdaLN-Zero: y = gate * ((x - mean) / sqrt(var + eps) * (1 + scale) + shift)
 *
 * Used in DiT, SD3, Flux for timestep conditioning.
 * scale, shift, gate come from the timestep/class embedding.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

// AdaLN kernel - applies adaptive layer normalization
// Input shape: [B, N, D] (batch, sequence, features)
// Scale/Shift shape: [B, D] or [B, 1, D] (per-sample modulation)
__global__ void adaln_f32_kernel(
    const float* __restrict__ input,     // [B, N, D]
    const float* __restrict__ scale,     // [B, D]
    const float* __restrict__ shift,     // [B, D]
    float* __restrict__ output,          // [B, N, D]
    int B, int N, int D,
    float eps
) {
    // Each block handles one row [batch, seq_pos]
    int row = blockIdx.x;
    int batch_idx = row / N;
    int seq_idx = row % N;

    if (batch_idx >= B) return;

    const float* row_input = input + row * D;
    const float* row_scale = scale + batch_idx * D;
    const float* row_shift = shift + batch_idx * D;
    float* row_output = output + row * D;

    // Step 1: Compute mean
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

    // Step 2: Compute variance
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

    // Step 3: Normalize and apply adaptive scale/shift
    // y = (x - mean) * inv_std * (1 + scale) + shift
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = row_input[i];
        float normalized = (x - mean) * inv_std;
        float s = row_scale[i];
        float sh = row_shift[i];
        row_output[i] = normalized * (1.0f + s) + sh;
    }
}

// AdaLN-Zero kernel - includes gate for residual connections
// y = residual + gate * ((x - mean) / sqrt(var + eps) * (1 + scale) + shift)
__global__ void adaln_zero_f32_kernel(
    const float* __restrict__ input,     // [B, N, D]
    const float* __restrict__ scale,     // [B, D]
    const float* __restrict__ shift,     // [B, D]
    const float* __restrict__ gate,      // [B, D]
    const float* __restrict__ residual,  // [B, N, D]
    float* __restrict__ output,          // [B, N, D]
    int B, int N, int D,
    float eps
) {
    int row = blockIdx.x;
    int batch_idx = row / N;
    int seq_idx = row % N;

    if (batch_idx >= B) return;

    const float* row_input = input + row * D;
    const float* row_scale = scale + batch_idx * D;
    const float* row_shift = shift + batch_idx * D;
    const float* row_gate = gate + batch_idx * D;
    const float* row_residual = residual + row * D;
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

    // Normalize with gate: residual + gate * (normalized * (1 + scale) + shift)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = row_input[i];
        float normalized = (x - mean) * inv_std;
        float s = row_scale[i];
        float sh = row_shift[i];
        float g = row_gate[i];
        float res = row_residual[i];
        row_output[i] = res + g * (normalized * (1.0f + s) + sh);
    }
}

// BF16 AdaLN
__global__ void adaln_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int D,
    float eps
) {
    int row = blockIdx.x;
    int batch_idx = row / N;

    if (batch_idx >= B) return;

    const __nv_bfloat16* row_input = input + row * D;
    const __nv_bfloat16* row_scale = scale + batch_idx * D;
    const __nv_bfloat16* row_shift = shift + batch_idx * D;
    __nv_bfloat16* row_output = output + row * D;

    // Compute mean in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += __bfloat162float(row_input[i]);
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

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = __bfloat162float(row_input[i]) - mean;
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

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        float normalized = (x - mean) * inv_std;
        float s = __bfloat162float(row_scale[i]);
        float sh = __bfloat162float(row_shift[i]);
        row_output[i] = __float2bfloat16(normalized * (1.0f + s) + sh);
    }
}

// BF16 AdaLN-Zero
__global__ void adaln_zero_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int D,
    float eps
) {
    int row = blockIdx.x;
    int batch_idx = row / N;

    if (batch_idx >= B) return;

    const __nv_bfloat16* row_input = input + row * D;
    const __nv_bfloat16* row_scale = scale + batch_idx * D;
    const __nv_bfloat16* row_shift = shift + batch_idx * D;
    const __nv_bfloat16* row_gate = gate + batch_idx * D;
    const __nv_bfloat16* row_residual = residual + row * D;
    __nv_bfloat16* row_output = output + row * D;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += __bfloat162float(row_input[i]);
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

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = __bfloat162float(row_input[i]) - mean;
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

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        float normalized = (x - mean) * inv_std;
        float s = __bfloat162float(row_scale[i]);
        float sh = __bfloat162float(row_shift[i]);
        float g = __bfloat162float(row_gate[i]);
        float res = __bfloat162float(row_residual[i]);
        row_output[i] = __float2bfloat16(res + g * (normalized * (1.0f + s) + sh));
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
