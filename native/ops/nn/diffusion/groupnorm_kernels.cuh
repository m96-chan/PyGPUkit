/**
 * GroupNorm kernels for diffusion models
 *
 * GroupNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
 * Normalizes over groups of channels for each spatial location.
 * Input: [N, C, H, W], normalizes over (C/G, H, W) for each group G.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

// GroupNorm kernel - one block per (batch, group)
// Input shape: [N, C, H, W]
// Normalizes over (C/num_groups, H, W) for each group
__global__ void groupnorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ output,
    int N, int C, int H, int W,
    int num_groups,
    float eps
) {
    // Each block handles one (batch, group) pair
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;

    if (batch_idx >= N) return;

    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;
    int channel_start = group_idx * channels_per_group;

    // Pointer to start of this group's data
    const float* group_input = input + batch_idx * C * H * W + channel_start * H * W;
    float* group_output = output + batch_idx * C * H * W + channel_start * H * W;

    // Step 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        sum += group_input[c_local * H * W + spatial];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = sum / group_size;
    }
    __syncthreads();

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        float diff = group_input[c_local * H * W + spatial] - mean;
        var_sum += diff * diff;
    }

    // Warp reduction for variance
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum / group_size + eps);
    }
    __syncthreads();

    // Step 3: Normalize and apply affine transform
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        int c_global = channel_start + c_local;

        float x = group_input[c_local * H * W + spatial];
        float normalized = (x - mean) * inv_std;
        group_output[c_local * H * W + spatial] = normalized * gamma[c_global] + beta[c_global];
    }
}

// BF16 GroupNorm (compute in FP32 for precision)
__global__ void groupnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    int N, int C, int H, int W,
    int num_groups,
    float eps
) {
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;

    if (batch_idx >= N) return;

    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;
    int channel_start = group_idx * channels_per_group;

    const __nv_bfloat16* group_input = input + batch_idx * C * H * W + channel_start * H * W;
    __nv_bfloat16* group_output = output + batch_idx * C * H * W + channel_start * H * W;

    // Compute mean in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        sum += __bfloat162float(group_input[c_local * H * W + spatial]);
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
    if (threadIdx.x == 0) mean = sum / group_size;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        float diff = __bfloat162float(group_input[c_local * H * W + spatial]) - mean;
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
    if (threadIdx.x == 0) inv_std = rsqrtf(var_sum / group_size + eps);
    __syncthreads();

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        int c_global = channel_start + c_local;

        float x = __bfloat162float(group_input[c_local * H * W + spatial]);
        float normalized = (x - mean) * inv_std;
        float g = __bfloat162float(gamma[c_global]);
        float b = __bfloat162float(beta[c_global]);
        group_output[c_local * H * W + spatial] = __float2bfloat16(normalized * g + b);
    }
}

// FP16 GroupNorm (compute in FP32 for precision)
__global__ void groupnorm_f16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ output,
    int N, int C, int H, int W,
    int num_groups,
    float eps
) {
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;

    if (batch_idx >= N) return;

    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;
    int channel_start = group_idx * channels_per_group;

    const __half* group_input = input + batch_idx * C * H * W + channel_start * H * W;
    __half* group_output = output + batch_idx * C * H * W + channel_start * H * W;

    // Compute mean in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        sum += __half2float(group_input[c_local * H * W + spatial]);
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
    if (threadIdx.x == 0) mean = sum / group_size;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        float diff = __half2float(group_input[c_local * H * W + spatial]) - mean;
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
    if (threadIdx.x == 0) inv_std = rsqrtf(var_sum / group_size + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int spatial = i % (H * W);
        int c_global = channel_start + c_local;

        float x = __half2float(group_input[c_local * H * W + spatial]);
        float normalized = (x - mean) * inv_std;
        float g = __half2float(gamma[c_global]);
        float b = __half2float(beta[c_global]);
        group_output[c_local * H * W + spatial] = __float2half(normalized * g + b);
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
