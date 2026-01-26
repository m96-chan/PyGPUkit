// Conv1d CUDA Kernels
// native/ops/conv/conv1d_kernels.cuh

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace pygpukit {
namespace ops {

// Conv1d kernel: each thread computes one output element
// Input:  [batch, in_channels, length]
// Weight: [out_channels, in_channels, kernel_size]
// Bias:   [out_channels] (optional)
// Output: [batch, out_channels, out_length]
template <typename T>
__global__ void conv1d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch,
    int in_channels,
    int length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int out_length
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_length;

    if (idx >= total) return;

    // Decode indices: [b, oc, ol]
    int ol = idx % out_length;
    int tmp = idx / out_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Compute convolution for this output element
    float sum = 0.0f;

    // Input start position (with padding)
    int in_start = ol * stride - padding;

    // Weight offset for this output channel
    int weight_base = oc * in_channels * kernel_size;

    // Input batch offset
    int input_batch_offset = b * in_channels * length;

    for (int ic = 0; ic < in_channels; ic++) {
        int input_channel_offset = input_batch_offset + ic * length;
        int weight_channel_offset = weight_base + ic * kernel_size;

        for (int k = 0; k < kernel_size; k++) {
            int in_pos = in_start + k;

            // Check bounds (padding uses zero)
            if (in_pos >= 0 && in_pos < length) {
                float in_val = static_cast<float>(input[input_channel_offset + in_pos]);
                float w_val = static_cast<float>(weight[weight_channel_offset + k]);
                sum += in_val * w_val;
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += static_cast<float>(bias[oc]);
    }

    // Write output
    output[idx] = static_cast<T>(sum);
}

// Specialization for float32 - avoid unnecessary casts
__global__ void conv1d_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int out_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_length;

    if (idx >= total) return;

    // Decode indices
    int ol = idx % out_length;
    int tmp = idx / out_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int in_start = ol * stride - padding;
    int weight_base = oc * in_channels * kernel_size;
    int input_batch_offset = b * in_channels * length;

    for (int ic = 0; ic < in_channels; ic++) {
        int input_channel_offset = input_batch_offset + ic * length;
        int weight_channel_offset = weight_base + ic * kernel_size;

        #pragma unroll 4
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = in_start + k;
            if (in_pos >= 0 && in_pos < length) {
                sum += input[input_channel_offset + in_pos] * weight[weight_channel_offset + k];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[idx] = sum;
}

// BF16 kernel with float accumulation
__global__ void conv1d_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int batch,
    int in_channels,
    int length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int out_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_length;

    if (idx >= total) return;

    int ol = idx % out_length;
    int tmp = idx / out_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int in_start = ol * stride - padding;
    int weight_base = oc * in_channels * kernel_size;
    int input_batch_offset = b * in_channels * length;

    for (int ic = 0; ic < in_channels; ic++) {
        int input_channel_offset = input_batch_offset + ic * length;
        int weight_channel_offset = weight_base + ic * kernel_size;

        for (int k = 0; k < kernel_size; k++) {
            int in_pos = in_start + k;
            if (in_pos >= 0 && in_pos < length) {
                sum += __bfloat162float(input[input_channel_offset + in_pos])
                     * __bfloat162float(weight[weight_channel_offset + k]);
            }
        }
    }

    if (bias != nullptr) {
        sum += __bfloat162float(bias[oc]);
    }

    output[idx] = __float2bfloat16(sum);
}

// FP16 kernel with float accumulation
__global__ void conv1d_f16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int batch,
    int in_channels,
    int length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int out_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_length;

    if (idx >= total) return;

    int ol = idx % out_length;
    int tmp = idx / out_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int in_start = ol * stride - padding;
    int weight_base = oc * in_channels * kernel_size;
    int input_batch_offset = b * in_channels * length;

    for (int ic = 0; ic < in_channels; ic++) {
        int input_channel_offset = input_batch_offset + ic * length;
        int weight_channel_offset = weight_base + ic * kernel_size;

        for (int k = 0; k < kernel_size; k++) {
            int in_pos = in_start + k;
            if (in_pos >= 0 && in_pos < length) {
                sum += __half2float(input[input_channel_offset + in_pos])
                     * __half2float(weight[weight_channel_offset + k]);
            }
        }
    }

    if (bias != nullptr) {
        sum += __half2float(bias[oc]);
    }

    output[idx] = __float2half(sum);
}

}  // namespace ops
}  // namespace pygpukit
