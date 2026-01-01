/**
 * 2D Convolution kernels for diffusion models (VAE, UNet)
 *
 * Implements im2col + GEMM approach for Conv2D.
 * For production, consider using cuDNN's convolution routines.
 *
 * Input: [N, C_in, H, W]
 * Weight: [C_out, C_in/groups, K_h, K_w]
 * Output: [N, C_out, H_out, W_out]
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {

// im2col kernel - extracts patches for convolution
// Converts [N, C, H, W] + convolution params -> [N, C*K*K, H_out*W_out]
__global__ void im2col_f32_kernel(
    const float* __restrict__ input,   // [N, C, H, W]
    float* __restrict__ output,        // [N, C*K_h*K_w, H_out*W_out]
    int N, int C, int H, int W,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int H_out, int W_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * K_h * K_w * H_out * W_out;

    if (index >= total) return;

    // Decode index
    int w_out = index % W_out;
    int temp = index / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int k_col = temp % (K_h * K_w);  // Which position in the kernel
    temp = temp / (K_h * K_w);
    int c = temp % C;
    int n = temp / C;

    int k_h = k_col / K_w;
    int k_w = k_col % K_w;

    // Input position with dilation
    int h_in = h_out * stride_h - pad_h + k_h * dil_h;
    int w_in = w_out * stride_w - pad_w + k_w * dil_w;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((n * C + c) * H + h_in) * W + w_in];
    }

    // Output: [N, C*K_h*K_w, H_out*W_out]
    int col_idx = (c * K_h * K_w + k_col);
    int spatial_idx = h_out * W_out + w_out;
    output[(n * C * K_h * K_w + col_idx) * (H_out * W_out) + spatial_idx] = val;
}

// col2im kernel - for transposed convolution (deconvolution)
// Converts [N, C*K_h*K_w, H_out*W_out] back to [N, C, H, W]
__global__ void col2im_f32_kernel(
    const float* __restrict__ input,   // [N, C*K_h*K_w, H_in*W_in]
    float* __restrict__ output,        // [N, C, H, W]
    int N, int C, int H, int W,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int H_in, int W_in  // Input spatial dimensions (before transpose)
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (index >= total) return;

    // Decode output index
    int w = index % W;
    int temp = index / W;
    int h = temp % H;
    temp = temp / H;
    int c = temp % C;
    int n = temp / C;

    float sum = 0.0f;

    // Accumulate contributions from all kernel positions
    for (int k_h = 0; k_h < K_h; k_h++) {
        for (int k_w = 0; k_w < K_w; k_w++) {
            // Find which input position contributes to this output
            int h_in_offset = h + pad_h - k_h * dil_h;
            int w_in_offset = w + pad_w - k_w * dil_w;

            // Check if this is a valid strided position
            if (h_in_offset % stride_h == 0 && w_in_offset % stride_w == 0) {
                int h_in = h_in_offset / stride_h;
                int w_in = w_in_offset / stride_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int k_col = k_h * K_w + k_w;
                    int col_idx = c * K_h * K_w + k_col;
                    int spatial_idx = h_in * W_in + w_in;
                    sum += input[(n * C * K_h * K_w + col_idx) * (H_in * W_in) + spatial_idx];
                }
            }
        }
    }

    output[((n * C + c) * H + h) * W + w] = sum;
}

// Simple direct convolution kernel for small kernels (3x3, 1x1)
// More efficient than im2col for these cases
__global__ void conv2d_direct_3x3_f32_kernel(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in, 3, 3]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ output,        // [N, C_out, H_out, W_out]
    int N, int C_in, int C_out, int H, int W,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;

    if (out_idx >= total) return;

    // Decode output index
    int w_out = out_idx % W_out;
    int temp = out_idx / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;

    // 3x3 convolution
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int k_h = 0; k_h < 3; k_h++) {
            for (int k_w = 0; k_w < 3; k_w++) {
                int h_in = h_out * stride_h - pad_h + k_h;
                int w_in = w_out * stride_w - pad_w + k_w;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float in_val = input[((n * C_in + c_in) * H + h_in) * W + w_in];
                    float w_val = weight[((c_out * C_in + c_in) * 3 + k_h) * 3 + k_w];
                    sum += in_val * w_val;
                }
            }
        }
    }

    output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = sum;
}

// 1x1 convolution (pointwise) - very common in VAE and UNet
__global__ void conv2d_1x1_f32_kernel(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ output,        // [N, C_out, H, W]
    int N, int C_in, int C_out, int H, int W
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;

    if (out_idx >= total) return;

    int w = out_idx % W;
    int temp = out_idx / W;
    int h = temp % H;
    temp = temp / H;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        float in_val = input[((n * C_in + c_in) * H + h) * W + w];
        float w_val = weight[c_out * C_in + c_in];
        sum += in_val * w_val;
    }

    output[((n * C_out + c_out) * H + h) * W + w] = sum;
}

// BF16 versions
__global__ void im2col_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int C, int H, int W,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int H_out, int W_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * K_h * K_w * H_out * W_out;

    if (index >= total) return;

    int w_out = index % W_out;
    int temp = index / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int k_col = temp % (K_h * K_w);
    temp = temp / (K_h * K_w);
    int c = temp % C;
    int n = temp / C;

    int k_h = k_col / K_w;
    int k_w = k_col % K_w;

    int h_in = h_out * stride_h - pad_h + k_h * dil_h;
    int w_in = w_out * stride_w - pad_w + k_w * dil_w;

    __nv_bfloat16 val = __float2bfloat16(0.0f);
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((n * C + c) * H + h_in) * W + w_in];
    }

    int col_idx = (c * K_h * K_w + k_col);
    int spatial_idx = h_out * W_out + w_out;
    output[(n * C * K_h * K_w + col_idx) * (H_out * W_out) + spatial_idx] = val;
}

__global__ void conv2d_1x1_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int N, int C_in, int C_out, int H, int W
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;

    if (out_idx >= total) return;

    int w = out_idx % W;
    int temp = out_idx / W;
    int h = temp % H;
    temp = temp / H;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float sum = (bias != nullptr) ? __bfloat162float(bias[c_out]) : 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        float in_val = __bfloat162float(input[((n * C_in + c_in) * H + h) * W + w]);
        float w_val = __bfloat162float(weight[c_out * C_in + c_in]);
        sum += in_val * w_val;
    }

    output[((n * C_out + c_out) * H + h) * W + w] = __float2bfloat16(sum);
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
