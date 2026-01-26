/**
 * Fused NN kernels for improved performance
 *
 * 1. Fused RMSNorm + Residual: y = rmsnorm(x + residual)
 * 2. Fused SwiGLU: y = silu(gate) * up
 *
 * These fusions reduce kernel launch overhead and memory bandwidth.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Use existing device helper functions from activation_kernels.cuh
// silu_f32, gelu_f32 are already defined there as __device__ __forceinline__
#include "activation_kernels.cuh"

namespace pygpukit {
namespace ops {
namespace nn {

// ============================================================================
// Fused RMSNorm + Residual
// ============================================================================
// Computes: y = rmsnorm(x + residual) * gamma
// Where: rmsnorm(z) = z / sqrt(mean(z^2) + eps)
//
// This fuses:
//   1. Residual addition: z = x + residual
//   2. RMSNorm: y = z / sqrt(mean(z^2) + eps) * gamma
//
// Memory: Reads x, residual, gamma; Writes output (no intermediate buffer)
// Perf gain: ~1.5-2x vs separate kernels (fewer memory round-trips)

__global__ void rmsnorm_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,      // [batch, features]
    const __nv_bfloat16* __restrict__ residual,   // [batch, features]
    const __nv_bfloat16* __restrict__ gamma,      // [features]
    __nv_bfloat16* __restrict__ output,           // [batch, features]
    size_t batch_size,
    size_t features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_input = input + row * features;
    const __nv_bfloat16* row_residual = residual + row * features;
    __nv_bfloat16* row_output = output + row * features;

    // Phase 1: Compute sum of squares of (x + residual) using parallel reduction
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        float r = __bfloat162float(row_residual[i]);
        float z = x + r;
        sum_sq += z * z;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block-level reduction using shared memory
    __shared__ float shared_sum[32];  // Max 32 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    // Phase 2: Normalize and apply scale (gamma)
    // Re-read input and residual, apply normalization
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __bfloat162float(row_input[i]);
        float r = __bfloat162float(row_residual[i]);
        float z = x + r;
        float g = __bfloat162float(gamma[i]);
        row_output[i] = __float2bfloat16(z * inv_rms * g);
    }
}

// FP16 version
__global__ void rmsnorm_residual_f16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ residual,
    const __half* __restrict__ gamma,
    __half* __restrict__ output,
    size_t batch_size,
    size_t features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* row_input = input + row * features;
    const __half* row_residual = residual + row * features;
    __half* row_output = output + row * features;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __half2float(row_input[i]);
        float r = __half2float(row_residual[i]);
        float z = x + r;
        sum_sq += z * z;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x = __half2float(row_input[i]);
        float r = __half2float(row_residual[i]);
        float z = x + r;
        float g = __half2float(gamma[i]);
        row_output[i] = __float2half(z * inv_rms * g);
    }
}

// FP32 version
__global__ void rmsnorm_residual_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    size_t batch_size,
    size_t features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * features;
    const float* row_residual = residual + row * features;
    float* row_output = output + row * features;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float z = row_input[i] + row_residual[i];
        sum_sq += z * z;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / features + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float z = row_input[i] + row_residual[i];
        row_output[i] = z * inv_rms * gamma[i];
    }
}

// ============================================================================
// Fused SwiGLU Activation
// ============================================================================
// Computes: y = silu(gate) * up
// Where: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Used in: Qwen, LLaMA3, Mistral FFN layers
// FFN computation: y = (silu(x @ W_gate) * (x @ W_up)) @ W_down
//
// This kernel fuses the element-wise part after the two projections:
//   gate_proj = x @ W_gate  (computed separately via matmul)
//   up_proj = x @ W_up      (computed separately via matmul)
//   y = silu(gate_proj) * up_proj  <-- THIS KERNEL
//
// Memory: Reads gate_proj, up_proj; Writes output
// Perf gain: 2x vs separate silu + multiply kernels

__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_proj,  // [batch, features]
    const __nv_bfloat16* __restrict__ up_proj,    // [batch, features]
    __nv_bfloat16* __restrict__ output,           // [batch, features]
    size_t n                                       // total elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate = __bfloat162float(gate_proj[idx]);
        float up = __bfloat162float(up_proj[idx]);
        float result = silu_f32(gate) * up;
        output[idx] = __float2bfloat16(result);
    }
}

// Vectorized BF16 version (float4 = 8 BF16 elements)
__global__ void swiglu_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    __nv_bfloat16* __restrict__ output,
    size_t n
) {
    // Process 8 BF16 elements per thread (2x float4)
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < n) {
        // Load 8 BF16 values as 2x float4
        float4 gate_vec1 = reinterpret_cast<const float4*>(gate_proj + idx)[0];
        float4 gate_vec2 = reinterpret_cast<const float4*>(gate_proj + idx + 4)[0];
        float4 up_vec1 = reinterpret_cast<const float4*>(up_proj + idx)[0];
        float4 up_vec2 = reinterpret_cast<const float4*>(up_proj + idx + 4)[0];

        // Unpack BF16 to float, compute SwiGLU, repack
        __nv_bfloat16* gate1 = reinterpret_cast<__nv_bfloat16*>(&gate_vec1);
        __nv_bfloat16* gate2 = reinterpret_cast<__nv_bfloat16*>(&gate_vec2);
        __nv_bfloat16* up1 = reinterpret_cast<__nv_bfloat16*>(&up_vec1);
        __nv_bfloat16* up2 = reinterpret_cast<__nv_bfloat16*>(&up_vec2);

        __nv_bfloat16 out[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float g = __bfloat162float(gate1[i]);
            float u = __bfloat162float(up1[i]);
            out[i] = __float2bfloat16(silu_f32(g) * u);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float g = __bfloat162float(gate2[i]);
            float u = __bfloat162float(up2[i]);
            out[i + 4] = __float2bfloat16(silu_f32(g) * u);
        }

        // Store
        reinterpret_cast<float4*>(output + idx)[0] = *reinterpret_cast<float4*>(out);
        reinterpret_cast<float4*>(output + idx + 4)[0] = *reinterpret_cast<float4*>(out + 4);
    } else {
        // Handle remainder with scalar code
        for (size_t i = idx; i < n; ++i) {
            float gate = __bfloat162float(gate_proj[i]);
            float up = __bfloat162float(up_proj[i]);
            output[i] = __float2bfloat16(silu_f32(gate) * up);
        }
    }
}

// FP16 version
__global__ void swiglu_f16_kernel(
    const __half* __restrict__ gate_proj,
    const __half* __restrict__ up_proj,
    __half* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate = __half2float(gate_proj[idx]);
        float up = __half2float(up_proj[idx]);
        float result = silu_f32(gate) * up;
        output[idx] = __float2half(result);
    }
}

// FP32 version
__global__ void swiglu_f32_kernel(
    const float* __restrict__ gate_proj,
    const float* __restrict__ up_proj,
    float* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate = gate_proj[idx];
        float up = up_proj[idx];
        output[idx] = silu_f32(gate) * up;
    }
}

// ============================================================================
// Fused GeGLU Activation (GELU variant)
// ============================================================================
// Computes: y = gelu(gate) * up
// Used in some transformer variants
// Note: gelu_f32 helper function is defined in activation_kernels.cuh

__global__ void geglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    __nv_bfloat16* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate = __bfloat162float(gate_proj[idx]);
        float up = __bfloat162float(up_proj[idx]);
        float result = gelu_f32(gate) * up;
        output[idx] = __float2bfloat16(result);
    }
}

__global__ void geglu_f16_kernel(
    const __half* __restrict__ gate_proj,
    const __half* __restrict__ up_proj,
    __half* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate = __half2float(gate_proj[idx]);
        float up = __half2float(up_proj[idx]);
        float result = gelu_f32(gate) * up;
        output[idx] = __float2half(result);
    }
}

__global__ void geglu_f32_kernel(
    const float* __restrict__ gate_proj,
    const float* __restrict__ up_proj,
    float* __restrict__ output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate = gate_proj[idx];
        float up = up_proj[idx];
        output[idx] = gelu_f32(gate) * up;
    }
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
