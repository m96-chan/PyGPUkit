/**
 * GPU Sampling Kernels for LLM Inference
 *
 * Provides efficient sampling operations on GPU:
 * - Greedy (argmax)
 * - Temperature scaling + multinomial
 * - Top-k / Top-p filtering
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

namespace pygpukit {
namespace ops {
namespace sampling {

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_max_sampling(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_sampling(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Argmax reduction helper
__device__ __forceinline__ void warp_reduce_argmax_helper(float& val, int& idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ============================================================================
// Greedy Sampling (Argmax) - FP32
// ============================================================================

__global__ void sample_argmax_f32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size
) {
    __shared__ float shared_max[32];
    __shared__ int shared_idx[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    // Grid-stride loop to find local max
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Warp-level reduction
    warp_reduce_argmax_helper(local_max, local_idx);

    // Write warp results to shared memory
    if (lane == 0) {
        shared_max[warp_id] = local_max;
        shared_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_idx = (tid < num_warps) ? shared_idx[tid] : 0;
        warp_reduce_argmax_helper(local_max, local_idx);

        if (lane == 0) {
            *result = local_idx;
        }
    }
}

// ============================================================================
// Greedy Sampling (Argmax) - FP16
// ============================================================================

__global__ void sample_argmax_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size
) {
    __shared__ float shared_max[32];
    __shared__ int shared_idx[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    warp_reduce_argmax_helper(local_max, local_idx);

    if (lane == 0) {
        shared_max[warp_id] = local_max;
        shared_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_idx = (tid < num_warps) ? shared_idx[tid] : 0;
        warp_reduce_argmax_helper(local_max, local_idx);

        if (lane == 0) {
            *result = local_idx;
        }
    }
}

// ============================================================================
// Greedy Sampling (Argmax) - BF16
// ============================================================================

__global__ void sample_argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size
) {
    __shared__ float shared_max[32];
    __shared__ int shared_idx[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    warp_reduce_argmax_helper(local_max, local_idx);

    if (lane == 0) {
        shared_max[warp_id] = local_max;
        shared_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_idx = (tid < num_warps) ? shared_idx[tid] : 0;
        warp_reduce_argmax_helper(local_max, local_idx);

        if (lane == 0) {
            *result = local_idx;
        }
    }
}

// ============================================================================
// Multinomial Sampling - FP32
// ============================================================================

__global__ void sample_multinomial_f32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    float temperature,
    float random_val
) {
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    // Step 1: Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i] / temperature;
        local_max = fmaxf(local_max, val);
    }

    local_max = warp_reduce_max_sampling(local_max);
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_max = warp_reduce_max_sampling(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float max_val = shared_max[0];

    // Step 2: Compute sum of exp(logit/temp - max)
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i] / temperature - max_val;
        local_sum += expf(val);
    }

    local_sum = warp_reduce_sum_sampling(local_sum);
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (tid < num_warps) ? shared_sum[tid] : 0.0f;
        local_sum = warp_reduce_sum_sampling(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float total_sum = shared_sum[0];

    // Step 3: Sample from cumulative distribution (thread 0 only)
    if (tid == 0) {
        float threshold = random_val * total_sum;
        float cumsum = 0.0f;
        int sampled_idx = vocab_size - 1;

        for (int i = 0; i < vocab_size; i++) {
            float val = logits[i] / temperature - max_val;
            cumsum += expf(val);
            if (cumsum >= threshold) {
                sampled_idx = i;
                break;
            }
        }
        *result = sampled_idx;
    }
}

// ============================================================================
// Multinomial Sampling - FP16
// ============================================================================

__global__ void sample_multinomial_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    float temperature,
    float random_val
) {
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]) / temperature;
        local_max = fmaxf(local_max, val);
    }

    local_max = warp_reduce_max_sampling(local_max);
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_max = warp_reduce_max_sampling(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]) / temperature - max_val;
        local_sum += expf(val);
    }

    local_sum = warp_reduce_sum_sampling(local_sum);
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (tid < num_warps) ? shared_sum[tid] : 0.0f;
        local_sum = warp_reduce_sum_sampling(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float total_sum = shared_sum[0];

    if (tid == 0) {
        float threshold = random_val * total_sum;
        float cumsum = 0.0f;
        int sampled_idx = vocab_size - 1;

        for (int i = 0; i < vocab_size; i++) {
            float val = __half2float(logits[i]) / temperature - max_val;
            cumsum += expf(val);
            if (cumsum >= threshold) {
                sampled_idx = i;
                break;
            }
        }
        *result = sampled_idx;
    }
}

// ============================================================================
// Multinomial Sampling - BF16
// ============================================================================

__global__ void sample_multinomial_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    float temperature,
    float random_val
) {
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]) / temperature;
        local_max = fmaxf(local_max, val);
    }

    local_max = warp_reduce_max_sampling(local_max);
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_max = warp_reduce_max_sampling(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]) / temperature - max_val;
        local_sum += expf(val);
    }

    local_sum = warp_reduce_sum_sampling(local_sum);
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (tid < num_warps) ? shared_sum[tid] : 0.0f;
        local_sum = warp_reduce_sum_sampling(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float total_sum = shared_sum[0];

    if (tid == 0) {
        float threshold = random_val * total_sum;
        float cumsum = 0.0f;
        int sampled_idx = vocab_size - 1;

        for (int i = 0; i < vocab_size; i++) {
            float val = __bfloat162float(logits[i]) / temperature - max_val;
            cumsum += expf(val);
            if (cumsum >= threshold) {
                sampled_idx = i;
                break;
            }
        }
        *result = sampled_idx;
    }
}

// ============================================================================
// Top-K Sampling - FP32
// ============================================================================

__global__ void sample_topk_f32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    int top_k,
    float temperature,
    float random_val
) {
    // Shared memory for top-k values and indices
    extern __shared__ char shared_mem[];
    float* top_vals = reinterpret_cast<float*>(shared_mem);
    int* top_idxs = reinterpret_cast<int*>(top_vals + top_k);

    const int tid = threadIdx.x;

    // Initialize top-k array (thread 0 only for simplicity)
    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            top_vals[i] = -FLT_MAX;
            top_idxs[i] = 0;
        }
    }
    __syncthreads();

    // Each thread finds its local top-k candidates
    // Simplified: each thread scans and updates shared array atomically
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i] / temperature;

        // Find minimum in current top-k (simplified linear search)
        int min_idx = 0;
        float min_val = top_vals[0];
        for (int j = 1; j < top_k; j++) {
            if (top_vals[j] < min_val) {
                min_val = top_vals[j];
                min_idx = j;
            }
        }

        if (val > min_val) {
            atomicExch(&top_vals[min_idx], val);
            atomicExch(&top_idxs[min_idx], i);
        }
    }
    __syncthreads();

    // Thread 0: Sample from top-k
    if (tid == 0) {
        // Compute softmax over top-k
        float max_val = top_vals[0];
        for (int i = 1; i < top_k; i++) {
            max_val = fmaxf(max_val, top_vals[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            sum += expf(top_vals[i] - max_val);
        }

        // Sample
        float threshold = random_val * sum;
        float cumsum = 0.0f;
        int sampled_idx = top_idxs[top_k - 1];

        for (int i = 0; i < top_k; i++) {
            cumsum += expf(top_vals[i] - max_val);
            if (cumsum >= threshold) {
                sampled_idx = top_idxs[i];
                break;
            }
        }
        *result = sampled_idx;
    }
}

// ============================================================================
// Top-K Sampling - FP16
// ============================================================================

__global__ void sample_topk_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    int top_k,
    float temperature,
    float random_val
) {
    extern __shared__ char shared_mem[];
    float* top_vals = reinterpret_cast<float*>(shared_mem);
    int* top_idxs = reinterpret_cast<int*>(top_vals + top_k);

    const int tid = threadIdx.x;

    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            top_vals[i] = -FLT_MAX;
            top_idxs[i] = 0;
        }
    }
    __syncthreads();

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]) / temperature;

        int min_idx = 0;
        float min_val = top_vals[0];
        for (int j = 1; j < top_k; j++) {
            if (top_vals[j] < min_val) {
                min_val = top_vals[j];
                min_idx = j;
            }
        }

        if (val > min_val) {
            atomicExch(&top_vals[min_idx], val);
            atomicExch(&top_idxs[min_idx], i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        float max_val = top_vals[0];
        for (int i = 1; i < top_k; i++) {
            max_val = fmaxf(max_val, top_vals[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            sum += expf(top_vals[i] - max_val);
        }

        float threshold = random_val * sum;
        float cumsum = 0.0f;
        int sampled_idx = top_idxs[top_k - 1];

        for (int i = 0; i < top_k; i++) {
            cumsum += expf(top_vals[i] - max_val);
            if (cumsum >= threshold) {
                sampled_idx = top_idxs[i];
                break;
            }
        }
        *result = sampled_idx;
    }
}

// ============================================================================
// Top-K Sampling - BF16
// ============================================================================

__global__ void sample_topk_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    int top_k,
    float temperature,
    float random_val
) {
    extern __shared__ char shared_mem[];
    float* top_vals = reinterpret_cast<float*>(shared_mem);
    int* top_idxs = reinterpret_cast<int*>(top_vals + top_k);

    const int tid = threadIdx.x;

    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            top_vals[i] = -FLT_MAX;
            top_idxs[i] = 0;
        }
    }
    __syncthreads();

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]) / temperature;

        int min_idx = 0;
        float min_val = top_vals[0];
        for (int j = 1; j < top_k; j++) {
            if (top_vals[j] < min_val) {
                min_val = top_vals[j];
                min_idx = j;
            }
        }

        if (val > min_val) {
            atomicExch(&top_vals[min_idx], val);
            atomicExch(&top_idxs[min_idx], i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        float max_val = top_vals[0];
        for (int i = 1; i < top_k; i++) {
            max_val = fmaxf(max_val, top_vals[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            sum += expf(top_vals[i] - max_val);
        }

        float threshold = random_val * sum;
        float cumsum = 0.0f;
        int sampled_idx = top_idxs[top_k - 1];

        for (int i = 0; i < top_k; i++) {
            cumsum += expf(top_vals[i] - max_val);
            if (cumsum >= threshold) {
                sampled_idx = top_idxs[i];
                break;
            }
        }
        *result = sampled_idx;
    }
}

// ============================================================================
// Top-P (Nucleus) Sampling - FP32
// ============================================================================

__global__ void sample_topp_f32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    float top_p,
    float temperature,
    float random_val
) {
    if (threadIdx.x != 0) return;

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        max_val = fmaxf(max_val, logits[i] / temperature);
    }

    // Compute sum
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum += expf(logits[i] / temperature - max_val);
    }

    // Sample with top-p approximation
    float threshold = random_val * sum * top_p;
    float cumsum = 0.0f;
    int sampled_idx = 0;

    for (int i = 0; i < vocab_size; i++) {
        cumsum += expf(logits[i] / temperature - max_val);
        if (cumsum >= threshold) {
            sampled_idx = i;
            break;
        }
    }

    *result = sampled_idx;
}

// ============================================================================
// Top-P (Nucleus) Sampling - FP16
// ============================================================================

__global__ void sample_topp_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    float top_p,
    float temperature,
    float random_val
) {
    if (threadIdx.x != 0) return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        max_val = fmaxf(max_val, __half2float(logits[i]) / temperature);
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum += expf(__half2float(logits[i]) / temperature - max_val);
    }

    float threshold = random_val * sum * top_p;
    float cumsum = 0.0f;
    int sampled_idx = 0;

    for (int i = 0; i < vocab_size; i++) {
        cumsum += expf(__half2float(logits[i]) / temperature - max_val);
        if (cumsum >= threshold) {
            sampled_idx = i;
            break;
        }
    }

    *result = sampled_idx;
}

// ============================================================================
// Top-P (Nucleus) Sampling - BF16
// ============================================================================

__global__ void sample_topp_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size,
    float top_p,
    float temperature,
    float random_val
) {
    if (threadIdx.x != 0) return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        max_val = fmaxf(max_val, __bfloat162float(logits[i]) / temperature);
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum += expf(__bfloat162float(logits[i]) / temperature - max_val);
    }

    float threshold = random_val * sum * top_p;
    float cumsum = 0.0f;
    int sampled_idx = 0;

    for (int i = 0; i < vocab_size; i++) {
        cumsum += expf(__bfloat162float(logits[i]) / temperature - max_val);
        if (cumsum >= threshold) {
            sampled_idx = i;
            break;
        }
    }

    *result = sampled_idx;
}

// ============================================================================
// Top-K Sampling with Pointer-based random_val (CUDA Graph compatible)
// random_val is read from GPU buffer, allowing update before Graph replay
// ============================================================================

__global__ void sample_topk_f16_ptr_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ result,
    const float* __restrict__ random_val_ptr,
    int vocab_size,
    int top_k,
    float temperature
) {
    extern __shared__ char shared_mem[];
    float* top_vals = reinterpret_cast<float*>(shared_mem);
    int* top_idxs = reinterpret_cast<int*>(top_vals + top_k);

    const int tid = threadIdx.x;

    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            top_vals[i] = -FLT_MAX;
            top_idxs[i] = 0;
        }
    }
    __syncthreads();

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]) / temperature;

        int min_idx = 0;
        float min_val = top_vals[0];
        for (int j = 1; j < top_k; j++) {
            if (top_vals[j] < min_val) {
                min_val = top_vals[j];
                min_idx = j;
            }
        }

        if (val > min_val) {
            atomicExch(&top_vals[min_idx], val);
            atomicExch(&top_idxs[min_idx], i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        float max_val = top_vals[0];
        for (int i = 1; i < top_k; i++) {
            max_val = fmaxf(max_val, top_vals[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            sum += expf(top_vals[i] - max_val);
        }

        // Read random_val from GPU buffer (allows update before Graph replay)
        float random_val = *random_val_ptr;
        float threshold = random_val * sum;
        float cumsum = 0.0f;
        int sampled_idx = top_idxs[top_k - 1];

        for (int i = 0; i < top_k; i++) {
            cumsum += expf(top_vals[i] - max_val);
            if (cumsum >= threshold) {
                sampled_idx = top_idxs[i];
                break;
            }
        }
        *result = sampled_idx;
    }
}

} // namespace sampling
} // namespace ops
} // namespace pygpukit
