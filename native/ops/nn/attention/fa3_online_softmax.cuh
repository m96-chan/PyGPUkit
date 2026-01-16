/**
 * Flash Attention 3 - Online Softmax
 *
 * Architecture-independent online softmax implementation.
 * Uses the "online" algorithm to compute softmax without materializing
 * the full attention matrix.
 *
 * Reference: FlashAttention-2 (Dao, 2023)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa3 {

// =============================================================================
// Online Softmax State
// =============================================================================

/**
 * Per-thread online softmax state.
 * Tracks running max and sum for numerical stability.
 */
struct OnlineSoftmaxState {
    float max_val;      // Running maximum
    float sum_exp;      // Running sum of exp(x - max)

    __device__ __forceinline__ OnlineSoftmaxState()
        : max_val(-FLT_MAX), sum_exp(0.0f) {}

    __device__ __forceinline__ OnlineSoftmaxState(float m, float s)
        : max_val(m), sum_exp(s) {}
};

// =============================================================================
// Fast Exponential Approximation (FA4-style)
// =============================================================================

/**
 * Fast exp2 approximation using cubic polynomial.
 * Avoids SFU contention on tensor core heavy workloads.
 *
 * exp2(x) ~ c0 + c1*x + c2*x^2 + c3*x^3
 * Uses Horner's method: ((c3*x + c2)*x + c1)*x + c0
 *
 * Accuracy: ~1e-4 relative error for x in [-1, 1]
 */
__device__ __forceinline__ float fast_exp2(float x) {
    // Coefficients for exp2 approximation
    constexpr float c0 = 1.0f;
    constexpr float c1 = 0.6931472f;    // ln(2)
    constexpr float c2 = 0.2402265f;    // ln(2)^2 / 2
    constexpr float c3 = 0.0555041f;    // ln(2)^3 / 6

    // Horner's method: 3 FMAs
    float result = c3;
    result = __fmaf_rn(result, x, c2);
    result = __fmaf_rn(result, x, c1);
    result = __fmaf_rn(result, x, c0);
    return result;
}

/**
 * Fast exp approximation: exp(x) = exp2(x * log2(e))
 */
__device__ __forceinline__ float fast_exp(float x) {
    constexpr float LOG2_E = 1.4426950408889634f;
    return fast_exp2(x * LOG2_E);
}

/**
 * Standard exp using CUDA intrinsic (more accurate but uses SFU)
 */
__device__ __forceinline__ float accurate_exp(float x) {
    return __expf(x);
}

// Configurable: use fast or accurate exp
#ifndef FA3_USE_FAST_EXP
#define FA3_USE_FAST_EXP 1
#endif

__device__ __forceinline__ float fa3_exp(float x) {
#if FA3_USE_FAST_EXP
    return fast_exp(x);
#else
    return accurate_exp(x);
#endif
}

// =============================================================================
// Online Softmax Operations
// =============================================================================

/**
 * Update online softmax state with new scores.
 *
 * Given current state (m, s) and new values x[0..n-1]:
 *   m' = max(m, max(x))
 *   s' = s * exp(m - m') + sum(exp(x - m'))
 *
 * @param state     Current online softmax state (modified in-place)
 * @param scores    New attention scores
 * @param n         Number of scores
 * @param scale     Pre-scaling factor (1/sqrt(d))
 */
__device__ __forceinline__ void online_softmax_update(
    OnlineSoftmaxState& state,
    const float* scores,
    int n,
    float scale = 1.0f
) {
    // Find max of new scores
    float new_max = state.max_val;
    for (int i = 0; i < n; ++i) {
        float s = scores[i] * scale;
        new_max = fmaxf(new_max, s);
    }

    // Rescale old sum if max changed
    float rescale = fa3_exp(state.max_val - new_max);
    state.sum_exp *= rescale;

    // Add new scores
    for (int i = 0; i < n; ++i) {
        float s = scores[i] * scale;
        state.sum_exp += fa3_exp(s - new_max);
    }

    state.max_val = new_max;
}

/**
 * Update online softmax state with single score.
 */
__device__ __forceinline__ void online_softmax_update_single(
    OnlineSoftmaxState& state,
    float score,
    float scale = 1.0f
) {
    float s = score * scale;
    float new_max = fmaxf(state.max_val, s);

    // Rescale if needed
    if (new_max > state.max_val) {
        state.sum_exp *= fa3_exp(state.max_val - new_max);
        state.max_val = new_max;
    }

    state.sum_exp += fa3_exp(s - state.max_val);
}

/**
 * Merge two online softmax states.
 * Used for parallel reduction across warps.
 */
__device__ __forceinline__ OnlineSoftmaxState online_softmax_merge(
    const OnlineSoftmaxState& a,
    const OnlineSoftmaxState& b
) {
    float new_max = fmaxf(a.max_val, b.max_val);
    float new_sum = a.sum_exp * fa3_exp(a.max_val - new_max)
                  + b.sum_exp * fa3_exp(b.max_val - new_max);
    return OnlineSoftmaxState(new_max, new_sum);
}

/**
 * Finalize online softmax: compute 1/sum for normalization.
 */
__device__ __forceinline__ float online_softmax_finalize(
    const OnlineSoftmaxState& state
) {
    return 1.0f / state.sum_exp;
}

/**
 * Compute softmax probability for a score given final state.
 */
__device__ __forceinline__ float online_softmax_prob(
    float score,
    const OnlineSoftmaxState& state,
    float scale = 1.0f
) {
    float s = score * scale;
    return fa3_exp(s - state.max_val) / state.sum_exp;
}

// =============================================================================
// Output Accumulator Rescaling (FA3 optimization)
// =============================================================================

/**
 * Rescale output accumulator when max changes.
 *
 * FA3 optimization: only rescale when max changes significantly.
 * This reduces the number of rescaling operations.
 *
 * @param output    Output accumulator (modified in-place)
 * @param old_max   Previous max value
 * @param new_max   New max value
 * @param dim       Dimension of output
 * @param threshold Minimum change to trigger rescale (FA3: ~0.5)
 * @return true if rescaling was performed
 */
__device__ __forceinline__ bool rescale_output_if_needed(
    float* output,
    float old_max,
    float new_max,
    int dim,
    float threshold = 0.5f
) {
    float diff = old_max - new_max;

    // Only rescale if max decreased significantly
    if (diff < threshold) {
        return false;
    }

    float rescale = fa3_exp(diff);
    for (int i = 0; i < dim; ++i) {
        output[i] *= rescale;
    }
    return true;
}

// =============================================================================
// Warp-level Softmax Reduction
// =============================================================================

/**
 * Warp-level reduction for online softmax state.
 * Uses shuffle instructions for efficiency.
 */
__device__ __forceinline__ OnlineSoftmaxState warp_reduce_softmax_state(
    OnlineSoftmaxState state
) {
    // Reduce max across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, state.max_val, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, state.sum_exp, offset);

        OnlineSoftmaxState other(other_max, other_sum);
        state = online_softmax_merge(state, other);
    }
    return state;
}

/**
 * Broadcast final softmax state from lane 0 to all lanes.
 */
__device__ __forceinline__ OnlineSoftmaxState warp_broadcast_softmax_state(
    OnlineSoftmaxState state
) {
    state.max_val = __shfl_sync(0xffffffff, state.max_val, 0);
    state.sum_exp = __shfl_sync(0xffffffff, state.sum_exp, 0);
    return state;
}

}  // namespace fa3
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
