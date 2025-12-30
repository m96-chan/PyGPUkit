/**
 * LSTM Kernel Definitions
 *
 * Implements LSTM cell computation for TTS and other sequence models.
 * Supports unidirectional and bidirectional modes.
 *
 * LSTM equations:
 *   i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)
 *   f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)
 *   g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)
 *   o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)
 *   c_t = f_t * c_{t-1} + i_t * g_t
 *   h_t = o_t * tanh(c_t)
 *
 * PyTorch packs weights as:
 *   W_ih: [4*hidden_size, input_size]  (i, f, g, o gates)
 *   W_hh: [4*hidden_size, hidden_size]
 *   b_ih: [4*hidden_size]
 *   b_hh: [4*hidden_size]
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
// Device functions (prefixed to avoid collision with activation_kernels.cuh)
// ============================================================================

__device__ __forceinline__ float lstm_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float lstm_tanh(float x) {
    return tanhf(x);
}

// ============================================================================
// LSTM Cell Kernel - Single timestep, single batch element
// ============================================================================

/**
 * Compute LSTM gates for a single timestep.
 *
 * Input:
 *   gates_precomputed: W_ih @ x_t + b_ih + b_hh [4*hidden_size]
 *   h_prev: previous hidden state [hidden_size]
 *   c_prev: previous cell state [hidden_size]
 *   W_hh: hidden-to-hidden weights [4*hidden_size, hidden_size]
 *
 * Output:
 *   h_out: new hidden state [hidden_size]
 *   c_out: new cell state [hidden_size]
 */
__global__ void lstm_cell_f32_kernel(
    const float* __restrict__ gates_precomputed,  // [batch, 4*hidden]
    const float* __restrict__ h_prev,             // [batch, hidden]
    const float* __restrict__ c_prev,             // [batch, hidden]
    const float* __restrict__ W_hh,               // [4*hidden, hidden]
    float* __restrict__ h_out,                    // [batch, hidden]
    float* __restrict__ c_out,                    // [batch, hidden]
    int batch_size,
    int hidden_size
) {
    int batch_idx = blockIdx.y;
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || h_idx >= hidden_size) return;

    // Compute W_hh @ h_prev contribution for this hidden unit
    float gate_i = gates_precomputed[batch_idx * 4 * hidden_size + h_idx];
    float gate_f = gates_precomputed[batch_idx * 4 * hidden_size + hidden_size + h_idx];
    float gate_g = gates_precomputed[batch_idx * 4 * hidden_size + 2 * hidden_size + h_idx];
    float gate_o = gates_precomputed[batch_idx * 4 * hidden_size + 3 * hidden_size + h_idx];

    // Add W_hh @ h_prev
    for (int k = 0; k < hidden_size; ++k) {
        float h_k = h_prev[batch_idx * hidden_size + k];
        gate_i += W_hh[h_idx * hidden_size + k] * h_k;
        gate_f += W_hh[(hidden_size + h_idx) * hidden_size + k] * h_k;
        gate_g += W_hh[(2 * hidden_size + h_idx) * hidden_size + k] * h_k;
        gate_o += W_hh[(3 * hidden_size + h_idx) * hidden_size + k] * h_k;
    }

    // Apply activations
    float i = lstm_sigmoid(gate_i);
    float f = lstm_sigmoid(gate_f);
    float g = lstm_tanh(gate_g);
    float o = lstm_sigmoid(gate_o);

    // Update cell state
    float c_prev_val = c_prev[batch_idx * hidden_size + h_idx];
    float c_new = f * c_prev_val + i * g;

    // Compute hidden state
    float h_new = o * lstm_tanh(c_new);

    // Store outputs
    c_out[batch_idx * hidden_size + h_idx] = c_new;
    h_out[batch_idx * hidden_size + h_idx] = h_new;
}

// ============================================================================
// Optimized LSTM Cell - Uses shared memory for W_hh @ h_prev
// ============================================================================

template<int BLOCK_SIZE = 256, int TILE_K = 32>
__global__ void lstm_cell_tiled_f32_kernel(
    const float* __restrict__ gates_precomputed,  // [batch, 4*hidden]
    const float* __restrict__ h_prev,             // [batch, hidden]
    const float* __restrict__ c_prev,             // [batch, hidden]
    const float* __restrict__ W_hh,               // [4*hidden, hidden]
    float* __restrict__ h_out,                    // [batch, hidden]
    float* __restrict__ c_out,                    // [batch, hidden]
    int batch_size,
    int hidden_size
) {
    extern __shared__ float smem[];
    float* h_shared = smem;  // [TILE_K]

    int batch_idx = blockIdx.y;
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Initialize gate accumulators from precomputed values
    float gate_i = 0.0f, gate_f = 0.0f, gate_g = 0.0f, gate_o = 0.0f;

    if (h_idx < hidden_size) {
        gate_i = gates_precomputed[batch_idx * 4 * hidden_size + h_idx];
        gate_f = gates_precomputed[batch_idx * 4 * hidden_size + hidden_size + h_idx];
        gate_g = gates_precomputed[batch_idx * 4 * hidden_size + 2 * hidden_size + h_idx];
        gate_o = gates_precomputed[batch_idx * 4 * hidden_size + 3 * hidden_size + h_idx];
    }

    // Tiled computation of W_hh @ h_prev
    for (int tile_start = 0; tile_start < hidden_size; tile_start += TILE_K) {
        // Load h_prev tile to shared memory
        int load_idx = tile_start + threadIdx.x;
        if (threadIdx.x < TILE_K && load_idx < hidden_size) {
            h_shared[threadIdx.x] = h_prev[batch_idx * hidden_size + load_idx];
        }
        __syncthreads();

        // Compute partial sums
        if (h_idx < hidden_size) {
            int tile_end = min(TILE_K, hidden_size - tile_start);
            for (int k = 0; k < tile_end; ++k) {
                float h_k = h_shared[k];
                int k_global = tile_start + k;
                gate_i += W_hh[h_idx * hidden_size + k_global] * h_k;
                gate_f += W_hh[(hidden_size + h_idx) * hidden_size + k_global] * h_k;
                gate_g += W_hh[(2 * hidden_size + h_idx) * hidden_size + k_global] * h_k;
                gate_o += W_hh[(3 * hidden_size + h_idx) * hidden_size + k_global] * h_k;
            }
        }
        __syncthreads();
    }

    if (h_idx >= hidden_size) return;

    // Apply activations
    float i = lstm_sigmoid(gate_i);
    float f = lstm_sigmoid(gate_f);
    float g = lstm_tanh(gate_g);
    float o = lstm_sigmoid(gate_o);

    // Update cell state
    float c_prev_val = c_prev[batch_idx * hidden_size + h_idx];
    float c_new = f * c_prev_val + i * g;

    // Compute hidden state
    float h_new = o * lstm_tanh(c_new);

    // Store outputs
    c_out[batch_idx * hidden_size + h_idx] = c_new;
    h_out[batch_idx * hidden_size + h_idx] = h_new;
}

// ============================================================================
// LSTM Forward - Process full sequence
// ============================================================================

/**
 * LSTM forward pass for full sequence.
 *
 * Processes sequence timestep by timestep.
 * For bidirectional, call twice (forward and reverse).
 *
 * Input:
 *   x: input sequence [batch, seq_len, input_size]
 *   W_ih: input-to-hidden weights [4*hidden_size, input_size]
 *   W_hh: hidden-to-hidden weights [4*hidden_size, hidden_size]
 *   b_ih: input bias [4*hidden_size]
 *   b_hh: hidden bias [4*hidden_size]
 *   h0: initial hidden state [batch, hidden_size] (can be nullptr for zeros)
 *   c0: initial cell state [batch, hidden_size] (can be nullptr for zeros)
 *   reverse: if true, process sequence in reverse order
 *
 * Output:
 *   output: hidden states for all timesteps [batch, seq_len, hidden_size]
 *   h_n: final hidden state [batch, hidden_size]
 *   c_n: final cell state [batch, hidden_size]
 */

// Kernel to precompute W_ih @ x + b_ih + b_hh for all timesteps
// This is a batched GEMM: [4*H, I] @ [B, S, I]^T -> [B, S, 4*H]
__global__ void lstm_precompute_gates_f32_kernel(
    const float* __restrict__ x,        // [batch, seq_len, input_size]
    const float* __restrict__ W_ih,     // [4*hidden, input_size]
    const float* __restrict__ b_ih,     // [4*hidden]
    const float* __restrict__ b_hh,     // [4*hidden]
    float* __restrict__ gates,          // [batch, seq_len, 4*hidden]
    int batch_size,
    int seq_len,
    int input_size,
    int hidden_size
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int gate_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || gate_idx >= 4 * hidden_size) return;

    // Compute W_ih @ x[batch, seq, :]
    float sum = 0.0f;
    const float* x_ptr = x + batch_idx * seq_len * input_size + seq_idx * input_size;
    const float* w_ptr = W_ih + gate_idx * input_size;

    for (int i = 0; i < input_size; ++i) {
        sum += w_ptr[i] * x_ptr[i];
    }

    // Add biases
    sum += b_ih[gate_idx] + b_hh[gate_idx];

    // Store
    gates[batch_idx * seq_len * 4 * hidden_size + seq_idx * 4 * hidden_size + gate_idx] = sum;
}

// Fused LSTM cell that operates on precomputed gates
__global__ void lstm_step_f32_kernel(
    const float* __restrict__ gates,    // [batch, 4*hidden] precomputed for this timestep
    const float* __restrict__ h_prev,   // [batch, hidden]
    const float* __restrict__ c_prev,   // [batch, hidden]
    const float* __restrict__ W_hh,     // [4*hidden, hidden]
    float* __restrict__ h_out,          // [batch, hidden]
    float* __restrict__ c_out,          // [batch, hidden]
    int batch_size,
    int hidden_size
) {
    int batch_idx = blockIdx.y;
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || h_idx >= hidden_size) return;

    // Load precomputed gates
    int base = batch_idx * 4 * hidden_size;
    float gate_i = gates[base + h_idx];
    float gate_f = gates[base + hidden_size + h_idx];
    float gate_g = gates[base + 2 * hidden_size + h_idx];
    float gate_o = gates[base + 3 * hidden_size + h_idx];

    // Add W_hh @ h_prev contribution
    const float* W_hh_i = W_hh + h_idx * hidden_size;
    const float* W_hh_f = W_hh + (hidden_size + h_idx) * hidden_size;
    const float* W_hh_g = W_hh + (2 * hidden_size + h_idx) * hidden_size;
    const float* W_hh_o = W_hh + (3 * hidden_size + h_idx) * hidden_size;
    const float* h_ptr = h_prev + batch_idx * hidden_size;

    for (int k = 0; k < hidden_size; ++k) {
        float h_k = h_ptr[k];
        gate_i += W_hh_i[k] * h_k;
        gate_f += W_hh_f[k] * h_k;
        gate_g += W_hh_g[k] * h_k;
        gate_o += W_hh_o[k] * h_k;
    }

    // Apply activations
    float i = lstm_sigmoid(gate_i);
    float f = lstm_sigmoid(gate_f);
    float g = lstm_tanh(gate_g);
    float o = lstm_sigmoid(gate_o);

    // Update states
    float c_new = f * c_prev[batch_idx * hidden_size + h_idx] + i * g;
    float h_new = o * lstm_tanh(c_new);

    c_out[batch_idx * hidden_size + h_idx] = c_new;
    h_out[batch_idx * hidden_size + h_idx] = h_new;
}

// Zero initialization kernel
__global__ void zero_init_f32_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

// Simple copy kernel (replaces cudaMemcpy DtoD)
__global__ void copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Strided copy kernel for LSTM output
// Copies h_next[batch, hidden] to output[batch, seq_idx, hidden]
// output layout: [batch, seq_len, hidden_size]
__global__ void lstm_copy_to_output_f32_kernel(
    const float* __restrict__ h_next,   // [batch, hidden_size]
    float* __restrict__ output,          // [batch, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    int seq_idx
) {
    int batch_idx = blockIdx.y;
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || h_idx >= hidden_size) return;

    // src: h_next[batch_idx, h_idx]
    int src_offset = batch_idx * hidden_size + h_idx;
    // dst: output[batch_idx, seq_idx, h_idx]
    int dst_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + h_idx;

    output[dst_offset] = h_next[src_offset];
}

// Concatenation kernel for bidirectional LSTM output
// Copies fwd[batch, seq, hidden] and bwd[batch, seq, hidden] to output[batch, seq, 2*hidden]
__global__ void lstm_concat_bidirectional_f32_kernel(
    const float* __restrict__ fwd_out,   // [batch, seq_len, hidden_size]
    const float* __restrict__ bwd_out,   // [batch, seq_len, hidden_size]
    float* __restrict__ output,          // [batch, seq_len, 2*hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || h_idx >= hidden_size) return;

    int src_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + h_idx;
    int dst_offset_fwd = batch_idx * seq_len * 2 * hidden_size + seq_idx * 2 * hidden_size + h_idx;
    int dst_offset_bwd = dst_offset_fwd + hidden_size;

    output[dst_offset_fwd] = fwd_out[src_offset];
    output[dst_offset_bwd] = bwd_out[src_offset];
}

}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
