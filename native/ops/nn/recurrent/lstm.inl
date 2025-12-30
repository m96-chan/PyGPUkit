/**
 * LSTM dispatch implementation
 *
 * Provides high-level LSTM operations:
 * - lstm_forward: unidirectional LSTM
 * - lstm_bidirectional: bidirectional LSTM
 *
 * NOTE: Uses kernel-based copies instead of cudaMemcpy for Driver API compatibility.
 */

#include "lstm_kernels.cuh"
#include "../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {

using namespace nn;

// ============================================================================
// LSTM Forward - Unidirectional
// ============================================================================

/**
 * LSTM forward pass.
 *
 * Args:
 *   x: input [batch, seq_len, input_size]
 *   W_ih: [4*hidden_size, input_size]
 *   W_hh: [4*hidden_size, hidden_size]
 *   b_ih: [4*hidden_size]
 *   b_hh: [4*hidden_size]
 *   h0: initial hidden [batch, hidden_size] or empty for zeros
 *   c0: initial cell [batch, hidden_size] or empty for zeros
 *   reverse: process sequence in reverse order
 *
 * Returns:
 *   output: [batch, seq_len, hidden_size]
 *   h_n: [batch, hidden_size]
 *   c_n: [batch, hidden_size]
 */
std::tuple<GPUArray, GPUArray, GPUArray> lstm_forward(
    const GPUArray& x,
    const GPUArray& W_ih,
    const GPUArray& W_hh,
    const GPUArray& b_ih,
    const GPUArray& b_hh,
    const GPUArray& h0,
    const GPUArray& c0,
    bool reverse
) {
    // Validate inputs
    if (x.ndim() != 3) {
        throw std::runtime_error("lstm_forward: x must be 3D [batch, seq_len, input_size]");
    }
    if (x.dtype() != DataType::Float32) {
        throw std::runtime_error("lstm_forward: only float32 supported currently");
    }

    int batch_size = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int input_size = static_cast<int>(x.shape()[2]);
    int hidden_size = static_cast<int>(W_hh.shape()[1]);

    // Allocate outputs
    GPUArray output({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)}, DataType::Float32);
    GPUArray h_n({static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);
    GPUArray c_n({static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);

    // Allocate intermediate buffers
    GPUArray gates({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), static_cast<size_t>(4 * hidden_size)}, DataType::Float32);
    GPUArray h_curr({static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);
    GPUArray c_curr({static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);
    GPUArray h_next({static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);
    GPUArray c_next({static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);

    // Get stream for CUDA Graph compatibility
    cudaStream_t stream = internal::get_capture_stream();

    // Initialize h0, c0
    int state_size = batch_size * hidden_size;
    int block_init = 256;
    int grid_init = (state_size + block_init - 1) / block_init;

    if (h0.size() > 0) {
        copy_f32_kernel<<<grid_init, block_init, 0, stream>>>(
            static_cast<const float*>(h0.data()),
            static_cast<float*>(h_curr.data()), state_size);
    } else {
        zero_init_f32_kernel<<<grid_init, block_init, 0, stream>>>(
            static_cast<float*>(h_curr.data()), state_size);
    }

    if (c0.size() > 0) {
        copy_f32_kernel<<<grid_init, block_init, 0, stream>>>(
            static_cast<const float*>(c0.data()),
            static_cast<float*>(c_curr.data()), state_size);
    } else {
        zero_init_f32_kernel<<<grid_init, block_init, 0, stream>>>(
            static_cast<float*>(c_curr.data()), state_size);
    }

    // Precompute all gates: W_ih @ x + b_ih + b_hh
    {
        int gate_size = 4 * hidden_size;
        dim3 block(256);
        dim3 grid((gate_size + 255) / 256, seq_len, batch_size);

        lstm_precompute_gates_f32_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(x.data()),
            static_cast<const float*>(W_ih.data()),
            static_cast<const float*>(b_ih.data()),
            static_cast<const float*>(b_hh.data()),
            static_cast<float*>(gates.data()),
            batch_size, seq_len, input_size, hidden_size);
    }

    sync_and_check("lstm_precompute_gates failed");

    // Process sequence
    dim3 block_step(256);
    dim3 grid_step((hidden_size + 255) / 256, batch_size);

    for (int t = 0; t < seq_len; ++t) {
        int seq_idx = reverse ? (seq_len - 1 - t) : t;

        // Adjust for correct memory layout [batch, seq, 4*hidden]
        size_t gates_offset = static_cast<size_t>(seq_idx) * 4 * hidden_size;

        lstm_step_f32_kernel<<<grid_step, block_step, 0, stream>>>(
            static_cast<const float*>(gates.data()) + gates_offset,
            static_cast<const float*>(h_curr.data()),
            static_cast<const float*>(c_curr.data()),
            static_cast<const float*>(W_hh.data()),
            static_cast<float*>(h_next.data()),
            static_cast<float*>(c_next.data()),
            batch_size, hidden_size);

        // Copy to output using kernel (strided copy)
        lstm_copy_to_output_f32_kernel<<<grid_step, block_step, 0, stream>>>(
            static_cast<const float*>(h_next.data()),
            static_cast<float*>(output.data()),
            batch_size, seq_len, hidden_size, seq_idx);

        // Swap buffers
        std::swap(h_curr, h_next);
        std::swap(c_curr, c_next);
    }

    sync_and_check("lstm_forward failed");

    // Copy final states using kernel
    copy_f32_kernel<<<grid_init, block_init, 0, stream>>>(
        static_cast<const float*>(h_curr.data()),
        static_cast<float*>(h_n.data()), state_size);
    copy_f32_kernel<<<grid_init, block_init, 0, stream>>>(
        static_cast<const float*>(c_curr.data()),
        static_cast<float*>(c_n.data()), state_size);

    sync_and_check("lstm_forward final copy failed");

    return std::make_tuple(std::move(output), std::move(h_n), std::move(c_n));
}

// ============================================================================
// LSTM Bidirectional
// ============================================================================

/**
 * Bidirectional LSTM.
 *
 * Args:
 *   x: input [batch, seq_len, input_size]
 *   W_ih_fwd, W_hh_fwd, b_ih_fwd, b_hh_fwd: forward LSTM weights
 *   W_ih_bwd, W_hh_bwd, b_ih_bwd, b_hh_bwd: backward LSTM weights
 *
 * Returns:
 *   output: [batch, seq_len, 2*hidden_size] (concatenated forward and backward)
 *   h_n: [2, batch, hidden_size]
 *   c_n: [2, batch, hidden_size]
 */
std::tuple<GPUArray, GPUArray, GPUArray> lstm_bidirectional(
    const GPUArray& x,
    const GPUArray& W_ih_fwd, const GPUArray& W_hh_fwd,
    const GPUArray& b_ih_fwd, const GPUArray& b_hh_fwd,
    const GPUArray& W_ih_bwd, const GPUArray& W_hh_bwd,
    const GPUArray& b_ih_bwd, const GPUArray& b_hh_bwd
) {
    int batch_size = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);
    int hidden_size = static_cast<int>(W_hh_fwd.shape()[1]);

    // Get stream for CUDA Graph compatibility
    cudaStream_t stream = internal::get_capture_stream();

    // Empty initial states (zero-sized arrays)
    GPUArray empty_h0_fwd({0}, DataType::Float32);
    GPUArray empty_c0_fwd({0}, DataType::Float32);
    GPUArray empty_h0_bwd({0}, DataType::Float32);
    GPUArray empty_c0_bwd({0}, DataType::Float32);

    // Forward pass
    auto [out_fwd, h_fwd, c_fwd] = lstm_forward(
        x, W_ih_fwd, W_hh_fwd, b_ih_fwd, b_hh_fwd, empty_h0_fwd, empty_c0_fwd, false);

    // Backward pass
    auto [out_bwd, h_bwd, c_bwd] = lstm_forward(
        x, W_ih_bwd, W_hh_bwd, b_ih_bwd, b_hh_bwd, empty_h0_bwd, empty_c0_bwd, true);

    // Concatenate outputs: [batch, seq_len, 2*hidden]
    GPUArray output({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), static_cast<size_t>(2 * hidden_size)}, DataType::Float32);

    // Use concatenation kernel (single kernel launch instead of nested loops)
    {
        dim3 block(256);
        dim3 grid((hidden_size + 255) / 256, seq_len, batch_size);

        lstm_concat_bidirectional_f32_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(out_fwd.data()),
            static_cast<const float*>(out_bwd.data()),
            static_cast<float*>(output.data()),
            batch_size, seq_len, hidden_size);
    }

    // Stack h_n, c_n: [2, batch, hidden]
    GPUArray h_n({2, static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);
    GPUArray c_n({2, static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::Float32);

    int state_size = batch_size * hidden_size;
    int block_copy = 256;
    int grid_copy = (state_size + block_copy - 1) / block_copy;

    // Copy h_n[0] = h_fwd, h_n[1] = h_bwd
    copy_f32_kernel<<<grid_copy, block_copy, 0, stream>>>(
        static_cast<const float*>(h_fwd.data()),
        static_cast<float*>(h_n.data()), state_size);
    copy_f32_kernel<<<grid_copy, block_copy, 0, stream>>>(
        static_cast<const float*>(h_bwd.data()),
        static_cast<float*>(h_n.data()) + state_size, state_size);

    // Copy c_n[0] = c_fwd, c_n[1] = c_bwd
    copy_f32_kernel<<<grid_copy, block_copy, 0, stream>>>(
        static_cast<const float*>(c_fwd.data()),
        static_cast<float*>(c_n.data()), state_size);
    copy_f32_kernel<<<grid_copy, block_copy, 0, stream>>>(
        static_cast<const float*>(c_bwd.data()),
        static_cast<float*>(c_n.data()) + state_size, state_size);

    sync_and_check("lstm_bidirectional failed");

    return std::make_tuple(std::move(output), std::move(h_n), std::move(c_n));
}

}  // namespace ops
}  // namespace pygpukit
