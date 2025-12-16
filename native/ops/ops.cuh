/**
 * PyGPUkit Operations - Public API
 *
 * This header provides access to all GPU array operations:
 * - Elementwise: add, mul, sub, div
 * - Unary: exp, log, relu
 * - Reduction: sum, mean, max
 * - Matmul: matrix multiplication with TensorCore support
 */
#pragma once

#include "../core/memory.hpp"

namespace pygpukit {
namespace ops {

// ============================================================================
// Elementwise Operations
// ============================================================================

// Add: c = a + b
void add(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray add(const GPUArray& a, const GPUArray& b);

// Mul: c = a * b
void mul(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray mul(const GPUArray& a, const GPUArray& b);

// Sub: c = a - b
void sub(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray sub(const GPUArray& a, const GPUArray& b);

// Div: c = a / b
void div(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray div(const GPUArray& a, const GPUArray& b);

// ============================================================================
// Unary Operations
// ============================================================================

// Exp: c = exp(a)
void exp(const GPUArray& a, GPUArray& c);
GPUArray exp(const GPUArray& a);

// Log: c = log(a)
void log(const GPUArray& a, GPUArray& c);
GPUArray log(const GPUArray& a);

// ReLU: c = max(0, a)
void relu(const GPUArray& a, GPUArray& c);
GPUArray relu(const GPUArray& a);

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum: scalar sum of all elements
GPUArray sum(const GPUArray& a);

// Mean: scalar mean of all elements
GPUArray mean(const GPUArray& a);

// Max: scalar max of all elements
GPUArray max(const GPUArray& a);

// ============================================================================
// Matrix Multiplication
// ============================================================================

// Matmul: c = a @ b
// Automatically selects optimal kernel based on dtype and size:
// - FP32: L2-optimized, tiled, or Ampere-optimized kernel
// - FP32 + PYGPUKIT_ALLOW_TF32=1: TF32 TensorCore kernel
// - FP16/BF16: Simple or TensorCore kernel (PYGPUKIT_ALLOW_FP16_TC=1)
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray matmul(const GPUArray& a, const GPUArray& b);

// Matmul with explicit TF32 control
void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32);
GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32);

// ============================================================================
// Neural Network Operations
// ============================================================================

// Transpose: c = a.T
// input: [rows, cols], output: [cols, rows]
GPUArray transpose(const GPUArray& input);

// GELU: Gaussian Error Linear Unit activation
// y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
GPUArray gelu(const GPUArray& input);

// Bias Add: output[batch, features] += bias[features] (in-place)
void bias_add_inplace(GPUArray& output, const GPUArray& bias);

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// input: [batch, features], gamma/beta: [features]
GPUArray layernorm(const GPUArray& input, const GPUArray& gamma, const GPUArray& beta, float eps = 1e-5f);

// Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
// Applied row-wise: input [batch, features] -> output [batch, features]
GPUArray softmax(const GPUArray& input);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
// input: [batch, features], gamma: [features]
// Simpler than LayerNorm (no mean subtraction, no beta)
GPUArray rmsnorm(const GPUArray& input, const GPUArray& gamma, float eps = 1e-5f);

// SiLU (Swish) activation: y = x * sigmoid(x)
GPUArray silu(const GPUArray& input);

// RoPE (Rotary Position Embedding) - In-place
// q: [seq_len, n_heads_q, head_dim]
// k: [seq_len, n_heads_k, head_dim]
// cos, sin: [seq_len, head_dim]
void rope_inplace(GPUArray& q, GPUArray& k, const GPUArray& cos, const GPUArray& sin);

// Scaled Dot-Product Attention with Causal Mask
// Q: [n_heads, q_len, head_dim]
// K: [n_heads, kv_len, head_dim]
// V: [n_heads, kv_len, head_dim]
// Output: [n_heads, q_len, head_dim]
// scale: 1/sqrt(head_dim), computed automatically if <= 0
GPUArray sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, float scale = 0.0f);

// ============================================================================
// Fused Operations (CUTLASS Epilogue Fusion)
// ============================================================================

// Linear + BiasGELU: output = gelu(input @ weight^T + bias)
// Fused kernel for efficient MLP layers
// input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]
// output: [batch, out_features]
GPUArray linear_bias_gelu(const GPUArray& input, const GPUArray& weight, const GPUArray& bias);

// ============================================================================
// Tensor Manipulation Operations
// ============================================================================

// Concat two tensors along axis 0
// a: [dim0_a, ...], b: [dim0_b, ...] -> output: [dim0_a + dim0_b, ...]
GPUArray concat_axis0(const GPUArray& a, const GPUArray& b);

// Repeat interleave along axis 1 (for GQA expansion)
// input: [dim0, dim1, dim2] -> output: [dim0, dim1 * repeats, dim2]
GPUArray repeat_interleave_axis1(const GPUArray& input, size_t repeats);

// Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]
GPUArray transpose_3d_021(const GPUArray& input);

// Reshape with copy (creates contiguous tensor with new shape)
GPUArray reshape_copy(const GPUArray& input, const std::vector<size_t>& new_shape);

// ============================================================================
// Quantization Operations (#85)
// ============================================================================

// Dequantize INT8 to FP16/FP32: output = input_int8 * scale
// input: [rows, cols] INT8, scale: [cols] FP16/FP32, output: [rows, cols] FP16/FP32
GPUArray dequantize_int8(const GPUArray& input, const GPUArray& scale, DataType output_dtype);

// Quantized Linear: output = activation @ (weight_int8 * scale).T
// activation: [M, K] FP16, weight_int8: [N, K] INT8, scale: [N] FP16
// output: [M, N] FP16
// Dequantization happens on-the-fly (no intermediate buffer)
GPUArray linear_int8(
    const GPUArray& activation,
    const GPUArray& weight_int8,
    const GPUArray& scale,
    const GPUArray* bias = nullptr
);

// Quantize FP16/FP32 to INT8 with per-column scaling
// Returns (weight_int8, scale) pair
// weight_int8: [rows, cols] INT8, scale: [cols] FP16/FP32
std::pair<GPUArray, GPUArray> quantize_to_int8(const GPUArray& input);

// ============================================================================
// Paged Attention (#87)
// ============================================================================

// Paged Attention v1: single-query attention with paged KV cache
// Q: [num_seqs, num_heads, head_dim]
// K_cache, V_cache: [num_blocks, num_kv_heads, block_size, head_dim]
// block_tables: [num_seqs, max_num_blocks_per_seq] int32
// context_lens: [num_seqs] int32
GPUArray paged_attention_v1(
    const GPUArray& Q,
    const GPUArray& K_cache,
    const GPUArray& V_cache,
    const GPUArray& block_tables,
    const GPUArray& context_lens,
    float scale = 0.0f
);

// Copy new KV entries to paged cache (decode phase)
// K_new, V_new: [num_seqs, num_kv_heads, head_dim]
// slot_mapping: [num_seqs] int32 - physical slot indices
void copy_to_paged_cache(
    const GPUArray& K_new,
    const GPUArray& V_new,
    GPUArray& K_cache,
    GPUArray& V_cache,
    const GPUArray& slot_mapping
);

// Reshape and copy KV from prefill format to paged cache
// K, V: [batch * seq_len, num_kv_heads, head_dim] (flattened prefill output)
// slot_mapping: [total_tokens] int32
void reshape_and_cache(
    const GPUArray& K,
    const GPUArray& V,
    GPUArray& K_cache,
    GPUArray& V_cache,
    const GPUArray& slot_mapping
);

// Allocate KV cache blocks
// Returns: [num_blocks, num_kv_heads, block_size, head_dim] FP16
GPUArray allocate_kv_cache(int num_blocks, int num_kv_heads, int block_size, int head_dim);

} // namespace ops
} // namespace pygpukit
