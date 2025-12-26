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

// Clamp: c = clamp(a, min_val, max_val)
void clamp(const GPUArray& a, GPUArray& c, float min_val, float max_val);
GPUArray clamp(const GPUArray& a, float min_val, float max_val);

// Where: c = cond ? a : b (conditional select)
void where(const GPUArray& cond, const GPUArray& a, const GPUArray& b, GPUArray& c);
GPUArray where(const GPUArray& cond, const GPUArray& a, const GPUArray& b);

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

// Sin: c = sin(a)
void sin(const GPUArray& a, GPUArray& c);
GPUArray sin(const GPUArray& a);

// Cos: c = cos(a)
void cos(const GPUArray& a, GPUArray& c);
GPUArray cos(const GPUArray& a);

// Sqrt: c = sqrt(a)
void sqrt(const GPUArray& a, GPUArray& c);
GPUArray sqrt(const GPUArray& a);

// Rsqrt: c = 1/sqrt(a)
void rsqrt(const GPUArray& a, GPUArray& c);
GPUArray rsqrt(const GPUArray& a);

// Abs: c = |a|
void abs(const GPUArray& a, GPUArray& c);
GPUArray abs(const GPUArray& a);

// Neg: c = -a
void neg(const GPUArray& a, GPUArray& c);
GPUArray neg(const GPUArray& a);

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum: scalar sum of all elements
GPUArray sum(const GPUArray& a);

// Mean: scalar mean of all elements
GPUArray mean(const GPUArray& a);

// Max: scalar max of all elements
GPUArray max(const GPUArray& a);

// Min: scalar min of all elements
GPUArray min(const GPUArray& a);

// Argmax: index of maximum element
GPUArray argmax(const GPUArray& a);

// Sum with axis: sum along specified axis (0 or 1)
// input: [M, N], axis=0 -> output: [N], axis=1 -> output: [M]
GPUArray sum_axis(const GPUArray& a, int axis);

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

// RMSNorm with output buffer (for CUDA Graph capture)
void rmsnorm(const GPUArray& input, const GPUArray& gamma, GPUArray& out, float eps = 1e-5f);

// SiLU (Swish) activation: y = x * sigmoid(x)
GPUArray silu(const GPUArray& input);

// SiLU with output buffer (for CUDA Graph capture)
void silu(const GPUArray& input, GPUArray& out);

// Sigmoid activation: y = 1 / (1 + exp(-x))
GPUArray sigmoid(const GPUArray& input);
void sigmoid(const GPUArray& input, GPUArray& out);

// Tanh activation
GPUArray tanh(const GPUArray& input);
void tanh(const GPUArray& input, GPUArray& out);

// RoPE (Rotary Position Embedding) - In-place
// q: [seq_len, n_heads_q, head_dim]
// k: [seq_len, n_heads_k, head_dim]
// cos, sin: [seq_len, head_dim]
void rope_inplace(GPUArray& q, GPUArray& k, const GPUArray& cos, const GPUArray& sin);

// RoPE with FP32 cos/sin tables (higher precision for bf16/f16 Q/K)
// q: [seq_len, n_heads_q, head_dim] (bf16 or f16)
// k: [seq_len, n_heads_k, head_dim] (bf16 or f16)
// cos, sin: [seq_len, head_dim] (f32)
void rope_inplace_f32table(GPUArray& q, GPUArray& k, const GPUArray& cos, const GPUArray& sin);

// Split fused QKV projection output into separate Q, K, V tensors
// qkv: [seq_len, q_dim + k_dim + v_dim]
// q_out: [seq_len, q_dim] (can be pre-allocated buffer)
// k_out: [seq_len, k_dim]
// v_out: [seq_len, v_dim]
// Note: Output buffers must be pre-allocated for CUDA Graph compatibility
void split_qkv_batch(
    const GPUArray& qkv,
    GPUArray& q_out,
    GPUArray& k_out,
    GPUArray& v_out,
    int q_dim,
    int k_dim,
    int v_dim
);

// Scaled Dot-Product Attention with Causal Mask
// Q: [n_heads, q_len, head_dim]
// K: [n_heads, kv_len, head_dim]
// V: [n_heads, kv_len, head_dim]
// Output: [n_heads, q_len, head_dim]
// scale: 1/sqrt(head_dim), computed automatically if <= 0
GPUArray sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, float scale = 0.0f);

// SDPA with output buffer (for CUDA Graph capture)
void sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, GPUArray& out, float scale = 0.0f);

// SDPA with fixed-length KV cache support (for CUDA Graph with dynamic context)
// K/V are pre-allocated to max_seq_len, context_len specifies actual valid tokens
void sdpa_causal_fixed_cache(const GPUArray& Q, const GPUArray& K, const GPUArray& V,
                              GPUArray& out, int context_len, float scale = 0.0f);

// SDPA with pointer-based context_len (for CUDA Graph replay with dynamic context)
// context_len_buf: GPU int32 buffer containing actual context length
// max_kv_len: Maximum context length (for shared memory allocation at graph capture)
void sdpa_causal_fixed_cache_ptr(const GPUArray& Q, const GPUArray& K, const GPUArray& V,
                                   GPUArray& out, const GPUArray& context_len_buf,
                                   int max_kv_len, float scale = 0.0f);

// ============================================================================
// Fused Operations (CUTLASS Epilogue Fusion)
// ============================================================================

// Linear + BiasGELU: output = gelu(input @ weight^T + bias)
// Fused kernel for efficient MLP layers
// input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]
// output: [batch, out_features]
GPUArray linear_bias_gelu(const GPUArray& input, const GPUArray& weight, const GPUArray& bias);

// Strided Batched GEMM: C[b] = A[b] @ B[b] for b in [0, batch_count)
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N] (row-major)
// Uses CUTLASS TensorCore for high performance
void batched_matmul_fp32(const GPUArray& A, const GPUArray& B, GPUArray& C,
                         int M, int N, int K, int batch_count,
                         int64_t strideA, int64_t strideB, int64_t strideC);

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
// Transpose 3D tensor with output buffer (for CUDA Graph capture)
void transpose_3d_021(const GPUArray& input, GPUArray& out);

// Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
// Swaps axes 1 and 2 (common in attention: batch, seq, heads, dim -> batch, heads, seq, dim)
GPUArray transpose_4d_0213(const GPUArray& input);
// Transpose 4D tensor with output buffer (for CUDA Graph capture)
void transpose_4d_0213(const GPUArray& input, GPUArray& out);

// Transpose 3D tensor: [d0, d1, d2] -> [d0, d2, d1]
// Swaps last two axes (common in attention operations)
GPUArray transpose_3d_012(const GPUArray& input);
// Transpose 3D tensor with output buffer (for CUDA Graph capture)
void transpose_3d_012(const GPUArray& input, GPUArray& out);

// Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d1, d3, d2]
// Swaps last two axes (for K^T in attention)
GPUArray transpose_4d_0132(const GPUArray& input);
// Transpose 4D tensor with output buffer (for CUDA Graph capture)
void transpose_4d_0132(const GPUArray& input, GPUArray& out);

// Reshape with copy (creates contiguous tensor with new shape)
GPUArray reshape_copy(const GPUArray& input, const std::vector<size_t>& new_shape);
// Reshape with copy into output buffer (for CUDA Graph capture)
void reshape_copy(const GPUArray& input, GPUArray& out);

// ============================================================================
// Fixed-Length KV Cache Operations (CUDA Graph Support)
// ============================================================================

// Update KV cache at a single position (decode step)
// new_kv: [1, num_kv_heads, head_dim] - single token K or V
// cache: [max_seq_len, num_kv_heads, head_dim] - pre-allocated cache
// position: where to write in cache (0-indexed)
void kv_cache_update(const GPUArray& new_kv, GPUArray& cache, int position);

// Prefill KV cache from sequence (prefill step)
// new_kv: [seq_len, num_kv_heads, head_dim]
// cache: [max_seq_len, num_kv_heads, head_dim]
// start_pos: where to start writing in cache
void kv_cache_prefill(const GPUArray& new_kv, GPUArray& cache, int start_pos);

// GQA-expanded KV cache operations (for CUDA Graph optimization)
// These write to transposed, GQA-expanded cache: [num_heads, max_seq_len, head_dim]
void kv_cache_update_gqa(const GPUArray& new_kv, GPUArray& cache, int num_heads, int position);
void kv_cache_update_gqa_ptr(const GPUArray& new_kv, GPUArray& cache, int num_heads, const GPUArray& position_buf);
void kv_cache_prefill_gqa(const GPUArray& new_kv, GPUArray& cache, int num_heads, int start_pos);

// Embedding lookup - GPU-only, no CPU transfer
// embed_matrix: [vocab_size, hidden_size], out: [1, hidden_size], token_id: row index
void embedding_lookup(const GPUArray& embed_matrix, GPUArray& out, int token_id);
void embedding_lookup_ptr(const GPUArray& embed_matrix, GPUArray& out, const GPUArray& token_id_buf);
void embedding_lookup_batch(const GPUArray& embed_matrix, GPUArray& out, const GPUArray& token_ids_buf, int batch_size);

// Slice consecutive rows from table using GPU-stored start position
// Copies `count` rows starting from start_pos (read from GPU buffer)
// out[i, :] = table[start_pos + i, :]
void slice_rows_range_ptr(const GPUArray& table, GPUArray& out, const GPUArray& start_pos_buf, int count);

// In-place addition: a += b
void add_inplace(GPUArray& a, const GPUArray& b);

// In-place multiplication: a *= b
void mul_inplace(GPUArray& a, const GPUArray& b);

// GPU-to-GPU copy
void copy_to(const GPUArray& src, GPUArray& dst);

// ============================================================================
// Dtype Cast Operations
// ============================================================================

// Cast float32 to bfloat16 (round to nearest even)
GPUArray cast_f32_to_bf16(const GPUArray& src);
void cast_f32_to_bf16(const GPUArray& src, GPUArray& dst);

// Cast float32 to float16
GPUArray cast_f32_to_f16(const GPUArray& src);

// Cast bfloat16 to float32
GPUArray cast_bf16_to_f32(const GPUArray& src);

// Cast float16 to float32
GPUArray cast_f16_to_f32(const GPUArray& src);

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

// ============================================================================
// Continuous Batching (#86)
// ============================================================================

// Gather token embeddings for a batch
// token_ids: [total_tokens] int32
// embeddings: [vocab_size, hidden_size] FP16
// Returns: [total_tokens, hidden_size] FP16
GPUArray gather_embeddings(
    const GPUArray& token_ids,
    const GPUArray& embeddings,
    int total_tokens
);

// Scatter last-token logits from batch output
// logits: [batch_tokens, vocab_size] FP16
// Returns: [batch_size, vocab_size] FP16
GPUArray scatter_last_token_logits(
    const GPUArray& logits,
    const GPUArray& seq_start_positions,
    const GPUArray& seq_lens,
    int batch_size,
    int vocab_size
);

// Prepare position IDs for rotary embeddings
// Returns: [total_tokens] int32
GPUArray prepare_position_ids(
    const GPUArray& seq_start_positions,
    const GPUArray& seq_context_lens,
    const GPUArray& is_prefill,
    const GPUArray& input_lens,
    int batch_size,
    int total_tokens
);

// Argmax sampling from logits
// logits: [batch_size, vocab_size] FP16
// Returns: [batch_size] int32 - sampled token IDs
GPUArray argmax_sample(
    const GPUArray& logits,
    int batch_size,
    int vocab_size
);

// Check for EOS tokens
// tokens: [batch_size] int32
// Returns: [batch_size] int32 - 1 if EOS, 0 otherwise
GPUArray check_eos(const GPUArray& tokens, int eos_token_id);

// Compute exclusive prefix sum (for seq_start_positions)
GPUArray compute_cumsum(const GPUArray& input);

// Prepare batch inputs from Python lists
// Returns: (token_ids GPUArray, total_tokens count)
std::pair<GPUArray, int> prepare_batch_inputs(
    const std::vector<std::vector<int>>& token_lists
);

// ============================================================================
// GPU Sampling Operations (#v0.2.10)
// ============================================================================

// Greedy sampling (argmax)
// logits: [vocab_size] or [1, vocab_size]
// Returns: sampled token ID
int sample_greedy(const GPUArray& logits);

// Multinomial sampling with temperature
// logits: [vocab_size] or [1, vocab_size]
// temperature: > 0 (lower = more deterministic)
// Returns: sampled token ID
int sample_multinomial(const GPUArray& logits, float temperature);

// Top-K sampling
// Samples from top-k highest probability tokens
// top_k: number of tokens to consider (> 0)
int sample_topk(const GPUArray& logits, int top_k, float temperature);

// Top-K sampling (CUDA Graph compatible)
// Writes result to pre-allocated buffer, no sync/D2H
// result_buf: pre-allocated int32 buffer [1]
// random_val: pre-generated random value [0, 1)
void sample_topk_to_buf(
    const GPUArray& logits,
    GPUArray& result_buf,
    int top_k,
    float temperature,
    float random_val
);

// Top-K sampling with pointer (CUDA Graph replay compatible)
// random_val is read from GPU buffer, allowing update before replay
// result_buf: pre-allocated int32 buffer [1]
// random_val_buf: pre-allocated float32 buffer [1] (updated before replay)
void sample_topk_to_buf_ptr(
    const GPUArray& logits,
    GPUArray& result_buf,
    const GPUArray& random_val_buf,
    int top_k,
    float temperature
);

// Top-P (Nucleus) sampling
// Samples from smallest set of tokens whose cumulative probability >= top_p
// top_p: cumulative probability threshold (0 < p <= 1)
int sample_topp(const GPUArray& logits, float top_p, float temperature);

// Unified sampling API
// Automatically selects sampling method based on parameters:
// - temperature=0: greedy (argmax)
// - top_k > 0: top-k sampling
// - top_p < 1: top-p sampling
// - otherwise: multinomial with temperature
int sample_token_gpu(
    const GPUArray& logits,
    float temperature = 1.0f,
    int top_k = 0,
    float top_p = 1.0f
);

// Set random seed for reproducible sampling
void set_sampling_seed(unsigned int seed);

} // namespace ops
} // namespace pygpukit
