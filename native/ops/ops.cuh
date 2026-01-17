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
#include <tuple>
#include <utility>

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

// L2 Norm (Llama4TextL2Norm): y = x * rsqrt(mean(x^2) + eps)
// Unlike RMSNorm, no gamma scaling is applied.
// Used for QK normalization in Llama 4 attention.
GPUArray l2norm(const GPUArray& input, float eps = 1e-6f);
void l2norm(const GPUArray& input, GPUArray& out, float eps = 1e-6f);

// iRoPE Q scaling (Llama 4)
// Formula: scale = log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1.0
// Q: [seq_len, num_heads, head_dim], positions: [seq_len]
GPUArray irope_scale_q(const GPUArray& Q, const GPUArray& positions,
                       float attn_scale = 0.1f, float floor_scale = 8192.0f);

// SDPA with iRoPE temperature scaling (Llama 4)
// Q: [n_heads, q_len, head_dim], K/V: [n_kv_heads, kv_len, head_dim]
GPUArray sdpa_irope(const GPUArray& Q, const GPUArray& K, const GPUArray& V,
                    const GPUArray& positions, float attn_scale = 0.1f,
                    float floor_scale = 8192.0f, int causal_offset = 0);

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

// ReLU squared activation: y = (max(0, x))^2
GPUArray relu2(const GPUArray& input);
void relu2(const GPUArray& input, GPUArray& out);

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

// RoPE context extension: NTK-aware scaling
// Returns (cos_table, sin_table) each [max_seq_len, head_dim]
std::pair<GPUArray, GPUArray> rope_init_ntk_aware(
    int max_seq_len, int head_dim, float base = 10000.0f, float scale = 1.0f);

// RoPE context extension: YaRN dimension-wise interpolation
// Returns (cos_table, sin_table) each [max_seq_len, head_dim]
std::pair<GPUArray, GPUArray> rope_init_yarn(
    int max_seq_len, int head_dim, float base = 10000.0f, float scale = 1.0f,
    int original_max_len = 4096, float beta_fast = 32.0f, float beta_slow = 1.0f, float mscale = 0.1f);

// RoPE context extension: Linear position interpolation
// Returns (cos_table, sin_table) each [max_seq_len, head_dim]
std::pair<GPUArray, GPUArray> rope_init_linear(
    int max_seq_len, int head_dim, float base = 10000.0f, float scale = 1.0f);

// PoPE (Positional Encoding) - additive positional encoding
// Returns encoding tensor [max_seq_len, head_dim]
GPUArray pope_init_encoding(int max_seq_len, int head_dim, float base = 10000.0f);

// PoPE in-place application
// q: [seq_len, n_heads_q, head_dim]
// k: [seq_len, n_heads_k, head_dim]
// encoding: [max_seq_len, head_dim] (f32)
void pope_inplace(GPUArray& q, GPUArray& k, const GPUArray& encoding, int start_pos = 0);

// ALiBi (Attention with Linear Biases) - head-specific slopes
// Returns slopes tensor [num_heads]
GPUArray alibi_init_slopes(int num_heads);

// ALiBi bias matrix computation
// Returns bias tensor [num_heads, seq_len, seq_len]
GPUArray alibi_compute_bias(int seq_len, int num_heads, const GPUArray& slopes, bool causal = true);

// ALiBi in-place bias addition to attention scores
// scores: [batch, num_heads, q_len, kv_len]
// slopes: [num_heads]
void alibi_add_bias(GPUArray& scores, const GPUArray& slopes, int start_pos = 0);

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

// SDPA with kernel-only timing (for benchmarking)
// Returns kernel execution time in microseconds via kernel_time_us
// Uses cudaEvent to measure ONLY kernel execution, excluding host overhead
void sdpa_causal_timed(const GPUArray& Q, const GPUArray& K, const GPUArray& V,
                       GPUArray& out, float scale, float* kernel_time_us);

// TMA descriptor cache statistics (for debugging/benchmarking)
void print_tma_cache_stats();
void clear_tma_cache();

// FA3 FP8: FP8 Q@K^T with block-scale MMA, BF16 P@V (SM120+)
bool fa3_fp8_available();
void sdpa_causal_fp8(const GPUArray& Q, const GPUArray& K, const GPUArray& V,
                     GPUArray& out, float scale);

// FP8 MMA test function (for debugging C fragment layout)
void test_fp8_mma_direct();

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

// ============================================================================
// LSTM (Long Short-Term Memory)
// ============================================================================

// LSTM forward pass (unidirectional)
// x: [batch, seq_len, input_size]
// W_ih: [4*hidden_size, input_size], W_hh: [4*hidden_size, hidden_size]
// b_ih, b_hh: [4*hidden_size]
// h0, c0: [batch, hidden_size] or empty for zeros
// reverse: process sequence in reverse order
// Returns: (output[batch, seq_len, hidden], h_n[batch, hidden], c_n[batch, hidden])
std::tuple<GPUArray, GPUArray, GPUArray> lstm_forward(
    const GPUArray& x,
    const GPUArray& W_ih,
    const GPUArray& W_hh,
    const GPUArray& b_ih,
    const GPUArray& b_hh,
    const GPUArray& h0,
    const GPUArray& c0,
    bool reverse = false
);

// Bidirectional LSTM
// Returns: (output[batch, seq_len, 2*hidden], h_n[2, batch, hidden], c_n[2, batch, hidden])
std::tuple<GPUArray, GPUArray, GPUArray> lstm_bidirectional(
    const GPUArray& x,
    const GPUArray& W_ih_fwd, const GPUArray& W_hh_fwd,
    const GPUArray& b_ih_fwd, const GPUArray& b_hh_fwd,
    const GPUArray& W_ih_bwd, const GPUArray& W_hh_bwd,
    const GPUArray& b_ih_bwd, const GPUArray& b_hh_bwd
);

// ============================================================================
// Diffusion Model Operations (SD3, Flux, PixArt)
// ============================================================================

// GroupNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// input: [N, C, H, W], gamma/beta: [C], normalize over (C/num_groups, H, W)
GPUArray group_norm(
    const GPUArray& input,
    const GPUArray& gamma,
    const GPUArray& beta,
    int num_groups,
    float eps = 1e-5f
);

// AdaLN: y = (x - mean) / sqrt(var + eps) * (1 + scale) + shift
// input: [B, N, D], scale/shift: [B, D] (per-sample modulation from timestep embedding)
GPUArray adaln(
    const GPUArray& input,
    const GPUArray& scale,
    const GPUArray& shift,
    float eps = 1e-5f
);

// AdaLN-Zero: y = residual + gate * ((x - mean) / sqrt(var + eps) * (1 + scale) + shift)
// input: [B, N, D], scale/shift/gate: [B, D], residual: [B, N, D]
GPUArray adaln_zero(
    const GPUArray& input,
    const GPUArray& scale,
    const GPUArray& shift,
    const GPUArray& gate,
    const GPUArray& residual,
    float eps = 1e-5f
);

// Cross-Attention (no causal mask) for text-to-image conditioning
// Q: [n_heads, q_len, head_dim] (from image latents)
// K: [n_heads, kv_len, head_dim] (from text embeddings)
// V: [n_heads, kv_len, head_dim] (from text embeddings)
// Output: [n_heads, q_len, head_dim]
GPUArray cross_attention(
    const GPUArray& Q,
    const GPUArray& K,
    const GPUArray& V,
    float scale = 0.0f
);

// Conv2D 1x1 (pointwise convolution, common in VAE/UNet)
// input: [N, C_in, H, W], weight: [C_out, C_in], bias: [C_out] or nullptr
// output: [N, C_out, H, W]
GPUArray conv2d_1x1(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray* bias = nullptr
);

// Conv2D 3x3 direct (optimized for small kernels)
// input: [N, C_in, H, W], weight: [C_out, C_in, 3, 3], bias: [C_out] or nullptr
GPUArray conv2d_3x3(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray* bias = nullptr,
    int pad_h = 1,
    int pad_w = 1,
    int stride_h = 1,
    int stride_w = 1
);

// im2col for general convolution (use with GEMM for Conv2D)
// input: [N, C, H, W]
// output: [N, C*K_h*K_w, H_out*W_out]
GPUArray im2col(
    const GPUArray& input,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h = 1, int dil_w = 1
);

// col2im for transposed convolution (deconvolution)
// input: [N, C*K_h*K_w, H_in*W_in]
// output: [N, C, H, W]
GPUArray col2im(
    const GPUArray& input,
    int C, int H, int W,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h = 1, int dil_w = 1
);

// ============================================================================
// FLUX-specific Operations
// ============================================================================

// LayerNorm without learnable parameters
// input: [B, N, D] or [rows, D]
GPUArray layer_norm_simple(const GPUArray& input, float eps = 1e-6f);

// Modulate: y = x * (1 + scale) + shift
// input: [B, N, D], scale/shift: [B, D]
GPUArray modulate(
    const GPUArray& input,
    const GPUArray& scale,
    const GPUArray& shift
);

// Gated residual: y = residual + gate * value
// residual/value: [B, N, D], gate: [B, D]
GPUArray gated_residual(
    const GPUArray& residual,
    const GPUArray& gate,
    const GPUArray& value
);

// Scale tensor: y = x * scale
GPUArray scale_tensor(const GPUArray& input, float scale);

// Concat along axis 1: [B, N1, ...] + [B, N2, ...] -> [B, N1+N2, ...]
// Supports 3D [B, N, D] and 4D [B, N, H, D]
GPUArray concat_axis1(const GPUArray& a, const GPUArray& b);

// Split along axis 1: [B, N, ...] -> ([B, split_size, ...], [B, N-split_size, ...])
// Supports 3D [B, N, D] and 4D [B, N, H, D]
std::pair<GPUArray, GPUArray> split_axis1(const GPUArray& input, int split_size);

// Apply RoPE (Rotary Position Embedding)
// x: [B, N, H, D], cos/sin: [N, D]
GPUArray apply_rope(
    const GPUArray& x,
    const GPUArray& cos_freq,
    const GPUArray& sin_freq
);

// ============================================================================
// Convolution Operations
// ============================================================================

// Conv1d: 1D convolution
// input: [batch, in_channels, length]
// weight: [out_channels, in_channels, kernel_size]
// bias: [out_channels] (optional)
// output: [batch, out_channels, out_length]
void conv1d(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray* bias,
    GPUArray& output,
    int stride,
    int padding
);

GPUArray conv1d(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray* bias,
    int stride,
    int padding
);

// Convenience wrappers for pybind11
GPUArray conv1d_no_bias(
    const GPUArray& input,
    const GPUArray& weight,
    int stride,
    int padding
);

GPUArray conv1d_with_bias(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray& bias,
    int stride,
    int padding
);

} // namespace ops
} // namespace pygpukit
