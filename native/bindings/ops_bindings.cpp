#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../ops/ops.cuh"

namespace py = pybind11;
using namespace pygpukit;

void init_ops_bindings(py::module_& m) {
    // ========================================================================
    // Binary Element-wise operations
    // ========================================================================

    // Add
    m.def("add", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::add),
          py::arg("a"), py::arg("b"),
          "Element-wise addition of two GPUArrays");

    m.def("add_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::add),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise addition with output array");

    // Sub
    m.def("sub", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::sub),
          py::arg("a"), py::arg("b"),
          "Element-wise subtraction of two GPUArrays");

    m.def("sub_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::sub),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise subtraction with output array");

    // Mul
    m.def("mul", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::mul),
          py::arg("a"), py::arg("b"),
          "Element-wise multiplication of two GPUArrays");

    m.def("mul_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::mul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise multiplication with output array");

    // Div
    m.def("div", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::div),
          py::arg("a"), py::arg("b"),
          "Element-wise division of two GPUArrays");

    m.def("div_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::div),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Element-wise division with output array");

    // ========================================================================
    // Unary Element-wise operations (float only)
    // ========================================================================

    // Exp
    m.def("exp", py::overload_cast<const GPUArray&>(&ops::exp),
          py::arg("a"),
          "Element-wise exponential (float32/float64 only)");

    m.def("exp_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::exp),
          py::arg("a"), py::arg("out"),
          "Element-wise exponential with output array");

    // Log
    m.def("log", py::overload_cast<const GPUArray&>(&ops::log),
          py::arg("a"),
          "Element-wise natural logarithm (float32/float64 only)");

    m.def("log_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::log),
          py::arg("a"), py::arg("out"),
          "Element-wise natural logarithm with output array");

    // ReLU
    m.def("relu", py::overload_cast<const GPUArray&>(&ops::relu),
          py::arg("a"),
          "Element-wise ReLU: max(0, x) (float32/float64 only)");

    m.def("relu_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::relu),
          py::arg("a"), py::arg("out"),
          "Element-wise ReLU with output array");

    // ========================================================================
    // Matrix operations
    // ========================================================================

    m.def("matmul", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"),
          "Matrix multiplication of two GPUArrays");

    m.def("matmul_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Matrix multiplication with output array");

    // TF32 variants
    m.def("matmul_tf32", py::overload_cast<const GPUArray&, const GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("use_tf32"),
          "Matrix multiplication with explicit TF32 control");

    m.def("matmul_tf32_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("use_tf32"),
          "Matrix multiplication with explicit TF32 control and output array");

    // ========================================================================
    // Reduction operations
    // ========================================================================

    m.def("sum", &ops::sum,
          py::arg("a"),
          "Sum of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("mean", &ops::mean,
          py::arg("a"),
          "Mean of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("max", &ops::max,
          py::arg("a"),
          "Max of all elements (float32/float64 only), returns scalar GPUArray");

    // ========================================================================
    // Neural Network operations
    // ========================================================================

    // Transpose
    m.def("transpose", &ops::transpose,
          py::arg("input"),
          "Matrix transpose: input [rows, cols] -> output [cols, rows]");

    // GELU activation
    m.def("gelu", &ops::gelu,
          py::arg("input"),
          "GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))");

    // Bias add (in-place)
    m.def("bias_add_inplace", &ops::bias_add_inplace,
          py::arg("output"), py::arg("bias"),
          "Add bias to output in-place: output[batch, features] += bias[features]");

    // LayerNorm
    m.def("layernorm", &ops::layernorm,
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f,
          "Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta");

    // Softmax
    m.def("softmax", &ops::softmax,
          py::arg("input"),
          "Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))\n"
          "Applied row-wise: input [batch, features] -> output [batch, features]");

    // RMSNorm
    m.def("rmsnorm", py::overload_cast<const GPUArray&, const GPUArray&, float>(&ops::rmsnorm),
          py::arg("input"), py::arg("gamma"), py::arg("eps") = 1e-5f,
          "RMS normalization: x / sqrt(mean(x^2) + eps) * gamma\n"
          "Simpler than LayerNorm (no mean subtraction, no beta)\n"
          "input: [batch, features], gamma: [features]");

    // RMSNorm with output buffer (for CUDA Graph capture)
    m.def("rmsnorm_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, float>(&ops::rmsnorm),
          py::arg("input"), py::arg("gamma"), py::arg("out"), py::arg("eps") = 1e-5f,
          "RMS normalization with output buffer (for CUDA Graph capture)");

    // ========================================================================
    // Fused Operations (CUTLASS Epilogue Fusion)
    // ========================================================================

    // Linear + BiasGELU (fused kernel)
    m.def("linear_bias_gelu", &ops::linear_bias_gelu,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          "Fused linear + bias + GELU: output = gelu(input @ weight^T + bias)\n"
          "Uses CUTLASS TensorCore epilogue fusion for efficiency.\n"
          "input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]");

    // ========================================================================
    // Additional Neural Network Operations
    // ========================================================================

    // SiLU (Swish) activation
    m.def("silu", py::overload_cast<const GPUArray&>(&ops::silu),
          py::arg("input"),
          "SiLU (Swish) activation: y = x * sigmoid(x)");

    // SiLU with output buffer (for CUDA Graph capture)
    m.def("silu_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::silu),
          py::arg("input"), py::arg("out"),
          "SiLU with output buffer (for CUDA Graph capture)");

    // RoPE (Rotary Position Embedding) - In-place
    m.def("rope_inplace", &ops::rope_inplace,
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          "Apply RoPE to Q and K tensors in-place.\n"
          "q: [seq_len, n_heads_q, head_dim]\n"
          "k: [seq_len, n_heads_k, head_dim]\n"
          "cos, sin: [seq_len, head_dim]");

    // Scaled Dot-Product Attention with Causal Mask
    m.def("sdpa_causal", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, float>(&ops::sdpa_causal),
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale") = 0.0f,
          "Scaled Dot-Product Attention with causal mask.\n"
          "Q: [n_heads, q_len, head_dim]\n"
          "K: [n_heads, kv_len, head_dim]\n"
          "V: [n_heads, kv_len, head_dim]\n"
          "Output: [n_heads, q_len, head_dim]\n"
          "scale: 1/sqrt(head_dim), auto-computed if <= 0");

    // SDPA with output buffer (for CUDA Graph capture)
    m.def("sdpa_causal_", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, GPUArray&, float>(&ops::sdpa_causal),
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"), py::arg("scale") = 0.0f,
          "SDPA with output buffer (for CUDA Graph capture)");

    // SDPA with fixed-length KV cache support
    m.def("sdpa_causal_fixed_cache", &ops::sdpa_causal_fixed_cache,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"),
          py::arg("context_len"), py::arg("scale") = 0.0f,
          "SDPA with fixed-length KV cache support.\n"
          "K/V are pre-allocated to max_seq_len, context_len specifies actual valid tokens.");

    // ========================================================================
    // Tensor Manipulation Operations
    // ========================================================================

    // Concat along axis 0
    m.def("concat_axis0", &ops::concat_axis0,
          py::arg("a"), py::arg("b"),
          "Concat two tensors along axis 0.\n"
          "a: [dim0_a, ...], b: [dim0_b, ...]\n"
          "Output: [dim0_a + dim0_b, ...]");

    // Repeat interleave along axis 1 (for GQA)
    m.def("repeat_interleave_axis1", &ops::repeat_interleave_axis1,
          py::arg("input"), py::arg("repeats"),
          "Repeat tensor along axis 1 (interleaved).\n"
          "input: [dim0, dim1, dim2] -> output: [dim0, dim1 * repeats, dim2]");

    // Transpose 3D: [d0, d1, d2] -> [d1, d0, d2]
    m.def("transpose_3d_021", &ops::transpose_3d_021,
          py::arg("input"),
          "Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]");

    // Reshape with copy
    m.def("reshape_copy", &ops::reshape_copy,
          py::arg("input"), py::arg("new_shape"),
          "Reshape tensor with copy (ensures contiguous output).");

    // ========================================================================
    // Fixed-Length KV Cache Operations (CUDA Graph Support)
    // ========================================================================

    m.def("kv_cache_update", &ops::kv_cache_update,
          py::arg("new_kv"), py::arg("cache"), py::arg("position"),
          "Update KV cache at a single position (decode step).\n"
          "new_kv: [1, num_kv_heads, head_dim]\n"
          "cache: [max_seq_len, num_kv_heads, head_dim]\n"
          "position: where to write in cache (0-indexed)");

    m.def("kv_cache_prefill", &ops::kv_cache_prefill,
          py::arg("new_kv"), py::arg("cache"), py::arg("start_pos"),
          "Prefill KV cache from sequence.\n"
          "new_kv: [seq_len, num_kv_heads, head_dim]\n"
          "cache: [max_seq_len, num_kv_heads, head_dim]\n"
          "start_pos: where to start writing in cache");

    // GQA-expanded KV cache operations (CUDA Graph optimization)
    m.def("kv_cache_update_gqa", &ops::kv_cache_update_gqa,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("position"),
          "Update GQA-expanded KV cache at single position.\n"
          "new_kv: [1, num_kv_heads, head_dim]\n"
          "cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)\n"
          "num_heads: total number of attention heads\n"
          "position: where to write in cache");

    m.def("kv_cache_prefill_gqa", &ops::kv_cache_prefill_gqa,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("start_pos"),
          "Prefill GQA-expanded KV cache from sequence.\n"
          "new_kv: [seq_len, num_kv_heads, head_dim]\n"
          "cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)\n"
          "num_heads: total number of attention heads\n"
          "start_pos: where to start writing in cache");

    // GPU-only embedding lookup (for CUDA Graph)
    m.def("embedding_lookup", &ops::embedding_lookup,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_id"),
          "Lookup embedding on GPU without CPU transfer.\n"
          "embed_matrix: [vocab_size, hidden_size]\n"
          "out: [1, hidden_size] pre-allocated buffer\n"
          "token_id: row index to copy");

    // In-place addition (for CUDA Graph)
    m.def("add_inplace", &ops::add_inplace,
          py::arg("a"), py::arg("b"),
          "In-place addition: a += b");

    // GPU-to-GPU copy (for CUDA Graph)
    m.def("copy_to", &ops::copy_to,
          py::arg("src"), py::arg("dst"),
          "Copy src to dst on GPU");

    // ========================================================================
    // Quantization Operations (#85)
    // ========================================================================

    // Dequantize INT8 to FP16/FP32
    m.def("dequantize_int8", &ops::dequantize_int8,
          py::arg("input"), py::arg("scale"), py::arg("output_dtype"),
          "Dequantize INT8 tensor to FP16/FP32.\n"
          "output = input_int8 * scale\n"
          "input: [rows, cols] INT8, scale: [cols], output_dtype: Float16 or Float32");

    // Quantized Linear (INT8 weight x FP16 activation)
    m.def("linear_int8", [](const GPUArray& activation, const GPUArray& weight_int8,
                            const GPUArray& scale, const GPUArray* bias) {
              return ops::linear_int8(activation, weight_int8, scale, bias);
          },
          py::arg("activation"), py::arg("weight_int8"), py::arg("scale"),
          py::arg("bias") = nullptr,
          "Quantized Linear layer with INT8 weights.\n"
          "output = activation @ (weight_int8 * scale).T\n"
          "activation: [M, K] FP16, weight_int8: [N, K] INT8, scale: [N] FP16\n"
          "Dequantization happens on-the-fly (memory efficient).");

    // Quantize to INT8
    m.def("quantize_to_int8", &ops::quantize_to_int8,
          py::arg("input"),
          "Quantize FP16/FP32 tensor to INT8 with per-column scaling.\n"
          "Returns (weight_int8, scale) tuple.\n"
          "weight_int8: [rows, cols] INT8, scale: [cols] same dtype as input");

    // ========================================================================
    // Paged Attention Operations (#87)
    // ========================================================================

    m.def("paged_attention_v1", &ops::paged_attention_v1,
          py::arg("Q"), py::arg("K_cache"), py::arg("V_cache"),
          py::arg("block_tables"), py::arg("context_lens"),
          py::arg("scale") = 0.0f,
          "Paged Attention v1: single-query attention with paged KV cache.\n"
          "Q: [num_seqs, num_heads, head_dim]\n"
          "K_cache, V_cache: [num_blocks, num_kv_heads, block_size, head_dim]\n"
          "block_tables: [num_seqs, max_num_blocks_per_seq] int32\n"
          "context_lens: [num_seqs] int32\n"
          "Output: [num_seqs, num_heads, head_dim]");

    m.def("copy_to_paged_cache", &ops::copy_to_paged_cache,
          py::arg("K_new"), py::arg("V_new"),
          py::arg("K_cache"), py::arg("V_cache"),
          py::arg("slot_mapping"),
          "Copy new KV entries to paged cache (decode phase).\n"
          "K_new, V_new: [num_seqs, num_kv_heads, head_dim]\n"
          "slot_mapping: [num_seqs] int32 - physical slot indices");

    m.def("reshape_and_cache", &ops::reshape_and_cache,
          py::arg("K"), py::arg("V"),
          py::arg("K_cache"), py::arg("V_cache"),
          py::arg("slot_mapping"),
          "Reshape and copy KV from prefill format to paged cache.\n"
          "K, V: [total_tokens, num_kv_heads, head_dim]\n"
          "slot_mapping: [total_tokens] int32");

    m.def("allocate_kv_cache", &ops::allocate_kv_cache,
          py::arg("num_blocks"), py::arg("num_kv_heads"),
          py::arg("block_size"), py::arg("head_dim"),
          "Allocate KV cache blocks.\n"
          "Returns: [num_blocks, num_kv_heads, block_size, head_dim] FP16");

    // ========================================================================
    // Continuous Batching Operations (#86)
    // ========================================================================

    m.def("gather_embeddings", &ops::gather_embeddings,
          py::arg("token_ids"), py::arg("embeddings"), py::arg("total_tokens"),
          "Gather token embeddings for a batch.\n"
          "token_ids: [total_tokens] int32\n"
          "embeddings: [vocab_size, hidden_size] FP16\n"
          "Returns: [total_tokens, hidden_size] FP16");

    m.def("scatter_last_token_logits", &ops::scatter_last_token_logits,
          py::arg("logits"), py::arg("seq_start_positions"),
          py::arg("seq_lens"), py::arg("batch_size"), py::arg("vocab_size"),
          "Scatter last-token logits from batch output.\n"
          "logits: [batch_tokens, vocab_size] FP16\n"
          "Returns: [batch_size, vocab_size] FP16");

    m.def("prepare_position_ids", &ops::prepare_position_ids,
          py::arg("seq_start_positions"), py::arg("seq_context_lens"),
          py::arg("is_prefill"), py::arg("input_lens"),
          py::arg("batch_size"), py::arg("total_tokens"),
          "Prepare position IDs for rotary embeddings.\n"
          "Returns: [total_tokens] int32");

    m.def("argmax_sample", &ops::argmax_sample,
          py::arg("logits"), py::arg("batch_size"), py::arg("vocab_size"),
          "Argmax sampling from logits.\n"
          "logits: [batch_size, vocab_size] FP16\n"
          "Returns: [batch_size] int32 - sampled token IDs");

    m.def("check_eos", &ops::check_eos,
          py::arg("tokens"), py::arg("eos_token_id"),
          "Check for EOS tokens.\n"
          "tokens: [batch_size] int32\n"
          "Returns: [batch_size] int32 - 1 if EOS, 0 otherwise");

    m.def("compute_cumsum", &ops::compute_cumsum,
          py::arg("input"),
          "Compute exclusive prefix sum.\n"
          "input: [n] int32\n"
          "Returns: [n] int32");

    m.def("prepare_batch_inputs", &ops::prepare_batch_inputs,
          py::arg("token_lists"),
          "Prepare batch inputs from Python lists.\n"
          "token_lists: List of token ID lists\n"
          "Returns: (token_ids GPUArray, total_tokens count)");
}
