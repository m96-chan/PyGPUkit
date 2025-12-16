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
    m.def("rmsnorm", &ops::rmsnorm,
          py::arg("input"), py::arg("gamma"), py::arg("eps") = 1e-5f,
          "RMS normalization: x / sqrt(mean(x^2) + eps) * gamma\n"
          "Simpler than LayerNorm (no mean subtraction, no beta)\n"
          "input: [batch, features], gamma: [features]");

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
    m.def("silu", &ops::silu,
          py::arg("input"),
          "SiLU (Swish) activation: y = x * sigmoid(x)");

    // RoPE (Rotary Position Embedding) - In-place
    m.def("rope_inplace", &ops::rope_inplace,
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          "Apply RoPE to Q and K tensors in-place.\n"
          "q: [seq_len, n_heads_q, head_dim]\n"
          "k: [seq_len, n_heads_k, head_dim]\n"
          "cos, sin: [seq_len, head_dim]");

    // Scaled Dot-Product Attention with Causal Mask
    m.def("sdpa_causal", &ops::sdpa_causal,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale") = 0.0f,
          "Scaled Dot-Product Attention with causal mask.\n"
          "Q: [n_heads, q_len, head_dim]\n"
          "K: [n_heads, kv_len, head_dim]\n"
          "V: [n_heads, kv_len, head_dim]\n"
          "Output: [n_heads, q_len, head_dim]\n"
          "scale: 1/sqrt(head_dim), auto-computed if <= 0");

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
}
