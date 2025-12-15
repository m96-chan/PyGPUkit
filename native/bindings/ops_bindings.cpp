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
}
