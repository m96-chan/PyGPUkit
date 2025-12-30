/**
 * NN activation functions: gelu, silu, sigmoid, tanh, linear_bias_gelu
 */
#include "../bindings_common.hpp"

void init_nn_activation(py::module_& m) {
    // GELU activation
    m.def("gelu", &ops::gelu,
          py::arg("input"),
          "GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))");

    // SiLU (Swish) activation
    m.def("silu", py::overload_cast<const GPUArray&>(&ops::silu),
          py::arg("input"),
          "SiLU (Swish) activation: y = x * sigmoid(x)");

    m.def("silu_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::silu),
          py::arg("input"), py::arg("out"),
          "SiLU with output buffer (for CUDA Graph capture)");

    // Sigmoid activation
    m.def("sigmoid", py::overload_cast<const GPUArray&>(&ops::sigmoid),
          py::arg("input"),
          "Sigmoid activation: y = 1 / (1 + exp(-x))");

    m.def("sigmoid_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::sigmoid),
          py::arg("input"), py::arg("out"),
          "Sigmoid with output buffer (for CUDA Graph capture)");

    // Tanh activation
    m.def("tanh", py::overload_cast<const GPUArray&>(&ops::tanh),
          py::arg("input"),
          "Tanh activation");

    m.def("tanh_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::tanh),
          py::arg("input"), py::arg("out"),
          "Tanh with output buffer (for CUDA Graph capture)");

    // Fused Linear + BiasGELU (CUTLASS epilogue fusion)
    m.def("linear_bias_gelu", &ops::linear_bias_gelu,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          "Fused linear + bias + GELU: output = gelu(input @ weight^T + bias)\n"
          "Uses CUTLASS TensorCore epilogue fusion for efficiency.\n"
          "input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]");
}
