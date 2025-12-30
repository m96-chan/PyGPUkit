/**
 * NN normalization operations: layernorm, rmsnorm, bias_add_inplace
 */
#include "../bindings_common.hpp"

void init_nn_norm(py::module_& m) {
    // Bias add (in-place)
    m.def("bias_add_inplace", &ops::bias_add_inplace,
          py::arg("output"), py::arg("bias"),
          "Add bias to output in-place: output[batch, features] += bias[features]");

    // LayerNorm
    m.def("layernorm", &ops::layernorm,
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f,
          "Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta");

    // RMSNorm
    m.def("rmsnorm", py::overload_cast<const GPUArray&, const GPUArray&, float>(&ops::rmsnorm),
          py::arg("input"), py::arg("gamma"), py::arg("eps") = 1e-5f,
          "RMS normalization: x / sqrt(mean(x^2) + eps) * gamma\n"
          "Simpler than LayerNorm (no mean subtraction, no beta)\n"
          "input: [batch, features], gamma: [features]");

    m.def("rmsnorm_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, float>(&ops::rmsnorm),
          py::arg("input"), py::arg("gamma"), py::arg("out"), py::arg("eps") = 1e-5f,
          "RMS normalization with output buffer (for CUDA Graph capture)");
}
