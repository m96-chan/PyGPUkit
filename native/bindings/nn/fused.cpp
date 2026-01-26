/**
 * NN fused operations: rmsnorm_residual, swiglu, geglu
 */
#include "../bindings_common.hpp"

void init_nn_fused(py::module_& m) {
    // Fused RMSNorm + Residual
    m.def("rmsnorm_residual",
          py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, float>(
              &ops::rmsnorm_residual),
          py::arg("input"), py::arg("residual"), py::arg("gamma"),
          py::arg("eps") = 1e-5f,
          "Fused RMSNorm + Residual: y = rmsnorm(x + residual) * gamma\n"
          "input: [batch, features], residual: [batch, features], gamma: [features]\n"
          "Fuses residual addition and RMSNorm into a single kernel.");

    m.def("rmsnorm_residual_",
          py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, GPUArray&, float>(
              &ops::rmsnorm_residual),
          py::arg("input"), py::arg("residual"), py::arg("gamma"), py::arg("out"),
          py::arg("eps") = 1e-5f,
          "Fused RMSNorm + Residual with output buffer (for CUDA Graph capture)");

    // Fused SwiGLU
    m.def("swiglu",
          py::overload_cast<const GPUArray&, const GPUArray&>(&ops::swiglu),
          py::arg("gate_proj"), py::arg("up_proj"),
          "Fused SwiGLU: y = silu(gate_proj) * up_proj\n"
          "Used in Qwen, LLaMA3, Mistral FFN layers.\n"
          "Fuses SiLU activation and element-wise multiply into one kernel.");

    m.def("swiglu_",
          py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::swiglu),
          py::arg("gate_proj"), py::arg("up_proj"), py::arg("out"),
          "Fused SwiGLU with output buffer (for CUDA Graph capture)");

    // Fused GeGLU (GELU variant)
    m.def("geglu",
          py::overload_cast<const GPUArray&, const GPUArray&>(&ops::geglu),
          py::arg("gate_proj"), py::arg("up_proj"),
          "Fused GeGLU: y = gelu(gate_proj) * up_proj\n"
          "GELU variant of gated linear unit, used in some transformer architectures.");

    m.def("geglu_",
          py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::geglu),
          py::arg("gate_proj"), py::arg("up_proj"), py::arg("out"),
          "Fused GeGLU with output buffer (for CUDA Graph capture)");
}
