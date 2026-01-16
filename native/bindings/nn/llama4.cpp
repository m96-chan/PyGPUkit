/**
 * Llama4 architecture specific operations
 * - L2 norm (QK normalization)
 * - iRoPE temperature scaling
 */
#include "../bindings_common.hpp"

void init_nn_llama4(py::module_& m) {
    // L2 norm
    m.def("l2norm", py::overload_cast<const GPUArray&, float>(&ops::l2norm),
          py::arg("input"), py::arg("eps") = 1e-6f,
          "L2 normalization (Llama4TextL2Norm): x * rsqrt(mean(x^2) + eps)\n"
          "Used for QK normalization in Llama 4 attention.\n"
          "Unlike RMSNorm, no gamma scaling is applied.");

    m.def("l2norm_", py::overload_cast<const GPUArray&, GPUArray&, float>(&ops::l2norm),
          py::arg("input"), py::arg("out"), py::arg("eps") = 1e-6f,
          "L2 normalization with output buffer (for CUDA Graph capture)");

    // iRoPE Q scaling
    m.def("irope_scale_q", &ops::irope_scale_q,
          py::arg("Q"), py::arg("positions"),
          py::arg("attn_scale") = 0.1f, py::arg("floor_scale") = 8192.0f,
          "Apply iRoPE temperature scaling to Q tensor.\n"
          "Formula: scale = log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1.0\n"
          "Q: [seq_len, num_heads, head_dim], positions: [seq_len]");

    // SDPA with iRoPE
    m.def("sdpa_irope", &ops::sdpa_irope,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("positions"),
          py::arg("attn_scale") = 0.1f, py::arg("floor_scale") = 8192.0f,
          py::arg("causal_offset") = 0,
          "Scaled dot-product attention with iRoPE temperature scaling.\n"
          "Fuses temperature scaling into attention computation.\n"
          "Q: [n_heads, q_len, head_dim], K/V: [n_kv_heads, kv_len, head_dim]");
}
