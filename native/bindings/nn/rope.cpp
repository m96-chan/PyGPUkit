/**
 * RoPE (Rotary Position Embedding) operations
 */
#include "../bindings_common.hpp"

void init_nn_rope(py::module_& m) {
    // RoPE (Rotary Position Embedding) - In-place
    m.def("rope_inplace", &ops::rope_inplace,
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          "Apply RoPE to Q and K tensors in-place.\n"
          "q: [seq_len, n_heads_q, head_dim]\n"
          "k: [seq_len, n_heads_k, head_dim]\n"
          "cos, sin: [seq_len, head_dim]");

    // RoPE with FP32 cos/sin tables (higher precision for bf16/f16)
    m.def("rope_inplace_f32table", &ops::rope_inplace_f32table,
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          "Apply RoPE with FP32 cos/sin tables (higher precision).\n"
          "q: [seq_len, n_heads_q, head_dim] (bf16 or f16)\n"
          "k: [seq_len, n_heads_k, head_dim] (bf16 or f16)\n"
          "cos, sin: [seq_len, head_dim] (f32)");
}
