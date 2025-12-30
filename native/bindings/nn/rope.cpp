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

    // NTK-aware RoPE initialization
    m.def("rope_init_ntk_aware", &ops::rope_init_ntk_aware,
          py::arg("max_seq_len"), py::arg("head_dim"),
          py::arg("base") = 10000.0f, py::arg("scale") = 1.0f,
          "Initialize RoPE with NTK-aware frequency scaling.\n"
          "Scales base frequency for context extension: base' = base * scale^(dim/(dim-2))\n"
          "Returns: tuple of (cos_table, sin_table) each [max_seq_len, head_dim]");

    // YaRN RoPE initialization
    m.def("rope_init_yarn", &ops::rope_init_yarn,
          py::arg("max_seq_len"), py::arg("head_dim"),
          py::arg("base") = 10000.0f, py::arg("scale") = 1.0f,
          py::arg("original_max_len") = 4096, py::arg("beta_fast") = 32.0f,
          py::arg("beta_slow") = 1.0f, py::arg("mscale") = 0.1f,
          "Initialize RoPE with YaRN dimension-wise interpolation.\n"
          "Different scaling for different frequency bands (low/mid/high).\n"
          "Returns: tuple of (cos_table, sin_table) each [max_seq_len, head_dim]");

    // Linear position interpolation
    m.def("rope_init_linear", &ops::rope_init_linear,
          py::arg("max_seq_len"), py::arg("head_dim"),
          py::arg("base") = 10000.0f, py::arg("scale") = 1.0f,
          "Initialize RoPE with linear position interpolation.\n"
          "Simple baseline: pos' = pos / scale. Degrades at high scales.\n"
          "Returns: tuple of (cos_table, sin_table) each [max_seq_len, head_dim]");

    // PoPE (Positional Encoding) - Alternative to RoPE
    m.def("pope_init_encoding", &ops::pope_init_encoding,
          py::arg("max_seq_len"), py::arg("head_dim"), py::arg("base") = 10000.0f,
          "Initialize sinusoidal positional encoding table.\n"
          "Returns: encoding tensor [max_seq_len, head_dim]");

    m.def("pope_inplace", &ops::pope_inplace,
          py::arg("q"), py::arg("k"), py::arg("encoding"), py::arg("start_pos") = 0,
          "Apply additive positional encoding to Q and K in-place.\n"
          "q: [seq_len, n_heads_q, head_dim]\n"
          "k: [seq_len, n_heads_k, head_dim]\n"
          "encoding: [max_seq_len, head_dim] (f32)");

    // ALiBi (Attention with Linear Biases)
    m.def("alibi_init_slopes", &ops::alibi_init_slopes,
          py::arg("num_heads"),
          "Initialize ALiBi head-specific slopes.\n"
          "m_h = 2^(-8 * h / num_heads)\n"
          "Returns: slopes tensor [num_heads]");

    m.def("alibi_compute_bias", &ops::alibi_compute_bias,
          py::arg("seq_len"), py::arg("num_heads"), py::arg("slopes"),
          py::arg("causal") = true,
          "Compute ALiBi bias matrix for attention.\n"
          "Returns: bias tensor [num_heads, seq_len, seq_len]");

    m.def("alibi_add_bias", &ops::alibi_add_bias,
          py::arg("scores"), py::arg("slopes"), py::arg("start_pos") = 0,
          "Add ALiBi bias to attention scores in-place.\n"
          "scores: [batch, num_heads, q_len, kv_len]\n"
          "slopes: [num_heads]");
}
