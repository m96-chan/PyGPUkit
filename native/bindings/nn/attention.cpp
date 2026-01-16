/**
 * NN attention operations: SDPA, split_qkv
 */
#include "../bindings_common.hpp"

void init_nn_attention(py::module_& m) {
    // Split fused QKV projection output into separate Q, K, V tensors
    m.def("split_qkv_batch", &ops::split_qkv_batch,
          py::arg("qkv"), py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
          "Split fused QKV projection [seq_len, q_dim+k_dim+v_dim] into Q, K, V.\n"
          "Output buffers must be pre-allocated for CUDA Graph compatibility.");

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

    m.def("sdpa_causal_fixed_cache_ptr", &ops::sdpa_causal_fixed_cache_ptr,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"),
          py::arg("context_len_buf"), py::arg("max_kv_len"), py::arg("scale") = 0.0f,
          "SDPA with pointer-based context_len for CUDA Graph support.\n"
          "context_len_buf: GPU int32 buffer containing actual context_len.\n"
          "max_kv_len: Max context length (for shared memory allocation at graph capture).");

    // Timed SDPA for benchmarking (kernel-only time via cudaEvent)
    m.def("sdpa_causal_timed", [](const GPUArray& Q, const GPUArray& K, const GPUArray& V,
                                   GPUArray& out, float scale) -> float {
        float kernel_time_us = 0.0f;
        ops::sdpa_causal_timed(Q, K, V, out, scale, &kernel_time_us);
        return kernel_time_us;
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("out"), py::arg("scale") = 0.0f,
    "SDPA with kernel-only timing (for benchmarking).\n"
    "Returns kernel execution time in microseconds (excludes host overhead).\n"
    "Only supports BFloat16, requires SM90+ (TMA).");

    // TMA cache utilities
    m.def("print_tma_cache_stats", &ops::print_tma_cache_stats,
          "Print TMA descriptor cache statistics (hits, misses, size).");
    m.def("clear_tma_cache", &ops::clear_tma_cache,
          "Clear all cached TMA descriptors.");
}
