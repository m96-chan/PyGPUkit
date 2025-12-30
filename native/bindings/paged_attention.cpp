/**
 * Paged Attention operations for continuous batching
 */
#include "bindings_common.hpp"

void init_paged_attention(py::module_& m) {
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
}
