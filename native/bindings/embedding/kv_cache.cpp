/**
 * KV cache operations for LLM inference
 */
#include "../bindings_common.hpp"

void init_embedding_kv_cache(py::module_& m) {
    m.def("kv_cache_update", &ops::kv_cache_update,
          py::arg("new_kv"), py::arg("cache"), py::arg("position"),
          "Update KV cache at a single position (decode step).\n"
          "new_kv: [1, num_kv_heads, head_dim]\n"
          "cache: [max_seq_len, num_kv_heads, head_dim]\n"
          "position: where to write in cache (0-indexed)");

    m.def("kv_cache_prefill", &ops::kv_cache_prefill,
          py::arg("new_kv"), py::arg("cache"), py::arg("start_pos"),
          "Prefill KV cache from sequence.\n"
          "new_kv: [seq_len, num_kv_heads, head_dim]\n"
          "cache: [max_seq_len, num_kv_heads, head_dim]\n"
          "start_pos: where to start writing in cache");

    // GQA-expanded KV cache operations (CUDA Graph optimization)
    m.def("kv_cache_update_gqa", &ops::kv_cache_update_gqa,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("position"),
          "Update GQA-expanded KV cache at single position.\n"
          "new_kv: [1, num_kv_heads, head_dim]\n"
          "cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)\n"
          "num_heads: total number of attention heads\n"
          "position: where to write in cache");

    m.def("kv_cache_prefill_gqa", &ops::kv_cache_prefill_gqa,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("start_pos"),
          "Prefill GQA-expanded KV cache from sequence.\n"
          "new_kv: [seq_len, num_kv_heads, head_dim]\n"
          "cache: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)\n"
          "num_heads: total number of attention heads\n"
          "start_pos: where to start writing in cache");

    // GPU position pointer variants (for CUDA Graph replay without recapture)
    m.def("kv_cache_update_gqa_ptr", &ops::kv_cache_update_gqa_ptr,
          py::arg("new_kv"), py::arg("cache"), py::arg("num_heads"), py::arg("position_buf"),
          "Update GQA-expanded KV cache reading position from GPU buffer.\n"
          "position_buf: GPUArray[1] int32 containing position value");
}
