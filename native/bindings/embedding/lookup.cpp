/**
 * Embedding lookup operations
 */
#include "../bindings_common.hpp"

void init_embedding_lookup(py::module_& m) {
    // GPU-only embedding lookup (for CUDA Graph)
    m.def("embedding_lookup", &ops::embedding_lookup,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_id"),
          "Lookup embedding on GPU without CPU transfer.\n"
          "embed_matrix: [vocab_size, hidden_size]\n"
          "out: [1, hidden_size] pre-allocated buffer\n"
          "token_id: row index to copy");

    m.def("embedding_lookup_ptr", &ops::embedding_lookup_ptr,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_id_buf"),
          "Lookup embedding reading index from GPU buffer.\n"
          "token_id_buf: GPUArray[1] int32 containing token/position value");

    m.def("embedding_lookup_batch", &ops::embedding_lookup_batch,
          py::arg("embed_matrix"), py::arg("out"), py::arg("token_ids_buf"),
          py::arg("batch_size"),
          "Batch embedding lookup from GPU token ID array.\n"
          "Looks up multiple rows: out[i, :] = embed_matrix[token_ids[i], :]");

    m.def("slice_rows_range_ptr", &ops::slice_rows_range_ptr,
          py::arg("table"), py::arg("out"), py::arg("start_pos_buf"),
          py::arg("count"),
          "Slice consecutive rows from table using GPU-stored start position.\n"
          "Copies `count` rows: out[i, :] = table[start_pos + i, :]");
}
