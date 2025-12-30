/**
 * Top-K sampling operations (CUDA Graph compatible)
 */
#include "../bindings_common.hpp"

void init_sampling_topk(py::module_& m) {
    m.def("sample_topk", &ops::sample_topk,
          py::arg("logits"), py::arg("top_k"), py::arg("temperature"),
          "Top-K sampling.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "top_k: number of top tokens to consider\n"
          "temperature: > 0\n"
          "Returns: sampled token ID (int)");

    m.def("sample_topk_to_buf", &ops::sample_topk_to_buf,
          py::arg("logits"), py::arg("result_buf"), py::arg("top_k"),
          py::arg("temperature"), py::arg("random_val"),
          "Top-K sampling (CUDA Graph compatible).\n"
          "Writes result to pre-allocated buffer, no sync/D2H.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "result_buf: pre-allocated int32 buffer [1]\n"
          "top_k: number of top tokens to consider\n"
          "temperature: > 0\n"
          "random_val: pre-generated random value [0, 1)");

    m.def("sample_topk_to_buf_ptr", &ops::sample_topk_to_buf_ptr,
          py::arg("logits"), py::arg("result_buf"), py::arg("random_val_buf"),
          py::arg("top_k"), py::arg("temperature"),
          "Top-K sampling with pointer (CUDA Graph replay compatible).\n"
          "random_val is read from GPU buffer, allowing update before replay.\n"
          "logits: [vocab_size] or [1, vocab_size] (float16 only)\n"
          "result_buf: pre-allocated int32 buffer [1]\n"
          "random_val_buf: pre-allocated float32 buffer [1]\n"
          "top_k: number of top tokens to consider\n"
          "temperature: > 0");
}
