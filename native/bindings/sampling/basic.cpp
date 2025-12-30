/**
 * Basic sampling operations: greedy, multinomial, topp
 */
#include "../bindings_common.hpp"

void init_sampling_basic(py::module_& m) {
    m.def("sample_greedy", &ops::sample_greedy,
          py::arg("logits"),
          "Greedy sampling (argmax) from logits.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "Returns: sampled token ID (int)");

    m.def("sample_multinomial", &ops::sample_multinomial,
          py::arg("logits"), py::arg("temperature"),
          "Multinomial sampling with temperature.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "temperature: > 0 (lower = more deterministic)\n"
          "Returns: sampled token ID (int)");

    m.def("sample_topp", &ops::sample_topp,
          py::arg("logits"), py::arg("top_p"), py::arg("temperature"),
          "Top-P (nucleus) sampling.\n"
          "logits: [vocab_size] or [1, vocab_size]\n"
          "top_p: cumulative probability threshold (0 < p <= 1)\n"
          "temperature: > 0\n"
          "Returns: sampled token ID (int)");

    m.def("sample_token_gpu", &ops::sample_token_gpu,
          py::arg("logits"),
          py::arg("temperature") = 1.0f,
          py::arg("top_k") = 0,
          py::arg("top_p") = 1.0f,
          "Unified GPU sampling API.\n"
          "Automatically selects sampling method:\n"
          "- temperature=0: greedy (argmax)\n"
          "- top_k > 0: top-k sampling\n"
          "- top_p < 1: top-p sampling\n"
          "- otherwise: multinomial with temperature\n"
          "Returns: sampled token ID (int)");
}
