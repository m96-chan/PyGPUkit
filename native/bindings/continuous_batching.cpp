/**
 * Continuous Batching operations for LLM inference
 */
#include "bindings_common.hpp"

void init_continuous_batching(py::module_& m) {
    m.def("gather_embeddings", &ops::gather_embeddings,
          py::arg("token_ids"), py::arg("embeddings"), py::arg("total_tokens"),
          "Gather embeddings for token IDs.\n"
          "token_ids: [total_tokens] int32\n"
          "embeddings: [vocab_size, hidden_size] FP16\n"
          "Returns: [total_tokens, hidden_size] FP16");

    m.def("scatter_last_token_logits", &ops::scatter_last_token_logits,
          py::arg("logits"), py::arg("seq_start_positions"),
          py::arg("seq_lens"), py::arg("batch_size"), py::arg("vocab_size"),
          "Scatter last-token logits from batch output.\n"
          "logits: [batch_tokens, vocab_size] FP16\n"
          "Returns: [batch_size, vocab_size] FP16");

    m.def("prepare_position_ids", &ops::prepare_position_ids,
          py::arg("seq_start_positions"), py::arg("seq_context_lens"),
          py::arg("is_prefill"), py::arg("input_lens"),
          py::arg("batch_size"), py::arg("total_tokens"),
          "Prepare position IDs for rotary embeddings.\n"
          "Returns: [total_tokens] int32");

    m.def("argmax_sample", &ops::argmax_sample,
          py::arg("logits"), py::arg("batch_size"), py::arg("vocab_size"),
          "Argmax sampling from logits.\n"
          "logits: [batch_size, vocab_size] FP16\n"
          "Returns: [batch_size] int32 - sampled token IDs");

    m.def("check_eos", &ops::check_eos,
          py::arg("tokens"), py::arg("eos_token_id"),
          "Check for EOS tokens.\n"
          "tokens: [batch_size] int32\n"
          "Returns: [batch_size] int32 - 1 if EOS, 0 otherwise");

    m.def("compute_cumsum", &ops::compute_cumsum,
          py::arg("input"),
          "Compute exclusive prefix sum.\n"
          "input: [n] int32\n"
          "Returns: [n] int32");
}
