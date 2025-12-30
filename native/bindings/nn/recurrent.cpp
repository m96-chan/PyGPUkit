/**
 * NN recurrent layers: LSTM
 */
#include "../bindings_common.hpp"

void init_nn_recurrent(py::module_& m) {
    // LSTM forward (unidirectional)
    m.def("lstm_forward", &ops::lstm_forward,
          py::arg("x"),
          py::arg("W_ih"), py::arg("W_hh"),
          py::arg("b_ih"), py::arg("b_hh"),
          py::arg("h0"), py::arg("c0"),
          py::arg("reverse") = false,
          "LSTM forward pass (unidirectional).\n\n"
          "Args:\n"
          "    x: input [batch, seq_len, input_size]\n"
          "    W_ih: input-to-hidden weights [4*hidden_size, input_size]\n"
          "    W_hh: hidden-to-hidden weights [4*hidden_size, hidden_size]\n"
          "    b_ih: input bias [4*hidden_size]\n"
          "    b_hh: hidden bias [4*hidden_size]\n"
          "    h0: initial hidden state [batch, hidden_size] or empty\n"
          "    c0: initial cell state [batch, hidden_size] or empty\n"
          "    reverse: process sequence in reverse order\n\n"
          "Returns:\n"
          "    tuple of (output, h_n, c_n)\n"
          "    output: [batch, seq_len, hidden_size]\n"
          "    h_n: [batch, hidden_size]\n"
          "    c_n: [batch, hidden_size]");

    // LSTM bidirectional
    m.def("lstm_bidirectional", &ops::lstm_bidirectional,
          py::arg("x"),
          py::arg("W_ih_fwd"), py::arg("W_hh_fwd"),
          py::arg("b_ih_fwd"), py::arg("b_hh_fwd"),
          py::arg("W_ih_bwd"), py::arg("W_hh_bwd"),
          py::arg("b_ih_bwd"), py::arg("b_hh_bwd"),
          "Bidirectional LSTM.\n\n"
          "Args:\n"
          "    x: input [batch, seq_len, input_size]\n"
          "    W_ih_fwd, W_hh_fwd, b_ih_fwd, b_hh_fwd: forward LSTM weights\n"
          "    W_ih_bwd, W_hh_bwd, b_ih_bwd, b_hh_bwd: backward LSTM weights\n\n"
          "Returns:\n"
          "    tuple of (output, h_n, c_n)\n"
          "    output: [batch, seq_len, 2*hidden_size] (concatenated fwd/bwd)\n"
          "    h_n: [2, batch, hidden_size]\n"
          "    c_n: [2, batch, hidden_size]");
}
