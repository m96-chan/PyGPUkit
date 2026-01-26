/**
 * Conv1d pybind11 bindings
 * native/bindings/nn/conv.cpp
 */
#include "../bindings_common.hpp"

void init_nn_conv(py::module_& m) {
    // Conv1d without bias
    m.def("conv1d", &ops::conv1d_no_bias,
          py::arg("input"),
          py::arg("weight"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          R"pbdoc(
1D convolution without bias.

Args:
    input: Input tensor [batch, in_channels, length]
    weight: Weight tensor [out_channels, in_channels, kernel_size]
    stride: Convolution stride (default: 1)
    padding: Input padding (default: 0)

Returns:
    Output tensor [batch, out_channels, out_length]
)pbdoc");

    // Conv1d with bias
    m.def("conv1d_bias", &ops::conv1d_with_bias,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          R"pbdoc(
1D convolution with bias.

Args:
    input: Input tensor [batch, in_channels, length]
    weight: Weight tensor [out_channels, in_channels, kernel_size]
    bias: Bias tensor [out_channels]
    stride: Convolution stride (default: 1)
    padding: Input padding (default: 0)

Returns:
    Output tensor [batch, out_channels, out_length]
)pbdoc");
}
