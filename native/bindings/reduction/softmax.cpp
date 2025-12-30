/**
 * Softmax reduction operation
 */
#include "../bindings_common.hpp"

void init_reduction_softmax(py::module_& m) {
    m.def("softmax", &ops::softmax,
          py::arg("input"),
          "Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))\n"
          "Applied row-wise: input [batch, features] -> output [batch, features]");
}
