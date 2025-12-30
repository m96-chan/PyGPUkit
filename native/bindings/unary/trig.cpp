/**
 * Unary trigonometric operations: sin, cos
 */
#include "../bindings_common.hpp"

void init_unary_trig(py::module_& m) {
    // Sin
    m.def("sin", py::overload_cast<const GPUArray&>(&ops::sin),
          py::arg("a"),
          "Element-wise sine");

    m.def("sin_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::sin),
          py::arg("a"), py::arg("out"),
          "Element-wise sine with output array");

    // Cos
    m.def("cos", py::overload_cast<const GPUArray&>(&ops::cos),
          py::arg("a"),
          "Element-wise cosine");

    m.def("cos_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::cos),
          py::arg("a"), py::arg("out"),
          "Element-wise cosine with output array");
}
