/**
 * Comparison and conditional operations: clamp, where
 */
#include "../bindings_common.hpp"

void init_elementwise_compare(py::module_& m) {
    // Clamp
    m.def("clamp", py::overload_cast<const GPUArray&, float, float>(&ops::clamp),
          py::arg("a"), py::arg("min_val"), py::arg("max_val"),
          "Element-wise clamp: clamp(x, min, max)");

    m.def("clamp_", py::overload_cast<const GPUArray&, GPUArray&, float, float>(&ops::clamp),
          py::arg("a"), py::arg("out"), py::arg("min_val"), py::arg("max_val"),
          "Element-wise clamp with output array");

    // Where (conditional select)
    m.def("where", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&>(&ops::where),
          py::arg("cond"), py::arg("a"), py::arg("b"),
          "Conditional select: where(cond, a, b) = cond ? a : b");

    m.def("where_", py::overload_cast<const GPUArray&, const GPUArray&, const GPUArray&, GPUArray&>(&ops::where),
          py::arg("cond"), py::arg("a"), py::arg("b"), py::arg("out"),
          "Conditional select with output array");
}
