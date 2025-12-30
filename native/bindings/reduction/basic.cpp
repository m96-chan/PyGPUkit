/**
 * Basic reduction operations: sum, mean, max, min, sum_axis
 */
#include "../bindings_common.hpp"

void init_reduction_basic(py::module_& m) {
    m.def("sum", &ops::sum,
          py::arg("a"),
          "Sum of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("mean", &ops::mean,
          py::arg("a"),
          "Mean of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("max", &ops::max,
          py::arg("a"),
          "Max of all elements (float32/float64 only), returns scalar GPUArray");

    m.def("min", &ops::min,
          py::arg("a"),
          "Min of all elements, returns scalar GPUArray");

    m.def("sum_axis", &ops::sum_axis,
          py::arg("a"), py::arg("axis"),
          "Sum along specified axis (0 or 1) for 2D tensors.\n"
          "axis=0: sum rows -> [N], axis=1: sum columns -> [M]");
}
