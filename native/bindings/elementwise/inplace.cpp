/**
 * In-place element-wise operations: add_inplace, mul_inplace, copy_to
 */
#include "../bindings_common.hpp"

void init_elementwise_inplace(py::module_& m) {
    // In-place addition (for CUDA Graph)
    m.def("add_inplace", &ops::add_inplace,
          py::arg("a"), py::arg("b"),
          "In-place addition: a += b");

    // In-place multiplication (for CUDA Graph)
    m.def("mul_inplace", &ops::mul_inplace,
          py::arg("a"), py::arg("b"),
          "In-place multiplication: a *= b");

    // GPU-to-GPU copy (for CUDA Graph)
    m.def("copy_to", &ops::copy_to,
          py::arg("src"), py::arg("dst"),
          "Copy src to dst on GPU");
}
