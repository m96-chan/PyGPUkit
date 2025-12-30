/**
 * Argmax/argmin reduction operations
 */
#include "../bindings_common.hpp"

void init_reduction_argmax(py::module_& m) {
    m.def("argmax", &ops::argmax,
          py::arg("a"),
          "Index of maximum element, returns int64 GPUArray");
}
