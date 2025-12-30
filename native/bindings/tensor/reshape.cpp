/**
 * Reshape operations
 */
#include "../bindings_common.hpp"

void init_tensor_reshape(py::module_& m) {
    m.def("reshape_copy", py::overload_cast<const GPUArray&, const std::vector<size_t>&>(&ops::reshape_copy),
          py::arg("input"), py::arg("new_shape"),
          "Reshape tensor with copy (ensures contiguous output).");

    m.def("reshape_copy_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::reshape_copy),
          py::arg("input"), py::arg("out"),
          "Reshape with copy into output buffer (for CUDA Graph capture).");
}
