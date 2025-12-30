/**
 * Transpose operations: 2D, 3D, 4D
 */
#include "../bindings_common.hpp"

void init_tensor_transpose(py::module_& m) {
    // 2D transpose
    m.def("transpose", &ops::transpose,
          py::arg("input"),
          "Matrix transpose: input [rows, cols] -> output [cols, rows]");

    // 3D transpose: [d0, d1, d2] -> [d1, d0, d2]
    m.def("transpose_3d_021", py::overload_cast<const GPUArray&>(&ops::transpose_3d_021),
          py::arg("input"),
          "Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2]");

    m.def("transpose_3d_021_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_3d_021),
          py::arg("input"), py::arg("out"),
          "Transpose 3D tensor with output buffer (for CUDA Graph capture)");

    // 4D transpose: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
    m.def("transpose_4d_0213", py::overload_cast<const GPUArray&>(&ops::transpose_4d_0213),
          py::arg("input"),
          "Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d2, d1, d3] (swap axes 1 and 2)");

    m.def("transpose_4d_0213_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_4d_0213),
          py::arg("input"), py::arg("out"),
          "Transpose 4D tensor with output buffer (for CUDA Graph capture)");

    // 3D transpose: [d0, d1, d2] -> [d0, d2, d1]
    m.def("transpose_3d_012", py::overload_cast<const GPUArray&>(&ops::transpose_3d_012),
          py::arg("input"),
          "Transpose 3D tensor: [d0, d1, d2] -> [d0, d2, d1] (swap last two axes)");

    m.def("transpose_3d_012_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_3d_012),
          py::arg("input"), py::arg("out"),
          "Transpose 3D tensor with output buffer (for CUDA Graph capture)");

    // 4D transpose: [d0, d1, d2, d3] -> [d0, d1, d3, d2]
    m.def("transpose_4d_0132", py::overload_cast<const GPUArray&>(&ops::transpose_4d_0132),
          py::arg("input"),
          "Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d1, d3, d2] (swap last two axes)");

    m.def("transpose_4d_0132_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::transpose_4d_0132),
          py::arg("input"), py::arg("out"),
          "Transpose 4D tensor with output buffer (for CUDA Graph capture)");
}
