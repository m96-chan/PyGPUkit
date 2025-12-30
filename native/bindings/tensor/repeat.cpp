/**
 * Repeat and concat operations
 */
#include "../bindings_common.hpp"

void init_tensor_repeat(py::module_& m) {
    // Concat along axis 0
    m.def("concat_axis0", &ops::concat_axis0,
          py::arg("a"), py::arg("b"),
          "Concatenate two tensors along axis 0.\n"
          "a: [dim0_a, ...], b: [dim0_b, ...]\n"
          "Output: [dim0_a + dim0_b, ...]");

    // Repeat interleave along axis 1 (for GQA)
    m.def("repeat_interleave_axis1", &ops::repeat_interleave_axis1,
          py::arg("input"), py::arg("repeats"),
          "Repeat tensor along axis 1 (interleaved).\n"
          "input: [dim0, dim1, dim2] -> output: [dim0, dim1 * repeats, dim2]");
}
