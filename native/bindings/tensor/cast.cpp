/**
 * Dtype cast operations
 */
#include "../bindings_common.hpp"

void init_tensor_cast(py::module_& m) {
    m.def("cast_f32_to_bf16", py::overload_cast<const GPUArray&>(&ops::cast_f32_to_bf16),
          py::arg("src"),
          "Cast float32 to bfloat16 on GPU (round to nearest even)");

    m.def("cast_f32_to_bf16_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::cast_f32_to_bf16),
          py::arg("src"), py::arg("dst"),
          "Cast float32 to bfloat16 on GPU (in-place version)");

    m.def("cast_f32_to_f16", &ops::cast_f32_to_f16,
          py::arg("src"),
          "Cast float32 to float16 on GPU");

    m.def("cast_bf16_to_f32", &ops::cast_bf16_to_f32,
          py::arg("src"),
          "Cast bfloat16 to float32 on GPU");

    m.def("cast_f16_to_f32", &ops::cast_f16_to_f32,
          py::arg("src"),
          "Cast float16 to float32 on GPU");
}
