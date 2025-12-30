/**
 * Quantization operations: INT8 quantization/dequantization
 */
#include "bindings_common.hpp"

void init_quantize(py::module_& m) {
    // Dequantize INT8 to FP16/FP32
    m.def("dequantize_int8", &ops::dequantize_int8,
          py::arg("input"), py::arg("scale"), py::arg("output_dtype"),
          "Dequantize INT8 tensor to FP16/FP32.\n"
          "output = input_int8 * scale\n"
          "input: [rows, cols] INT8, scale: [cols], output_dtype: Float16 or Float32");

    // Fused INT8 linear (dequantize + matmul)
    m.def("linear_int8", [](const GPUArray& activation, const GPUArray& weight_int8,
                            const GPUArray& scale, const GPUArray* bias) {
              return ops::linear_int8(activation, weight_int8, scale, bias);
          },
          py::arg("activation"), py::arg("weight_int8"), py::arg("scale"),
          py::arg("bias") = nullptr,
          "Fused INT8 linear layer: output = activation @ (weight_int8 * scale)^T\n"
          "activation: [M, K] FP16, weight_int8: [N, K] INT8, scale: [N] FP16\n"
          "Dequantization happens on-the-fly (memory efficient).");

    // Quantize to INT8
    m.def("quantize_to_int8", &ops::quantize_to_int8,
          py::arg("input"),
          "Quantize FP16/FP32 tensor to INT8 with per-column scaling.\n"
          "Returns (weight_int8, scale) tuple.\n"
          "weight_int8: [rows, cols] INT8, scale: [cols] same dtype as input");
}
