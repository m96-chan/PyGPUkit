/**
 * Generic GEMM operations: matmul, strided batched GEMM
 */
#include "../bindings_common.hpp"

void init_gemm_generic(py::module_& m) {
    // Basic matmul
    m.def("matmul", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"),
          "Matrix multiplication of two GPUArrays");

    m.def("matmul_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "Matrix multiplication with output array");

    // TF32 variants
    m.def("matmul_tf32", py::overload_cast<const GPUArray&, const GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("use_tf32"),
          "Matrix multiplication with explicit TF32 control");

    m.def("matmul_tf32_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("use_tf32"),
          "Matrix multiplication with explicit TF32 control and output array");

    // Strided Batched GEMM
    m.def("gemm_strided_batched_fp32", &ops::batched_matmul_fp32,
       py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("batch_count"),
       py::arg("strideA"), py::arg("strideB"), py::arg("strideC"),
       "Strided batched GEMM: C[b] = A[b] @ B[b] for b in [0, batch_count)");
}
