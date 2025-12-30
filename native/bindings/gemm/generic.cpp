/**
 * Generic GEMM operations: matmul, strided batched GEMM
 */
#include "../bindings_common.hpp"

void init_gemm_generic(py::module_& m) {
    // ============================================================
    // Basic matmul (F32 -> F32)
    // New name: gemm_f32_f32, alias: matmul
    // ============================================================
    m.def("gemm_f32_f32", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"),
          "GEMM F32->F32: Matrix multiplication of two GPUArrays");
    m.def("matmul", py::overload_cast<const GPUArray&, const GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"),
          "[Alias for gemm_f32_f32] Matrix multiplication of two GPUArrays");

    m.def("gemm_f32_f32_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "GEMM F32->F32: Matrix multiplication with output array");
    m.def("matmul_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"),
          "[Alias for gemm_f32_f32_] Matrix multiplication with output array");

    // ============================================================
    // TF32 variants (TF32 compute -> F32 output)
    // New name: gemm_tf32_f32, alias: matmul_tf32
    // ============================================================
    m.def("gemm_tf32_f32", py::overload_cast<const GPUArray&, const GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("use_tf32"),
          "GEMM TF32->F32: Matrix multiplication with TF32 TensorCore");
    m.def("matmul_tf32", py::overload_cast<const GPUArray&, const GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("use_tf32"),
          "[Alias for gemm_tf32_f32] Matrix multiplication with explicit TF32 control");

    m.def("gemm_tf32_f32_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("use_tf32"),
          "GEMM TF32->F32: Matrix multiplication with TF32 TensorCore and output array");
    m.def("matmul_tf32_", py::overload_cast<const GPUArray&, const GPUArray&, GPUArray&, bool>(&ops::matmul),
          py::arg("a"), py::arg("b"), py::arg("out"), py::arg("use_tf32"),
          "[Alias for gemm_tf32_f32_] Matrix multiplication with explicit TF32 control and output array");

    // ============================================================
    // Strided Batched GEMM (F32 -> F32)
    // New name: gemm_f32_f32_batched, alias: gemm_strided_batched_fp32
    // ============================================================
    m.def("gemm_f32_f32_batched", &ops::batched_matmul_fp32,
       py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("batch_count"),
       py::arg("strideA"), py::arg("strideB"), py::arg("strideC"),
       "GEMM F32->F32 batched: C[b] = A[b] @ B[b] for b in [0, batch_count)");
    m.def("gemm_strided_batched_fp32", &ops::batched_matmul_fp32,
       py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("batch_count"),
       py::arg("strideA"), py::arg("strideB"), py::arg("strideC"),
       "[Alias for gemm_f32_f32_batched] Strided batched GEMM: C[b] = A[b] @ B[b] for b in [0, batch_count)");
}
