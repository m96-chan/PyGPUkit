/**
 * cuBLASLt debug/utility functions
 */
#include "bindings_common.hpp"

void init_cublaslt(py::module_& m) {
    m.def("cublaslt_is_available", &cublaslt::is_available,
          "Check if cuBLASLt is dynamically loaded and available.");

    m.def("cublaslt_get_library_path", &cublaslt::get_library_path,
          "Get the path to the loaded cuBLASLt library.");

    m.def("cublaslt_get_version", []() {
        auto [major, minor, patch] = cublaslt::get_version();
        return py::make_tuple(major, minor, patch);
    }, "Get cuBLASLt version as (major, minor, patch) tuple.");

    m.def("cublaslt_test_gemm", [](const GPUArray& a, const GPUArray& b) {
        // Test GEMM and return status code
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        GPUArray c({M, N}, a.dtype());

        cudaError_t err = cublaslt::gemm_fp16(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(c.data()),
            M, N, K, nullptr);

        return static_cast<int>(err);
    }, py::arg("a"), py::arg("b"),
       "Test cuBLASLt FP16 GEMM and return error code (0 = success).");

    m.def("cublaslt_get_last_error", &cublaslt::get_last_cublaslt_error,
          "Get last cuBLASLt status code for debugging.");

    m.def("cublaslt_get_last_step", &cublaslt::get_last_cublaslt_step,
          "Get which step failed (1=handle, 2=desc, 3-5=layout, 6=matmul).");

    m.def("cublaslt_get_handle", []() {
        auto handle = cublaslt::get_handle();
        return reinterpret_cast<uintptr_t>(handle);
    }, "Get cuBLASLt handle address for debugging (0 if not available).");
}
