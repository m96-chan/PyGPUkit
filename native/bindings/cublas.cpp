/**
 * cuBLAS debug/utility functions
 *
 * PyGPUkit v0.2.19+
 */
#include "bindings_common.hpp"

void init_cublas(py::module_& m) {
    m.def("cublas_is_available", &cublas::is_available,
          "Check if cuBLAS is dynamically loaded and available.");

    m.def("cublas_get_library_path", &cublas::get_library_path,
          "Get the path to the loaded cuBLAS library.");

    m.def("cublas_get_version", []() {
        auto [major, minor, patch] = cublas::get_version();
        return py::make_tuple(major, minor, patch);
    }, "Get cuBLAS version as (major, minor, patch) tuple.");

    m.def("cublas_test_sgemm", [](const GPUArray& a, const GPUArray& b) {
        // Test SGEMM and return status code
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        GPUArray c({M, N}, a.dtype());

        cudaError_t err = cublas::gemm_fp32(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K, nullptr);

        return static_cast<int>(err);
    }, py::arg("a"), py::arg("b"),
       "Test cuBLAS FP32 SGEMM and return error code (0 = success).");

    m.def("cublas_test_dgemm", [](const GPUArray& a, const GPUArray& b) {
        // Test DGEMM and return status code
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        GPUArray c({M, N}, a.dtype());

        cudaError_t err = cublas::gemm_fp64(
            static_cast<const double*>(a.data()),
            static_cast<const double*>(b.data()),
            static_cast<double*>(c.data()),
            M, N, K, nullptr);

        return static_cast<int>(err);
    }, py::arg("a"), py::arg("b"),
       "Test cuBLAS FP64 DGEMM and return error code (0 = success).");

    m.def("cublas_test_hgemm", [](const GPUArray& a, const GPUArray& b) {
        // Test HGEMM and return status code
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        GPUArray c({M, N}, a.dtype());

        cudaError_t err = cublas::gemm_fp16(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(c.data()),
            M, N, K, nullptr);

        return static_cast<int>(err);
    }, py::arg("a"), py::arg("b"),
       "Test cuBLAS FP16 HGEMM and return error code (0 = success).");

    m.def("cublas_test_bf16gemm", [](const GPUArray& a, const GPUArray& b) {
        // Test BF16 GEMM via GemmEx and return status code
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        GPUArray c({M, N}, a.dtype());

        cudaError_t err = cublas::gemm_bf16(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(c.data()),
            M, N, K, nullptr);

        return static_cast<int>(err);
    }, py::arg("a"), py::arg("b"),
       "Test cuBLAS BF16 GEMM (via GemmEx) and return error code (0 = success).");

    m.def("cublas_get_last_error", &cublas::get_last_error,
          "Get last cuBLAS status code for debugging.");

    m.def("cublas_get_handle", []() {
        auto handle = cublas::get_handle();
        return reinterpret_cast<uintptr_t>(handle);
    }, "Get cuBLAS handle address for debugging (0 if not available).");
}
