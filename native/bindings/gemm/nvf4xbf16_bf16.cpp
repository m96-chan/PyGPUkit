/**
 * NVF4 (4-bit) GEMM for SM120 with BF16 I/O
 */
#include "../bindings_common.hpp"

// Extern declarations for NVF4 functions
extern "C" {
    cudaError_t pygpukit_gemm_nvf4_bf16_sm120(
        const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_nvf4_bf16_sm120_available();
    bool pygpukit_nvf4_nvf4_sm120_available();

    cudaError_t pygpukit_benchmark_gemm_nvf4_sm120(
        __nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
}

void init_gemm_nvf4xbf16_bf16(py::module_& m) {
    m.def("nvf4_bf16_sm120_available", []() {
        return pygpukit_nvf4_bf16_sm120_available();
    }, "Check if NVF4 BF16 GEMM is available on SM120");

    m.def("gemm_nvf4_bf16_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || B.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: all inputs must be bfloat16");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_nvf4_bf16_sm120(
            static_cast<const __nv_bfloat16*>(A.data()),
            static_cast<const __nv_bfloat16*>(B.data()),
            static_cast<__nv_bfloat16*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_nvf4_bf16_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "NVF4 (4-bit) GEMM for SM120 with BF16 I/O: D = A @ B (BF16 -> NVF4 quantize -> GEMM -> BF16)");

    m.def("nvf4_nvf4_sm120_available", []() {
        return pygpukit_nvf4_nvf4_sm120_available();
    }, "Check if pure NVF4 GEMM is available (SM120+)");

    m.def("benchmark_gemm_nvf4_sm120", [](GPUArray& D, int M, int N, int K) {
        if (D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("benchmark_gemm_nvf4_sm120: D must be bfloat16");
        }
        if (D.ndim() != 2) {
            throw std::runtime_error("benchmark_gemm_nvf4_sm120: D must be 2D");
        }

        cudaError_t err = pygpukit_benchmark_gemm_nvf4_sm120(
            static_cast<__nv_bfloat16*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("benchmark_gemm_nvf4_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"),
       "Benchmark pure NVF4 GEMM (pre-allocated data, no quantization overhead)");
}
