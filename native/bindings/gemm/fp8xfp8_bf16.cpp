/**
 * FP8 GEMM with F32 I/O: FP8 internally quantized, F32 input/output
 * For SM90 (Hopper), SM100 (Blackwell datacenter), SM120 (Blackwell GeForce)
 */
#include "../bindings_common.hpp"

// Extern declarations for FP8 functions
extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm90(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_sm90_available();

    cudaError_t pygpukit_gemm_fp8_sm100(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_sm100_available();

    cudaError_t pygpukit_gemm_fp8_sm120(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_sm120_available();
}

void init_gemm_fp8xfp8_bf16(py::module_& m) {
    // ============================================================
    // SM90 (Hopper) - FP8 internally, F32 I/O
    // New name: gemm_fp8_f32_sm90_available, alias: fp8_sm90_available
    // ============================================================
    m.def("gemm_fp8_f32_sm90_available", []() {
        return pygpukit_fp8_sm90_available();
    }, "Check if FP8 GEMM (F32 I/O) is available on SM90 (Hopper)");
    m.def("fp8_sm90_available", []() {
        return pygpukit_fp8_sm90_available();
    }, "[Alias for gemm_fp8_f32_sm90_available] Check if FP8 GEMM is available on SM90 (Hopper)");

    // New name: gemm_fp8_f32_sm90, alias: gemm_fp8_sm90
    m.def("gemm_fp8_f32_sm90", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm90: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm90: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm90: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm90: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_sm90(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_f32_sm90 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "GEMM FP8 (F32 I/O) for SM90: D = A @ B (FP8 quantization internally)");
    // Alias: gemm_fp8_sm90
    m.def("gemm_fp8_sm90", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm90: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm90: all inputs must be 2D");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];
        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm90: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm90: D shape mismatch");
        }
        cudaError_t err = pygpukit_gemm_fp8_sm90(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_sm90 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "[Alias for gemm_fp8_f32_sm90] FP8 GEMM for SM90 (Hopper)");

    // ============================================================
    // SM100 (Blackwell datacenter) - FP8 internally, F32 I/O
    // New name: gemm_fp8_f32_sm100_available, alias: fp8_sm100_available
    // ============================================================
    m.def("gemm_fp8_f32_sm100_available", []() {
        return pygpukit_fp8_sm100_available();
    }, "Check if FP8 GEMM (F32 I/O) is available on SM100 (Blackwell datacenter)");
    m.def("fp8_sm100_available", []() {
        return pygpukit_fp8_sm100_available();
    }, "[Alias for gemm_fp8_f32_sm100_available] Check if FP8 GEMM is available on SM100 (Blackwell datacenter)");

    // New name: gemm_fp8_f32_sm100, alias: gemm_fp8_sm100
    m.def("gemm_fp8_f32_sm100", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm100: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm100: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm100: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm100: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_sm100(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_f32_sm100 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "GEMM FP8 (F32 I/O) for SM100: D = A @ B (FP8 quantization internally)");
    // Alias: gemm_fp8_sm100
    m.def("gemm_fp8_sm100", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm100: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm100: all inputs must be 2D");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];
        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm100: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm100: D shape mismatch");
        }
        cudaError_t err = pygpukit_gemm_fp8_sm100(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_sm100 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "[Alias for gemm_fp8_f32_sm100] FP8 GEMM for SM100 (Blackwell datacenter)");

    // ============================================================
    // SM120 (Blackwell GeForce) - FP8 internally, F32 I/O
    // New name: gemm_fp8_f32_sm120_available, alias: fp8_sm120_available
    // ============================================================
    m.def("gemm_fp8_f32_sm120_available", []() {
        return pygpukit_fp8_sm120_available();
    }, "Check if FP8 GEMM (F32 I/O) is available on SM120 (Blackwell GeForce)");
    m.def("fp8_sm120_available", []() {
        return pygpukit_fp8_sm120_available();
    }, "[Alias for gemm_fp8_f32_sm120_available] Check if FP8 GEMM is available on SM120");

    // New name: gemm_fp8_f32_sm120, alias: gemm_fp8_sm120
    m.def("gemm_fp8_f32_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm120: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm120: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_sm120(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_f32_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "GEMM FP8 (F32 I/O) for SM120: D = A @ B (FP8 quantization internally)");
    // Alias: gemm_fp8_sm120
    m.def("gemm_fp8_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || D.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_sm120: all inputs must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_sm120: all inputs must be 2D");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];
        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_sm120: D shape mismatch");
        }
        cudaError_t err = pygpukit_gemm_fp8_sm120(
            static_cast<const float*>(A.data()),
            static_cast<const float*>(B.data()),
            static_cast<float*>(D.data()),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "[Alias for gemm_fp8_f32_sm120] FP8 GEMM for SM120");
}
