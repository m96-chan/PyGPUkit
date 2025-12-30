/**
 * Int8/Int4 GEMM operations using dp4a CUDA cores (SM120)
 */
#include "../bindings_common.hpp"

// Extern declarations for Int8/Int4 GEMM functions
extern "C" {
    cudaError_t pygpukit_gemm_int8_native_sm120(
        const int8_t* A, const int8_t* B, int32_t* D,
        int M, int N, int K,
        cudaStream_t stream
    );
    bool pygpukit_int8_native_gemm_available();

    bool pygpukit_int4_gemm_sm120_available();
    cudaError_t pygpukit_gemm_int4_int4_int32_sm120(
        const uint8_t* A, const uint8_t* B, int32_t* D,
        int M, int N, int K,
        float scale_A, float scale_B, float descale_D,
        cudaStream_t stream
    );
    cudaError_t pygpukit_gemm_int4_int4_int8_sm120(
        const uint8_t* A, const uint8_t* B, int8_t* D,
        int M, int N, int K,
        float scale_A, float scale_B, float descale_D,
        cudaStream_t stream
    );
}

void init_gemm_int(py::module_& m) {
    // Int8 GEMM
    m.def("int8_native_gemm_available", []() {
        return pygpukit_int8_native_gemm_available();
    }, "Check if native Int8 GEMM is available (uses dp4a CUDA cores)");

    m.def("int8_native_gemm_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D
    ) {
        if (A.dtype() != DataType::Int8) {
            throw std::runtime_error("int8_native_gemm_sm120: A must be int8");
        }
        if (B.dtype() != DataType::Int8) {
            throw std::runtime_error("int8_native_gemm_sm120: B must be int8");
        }
        if (D.dtype() != DataType::Int32) {
            throw std::runtime_error("int8_native_gemm_sm120: D must be int32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("int8_native_gemm_sm120: A[M,K], B[N,K], D[M,N] required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[0];

        if (B.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("int8_native_gemm_sm120: K dimension mismatch");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("int8_native_gemm_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_int8_native_sm120(
            reinterpret_cast<const int8_t*>(A.data()),
            reinterpret_cast<const int8_t*>(B.data()),
            reinterpret_cast<int32_t*>(D.data()),
            M, N, K,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int8_native_gemm_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "Native Int8 GEMM using dp4a: D[M,N] = A[M,K] @ B[N,K]^T with exact Int32 output");

    // Int4 GEMM
    m.def("int4_gemm_available", []() {
        return pygpukit_int4_gemm_sm120_available();
    }, "Check if Int4 GEMM is available (SM120 via Int8/FP8 approximation)");

    m.def("int4_gemm_int32_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D,
        float scale_A, float scale_B, float descale_D
    ) {
        if (A.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int32_sm120: A must be uint8 (packed int4)");
        }
        if (B.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int32_sm120: B must be uint8 (packed int4)");
        }
        if (D.dtype() != DataType::Int32) {
            throw std::runtime_error("int4_gemm_int32_sm120: D must be int32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("int4_gemm_int32_sm120: A[M,K/2], B[N,K/2], D[M,N] required");
        }

        int M = A.shape()[0];
        int K_packed = A.shape()[1];
        int K = K_packed * 2;
        int N = B.shape()[0];

        if (B.shape()[1] != static_cast<size_t>(K_packed)) {
            throw std::runtime_error("int4_gemm_int32_sm120: K dimension mismatch");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("int4_gemm_int32_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_int4_int4_int32_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B.data()),
            reinterpret_cast<int32_t*>(D.data()),
            M, N, K,
            scale_A, scale_B, descale_D,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int4_gemm_int32_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("scale_A") = 1.0f, py::arg("scale_B") = 1.0f, py::arg("descale_D") = 1.0f,
       "Int4 GEMM via Int8/FP8: D[M,N] = A[M,K] @ B[N,K]^T with Int32 output. Input is packed int4.");

    m.def("int4_gemm_int8_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D,
        float scale_A, float scale_B, float descale_D
    ) {
        if (A.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int8_sm120: A must be uint8 (packed int4)");
        }
        if (B.dtype() != DataType::UInt8) {
            throw std::runtime_error("int4_gemm_int8_sm120: B must be uint8 (packed int4)");
        }
        if (D.dtype() != DataType::Int8) {
            throw std::runtime_error("int4_gemm_int8_sm120: D must be int8");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("int4_gemm_int8_sm120: A[M,K/2], B[N,K/2], D[M,N] required");
        }

        int M = A.shape()[0];
        int K_packed = A.shape()[1];
        int K = K_packed * 2;
        int N = B.shape()[0];

        if (B.shape()[1] != static_cast<size_t>(K_packed)) {
            throw std::runtime_error("int4_gemm_int8_sm120: K dimension mismatch");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("int4_gemm_int8_sm120: output shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_int4_int4_int8_sm120(
            reinterpret_cast<const uint8_t*>(A.data()),
            reinterpret_cast<const uint8_t*>(B.data()),
            reinterpret_cast<int8_t*>(D.data()),
            M, N, K,
            scale_A, scale_B, descale_D,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("int4_gemm_int8_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("scale_A") = 1.0f, py::arg("scale_B") = 1.0f, py::arg("descale_D") = 1.0f,
       "Int4 GEMM via Int8/FP8: D[M,N] = A[M,K] @ B[N,K]^T with Int8 output. Input is packed int4.");
}
