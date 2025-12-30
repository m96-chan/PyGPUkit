/**
 * W8A16 GEMM: FP8 weight x BF16 activation -> BF16 output (SM120)
 */
#include "../bindings_common.hpp"

// Extern declarations for W8A16 functions
extern "C" {
    cudaError_t pygpukit_w8a16_gemm_init_lut();
    cudaError_t pygpukit_w8a16_gemm_sm120(
        const void* A, const void* B_fp8, const void* B_scale, void* C,
        int M, int N, int K, int scale_stride_n, cudaStream_t stream
    );
    cudaError_t pygpukit_w8a16_cutlass_sm120(
        const void* A, const void* B, void* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    cudaError_t pygpukit_w8a16_blockwise_sm120(
        const void* A, const void* B, void* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    cudaError_t pygpukit_gemm_w8a16_optimized_sm120(
        const void* A, const uint8_t* B,
        void* D, const float* scale_A, const float* scale_B,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
}

void init_gemm_fp8xbf16_bf16(py::module_& m) {
    // ============================================================
    // W8A16 GEMM: FP8 weight x BF16 activation -> BF16 output (SM120)
    // New name: gemm_w8a16_init_lut, alias: w8a16_gemm_init_lut
    // ============================================================
    m.def("gemm_w8a16_init_lut", []() {
        cudaError_t err = pygpukit_w8a16_gemm_init_lut();
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_w8a16_init_lut failed: " + std::string(cudaGetErrorString(err)));
        }
    }, "Initialize FP8->F32 LUT for W8A16 GEMM");
    m.def("w8a16_gemm_init_lut", []() {
        cudaError_t err = pygpukit_w8a16_gemm_init_lut();
        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_gemm_init_lut failed: " + std::string(cudaGetErrorString(err)));
        }
    }, "[Alias for gemm_w8a16_init_lut] Initialize FP8->F32 LUT for W8A16 GEMM");

    // ============================================================
    // W8A16 GEMM with block-wise scale
    // New name: gemm_w8a16_bf16_sm120, alias: w8a16_gemm_sm120
    // ============================================================
    m.def("gemm_w8a16_bf16_sm120", [](const GPUArray& A, const GPUArray& B_fp8, const GPUArray& B_scale, GPUArray& C) {
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120: A and C must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120: B_scale must be bfloat16");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || C.ndim() != 2) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120: A[M,K], B_fp8[K,N], C[M,N] dimensions required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[1];
        int scale_stride_n = (N + 127) / 128;
        if (B_fp8.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_w8a16_gemm_sm120(
            A.data(), B_fp8.data(), B_scale.data(), C.data(),
            M, N, K, scale_stride_n, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_w8a16_bf16_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("B_scale"), py::arg("C"),
       "GEMM W8A16->BF16 for SM120: C[M,N] = A[M,K] @ B_fp8[K,N] (FP8 weight x BF16 activation with block-wise scale)");
    // Alias: w8a16_gemm_sm120
    m.def("w8a16_gemm_sm120", [](const GPUArray& A, const GPUArray& B_fp8, const GPUArray& B_scale, GPUArray& C) {
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_gemm_sm120: A and C must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_gemm_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_gemm_sm120: B_scale must be bfloat16");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || C.ndim() != 2) {
            throw std::runtime_error("w8a16_gemm_sm120: A[M,K], B_fp8[K,N], C[M,N] dimensions required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[1];
        int scale_stride_n = (N + 127) / 128;
        if (B_fp8.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_gemm_sm120: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_gemm_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_w8a16_gemm_sm120(
            A.data(), B_fp8.data(), B_scale.data(), C.data(),
            M, N, K, scale_stride_n, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_gemm_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("B_scale"), py::arg("C"),
       "[Alias for gemm_w8a16_bf16_sm120] W8A16 GEMM: C[M,N] = A[M,K] @ B_fp8[K,N]");

    // ============================================================
    // W8A16 CUTLASS variant
    // New name: gemm_w8a16_bf16_cutlass_sm120, alias: w8a16_cutlass_sm120
    // ============================================================
    m.def("gemm_w8a16_bf16_cutlass_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_w8a16_bf16_cutlass_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_w8a16_bf16_cutlass_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_w8a16_bf16_cutlass_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];
        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_w8a16_bf16_cutlass_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_w8a16_bf16_cutlass_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_w8a16_cutlass_sm120(
            A.data(), B_fp8.data(), D.data(),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_w8a16_bf16_cutlass_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "GEMM W8A16->BF16 (CUTLASS) for SM120: D[M,N] = A[M,K] @ B_fp8[N,K]");
    // Alias: w8a16_cutlass_sm120
    m.def("w8a16_cutlass_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_cutlass_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_cutlass_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("w8a16_cutlass_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];
        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_cutlass_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_cutlass_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_w8a16_cutlass_sm120(
            A.data(), B_fp8.data(), D.data(),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_cutlass_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "[Alias for gemm_w8a16_bf16_cutlass_sm120] W8A16 GEMM using CUTLASS");

    // ============================================================
    // W8A16 blockwise variant
    // New name: gemm_w8a16_bf16_blockwise_sm120, alias: w8a16_blockwise_sm120
    // ============================================================
    m.def("gemm_w8a16_bf16_blockwise_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_w8a16_bf16_blockwise_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_w8a16_bf16_blockwise_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_w8a16_bf16_blockwise_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];
        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_w8a16_bf16_blockwise_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_w8a16_bf16_blockwise_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_w8a16_blockwise_sm120(
            A.data(), B_fp8.data(), D.data(),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_w8a16_bf16_blockwise_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "GEMM W8A16->BF16 (blockwise) for SM120: D[M,N] = A[M,K] @ B_fp8[N,K]");
    // Alias: w8a16_blockwise_sm120
    m.def("w8a16_blockwise_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_blockwise_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_blockwise_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("w8a16_blockwise_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];
        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_blockwise_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_blockwise_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_w8a16_blockwise_sm120(
            A.data(), B_fp8.data(), D.data(),
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_blockwise_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "[Alias for gemm_w8a16_bf16_blockwise_sm120] W8A16 GEMM using blockwise");

    // ============================================================
    // W8A16 optimized variant
    // New name: gemm_w8a16_bf16_optimized_sm120, alias: w8a16_optimized_sm120
    // ============================================================
    m.def("gemm_w8a16_bf16_optimized_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemm_w8a16_bf16_optimized_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_w8a16_bf16_optimized_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_w8a16_bf16_optimized_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];
        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_w8a16_bf16_optimized_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_w8a16_bf16_optimized_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_gemm_w8a16_optimized_sm120(
            A.data(),
            reinterpret_cast<const uint8_t*>(B_fp8.data()),
            D.data(),
            nullptr, nullptr,
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_w8a16_bf16_optimized_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "GEMM W8A16->BF16 (optimized) for SM120: D[M,N] = A[M,K] @ B_fp8[N,K] (~220+ TFLOPS)");
    // Alias: w8a16_optimized_sm120
    m.def("w8a16_optimized_sm120", [](const GPUArray& A, const GPUArray& B_fp8, GPUArray& D) {
        if (A.dtype() != DataType::BFloat16 || D.dtype() != DataType::BFloat16) {
            throw std::runtime_error("w8a16_optimized_sm120: A and D must be bfloat16");
        }
        if (B_fp8.dtype() != DataType::UInt8) {
            throw std::runtime_error("w8a16_optimized_sm120: B_fp8 must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B_fp8.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("w8a16_optimized_sm120: A[M,K], B_fp8[N,K], D[M,N] required");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_fp8.shape()[0];
        if (B_fp8.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("w8a16_optimized_sm120: K dimension mismatch (B_fp8 should be [N,K])");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("w8a16_optimized_sm120: output shape mismatch");
        }
        cudaError_t err = pygpukit_gemm_w8a16_optimized_sm120(
            A.data(),
            reinterpret_cast<const uint8_t*>(B_fp8.data()),
            D.data(),
            nullptr, nullptr,
            M, N, K, 1.0f, 0.0f, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("w8a16_optimized_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_fp8"), py::arg("D"),
       "[Alias for gemm_w8a16_bf16_optimized_sm120] Optimized W8A16 GEMM");
}
