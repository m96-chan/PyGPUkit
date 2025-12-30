/**
 * Grouped GEMM for MoE: FP8 weights x BF16 activations -> BF16 output
 */
#include "../bindings_common.hpp"

// Extern declarations for grouped GEMM functions
extern "C" {
    cudaError_t pygpukit_grouped_gemm_init_lut();
    cudaError_t pygpukit_grouped_gemm_fp8_bf16(
        const void* A, const void* B_stacked, const void* B_scale,
        void* C, const int* row_expert_ids,
        int M, int N, int K, cudaStream_t stream
    );
}

void init_gemm_grouped(py::module_& m) {
    // ============================================================
    // Grouped GEMM for MoE: FP8 weights x BF16 activations -> BF16 output
    // Functions already follow convention, just add _sm120 suffix where missing
    // ============================================================
    m.def("grouped_gemm_init_lut", []() {
        cudaError_t err = pygpukit_grouped_gemm_init_lut();
        if (err != cudaSuccess) {
            throw std::runtime_error("grouped_gemm_init_lut failed: " + std::string(cudaGetErrorString(err)));
        }
    }, "Initialize FP8->BF16 LUT for grouped GEMM");

    // New name: grouped_gemm_fp8_bf16_sm120, alias: grouped_gemm_fp8_bf16
    m.def("grouped_gemm_fp8_bf16_sm120", [](
        const GPUArray& A,
        const GPUArray& B_stacked,
        const GPUArray& B_scale,
        GPUArray& C,
        const GPUArray& row_expert_ids
    ) {
        // Validate dtypes
        if (A.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: A must be bfloat16");
        }
        if (B_stacked.dtype() != DataType::UInt8) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: B_stacked must be uint8 (FP8)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: B_scale must be bfloat16");
        }
        if (C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: C must be bfloat16");
        }
        if (row_expert_ids.dtype() != DataType::Int32) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: row_expert_ids must be int32");
        }

        // Validate dimensions
        if (A.ndim() != 2 || B_stacked.ndim() != 3 || C.ndim() != 2) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: invalid dimensions");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_stacked.shape()[1];

        if (B_stacked.shape()[2] != static_cast<size_t>(K)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: output shape mismatch");
        }
        if (row_expert_ids.ndim() != 1 || row_expert_ids.shape()[0] != static_cast<size_t>(M)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: row_expert_ids size mismatch");
        }

        cudaError_t err = pygpukit_grouped_gemm_fp8_bf16(
            A.data(), B_stacked.data(), B_scale.data(), C.data(),
            reinterpret_cast<const int*>(row_expert_ids.data()),
            M, N, K, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("grouped_gemm_fp8_bf16_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_stacked"), py::arg("B_scale"), py::arg("C"), py::arg("row_expert_ids"),
       "Grouped GEMM FP8->BF16 for SM120: C[M,N] = A[M,K] @ B_stacked[experts,N,K] with per-row expert IDs");
    // Alias: grouped_gemm_fp8_bf16
    m.def("grouped_gemm_fp8_bf16", [](
        const GPUArray& A,
        const GPUArray& B_stacked,
        const GPUArray& B_scale,
        GPUArray& C,
        const GPUArray& row_expert_ids
    ) {
        if (A.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: A must be bfloat16");
        }
        if (B_stacked.dtype() != DataType::UInt8) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: B_stacked must be uint8 (FP8)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: B_scale must be bfloat16");
        }
        if (C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: C must be bfloat16");
        }
        if (row_expert_ids.dtype() != DataType::Int32) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: row_expert_ids must be int32");
        }
        if (A.ndim() != 2 || B_stacked.ndim() != 3 || C.ndim() != 2) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: invalid dimensions");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_stacked.shape()[1];
        if (B_stacked.shape()[2] != static_cast<size_t>(K)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: output shape mismatch");
        }
        if (row_expert_ids.ndim() != 1 || row_expert_ids.shape()[0] != static_cast<size_t>(M)) {
            throw std::runtime_error("grouped_gemm_fp8_bf16: row_expert_ids size mismatch");
        }
        cudaError_t err = pygpukit_grouped_gemm_fp8_bf16(
            A.data(), B_stacked.data(), B_scale.data(), C.data(),
            reinterpret_cast<const int*>(row_expert_ids.data()),
            M, N, K, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("grouped_gemm_fp8_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_stacked"), py::arg("B_scale"), py::arg("C"), py::arg("row_expert_ids"),
       "[Alias for grouped_gemm_fp8_bf16_sm120] Grouped GEMM for MoE");
}
