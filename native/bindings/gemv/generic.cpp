/**
 * Generic GEMV operations: BF16 optimized GEMV
 */
#include "../bindings_common.hpp"

// Extern declarations for GEMV functions
extern "C" {
    cudaError_t pygpukit_gemv_bf16_opt_sm120(
        const __nv_bfloat16* A, const __nv_bfloat16* B_nk, __nv_bfloat16* C,
        int K, int N, cudaStream_t stream
    );
    bool pygpukit_gemv_bf16_opt_sm120_available();
}

void init_gemv_generic(py::module_& m) {
    // ============================================================
    // BF16 GEMV: BF16 x BF16 -> BF16 (SM120)
    // New name: gemv_bf16_bf16_sm120, alias: gemv_bf16_opt_sm120
    // ============================================================
    m.def("gemv_bf16_bf16_sm120", [](const GPUArray& A, const GPUArray& B_nk, GPUArray& C) {
        if (A.dtype() != DataType::BFloat16 || B_nk.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_bf16_bf16_sm120: all inputs must be bfloat16");
        }
        if (A.ndim() != 1 || B_nk.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_bf16_bf16_sm120: A[K], B_nk[N,K], C[N] dimensions required");
        }
        int K = A.shape()[0];
        int N = B_nk.shape()[0];
        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_bf16_bf16_sm120: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_bf16_bf16_sm120: N dimension mismatch");
        }
        cudaError_t err = pygpukit_gemv_bf16_opt_sm120(
            reinterpret_cast<const __nv_bfloat16*>(A.data()),
            reinterpret_cast<const __nv_bfloat16*>(B_nk.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_bf16_bf16_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("C"),
       "GEMV BF16->BF16 for SM120: C[N] = A[K] @ B_nk[N,K]^T (warp-reduce optimized)");
    // Alias: gemv_bf16_opt_sm120
    m.def("gemv_bf16_opt_sm120", [](const GPUArray& A, const GPUArray& B_nk, GPUArray& C) {
        if (A.dtype() != DataType::BFloat16 || B_nk.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_bf16_opt_sm120: all inputs must be bfloat16");
        }
        if (A.ndim() != 1 || B_nk.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_bf16_opt_sm120: A[K], B_nk[N,K], C[N] dimensions required");
        }
        int K = A.shape()[0];
        int N = B_nk.shape()[0];
        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_bf16_opt_sm120: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_bf16_opt_sm120: N dimension mismatch");
        }
        cudaError_t err = pygpukit_gemv_bf16_opt_sm120(
            reinterpret_cast<const __nv_bfloat16*>(A.data()),
            reinterpret_cast<const __nv_bfloat16*>(B_nk.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_bf16_opt_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("C"),
       "[Alias for gemv_bf16_bf16_sm120] Optimized BF16 GEMV");

    // New name: gemv_bf16_bf16_available, alias: gemv_bf16_opt_available
    m.def("gemv_bf16_bf16_available", []() {
        return pygpukit_gemv_bf16_opt_sm120_available();
    }, "Check if BF16 GEMV is available (SM80+)");
    m.def("gemv_bf16_opt_available", []() {
        return pygpukit_gemv_bf16_opt_sm120_available();
    }, "[Alias for gemv_bf16_bf16_available] Check if optimized BF16 GEMV is available");
}
