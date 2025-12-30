/**
 * Optimized FP8 GEMV: FP8 weights x BF16 activations -> BF16 output
 */
#include "../bindings_common.hpp"

// Forward declaration for namespace-scoped functions
namespace pygpukit {
namespace ops {
namespace gemv {
    cudaError_t launch_gemv_fp8_opt(
        const __nv_bfloat16* A, const uint8_t* B_nk, const __nv_bfloat16* B_scale, __nv_bfloat16* C,
        int K, int N, cudaStream_t stream
    );
    cudaError_t launch_gemv_fp8_opt_batched(
        const __nv_bfloat16* A, const uint8_t* B_nk, const __nv_bfloat16* B_scale, __nv_bfloat16* C,
        int K, int N, int M, cudaStream_t stream
    );
}
}
}

void init_gemv_fp8xfp8_bf16(py::module_& m) {
    m.def("gemv_fp8_bf16_opt", [](const GPUArray& A, const GPUArray& B_nk, const GPUArray& B_scale, GPUArray& C) {
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt: A and C must be bfloat16");
        }
        if (B_nk.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_bf16_opt: B_nk must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt: B_scale must be bfloat16");
        }
        if (A.ndim() != 1 || B_nk.ndim() != 2 || C.ndim() != 1) {
            throw std::runtime_error("gemv_fp8_bf16_opt: A[K], B_nk[N,K], C[N] dimensions required");
        }

        int K = A.shape()[0];
        int N = B_nk.shape()[0];

        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_fp8_bf16_opt: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_fp8_bf16_opt: N dimension mismatch");
        }

        cudaError_t err = pygpukit::ops::gemv::launch_gemv_fp8_opt(
            reinterpret_cast<const __nv_bfloat16*>(A.data()),
            reinterpret_cast<const uint8_t*>(B_nk.data()),
            reinterpret_cast<const __nv_bfloat16*>(B_scale.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_fp8_bf16_opt failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("B_scale"), py::arg("C"),
       "Optimized FP8 GEMV: C[N] = A[K] @ B_nk[N,K]^T (warp-reduce, smem, vec4)");

    m.def("gemv_fp8_bf16_opt_batched", [](const GPUArray& A, const GPUArray& B_nk, const GPUArray& B_scale, GPUArray& C) {
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: A and C must be bfloat16");
        }
        if (B_nk.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: B_nk must be uint8 (FP8 E4M3)");
        }
        if (B_scale.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: B_scale must be bfloat16");
        }
        if (A.ndim() != 2 || B_nk.ndim() != 2 || C.ndim() != 2) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: A[M,K], B_nk[N,K], C[M,N] dimensions required");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B_nk.shape()[0];

        if (B_nk.shape()[1] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: K dimension mismatch");
        }
        if (C.shape()[0] != static_cast<size_t>(M) || C.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched: output shape mismatch");
        }

        cudaError_t err = pygpukit::ops::gemv::launch_gemv_fp8_opt_batched(
            reinterpret_cast<const __nv_bfloat16*>(A.data()),
            reinterpret_cast<const uint8_t*>(B_nk.data()),
            reinterpret_cast<const __nv_bfloat16*>(B_scale.data()),
            reinterpret_cast<__nv_bfloat16*>(C.data()),
            K, N, M, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_fp8_bf16_opt_batched failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_nk"), py::arg("B_scale"), py::arg("C"),
       "Optimized batched FP8 GEMV: C[M,N] = A[M,K] @ B_nk[N,K]^T (warp-reduce, smem, vec4)");
}
