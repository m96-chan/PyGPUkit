/**
 * Pure FP8 I/O GEMM: FP8 input/output for SM120 (Blackwell GeForce)
 */
#include "../bindings_common.hpp"

// Extern declarations for pure FP8 functions
extern "C" {
    cudaError_t pygpukit_gemm_fp8_fp8_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    bool pygpukit_fp8_fp8_sm120_available();

    cudaError_t pygpukit_gemm_fp8_fp8_blockwise_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        const float* scale_A, const float* scale_B,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    );
    void pygpukit_fp8_fp8_get_scale_sizes(
        int M, int N, int K,
        size_t* sfa_size, size_t* sfb_size
    );

    // Tile variants
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v2(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v3(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v4(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);

    // Optimized variants (V5-V8)
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v5(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v6(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v7(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    cudaError_t pygpukit_gemm_fp8_fp8_sm120_v8(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t);
    void pygpukit_gemm_fp8_fp8_sm120_cleanup();
}

void init_gemm_fp8xfp8_fp8(py::module_& m) {
    m.def("fp8_fp8_sm120_available", []() {
        return pygpukit_fp8_fp8_sm120_available();
    }, "Check if Pure FP8 I/O GEMM is available on SM120");

    m.def("gemm_fp8_fp8_sm120", [](const GPUArray& A, const GPUArray& B, GPUArray& D) {
        if (A.dtype() != DataType::UInt8 || B.dtype() != DataType::UInt8 || D.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: all inputs must be uint8 (FP8 E4M3)");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: all inputs must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_fp8_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_fp8_sm120(
            static_cast<const uint8_t*>(A.data()),
            static_cast<const uint8_t*>(B.data()),
            static_cast<uint8_t*>(D.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_fp8_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       "Pure FP8 I/O GEMM for SM120: D = A @ B (FP8 E4M3 input/output)");

    // Tile variant helper
    auto bind_fp8_tile = [&m](const char* name, auto func, const char* doc) {
        m.def(name, [func, name](const GPUArray& A, const GPUArray& B, GPUArray& D) {
            if (A.dtype() != DataType::UInt8 || B.dtype() != DataType::UInt8 || D.dtype() != DataType::UInt8) {
                throw std::runtime_error("FP8 GEMM: all inputs must be uint8");
            }
            int M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
            if (B.shape()[0] != static_cast<size_t>(K)) throw std::runtime_error("Shape mismatch");
            cudaError_t err = func(
                static_cast<const uint8_t*>(A.data()),
                static_cast<const uint8_t*>(B.data()),
                static_cast<uint8_t*>(D.data()),
                M, N, K, 1.0f, 0.0f, nullptr);
            if (err != cudaSuccess) throw std::runtime_error(std::string(name) + " failed");
        }, py::arg("A"), py::arg("B"), py::arg("D"), doc);
    };

    bind_fp8_tile("gemm_fp8_fp8_sm120_v2", pygpukit_gemm_fp8_fp8_sm120_v2, "FP8 GEMM 128x256x64");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v3", pygpukit_gemm_fp8_fp8_sm120_v3, "FP8 GEMM 256x128x64");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v4", pygpukit_gemm_fp8_fp8_sm120_v4, "FP8 GEMM 128x128x64");

    // Optimized FP8 GEMM (V5-V8) - Cached scale buffers
    bind_fp8_tile("gemm_fp8_fp8_sm120_v5", pygpukit_gemm_fp8_fp8_sm120_v5, "FP8 GEMM 128x128x128 cached");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v6", pygpukit_gemm_fp8_fp8_sm120_v6, "FP8 GEMM 128x256x64 cached");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v7", pygpukit_gemm_fp8_fp8_sm120_v7, "FP8 GEMM 256x128x64 cached");
    bind_fp8_tile("gemm_fp8_fp8_sm120_v8", pygpukit_gemm_fp8_fp8_sm120_v8, "FP8 GEMM 128x128x64 cached");

    // Blockwise scaled FP8 GEMM
    m.def("gemm_fp8_fp8_blockwise_sm120", [](
        const GPUArray& A, const GPUArray& B, GPUArray& D,
        const GPUArray& scale_A, const GPUArray& scale_B
    ) {
        if (A.dtype() != DataType::UInt8 || B.dtype() != DataType::UInt8 || D.dtype() != DataType::UInt8) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: A, B, D must be uint8 (FP8 E4M3)");
        }
        if (scale_A.dtype() != DataType::Float32 || scale_B.dtype() != DataType::Float32) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: scale_A, scale_B must be float32");
        }
        if (A.ndim() != 2 || B.ndim() != 2 || D.ndim() != 2) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: A, B, D must be 2D");
        }

        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];

        if (B.shape()[0] != static_cast<size_t>(K)) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: A.shape[1] must equal B.shape[0]");
        }
        if (D.shape()[0] != static_cast<size_t>(M) || D.shape()[1] != static_cast<size_t>(N)) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120: D shape mismatch");
        }

        cudaError_t err = pygpukit_gemm_fp8_fp8_blockwise_sm120(
            static_cast<const uint8_t*>(A.data()),
            static_cast<const uint8_t*>(B.data()),
            static_cast<uint8_t*>(D.data()),
            static_cast<const float*>(scale_A.data()),
            static_cast<const float*>(scale_B.data()),
            M, N, K,
            1.0f, 0.0f,
            nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemm_fp8_fp8_blockwise_sm120 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("scale_A"), py::arg("scale_B"),
       "Blockwise scaled FP8 I/O GEMM for SM120: D = (A * scale_A) @ (B * scale_B)");

    // Get scale factor sizes for FP8 blockwise GEMM
    m.def("fp8_fp8_get_scale_sizes", [](int M, int N, int K) {
        size_t sfa_size, sfb_size;
        pygpukit_fp8_fp8_get_scale_sizes(M, N, K, &sfa_size, &sfb_size);
        return py::make_tuple(sfa_size, sfb_size);
    }, py::arg("M"), py::arg("N"), py::arg("K"),
       "Get scale factor sizes for FP8 blockwise GEMM (returns (sfa_size, sfb_size))");
}
