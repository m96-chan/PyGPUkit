/**
 * NVF4 GEMV: NVF4 weights x BF16 activations -> BF16 output (SM120)
 */
#include "../bindings_common.hpp"

// Extern declarations for NVF4 GEMV functions
extern "C" {
    bool pygpukit_gemv_nvf4_available();
    void pygpukit_nvf4_get_sizes(int K, int N, size_t* data_size, size_t* scale_size);
    cudaError_t pygpukit_quantize_bf16_to_nvf4(
        const void* input, void* out_data, void* out_scale,
        int K, int N, cudaStream_t stream
    );
    cudaError_t pygpukit_quantize_bf16_to_nvf4_rowmajor(
        const void* input, void* out_data, void* out_scale,
        int K, int N, cudaStream_t stream
    );
    cudaError_t pygpukit_gemv_nvf4_bf16(
        const void* A, const void* B_data, const void* B_scale, void* C,
        int K, int N, float alpha, cudaStream_t stream
    );
}

void init_gemv_nvf4xbf16_bf16(py::module_& m) {
    m.def("gemv_nvf4_available", []() {
        return pygpukit_gemv_nvf4_available();
    }, "Check if NVF4 GEMV is available (SM120+)");

    m.def("nvf4_get_sizes", [](int K, int N) {
        size_t data_size, scale_size;
        pygpukit_nvf4_get_sizes(K, N, &data_size, &scale_size);
        return py::make_tuple(data_size, scale_size);
    }, py::arg("K"), py::arg("N"),
       "Get buffer sizes for NVF4 quantization: returns (data_size, scale_size)");

    m.def("quantize_bf16_to_nvf4", [](const GPUArray& input, GPUArray& out_data, GPUArray& out_scale) {
        if (input.dtype() != DataType::BFloat16) {
            throw std::runtime_error("quantize_bf16_to_nvf4: input must be bfloat16");
        }
        if (input.ndim() != 2) {
            throw std::runtime_error("quantize_bf16_to_nvf4: input must be 2D [K, N]");
        }

        int K = input.shape()[0];
        int N = input.shape()[1];

        cudaError_t err = pygpukit_quantize_bf16_to_nvf4(
            input.data(), out_data.data(), out_scale.data(),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("quantize_bf16_to_nvf4 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("input"), py::arg("out_data"), py::arg("out_scale"),
       "Quantize BF16 weights to NVF4 format (column-major output [K/2,N]) for SM120 W4A16 GEMV");

    m.def("quantize_bf16_to_nvf4_rowmajor", [](const GPUArray& input, GPUArray& out_data, GPUArray& out_scale) {
        if (input.dtype() != DataType::BFloat16) {
            throw std::runtime_error("quantize_bf16_to_nvf4_rowmajor: input must be bfloat16");
        }
        if (input.ndim() != 2) {
            throw std::runtime_error("quantize_bf16_to_nvf4_rowmajor: input must be 2D [K, N]");
        }

        int K = input.shape()[0];
        int N = input.shape()[1];

        cudaError_t err = pygpukit_quantize_bf16_to_nvf4_rowmajor(
            input.data(), out_data.data(), out_scale.data(),
            K, N, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("quantize_bf16_to_nvf4_rowmajor failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("input"), py::arg("out_data"), py::arg("out_scale"),
       "Quantize BF16 weights to NVF4 format (row-major output [N,K/2]) for pure NVF4/NVF4 GEMV");

    m.def("gemv_nvf4_bf16", [](const GPUArray& A, const GPUArray& B_data, const GPUArray& B_scale, GPUArray& C, float alpha) {
        if (A.dtype() != DataType::BFloat16 || C.dtype() != DataType::BFloat16) {
            throw std::runtime_error("gemv_nvf4_bf16: A and C must be bfloat16");
        }
        if (A.ndim() != 1) {
            throw std::runtime_error("gemv_nvf4_bf16: A must be 1D [K]");
        }

        int K = A.shape()[0];
        int N = C.shape()[0];

        cudaError_t err = pygpukit_gemv_nvf4_bf16(
            A.data(), B_data.data(), B_scale.data(), C.data(),
            K, N, alpha, nullptr
        );

        if (err != cudaSuccess) {
            throw std::runtime_error("gemv_nvf4_bf16 failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("A"), py::arg("B_data"), py::arg("B_scale"), py::arg("C"), py::arg("alpha") = 1.0f,
       "NVF4 GEMV for SM120: C[N] = alpha * A[K] @ B[K,N] (NVF4 quantized weights)");
}
