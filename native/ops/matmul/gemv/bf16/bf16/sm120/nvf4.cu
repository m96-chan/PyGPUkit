/**
 * NVF4 GEMV Implementation for SM120 with BF16 I/O
 *
 * This file provides:
 * 1. NVF4 GEMV kernel dispatch
 * 2. BF16 -> NVF4 weight quantization
 * 3. Automatic dispatch based on GPU architecture
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

// Include BF16 and NVF4 GEMV kernels
#include "../generic/bf16_cutlass.cuh"
#include "nvf4.cuh"

namespace pygpukit {
namespace ops {
namespace gemv_dispatch {

// ============================================================================
// GPU Architecture Detection
// ============================================================================

static int cached_sm_version = -1;

inline int get_sm_version() {
    if (cached_sm_version < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        cached_sm_version = props.major * 10 + props.minor;
    }
    return cached_sm_version;
}

inline bool is_sm120() {
    int sm = get_sm_version();
    return (sm == 120 || sm == 121);
}

// ============================================================================
// NVF4 Weight Storage
// ============================================================================

/**
 * Container for NVF4-quantized weights
 */
struct NVF4Weights {
    uint8_t* data;      // [K/2, N] packed NVF4
    uint8_t* scale;     // [K/32, N] scale factors
    int K;
    int N;
    bool owns_memory;

    NVF4Weights() : data(nullptr), scale(nullptr), K(0), N(0), owns_memory(false) {}

    ~NVF4Weights() {
        if (owns_memory) {
            if (data) cudaFree(data);
            if (scale) cudaFree(scale);
        }
    }

    // Calculate memory sizes
    size_t data_size() const { return (K / 2) * N; }
    size_t scale_size() const { return ((K + 31) / 32) * N; }
    size_t total_size() const { return data_size() + scale_size(); }

    // Memory savings vs BF16
    float compression_ratio() const {
        size_t bf16_size = K * N * 2;  // 2 bytes per BF16
        return (float)bf16_size / total_size();
    }
};

// ============================================================================
// Exported Functions
// ============================================================================

}  // namespace gemv_dispatch
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// C API for Python Bindings
// ============================================================================

extern "C" {

/**
 * Check if NVF4 GEMV is available
 */
bool pygpukit_gemv_nvf4_available() {
    return pygpukit::ops::gemv_nvf4::is_available();
}

/**
 * Quantize BF16 weights to NVF4 format
 *
 * @param input      [K, N] BF16 row-major
 * @param out_data   [K/2, N] packed NVF4 (pre-allocated)
 * @param out_scale  [K/32, N] scale factors (pre-allocated)
 * @param K          Inner dimension
 * @param N          Output dimension
 */
cudaError_t pygpukit_quantize_bf16_to_nvf4(
    const void* input,
    void* out_data,
    void* out_scale,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv_nvf4::quantize_bf16_to_nvf4(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<uint8_t*>(out_data),
        static_cast<uint8_t*>(out_scale),
        K, N, stream
    );
}

/**
 * Quantize BF16 weights to NVF4 format (row-major layout)
 * For pure NVF4/NVF4 GEMV - better memory coalescing
 *
 * @param input      [K, N] BF16 row-major
 * @param out_data   [N, K/2] packed NVF4 row-major (pre-allocated)
 * @param out_scale  [N, K/32] scale factors row-major (pre-allocated)
 * @param K          Inner dimension
 * @param N          Output dimension
 */
cudaError_t pygpukit_quantize_bf16_to_nvf4_rowmajor(
    const void* input,
    void* out_data,
    void* out_scale,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv_nvf4::quantize_bf16_to_nvf4_rowmajor(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<uint8_t*>(out_data),
        static_cast<uint8_t*>(out_scale),
        K, N, stream
    );
}

/**
 * NVF4 GEMV: C[1,N] = A[1,K] @ B[K,N] (NVF4 quantized)
 *
 * @param A         [K] BF16 input vector
 * @param B_data    [K/2, N] packed NVF4 weights
 * @param B_scale   [K/32, N] scale factors
 * @param C         [N] BF16 output vector
 * @param K         Inner dimension
 * @param N         Output dimension
 * @param alpha     Scaling factor
 */
cudaError_t pygpukit_gemv_nvf4_bf16(
    const void* A,
    const void* B_data,
    const void* B_scale,
    void* C,
    int K,
    int N,
    float alpha,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv_nvf4::launch_gemv_nvf4_bf16(
        static_cast<const __nv_bfloat16*>(A),
        static_cast<const uint8_t*>(B_data),
        static_cast<const uint8_t*>(B_scale),
        static_cast<__nv_bfloat16*>(C),
        K, N, alpha, stream
    );
}

/**
 * BF16 GEMV (standard, no quantization)
 */
cudaError_t pygpukit_gemv_bf16(
    const void* A,
    const void* B,
    void* C,
    int K,
    int N,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv::launch_gemv_bf16(
        static_cast<const __nv_bfloat16*>(A),
        static_cast<const __nv_bfloat16*>(B),
        static_cast<__nv_bfloat16*>(C),
        K, N, alpha, beta, stream
    );
}

/**
 * Auto-dispatch GEMV: Uses NVF4 on SM120 if weights are pre-quantized
 * Falls back to BF16 GEMV otherwise
 */
cudaError_t pygpukit_gemv_bf16_auto(
    const void* A,
    const void* B,
    void* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Only dispatch GEMV for M=1
    if (M != 1) {
        return cudaErrorInvalidValue;  // Use GEMM instead
    }

    // Use standard BF16 GEMV (NVF4 requires pre-quantized weights)
    return pygpukit::ops::gemv::launch_gemv_bf16(
        static_cast<const __nv_bfloat16*>(A),
        static_cast<const __nv_bfloat16*>(B),
        static_cast<__nv_bfloat16*>(C),
        K, N, alpha, beta, stream
    );
}

/**
 * Get memory sizes for NVF4 quantization
 */
void pygpukit_nvf4_get_sizes(
    int K,
    int N,
    size_t* data_size,
    size_t* scale_size
) {
    *data_size = (K / 2) * N;
    *scale_size = ((K + 31) / 32) * N;
}

}  // extern "C"
