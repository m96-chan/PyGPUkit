/**
 * Pure FP8/FP8/FP8 GEMV Launch Functions (SM120)
 */

#include "fp8_gemv.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_fp8_pure(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    using Config = GemvFP8PureConfig;

    dim3 block(Config::BLOCK_SIZE);  // 256 threads
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK);

    // Shared memory for A (FP8 = 1 byte per element)
    size_t smem_size = K * sizeof(uint8_t);

    // Kernel selection based on K size:
    // - K >= 512: Use optimized kernel (128-bit loads, __ldg, multi-accumulators)
    // - K >= 256: Use vec8 kernel
    // - K < 256: Use scalar kernel
    if (K >= 512) {
        gemv_fp8_pure_opt_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, scale_A, scale_B, C, K, N
        );
    } else if (K >= 256) {
        gemv_fp8_pure_vec8_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, scale_A, scale_B, C, K, N
        );
    } else {
        gemv_fp8_pure_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, scale_A, scale_B, C, K, N
        );
    }

    return cudaGetLastError();
}

cudaError_t launch_gemv_fp8_pure_fp8out(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    uint8_t* C,
    float scale_C,
    int K,
    int N,
    cudaStream_t stream
) {
    using Config = GemvFP8PureConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK);

    size_t smem_size = K * sizeof(uint8_t);

    gemv_fp8_pure_fp8out_kernel<Config><<<grid, block, smem_size, stream>>>(
        A, B_nk, scale_A, scale_B, C, scale_C, K, N
    );

    return cudaGetLastError();
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// Extern C Interface
// ============================================================================

extern "C" {

/**
 * Pure FP8 GEMV: A[K](FP8) x B[N,K](FP8) -> C[N](BF16)
 *
 * @param A         [K] FP8 E4M3 activation vector
 * @param B_nk      [N, K] FP8 E4M3 weight matrix (row-major)
 * @param scale_A   [K/128] FP32 scales for A (blockwise)
 * @param scale_B   [N/128, K/128] FP32 scales for B (blockwise)
 * @param C         [N] BF16 output vector
 * @param K         Inner dimension
 * @param N         Output dimension
 * @param stream    CUDA stream
 */
cudaError_t pygpukit_gemv_fp8_fp8_bf16_sm120(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv::launch_gemv_fp8_pure(
        A, B_nk, scale_A, scale_B, C, K, N, stream
    );
}

/**
 * Pure FP8 GEMV with FP8 output: A[K](FP8) x B[N,K](FP8) -> C[N](FP8)
 */
cudaError_t pygpukit_gemv_fp8_fp8_fp8_sm120(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    uint8_t* C,
    float scale_C,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv::launch_gemv_fp8_pure_fp8out(
        A, B_nk, scale_A, scale_B, C, scale_C, K, N, stream
    );
}

/**
 * Check if pure FP8 GEMV is available (SM120+)
 */
bool pygpukit_gemv_fp8_fp8_sm120_available() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return false;

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    int sm = major * 10 + minor;
    return sm >= 100;  // SM100+ (Blackwell)
#else
    return false;
#endif
}

}  // extern "C"
