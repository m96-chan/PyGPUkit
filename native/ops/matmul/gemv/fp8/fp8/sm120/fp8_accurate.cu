/**
 * Accurate FP8/FP8 GEMV Launch Functions (SM120) - Issue #123
 *
 * Target: <0.5% relative error (vs ~1-2% in fast version)
 * Trade-off: ~1.5-2x slower, 4x more scale memory
 */

#include "fp8_accurate.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_fp8_accurate(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    using Config = GemvFP8AccurateConfig;

    dim3 block(Config::BLOCK_SIZE);  // 256 threads
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK);

    // Shared memory for A (FP8 = 1 byte per element)
    size_t smem_size = K * sizeof(uint8_t);

    // Kernel selection based on K size:
    // - K >= 512: Use optimized kernel (double accumulator, vectorized loads)
    // - K < 512: Use Kahan summation kernel (simpler, stable)
    if (K >= 512) {
        gemv_fp8_accurate_opt_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, scale_A, scale_B, C, K, N
        );
    } else {
        gemv_fp8_accurate_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, scale_A, scale_B, C, K, N
        );
    }

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
 * Accurate FP8 GEMV: A[K](FP8) x B[N,K](FP8) -> C[N](BF16)
 *
 * Key differences from fast version:
 * 1. Smaller scale blocks: 32 elements (vs 128 in fast)
 * 2. Kahan summation or double accumulator for reduced error
 * 3. Target error: <0.5% (vs ~1-2% in fast)
 *
 * @param A         [K] FP8 E4M3 activation vector
 * @param B_nk      [N, K] FP8 E4M3 weight matrix (row-major)
 * @param scale_A   [K/32] FP32 scales for A (blockwise, 4x more than fast)
 * @param scale_B   [N/32, K/32] FP32 scales for B (blockwise, 16x more than fast)
 * @param C         [N] BF16 output vector
 * @param K         Inner dimension
 * @param N         Output dimension
 * @param stream    CUDA stream
 */
cudaError_t pygpukit_gemv_fp8_fp8_bf16_accurate_sm120(
    const uint8_t* A,
    const uint8_t* B_nk,
    const float* scale_A,
    const float* scale_B,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv::launch_gemv_fp8_accurate(
        A, B_nk, scale_A, scale_B, C, K, N, stream
    );
}

/**
 * Check if accurate FP8 GEMV is available (SM120+)
 */
bool pygpukit_gemv_fp8_fp8_accurate_sm120_available() {
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
