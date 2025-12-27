/**
 * Int4 GEMV Launch Functions (SM120)
 *
 * For M=1 decode in LLM inference with Int4 quantization.
 */

#include "int4_gemv.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_int4(
    const uint8_t* A,
    const uint8_t* B_nk,
    int32_t* C,
    int K,
    int N,
    float scale_A,
    float scale_B,
    cudaStream_t stream
) {
    using Config = GemvInt4Config;

    const int K_packed = K / 2;

    // Grid: each block handles WARPS_PER_BLOCK outputs
    dim3 block(Config::BLOCK_SIZE);  // 256 threads
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK);

    // Shared memory for A vector (packed)
    size_t smem_size = K_packed * sizeof(uint8_t);

    // Always use non-vectorized kernel for now (vectorized has a bug)
    // TODO: Fix vectorized kernel for K_packed >= 128
    gemv_int4_warp_reduce_kernel<Config><<<grid, block, smem_size, stream>>>(
        A, B_nk, C, K, N, scale_A, scale_B
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
 * Int4 x Int4 GEMV for M=1
 *
 * @param A         [K/2] packed Int4 activation vector
 * @param B_nk      [N, K/2] packed Int4 weights (row-major, transposed)
 * @param C         [N] Int32 output
 * @param K         Unpacked K dimension (must be even)
 * @param N         Output dimension
 * @param scale_A   Scale for A dequantization
 * @param scale_B   Scale for B dequantization
 * @param stream    CUDA stream
 */
cudaError_t pygpukit_gemv_int4_int4_int32_sm120(
    const uint8_t* A,
    const uint8_t* B_nk,
    int32_t* C,
    int K,
    int N,
    float scale_A,
    float scale_B,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv::launch_gemv_int4(
        A, B_nk, C, K, N, scale_A, scale_B, stream
    );
}

/**
 * Check if Int4 GEMV is available (SM120)
 */
bool pygpukit_int4_gemv_sm120_available() {
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
