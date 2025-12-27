/**
 * Pure NVF4/NVF4/NVF4 GEMV Launch Functions (SM120)
 */

#include "nvf4_gemv.cuh"

namespace pygpukit {
namespace ops {
namespace gemv_nvf4_pure {

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_nvf4_pure(
    const uint8_t* A_data,
    const uint8_t* A_scale,
    const uint8_t* B_data,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    using Config = GemvNvf4PureConfig;

    dim3 block(Config::BLOCK_SIZE);  // 256 threads
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N);  // 1 thread = 1 output

    // Use optimized kernel for aligned K, basic kernel otherwise
    if (K % Config::SCALE_BLOCK_SIZE == 0 && K >= Config::SCALE_BLOCK_SIZE) {
        gemv_nvf4_pure_opt_kernel<Config><<<grid, block, 0, stream>>>(
            A_data, A_scale, B_data, B_scale, C, K, N
        );
    } else {
        gemv_nvf4_pure_kernel<Config><<<grid, block, 0, stream>>>(
            A_data, A_scale, B_data, B_scale, C, K, N
        );
    }

    return cudaGetLastError();
}

}  // namespace gemv_nvf4_pure
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// Extern C Interface
// ============================================================================

extern "C" {

/**
 * Pure NVF4 GEMV: A[K](NVF4) x B[K,N](NVF4) -> C[N](BF16)
 *
 * @param A_data   [K/2] packed NVF4 activation (2 values per byte)
 * @param A_scale  [K/32] UE4M3 scales for A (blockwise)
 * @param B_data   [N, K/2] packed NVF4 weight matrix (row-major, use quantize_bf16_to_nvf4_rowmajor)
 * @param B_scale  [N, K/32] UE4M3 scales for B (row-major)
 * @param C        [N] BF16 output vector
 * @param K        Inner dimension (must be even)
 * @param N        Output dimension
 * @param stream   CUDA stream
 */
cudaError_t pygpukit_gemv_nvf4_nvf4_bf16_sm120(
    const uint8_t* A_data,
    const uint8_t* A_scale,
    const uint8_t* B_data,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv_nvf4_pure::launch_gemv_nvf4_pure(
        A_data, A_scale, B_data, B_scale, C, K, N, stream
    );
}

/**
 * Check if pure NVF4 GEMV is available (SM120+)
 */
bool pygpukit_gemv_nvf4_nvf4_sm120_available() {
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
