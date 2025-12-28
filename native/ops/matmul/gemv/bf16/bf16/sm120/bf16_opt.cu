/**
 * Optimized BF16 GEMV Launch Functions (SM120)
 */

#include "bf16_opt.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// Already defined in header as inline

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// Extern C Interface
// ============================================================================

extern "C" {

/**
 * Optimized BF16 GEMV: A[K] x B[N,K]^T -> C[N]
 *
 * Uses B[N,K] row-major layout for coalesced memory access.
 * Warp-level reduction over K dimension.
 *
 * @param A     [K] BF16 activation
 * @param B_nk  [N, K] BF16 weights (row-major)
 * @param C     [N] BF16 output
 * @param K     Inner dimension
 * @param N     Output dimension
 * @param stream CUDA stream
 */
cudaError_t pygpukit_gemv_bf16_opt_sm120(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B_nk,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    return pygpukit::ops::gemv::launch_gemv_bf16_opt(
        A, B_nk, C, K, N, stream
    );
}

/**
 * Check if optimized BF16 GEMV is available
 */
bool pygpukit_gemv_bf16_opt_sm120_available() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return false;

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    // SM80+ (Ampere and newer)
    return major * 10 + minor >= 80;
}

}  // extern "C"
