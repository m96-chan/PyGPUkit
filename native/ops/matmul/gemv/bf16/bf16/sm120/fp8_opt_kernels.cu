/**
 * Optimized FP8 GEMV Kernel Implementations
 */

#include "fp8_opt.cuh"

namespace pygpukit {
namespace ops {
namespace gemv {

// ============================================================================
// Launch Functions
// ============================================================================

cudaError_t launch_gemv_fp8_opt(
    const __nv_bfloat16* A,
    const uint8_t* B_nk,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    cudaStream_t stream
) {
    using Config = GemvFP8OptConfig;

    // Grid: each block handles WARPS_PER_BLOCK outputs
    dim3 block(Config::BLOCK_SIZE);  // 256 threads
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK);

    // Shared memory for A vector
    size_t smem_size = K * sizeof(__nv_bfloat16);

    // Use vectorized kernel for K >= 128
    if (K >= 128) {
        gemv_fp8_warp_reduce_vec4_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, B_scale, C, K, N
        );
    } else {
        gemv_fp8_warp_reduce_kernel<Config><<<grid, block, smem_size, stream>>>(
            A, B_nk, B_scale, C, K, N
        );
    }

    return cudaGetLastError();
}

cudaError_t launch_gemv_fp8_opt_batched(
    const __nv_bfloat16* A,
    const uint8_t* B_nk,
    const __nv_bfloat16* B_scale,
    __nv_bfloat16* C,
    int K,
    int N,
    int batch_count,
    cudaStream_t stream
) {
    using Config = GemvFP8OptConfig;

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((N + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK, batch_count);

    size_t smem_size = K * sizeof(__nv_bfloat16);

    gemv_fp8_warp_reduce_batched_kernel<Config><<<grid, block, smem_size, stream>>>(
        A, B_nk, B_scale, C, K, N, batch_count
    );

    return cudaGetLastError();
}

}  // namespace gemv
}  // namespace ops
}  // namespace pygpukit
