// Grouped GEMM for MoE: FP8 weights x BF16 activations -> BF16 output
// Each row has an associated expert_id, weights are stacked per expert
// This version correctly handles rows belonging to different experts

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pygpukit {
namespace grouped_gemm {

// LUT for FP8 E4M3 -> BF16 conversion (256 entries)
__device__ __constant__ __nv_bfloat16 g_fp8_lut[256];

// FP8 block scaling parameters
constexpr int SCALE_BLOCK_H = 128;
constexpr int SCALE_BLOCK_W = 128;

// Simple per-row GEMM kernel
// Each thread computes one output element
// A: [M, K] BF16, B_stacked: [num_experts, N, K] FP8, C: [M, N] BF16
// row_expert_ids: [M] int32 - which expert each row uses
__global__ void grouped_gemm_simple_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B_stacked,
    const __nv_bfloat16* __restrict__ B_scale_stacked,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ row_expert_ids,
    int M,
    int N,
    int K
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Get expert ID for this row
    int expert_id = row_expert_ids[row];

    // Calculate pointers for this expert's weights
    size_t weight_offset = (size_t)expert_id * N * K;
    int scale_n = (N + SCALE_BLOCK_H - 1) / SCALE_BLOCK_H;
    int scale_k = (K + SCALE_BLOCK_W - 1) / SCALE_BLOCK_W;
    size_t scale_offset = (size_t)expert_id * scale_n * scale_k;

    const uint8_t* B = B_stacked + weight_offset;
    const __nv_bfloat16* B_scale = B_scale_stacked + scale_offset;

    // Compute dot product for C[row, col]
    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        // Load A[row, k]
        float a_val = __bfloat162float(A[row * K + k]);

        // Load and dequantize B[col, k] (B is [N, K])
        uint8_t fp8_val = B[col * K + k];
        int scale_row = col / SCALE_BLOCK_H;
        int scale_col = k / SCALE_BLOCK_W;
        __nv_bfloat16 scale = B_scale[scale_row * scale_k + scale_col];
        float b_val = __bfloat162float(__hmul(g_fp8_lut[fp8_val], scale));

        acc += a_val * b_val;
    }

    C[row * N + col] = __float2bfloat16(acc);
}

// Optimized tiled kernel with shared memory
// Block: (TILE_N threads), Grid: (M, ceil(N/TILE_N))
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;

__global__ void grouped_gemm_tiled_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B_stacked,
    const __nv_bfloat16* __restrict__ B_scale_stacked,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ row_expert_ids,
    int M,
    int N,
    int K
) {
    int row = blockIdx.x;
    int col_base = blockIdx.y * TILE_N;
    int tid = threadIdx.x;
    int col = col_base + tid;

    if (row >= M) return;

    // Get expert ID for this row
    int expert_id = row_expert_ids[row];

    // Calculate pointers for this expert's weights
    size_t weight_offset = (size_t)expert_id * N * K;
    int scale_n = (N + SCALE_BLOCK_H - 1) / SCALE_BLOCK_H;
    int scale_k = (K + SCALE_BLOCK_W - 1) / SCALE_BLOCK_W;
    size_t scale_offset = (size_t)expert_id * scale_n * scale_k;

    const uint8_t* B = B_stacked + weight_offset;
    const __nv_bfloat16* B_scale = B_scale_stacked + scale_offset;

    // Shared memory for A tile (one row, TILE_K columns)
    __shared__ float smem_A[TILE_K];

    float acc = 0.0f;

    // Loop over K in tiles
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        // Cooperative load of A[row, k_base:k_base+TILE_K]
        if (tid < TILE_K && k_base + tid < K) {
            smem_A[tid] = __bfloat162float(A[row * K + k_base + tid]);
        }
        __syncthreads();

        // Compute partial dot product
        if (col < N) {
            #pragma unroll 8
            for (int k = 0; k < TILE_K && k_base + k < K; ++k) {
                float a_val = smem_A[k];

                // Load and dequantize B[col, k_base + k]
                int global_k = k_base + k;
                uint8_t fp8_val = B[col * K + global_k];
                int scale_row = col / SCALE_BLOCK_H;
                int scale_col = global_k / SCALE_BLOCK_W;
                __nv_bfloat16 scale = B_scale[scale_row * scale_k + scale_col];
                float b_val = __bfloat162float(__hmul(g_fp8_lut[fp8_val], scale));

                acc += a_val * b_val;
            }
        }
        __syncthreads();
    }

    if (col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

}  // namespace grouped_gemm
}  // namespace pygpukit

// Initialize FP8 LUT
extern "C" cudaError_t pygpukit_grouped_gemm_init_lut() {
    __nv_bfloat16 h_lut[256];
    for (int i = 0; i < 256; ++i) {
        __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&i);
        h_lut[i] = __nv_bfloat16(fp8);
    }
    return cudaMemcpyToSymbol(
        pygpukit::grouped_gemm::g_fp8_lut, h_lut, 256 * sizeof(__nv_bfloat16)
    );
}

// Grouped GEMM: row_expert_ids per-row expert assignment
extern "C" cudaError_t pygpukit_grouped_gemm_fp8_bf16(
    const void* A,              // [M, K] BF16
    const void* B_stacked,      // [num_experts, N, K] FP8
    const void* B_scale,        // [num_experts, N/128, K/128] BF16
    void* C,                    // [M, N] BF16
    const int* row_expert_ids,  // [M] int32 - expert ID per row
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    using namespace pygpukit::grouped_gemm;

    if (M == 0) return cudaSuccess;

    // Use tiled kernel for better performance
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(TILE_N);

    grouped_gemm_tiled_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const uint8_t*>(B_stacked),
        reinterpret_cast<const __nv_bfloat16*>(B_scale),
        reinterpret_cast<__nv_bfloat16*>(C),
        row_expert_ids,
        M, N, K
    );

    return cudaGetLastError();
}
