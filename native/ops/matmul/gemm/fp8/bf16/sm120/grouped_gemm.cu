// Grouped GEMM for MoE: FP8 weights x BF16 activations -> BF16 output
// Each expert has different M (number of tokens), same N and K
// Weights are stacked: [num_experts, N, K] in FP8 with block-wise scaling

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pygpukit {
namespace grouped_gemm {

// Block sizes for output tiles
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

// Threads per block
constexpr int THREADS = 256;

// FP8 block scaling parameters
constexpr int SCALE_BLOCK_H = 128;
constexpr int SCALE_BLOCK_W = 128;

// LUT for FP8 E4M3 -> BF16 conversion (256 entries)
__device__ __constant__ __nv_bfloat16 g_fp8_lut[256];

// Binary search to find expert index for a given row
__device__ __forceinline__ int find_expert(const int* expert_offsets, int num_experts, int row) {
    int lo = 0, hi = num_experts;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (expert_offsets[mid] <= row) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Grouped GEMM kernel
// A: [M_total, K] in BF16 (input tokens sorted by expert)
// B_stacked: [num_experts, N, K] in FP8 (stacked expert weights, row-major)
// B_scale_stacked: [num_experts, N/128, K/128] in BF16 (block-wise scales)
// C: [M_total, N] in BF16 (output)
// expert_offsets: [num_experts + 1] - cumulative token counts per expert
template <bool USE_LUT = true>
__global__ void grouped_gemm_fp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B_stacked,
    const __nv_bfloat16* __restrict__ B_scale_stacked,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    int M_total,
    int N,
    int K,
    int num_experts
) {
    // Each block handles one BLOCK_M x BLOCK_N tile of output
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;

    int row_start = block_m * BLOCK_M;
    int col_start = block_n * BLOCK_N;

    // Skip if this block is entirely out of bounds
    if (row_start >= M_total) return;

    // Find which expert this block belongs to
    int expert_id = find_expert(expert_offsets, num_experts, row_start);
    int expert_row_start = expert_offsets[expert_id];
    int expert_row_end = expert_offsets[expert_id + 1];

    // Skip if expert has no tokens (shouldn't happen but safety check)
    if (expert_row_start >= expert_row_end) return;

    // Calculate pointers for this expert's weights
    size_t weight_offset = (size_t)expert_id * N * K;
    int scale_n = (N + SCALE_BLOCK_H - 1) / SCALE_BLOCK_H;
    int scale_k = (K + SCALE_BLOCK_W - 1) / SCALE_BLOCK_W;
    size_t scale_offset = (size_t)expert_id * scale_n * scale_k;

    const uint8_t* B = B_stacked + weight_offset;
    const __nv_bfloat16* B_scale = B_scale_stacked + scale_offset;

    // Thread indices
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory for tiles
    __shared__ __nv_bfloat16 smem_A[BLOCK_M][BLOCK_K + 4];  // +4 for bank conflict
    __shared__ __nv_bfloat16 smem_B[BLOCK_K][BLOCK_N + 4];  // B is transposed in smem

    // Accumulator registers - each thread handles a 4x4 tile
    // We have 256 threads covering 64x64 = 4096 elements
    // 256 threads * 4 = 1024 elements per row pass, need 4 row groups
    float acc[4][4] = {0.0f};

    // Each thread covers a portion of the output tile
    // 256 threads -> 16x16 thread grid covering 64x64 tile (4x4 per thread)
    int thread_row = (tid / 16) * 4;  // 0, 4, 8, ... 60
    int thread_col = (tid % 16) * 4;  // 0, 4, 8, ... 60

    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Cooperative loading of A tile [BLOCK_M, BLOCK_K]
        // 256 threads loading 64*32 = 2048 elements = 8 elements per thread
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS) {
            int local_m = i / BLOCK_K;
            int local_k = i % BLOCK_K;
            int global_m = row_start + local_m;
            int global_k = k_tile + local_k;

            if (global_m < M_total && global_k < K) {
                smem_A[local_m][local_k] = A[global_m * K + global_k];
            } else {
                smem_A[local_m][local_k] = __float2bfloat16(0.0f);
            }
        }

        // Cooperative loading of B tile [BLOCK_K, BLOCK_N] with FP8->BF16 dequant
        // B is [N, K] row-major, we want B^T[K, N]
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += THREADS) {
            int local_k = i / BLOCK_N;
            int local_n = i % BLOCK_N;
            int global_k = k_tile + local_k;
            int global_n = col_start + local_n;

            if (global_k < K && global_n < N) {
                // B is stored as [N, K], so B[global_n, global_k]
                uint8_t fp8_val = B[global_n * K + global_k];

                // Get scale for this block
                int scale_row = global_n / SCALE_BLOCK_H;
                int scale_col = global_k / SCALE_BLOCK_W;
                __nv_bfloat16 scale = B_scale[scale_row * scale_k + scale_col];

                // Dequantize FP8 to BF16
                __nv_bfloat16 bf16_val;
                if constexpr (USE_LUT) {
                    bf16_val = __hmul(g_fp8_lut[fp8_val], scale);
                } else {
                    // Direct conversion using CUDA intrinsic
                    __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&fp8_val);
                    bf16_val = __hmul(__nv_bfloat16(fp8), scale);
                }

                smem_B[local_k][local_n] = bf16_val;
            } else {
                smem_B[local_k][local_n] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // Compute: each thread computes its 4x4 output tile
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            // Load A values for this thread's rows
            float a_vals[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_vals[i] = __bfloat162float(smem_A[thread_row + i][k]);
            }

            // Load B values for this thread's columns
            float b_vals[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                b_vals[j] = __bfloat162float(smem_B[k][thread_col + j]);
            }

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int global_m = row_start + thread_row + i;
        if (global_m < M_total && global_m < expert_row_end) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int global_n = col_start + thread_col + j;
                if (global_n < N) {
                    C[global_m * N + global_n] = __float2bfloat16(acc[i][j]);
                }
            }
        }
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

// Main entry point
extern "C" cudaError_t pygpukit_grouped_gemm_fp8_bf16(
    const void* A,           // [M_total, K] BF16
    const void* B_stacked,   // [num_experts, N, K] FP8
    const void* B_scale,     // [num_experts, N/128, K/128] BF16
    void* C,                 // [M_total, N] BF16
    const int* expert_offsets,  // [num_experts + 1]
    int M_total,
    int N,
    int K,
    int num_experts,
    cudaStream_t stream
) {
    using namespace pygpukit::grouped_gemm;

    if (M_total == 0) return cudaSuccess;

    // Grid: one block per output tile
    int grid_m = (M_total + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(grid_m, grid_n);
    dim3 block(THREADS);

    grouped_gemm_fp8_bf16_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const uint8_t*>(B_stacked),
        reinterpret_cast<const __nv_bfloat16*>(B_scale),
        reinterpret_cast<__nv_bfloat16*>(C),
        expert_offsets,
        M_total, N, K, num_experts
    );

    return cudaGetLastError();
}
