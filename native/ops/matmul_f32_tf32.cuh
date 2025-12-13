#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace pygpukit {
namespace ops {
namespace tf32 {

constexpr int BM = 128;      // tile M
constexpr int BN = 128;      // tile N
constexpr int BK = 32;       // tile K (must align with TF32 MMA)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int WARPS_M = 4;   // 4 warp rows  (4*32=128 rows)
constexpr int WARPS_N = 2;   // 2 warp cols  (2*64=128 cols)

constexpr int WARP_TILES_M = 4;
constexpr int WARP_TILES_N = 2;

constexpr int A_PAD = 8;
constexpr int B_PAD = 8;

// ============================================================
// cp.async helpers
// ============================================================

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    unsigned smem_u32;
    asm volatile(
        "{ .reg .u64 smem64;      \n"
        "  cvta.to.shared.u64 smem64, %1; \n"
        "  cvt.u32.u64 %0, smem64;       \n"
        "}" : "=r"(smem_u32) : "l"(smem)
    );
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(smem_u32), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;");
}

// ============================================================
// TensorCore fragments
// ============================================================

using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// ============================================================
// Main Kernel
// ============================================================

__global__ void __launch_bounds__(256, 2)
sgemm_tf32_kernel(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warpId = tid >> 5;
    const int laneId = tid & 31;

    // tile positions
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    // warp positioning
    const int warp_m = (warpId / WARPS_N) * (WMMA_M * WARP_TILES_M);
    const int warp_n = (warpId % WARPS_N) * (WMMA_N * WARP_TILES_N);

    // ========================================================
    // Shared memory: 2-stage pipeline
    // ========================================================
    extern __shared__ float smem[];

    float* As = smem;
    float* Bs = As + (2 * BM * (BK + A_PAD));

    // Layout:
    // A_smem[2][128][32+A_PAD]
    // B_smem[2][32][128+B_PAD]

    // ========================================================
    // Warp accumulators
    // ========================================================

    FragC c[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(c[i][j], 0.0f);

    const int num_tiles_k = K / BK;

    // ========================================================
    // Helper: cp.async loaders
    // ========================================================

    auto load_A = [&](int stage, int kt){
        int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 256;     // 256 threads × 8 = 2048 / tile
            int m = idx / 2;
            int k = (idx % 2) * 16;
            if (m < BM) {
                cp_async_16(&As[stage * BM*(BK+A_PAD) + m*(BK+A_PAD) + k],
                            &A[(block_m + m)*K + k0 + k]);
            }
        }
    };

    auto load_B = [&](int stage, int kt){
        int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < 8; i++){
            int idx = tid + i * 256;
            int k = idx / 4;
            int n = (idx % 4) * 16;   // float4 per thread
            if (k < BK && n < BN) {
                const float4* src = reinterpret_cast<const float4*>(&B[(k0+k)*N + block_n + n]);
                float4 v = *src;

                float* dst = &Bs[stage * BK*(BN+B_PAD) + k*(BN+B_PAD) + n];
                dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
            }
        }
    };

    // ========================================================
    // PROLOGUE (load 2 tiles)
    // ========================================================
    load_A(0, 0);
    load_B(0, 0);
    load_A(1, 1);
    load_B(1, 1);
    cp_async_commit();
    cp_async_wait1();
    __syncthreads();

    // ========================================================
    // MAIN LOOP (double-buffered)
    // ========================================================
    for (int kt = 0; kt < num_tiles_k; kt++){
        int curr = kt & 1;
        int next = curr ^ 1;

        if (kt + 2 < num_tiles_k){
            load_A(next, kt+2);
            load_B(next, kt+2);
            cp_async_commit();
        }

        // -------- Compute: 32×(16x16x8) = 4 MMAs per BK
        #pragma unroll
        for (int kstep = 0; kstep < BK; kstep += WMMA_K) {

            FragA a[WARP_TILES_M];
            FragB b[WARP_TILES_N];

            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++) {
                int offA = curr*BM*(BK+A_PAD)
                         + (warp_m + i*WMMA_M)*(BK+A_PAD)
                         + kstep;
                wmma::load_matrix_sync(a[i], &As[offA], BK + A_PAD);
            }

            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                int offB = curr*BK*(BN+B_PAD)
                         + (kstep)*(BN+B_PAD)
                         + warp_n + j*WMMA_N;
                wmma::load_matrix_sync(b[j], &Bs[offB], BN + B_PAD);
            }

            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++)
                for (int j = 0; j < WARP_TILES_N; j++)
                    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }

        cp_async_wait1();
        __syncthreads();
    }

    // ========================================================
    // Store C
    // ========================================================

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++){
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++){
            int row = block_m + warp_m + i*WMMA_M;
            int col = block_n + warp_n + j*WMMA_N;

            if (row < M && col < N) {
                wmma::store_matrix_sync(&C[row*N + col], c[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

} // namespace tf32
} // namespace ops
} // namespace pygpukit
