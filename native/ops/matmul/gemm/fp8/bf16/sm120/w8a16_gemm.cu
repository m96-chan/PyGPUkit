/**
 * W8A16 GEMM for SM120 (Blackwell GeForce)
 *
 * FP8 Weight x BF16 Activation -> BF16 Output
 * - A: [M, K] BF16 activation (RowMajor)
 * - B: [K, N] FP8 E4M3 weight (RowMajor) + block-wise scale
 * - C: [M, N] BF16 output
 *
 * Approach: Dequantize FP8 weights to BF16, then use BF16 TensorCore MMA (m16n8k16)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Include FP8 LUT from GEMV
#include "../../../../gemv/bf16/bf16/sm120/fp8.cuh"

namespace pygpukit {
namespace ops {
namespace w8a16_gemm {

// Block tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;  // BF16 MMA K=16, 2 MMAs per iteration

// MMA tile dimensions (m16n8k16 for BF16)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Warp configuration
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;

// Padding to avoid bank conflicts
constexpr int A_PAD = 8;  // BF16 padding
constexpr int B_PAD = 8;

// Block size for FP8 scaling (128x128)
constexpr int SCALE_BLOCK = 128;

// ============================================================================
// FP8 to Float Dequantization (using shared LUT from gemv)
// ============================================================================
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t fp8) {
    return pygpukit::ops::gemv::FP8_E4M3_LUT[fp8];
}

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem64; "
        "  cvta.to.shared.u64 smem64, %1; "
        "  cvt.u32.u64 %0, smem64; }"
        : "=r"(addr) : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t addr = smem_u32(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(addr), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;");
}

// FP32 to BF16 conversion
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float f) {
    return __float2bfloat16(f);
}

// BF16 to uint16 for packing
__device__ __forceinline__ uint16_t bf16_to_u16(__nv_bfloat16 b) {
    return *reinterpret_cast<uint16_t*>(&b);
}

// ============================================================================
// W8A16 GEMM Kernel with BF16 TensorCore (dequantize FP8 weights)
// ============================================================================

__global__ void __launch_bounds__(256, 2)
w8a16_gemm_kernel_bf16tc(
    const __nv_bfloat16* __restrict__ A,  // [M, K] BF16 activation
    const uint8_t* __restrict__ B_fp8,     // [K, N] FP8 weight
    const __nv_bfloat16* __restrict__ B_scale,  // [K/128, N/128] BF16 scale
    __nv_bfloat16* __restrict__ C,         // [M, N] BF16 output
    int M, int N, int K,
    int scale_stride_n  // N/128
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int cta_m = blockIdx.y * BM;
    const int cta_n = blockIdx.x * BN;

    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    const int warp_m = warp_row * (WARP_TILES_M * MMA_M);
    const int warp_n = warp_col * (WARP_TILES_N * MMA_N);

    // Shared memory: store A as BF16, B as BF16 (dequantized from FP8)
    __shared__ __nv_bfloat16 smA[2][BM][BK + A_PAD];  // [M, K] BF16
    __shared__ __nv_bfloat16 smB[2][BN][BK + B_PAD];  // [N, K] transposed, BF16

    // Accumulators (FP32)
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // Fragment index mappings for m16n8k16 BF16 MMA
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    // ====== Load A (BF16 direct) ======
    auto load_A = [&](int stage, int kt) {
        // 256 threads, load BM*BK = 128*32 = 4096 BF16 = 8192 bytes
        // Each thread loads 16 BF16 = 32 bytes (2 x uint4)
        const int rows_per_iter = 256 / 2;  // 128 rows per iteration
        const int bf16_per_thread = 16;

        int local_row = tid / 2;  // 0-127
        int local_col = (tid % 2) * bf16_per_thread;  // 0 or 16

        if (local_row < BM) {
            int gm = cta_m + local_row;
            int gk = kt * BK + local_col;

            if (gm < M && gk + 15 < K) {
                // Load 16 BF16 values (32 bytes)
                uint4 bf16_8_0 = *reinterpret_cast<const uint4*>(&A[gm * K + gk]);
                uint4 bf16_8_1 = *reinterpret_cast<const uint4*>(&A[gm * K + gk + 8]);
                *reinterpret_cast<uint4*>(&smA[stage][local_row][local_col]) = bf16_8_0;
                *reinterpret_cast<uint4*>(&smA[stage][local_row][local_col + 8]) = bf16_8_1;
            } else {
                // Boundary handling
                for (int i = 0; i < 16; ++i) {
                    if (gm < M && gk + i < K) {
                        smA[stage][local_row][local_col + i] = A[gm * K + gk + i];
                    } else {
                        smA[stage][local_row][local_col + i] = __float2bfloat16(0.0f);
                    }
                }
            }
        }
    };

    // ====== Load B (FP8 -> dequantize to BF16) ======
    auto load_B = [&](int stage, int kt) {
        // 256 threads, load BK*BN = 32*128 = 4096 FP8 bytes
        // Dequantize to 4096 BF16 values
        // Need to load B[K,N] and transpose to smB[N,K] for col-major MMA access

        // Each thread handles 16 FP8 values
        const int fp8_per_thread = 16;
        const int threads_per_k = 256 / BK;  // 8 threads per K row
        const int n_per_thread = fp8_per_thread;

        int k_local = tid / 8;  // 0-31
        int n_base = (tid % 8) * n_per_thread;  // 0, 16, 32, ..., 112
        int gk = kt * BK + k_local;

        // Calculate scale for this K block
        int scale_k = gk / SCALE_BLOCK;

        if (gk < K && n_base + 15 < BN) {
            // Vectorized load of 16 FP8 bytes
            uint4 fp8_16 = *reinterpret_cast<const uint4*>(&B_fp8[gk * N + cta_n + n_base]);
            const uint8_t* fp8_bytes = reinterpret_cast<const uint8_t*>(&fp8_16);

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int n_local = n_base + i;
                int gn = cta_n + n_local;
                int scale_n = gn / SCALE_BLOCK;
                float scale = __bfloat162float(B_scale[scale_k * scale_stride_n + scale_n]);
                float dequant = fp8_e4m3_to_float(fp8_bytes[i]) * scale;
                // Transpose: store to smB[N, K]
                smB[stage][n_local][k_local] = __float2bfloat16(dequant);
            }
        } else if (gk < K) {
            // Boundary handling
            for (int i = 0; i < 16; ++i) {
                int n_local = n_base + i;
                int gn = cta_n + n_local;
                if (gn < N) {
                    int scale_n = gn / SCALE_BLOCK;
                    float scale = __bfloat162float(B_scale[scale_k * scale_stride_n + scale_n]);
                    float dequant = fp8_e4m3_to_float(B_fp8[gk * N + gn]) * scale;
                    smB[stage][n_local][k_local] = __float2bfloat16(dequant);
                } else {
                    smB[stage][n_local][k_local] = __float2bfloat16(0.0f);
                }
            }
        }
    };

    // ====== Prologue ======
    load_A(0, 0);
    load_B(0, 0);
    __syncthreads();

    // ====== Main loop ======
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile
        if (kt + 1 < num_k_tiles) {
            load_A(next, kt + 1);
            load_B(next, kt + 1);
        }

        // Process current tile with BF16 MMA
        #pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * MMA_M;

                // Load A fragment for m16n8k16 BF16
                // A: 16x16, each thread holds 8 BF16 values (4 registers)
                uint32_t a_frag[4];
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    // Row: groupID + 8 * (p / 2)
                    // Col: tid_in_group * 2 + (p % 2) * 8
                    int row = groupID + 8 * (p >> 1);
                    int col = (tid_in_group << 1) + ((p & 1) << 3);

                    // Load 2 consecutive BF16 as uint32
                    a_frag[p] = *reinterpret_cast<const uint32_t*>(&smA[curr][tile_m + row][kk + col]);
                }

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * MMA_N;

                    // Load B fragment for m16n8k16 BF16
                    // smB is [N, K] layout (transposed)
                    // B fragment: 16x8 (col-major for MMA)
                    uint32_t b_frag[2];
                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        // k_offset: tid_in_group * 2 + p * 8
                        // n_offset: groupID (0-7)
                        int k_offset = (tid_in_group << 1) + (p << 3);
                        int n_offset = groupID;

                        // smB[N, K] layout: load 2 BF16 from K dimension
                        b_frag[p] = *reinterpret_cast<const uint32_t*>(
                            &smB[curr][tile_n + n_offset][kk + k_offset]);
                    }

                    // BF16 MMA: m16n8k16
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                          "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]),
                          "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1])
                    );
                }
            }
        }

        __syncthreads();
    }

    // ====== Epilogue: Store results ======
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * MMA_M;
            int tile_n = cta_n + warp_n + wn * MMA_N;

            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                int row = groupID + 8 * pair;
                int col = tid_in_group * 2;
                int gm = tile_m + row;
                int gn = tile_n + col;

                if (gm < M && gn + 1 < N) {
                    // Convert to BF16 and store
                    __nv_bfloat16 v0 = f32_to_bf16(acc[wm][wn][pair * 2]);
                    __nv_bfloat16 v1 = f32_to_bf16(acc[wm][wn][pair * 2 + 1]);
                    uint32_t packed = bf16_to_u16(v0) | (uint32_t(bf16_to_u16(v1)) << 16);
                    *reinterpret_cast<uint32_t*>(&C[gm * N + gn]) = packed;
                } else if (gm < M) {
                    if (gn < N) C[gm * N + gn] = f32_to_bf16(acc[wm][wn][pair * 2]);
                    if (gn + 1 < N) C[gm * N + gn + 1] = f32_to_bf16(acc[wm][wn][pair * 2 + 1]);
                }
            }
        }
    }
}

}  // namespace w8a16_gemm
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// C API
// ============================================================================

extern "C" cudaError_t pygpukit_w8a16_gemm_sm120(
    const void* A,        // [M, K] BF16
    const void* B_fp8,    // [K, N] uint8 FP8
    const void* B_scale,  // [K/128, N/128] BF16
    void* C,              // [M, N] BF16
    int M, int N, int K,
    int scale_stride_n,
    cudaStream_t stream
) {
    using namespace pygpukit::ops::w8a16_gemm;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(256);

    w8a16_gemm_kernel_bf16tc<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const uint8_t*>(B_fp8),
        reinterpret_cast<const __nv_bfloat16*>(B_scale),
        reinterpret_cast<__nv_bfloat16*>(C),
        M, N, K, scale_stride_n
    );

    return cudaGetLastError();
}
