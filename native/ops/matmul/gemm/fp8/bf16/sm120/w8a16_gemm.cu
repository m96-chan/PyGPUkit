/**
 * W8A16 GEMM for SM120 (Blackwell GeForce)
 *
 * FP8 Weight x BF16 Activation -> BF16 Output
 * - A: [M, K] BF16 activation (RowMajor)
 * - B: [K, N] FP8 E4M3 weight (RowMajor) + block-wise scale
 * - C: [M, N] BF16 output
 *
 * Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 * FP8 weights are dequantized on-the-fly during shared memory load.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace w8a16_gemm {

// Block tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// MMA tile dimensions (m16n8k16)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Warp configuration
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;

// Padding to avoid bank conflicts
constexpr int A_PAD = 8;
constexpr int B_PAD = 8;

// Block size for FP8 scaling (128x128)
constexpr int SCALE_BLOCK = 128;

// ============================================================================
// FP8 E4M3 Lookup Table (compile-time initialized)
// ============================================================================
__device__ __constant__ float FP8_E4M3_LUT[256] = {
    // exp=0 (subnormal): mant * 2^(-9), positive
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    // exp=1-15, positive (0x08-0x7F)
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 480.0f,
    // exp=0-15, negative (0x80-0xFF)
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -480.0f,
};

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
// W8A16 GEMM Kernel
// ============================================================================

__global__ void __launch_bounds__(256, 2)
w8a16_gemm_kernel(
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

    // Shared memory
    __shared__ __nv_bfloat16 smA[2][BM][BK + A_PAD];
    __shared__ __nv_bfloat16 smB[2][BK][BN + B_PAD];

    // Accumulators (FP32)
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // Fragment index mappings
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    // ====== Load A (BF16) via cp.async ======
    auto load_A_async = [&](int stage, int kt) {
        const int elems_per_thread = (BM * BK) / 256;  // 16
        const int bf16_per_load = 8;

        #pragma unroll
        for (int i = 0; i < elems_per_thread / bf16_per_load; ++i) {
            int elem_idx = tid * (elems_per_thread / bf16_per_load) + i;
            int row = (elem_idx * bf16_per_load) / BK;
            int col = (elem_idx * bf16_per_load) % BK;
            int gm = cta_m + row;
            int gk = kt * BK + col;
            if (gm < M && gk + 7 < K) {
                cp_async_16(&smA[stage][row][col], &A[gm * K + gk]);
            }
        }
    };

    // ====== Load B (FP8 -> BF16 with scale) ======
    auto load_B_dequant = [&](int stage, int kt) {
        // 256 threads, load BK*BN = 32*128 = 4096 elements
        // Each thread loads 16 FP8 bytes, dequantizes to BF16
        const int elems_per_thread = (BK * BN) / 256;  // 16

        #pragma unroll
        for (int i = 0; i < elems_per_thread; ++i) {
            int elem_idx = tid * elems_per_thread + i;
            int row = elem_idx / BN;  // k index within tile
            int col = elem_idx % BN;  // n index within tile
            int gk = kt * BK + row;
            int gn = cta_n + col;

            if (gk < K && gn < N) {
                // Load FP8 byte
                uint8_t fp8_val = B_fp8[gk * N + gn];

                // Dequantize via LUT
                float f32_val = FP8_E4M3_LUT[fp8_val];

                // Get scale factor for this block
                int scale_k = gk / SCALE_BLOCK;
                int scale_n = gn / SCALE_BLOCK;
                __nv_bfloat16 scale_bf16 = B_scale[scale_k * scale_stride_n + scale_n];
                float scale_f32 = __bfloat162float(scale_bf16);

                // Apply scale and convert to BF16
                __nv_bfloat16 bf16_val = f32_to_bf16(f32_val * scale_f32);

                smB[stage][row][col] = bf16_val;
            }
        }
    };

    // ====== Prologue ======
    load_A_async(0, 0);
    load_B_dequant(0, 0);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    // ====== Main loop ======
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile
        if (kt + 1 < num_k_tiles) {
            load_A_async(next, kt + 1);
            load_B_dequant(next, kt + 1);
        }
        cp_async_commit();

        // Process current tile
        #pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * MMA_M;

                // Load A fragment
                uint32_t a_frag[4];
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    int i0 = p * 2;
                    int i1 = p * 2 + 1;
                    int row0 = groupID + 8 * ((i0 / 2) % 2);
                    int col0 = tid_in_group * 2 + (i0 % 2) + 8 * (i0 / 4);
                    int row1 = groupID + 8 * ((i1 / 2) % 2);
                    int col1 = tid_in_group * 2 + (i1 % 2) + 8 * (i1 / 4);

                    __nv_bfloat16 h0 = smA[curr][tile_m + row0][kk + col0];
                    __nv_bfloat16 h1 = smA[curr][tile_m + row1][kk + col1];
                    a_frag[p] = bf16_to_u16(h0) | (uint32_t(bf16_to_u16(h1)) << 16);
                }

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * MMA_N;

                    // Load B fragment
                    uint32_t b_frag[2];
                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        int i0 = p * 2;
                        int i1 = p * 2 + 1;
                        int row0 = tid_in_group * 2 + (i0 % 2) + 8 * (i0 / 2);
                        int col0 = groupID;
                        int row1 = tid_in_group * 2 + (i1 % 2) + 8 * (i1 / 2);
                        int col1 = groupID;

                        __nv_bfloat16 h0 = smB[curr][kk + row0][tile_n + col0];
                        __nv_bfloat16 h1 = smB[curr][kk + row1][tile_n + col1];
                        b_frag[p] = bf16_to_u16(h0) | (uint32_t(bf16_to_u16(h1)) << 16);
                    }

                    // MMA: m16n8k16 BF16
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

        cp_async_wait_0();
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
            for (int i = 0; i < 4; ++i) {
                int row = groupID + 8 * (i / 2);
                int col = tid_in_group * 2 + (i % 2);
                int gm = tile_m + row;
                int gn = tile_n + col;

                if (gm < M && gn < N) {
                    C[gm * N + gn] = f32_to_bf16(acc[wm][wn][i]);
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

    w8a16_gemm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const uint8_t*>(B_fp8),
        reinterpret_cast<const __nv_bfloat16*>(B_scale),
        reinterpret_cast<__nv_bfloat16*>(C),
        M, N, K, scale_stride_n
    );

    return cudaGetLastError();
}
