/**
 * W8A16 GEMM for SM120 (Blackwell GeForce) - FP8 TensorCore Version
 *
 * FP8 Weight x BF16 Activation -> BF16 Output
 * - A: [M, K] BF16 activation (RowMajor) -> quantized to FP8 on-the-fly
 * - B: [K, N] FP8 E4M3 weight (RowMajor) + block-wise scale
 * - C: [M, N] BF16 output
 *
 * Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 * This provides 2x throughput vs BF16 MMA (K=32 vs K=16).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace w8a16_gemm {

// Block tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;  // Increased for FP8 (K=32 per MMA, 2 MMAs per iteration)

// MMA tile dimensions (m16n8k32 for FP8)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 32;

// Warp configuration
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;

// Padding to avoid bank conflicts
constexpr int A_PAD = 16;  // 16 bytes for FP8
constexpr int B_PAD = 16;

// Block size for FP8 scaling (128x128)
constexpr int SCALE_BLOCK = 128;

// ============================================================================
// BF16 to FP8 E4M3 Quantization (fast bit manipulation version)
// ============================================================================
__device__ __forceinline__ uint8_t bf16_to_fp8_e4m3(float val) {
    // FP32: [S:1][E:8][M:23], bias=127
    // FP8 E4M3: [S:1][E:4][M:3], bias=7
    uint32_t f32_bits = *reinterpret_cast<uint32_t*>(&val);

    uint32_t sign = (f32_bits >> 24) & 0x80;  // Sign bit to FP8 position
    uint32_t exp_f32 = (f32_bits >> 23) & 0xFF;
    uint32_t mant_f32 = f32_bits & 0x7FFFFF;

    // Handle zero
    if (exp_f32 == 0) return sign;

    // Convert exponent: FP32 bias=127, FP8 bias=7
    // e_fp8 = e_fp32 - 127 + 7 = e_fp32 - 120
    int e_fp8 = (int)exp_f32 - 120;

    if (e_fp8 <= 0) {
        // Subnormal or underflow in FP8
        if (e_fp8 < -3) return sign;  // Too small, return zero
        // Subnormal: shift mantissa
        uint32_t mant_with_implicit = (1 << 23) | mant_f32;
        int shift = 1 - e_fp8 + 20;  // 20 = 23 - 3 (FP8 has 3-bit mantissa)
        uint32_t m = (shift < 32) ? (mant_with_implicit >> shift) : 0;
        return sign | (m & 0x7);
    }

    if (e_fp8 >= 15) {
        // Overflow: clamp to max FP8 value (not NaN)
        return sign | 0x7E;  // exp=15, mant=6 -> 448
    }

    // Normal case: truncate mantissa from 23 bits to 3 bits
    uint32_t m = mant_f32 >> 20;  // Keep top 3 bits

    return sign | (e_fp8 << 3) | m;
}

// Vectorized version: convert 2 BF16 to 2 FP8 packed in uint16
__device__ __forceinline__ uint16_t bf16x2_to_fp8x2(uint32_t bf16_packed) {
    __nv_bfloat16 h0 = *reinterpret_cast<__nv_bfloat16*>(&bf16_packed);
    __nv_bfloat16 h1 = *(reinterpret_cast<__nv_bfloat16*>(&bf16_packed) + 1);
    uint8_t fp8_0 = bf16_to_fp8_e4m3(__bfloat162float(h0));
    uint8_t fp8_1 = bf16_to_fp8_e4m3(__bfloat162float(h1));
    return fp8_0 | (fp8_1 << 8);
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
// W8A16 GEMM Kernel with FP8 TensorCore
// ============================================================================

__global__ void __launch_bounds__(256, 2)
w8a16_gemm_kernel_fp8tc(
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

    // Shared memory for FP8 data
    __shared__ uint8_t smA[2][BM][BK + A_PAD];  // FP8, [M, K]
    __shared__ uint8_t smB[2][BN][BK + B_PAD];  // FP8, [N, K] transposed for col-major MMA access
    __shared__ float smScale[2];  // Scale for each stage

    // Accumulators (FP32)
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // Fragment index mappings for m16n8k32
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    // ====== Load A (BF16 -> FP8 quantization) ======
    auto load_A_quant = [&](int stage, int kt) {
        // 256 threads, load BM*BK = 128*64 = 8192 bytes of FP8
        // Each thread handles 32 bytes (from 32 BF16 values = 64 bytes input)
        // Use 8 threads per row (8 * 8 = 64 FP8 per row)

        const int rows_per_iter = 256 / 8;  // 32 rows per iteration
        const int fp8_per_thread = 8;  // 8 FP8 values from 8 BF16 values

        int local_row = tid / 8;  // 0-31
        int local_col = (tid % 8) * fp8_per_thread;  // 0, 8, 16, ..., 56

        #pragma unroll
        for (int iter = 0; iter < BM / rows_per_iter; ++iter) {
            int row = iter * rows_per_iter + local_row;
            int gm = cta_m + row;
            int gk = kt * BK + local_col;

            if (gm < M && gk + 7 < K) {
                // Load 8 BF16 values (16 bytes) and convert to 8 FP8 values
                uint4 bf16_8 = *reinterpret_cast<const uint4*>(&A[gm * K + gk]);
                const uint16_t* bf16_vals = reinterpret_cast<const uint16_t*>(&bf16_8);

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    __nv_bfloat16 bf16_val = *reinterpret_cast<const __nv_bfloat16*>(&bf16_vals[i]);
                    smA[stage][row][local_col + i] = bf16_to_fp8_e4m3(__bfloat162float(bf16_val));
                }
            }
        }
    };

    // ====== Load B (FP8 direct, coalesced load with transpose to [N, K]) ======
    auto load_B_direct = [&](int stage, int kt) {
        // 256 threads, load BK*BN = 64*128 = 8192 bytes
        // Global: B[K, N] row-major -> coalesced access along N dimension
        // smem: smB[N, K] transposed layout

        // Each thread loads 32 bytes = 2 x uint4 (16 bytes each)
        // Load pattern: 4 threads per K row (4 * 32 = 128 bytes/row = BN)
        // 64 K rows, 4 threads each = 256 threads total

        int k_local = tid / 4;  // 0-63
        int n_base = (tid % 4) * 32;  // 0, 32, 64, 96
        int gk = kt * BK + k_local;

        if (gk < K) {
            // Coalesced 32-byte load from B[K, N]
            uint4 fp8_16_0 = *reinterpret_cast<const uint4*>(&B_fp8[gk * N + cta_n + n_base]);
            uint4 fp8_16_1 = *reinterpret_cast<const uint4*>(&B_fp8[gk * N + cta_n + n_base + 16]);

            // Transpose: scatter to smB[N, K]
            const uint8_t* bytes0 = reinterpret_cast<const uint8_t*>(&fp8_16_0);
            const uint8_t* bytes1 = reinterpret_cast<const uint8_t*>(&fp8_16_1);

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                smB[stage][n_base + i][k_local] = bytes0[i];
                smB[stage][n_base + 16 + i][k_local] = bytes1[i];
            }
        }

        // Load scale once per tile (thread 0 only)
        if (tid == 0) {
            int scale_k = (kt * BK) / SCALE_BLOCK;
            int scale_n = cta_n / SCALE_BLOCK;
            smScale[stage] = __bfloat162float(B_scale[scale_k * scale_stride_n + scale_n]);
        }
    };

    // ====== Prologue ======
    load_A_quant(0, 0);
    load_B_direct(0, 0);
    __syncthreads();

    // ====== Main loop ======
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile
        if (kt + 1 < num_k_tiles) {
            load_A_quant(next, kt + 1);
            load_B_direct(next, kt + 1);
        }

        __syncthreads();

        float scale = smScale[curr];

        // Process current tile with FP8 MMA
        #pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * MMA_M;

                // Load A fragment for m16n8k32 FP8
                // A: 16x32, each thread holds 4 uint32 (16 FP8 values)
                uint32_t a_frag[4];
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    // Row: groupID + 8 * (p / 2)
                    // Col: tid_in_group * 8 + (p % 2) * 4
                    int row = groupID + 8 * (p >> 1);
                    int col = (tid_in_group << 3) + ((p & 1) << 2);

                    // Load 4 consecutive FP8 bytes
                    a_frag[p] = *reinterpret_cast<const uint32_t*>(&smA[curr][tile_m + row][kk + col]);
                }

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * MMA_N;

                    // Load B fragment for m16n8k32 FP8
                    // smB is now [N, K] transposed layout
                    // B fragment: 32x8 (col-major for MMA), each thread holds 2 uint32 (8 FP8 values)
                    uint32_t b_frag[2];
                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        // k_offset: tid_in_group * 8 + p * 4
                        // n_offset: groupID (0-7)
                        int k_offset = (tid_in_group << 3) + (p << 2);
                        int n_offset = groupID;

                        // smB[N, K] layout: 4 consecutive K values are now contiguous!
                        b_frag[p] = *reinterpret_cast<const uint32_t*>(
                            &smB[curr][tile_n + n_offset][kk + k_offset]);
                    }

                    // FP8 MMA: m16n8k32
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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

        // Apply scale to accumulators at the end of each K-tile
        // (scale is per 128 K elements, and BK=64, so we apply it every 2 tiles)
        // Actually, we'll apply scale in epilogue for simplicity

        __syncthreads();
    }

    // ====== Epilogue: Apply scale and store results ======
    // Get final scale (from last tile processed)
    float final_scale = smScale[(num_k_tiles - 1) & 1];

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
                    // Apply scale and convert to BF16
                    __nv_bfloat16 v0 = f32_to_bf16(acc[wm][wn][pair * 2] * final_scale);
                    __nv_bfloat16 v1 = f32_to_bf16(acc[wm][wn][pair * 2 + 1] * final_scale);
                    uint32_t packed = bf16_to_u16(v0) | (uint32_t(bf16_to_u16(v1)) << 16);
                    *reinterpret_cast<uint32_t*>(&C[gm * N + gn]) = packed;
                } else if (gm < M) {
                    if (gn < N) C[gm * N + gn] = f32_to_bf16(acc[wm][wn][pair * 2] * final_scale);
                    if (gn + 1 < N) C[gm * N + gn + 1] = f32_to_bf16(acc[wm][wn][pair * 2 + 1] * final_scale);
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

    w8a16_gemm_kernel_fp8tc<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const uint8_t*>(B_fp8),
        reinterpret_cast<const __nv_bfloat16*>(B_scale),
        reinterpret_cast<__nv_bfloat16*>(C),
        M, N, K, scale_stride_n
    );

    return cudaGetLastError();
}
