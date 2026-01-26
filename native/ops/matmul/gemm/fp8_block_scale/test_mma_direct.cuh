/**
 * Direct MMA test to isolate FP8 block-scale MMA behavior.
 *
 * Test 1: All elements = 1.0, verify we get non-zero output
 * Test 2: Sparse test to verify fragment layout
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if __CUDACC_VER_MAJOR__ >= 13

namespace pygpukit {
namespace ops {
namespace matmul {
namespace fp8_mma_test {

// The MMA function
__device__ __forceinline__ void mma_fp8_16x8x32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint8_t sfa,
    uint8_t sfb
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidB = 0;
    static constexpr uint16_t tidB = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(uint32_t(sfa)), "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb)), "h"(bidB), "h"(tidB)
    );
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

/**
 * Test 1: All ones - verify MMA produces non-zero output
 *
 * If A = all 1.0 (16x32) and B = all 1.0 (32x8), then
 * C[m, n] = sum_k A[m, k] * B[k, n] = sum_k 1.0 * 1.0 = 32.0 for all m, n
 *
 * With scale = 1.0, output should be 32.0 everywhere.
 */
__global__ void test_mma_all_ones_kernel(
    float* __restrict__ output   // [32 lanes, 4 values]
) {
    int lane_id = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    // Scale = 1.0 (UE8M0 = 127)
    uint8_t scale_ue8m0 = 127;

    // FP8 E4M3 for 1.0 = 0x38
    uint8_t fp8_one = 0x38;

    // Fill all A registers with 1.0
    // a_frag[4] = 16 bytes per thread = 16 FP8 values, all 1.0
    uint32_t a_frag[4];
    uint32_t one_reg = fp8_one | (fp8_one << 8) | (fp8_one << 16) | (fp8_one << 24);
    a_frag[0] = one_reg;
    a_frag[1] = one_reg;
    a_frag[2] = one_reg;
    a_frag[3] = one_reg;

    // Fill all B registers with 1.0
    // b_frag[2] = 8 bytes per thread = 8 FP8 values, all 1.0
    uint32_t b_frag[2];
    b_frag[0] = one_reg;
    b_frag[1] = one_reg;

    // Execute MMA
    float d_frag[4] = {0, 0, 0, 0};
    mma_fp8_16x8x32(
        d_frag[0], d_frag[1], d_frag[2], d_frag[3],
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag[0], b_frag[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        scale_ue8m0, scale_ue8m0
    );

    // Store results
    for (int v = 0; v < 4; ++v) {
        output[lane_id * 4 + v] = d_frag[v];
    }
}

/**
 * Test 2: Sparse inputs using NAIVE layout (not CuTe)
 *
 * Try simple sequential filling without CuTe layout formulas.
 */
__global__ void test_mma_sequential_kernel(
    float* __restrict__ output,
    uint32_t* __restrict__ debug
) {
    int lane_id = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    uint8_t scale_ue8m0 = 127;
    uint8_t fp8_one = 0x38;
    uint8_t fp8_two = 0x40;

    // Just set all fragments to 1.0 first
    uint32_t one_reg = fp8_one | (fp8_one << 8) | (fp8_one << 16) | (fp8_one << 24);

    uint32_t a_frag[4] = {one_reg, one_reg, one_reg, one_reg};
    uint32_t b_frag[2] = {one_reg, one_reg};

    // Modify B for lane 0 only to have 2.0 in first byte
    // This should make column 0 different from column 1 (if layout is simple)
    if (lane_id == 0) {
        b_frag[0] = fp8_two | (fp8_one << 8) | (fp8_one << 16) | (fp8_one << 24);
        debug[0] = b_frag[0];
        debug[1] = b_frag[1];
    }

    // Execute MMA
    float d_frag[4] = {0, 0, 0, 0};
    mma_fp8_16x8x32(
        d_frag[0], d_frag[1], d_frag[2], d_frag[3],
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag[0], b_frag[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        scale_ue8m0, scale_ue8m0
    );

    // Store results
    for (int v = 0; v < 4; ++v) {
        output[lane_id * 4 + v] = d_frag[v];
    }
}

/**
 * Test 3: Test B fragment layout mapping with FIXED formula
 *
 * PROBLEM IDENTIFIED:
 * - C fragment layout: lane L outputs C columns (L%4)*2 and (L%4)*2+1
 * - Old B loading: grouped by t0=lane/8, loading B cols (t0, t0+4)
 * - This mismatch caused wrong B columns for each C column!
 *
 * NEW FORMULA:
 * - Group B loading by (lane%4) to match C layout
 * - n_idx = (lane%4)*2 + r   -> B columns 2*(lane%4) and 2*(lane%4)+1
 * - k_idx = (lane/4)*4 + b   -> 8 lanes (same lane%4) cover all 32 k values
 */
__global__ void test_mma_fa3_formula_kernel(
    float* __restrict__ output,
    float* __restrict__ expected
) {
    int lane_id = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    uint8_t scale_ue8m0 = 127;  // Scale = 1.0

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 32;
    constexpr int HEAD_DIM = 32;

    __shared__ uint8_t smem_Q[MMA_M * HEAD_DIM];
    __shared__ uint8_t smem_K[MMA_N * HEAD_DIM];

    uint8_t fp8_one = 0x38;
    uint8_t fp8_two = 0x40;
    uint8_t fp8_1_5 = 0x3C;

    if (lane_id == 0) {
        for (int i = 0; i < MMA_M * HEAD_DIM; ++i) smem_Q[i] = fp8_one;

        for (int n = 0; n < MMA_N; ++n) {
            uint8_t val;
            switch(n) {
                case 0: val = fp8_two; break;      // 2.0 -> C[m,0] = 64
                case 1: val = fp8_1_5; break;      // 1.5 -> C[m,1] = 48
                case 2: val = fp8_one; break;      // 1.0 -> C[m,2] = 32
                case 3: val = 0x34; break;         // 0.75 -> C[m,3] = 24
                case 4: val = 0x30; break;         // 0.5 -> C[m,4] = 16
                case 5: val = 0x2C; break;         // 0.375 -> C[m,5] = 12
                case 6: val = 0x28; break;         // 0.25 -> C[m,6] = 8
                case 7: val = 0x24; break;         // 0.1875 -> C[m,7] = 6
                default: val = fp8_one; break;
            }
            for (int k = 0; k < HEAD_DIM; ++k) {
                smem_K[n * HEAD_DIM + k] = val;
            }
        }
    }
    __syncthreads();

    int t0 = lane_id / 8;
    int t1 = lane_id % 8;

    // Load A fragment using the FA3 formula (unchanged)
    uint32_t a_frag[4];
    const uint8_t* A_ptr = smem_Q;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int v1 = r % 2;
        int v2 = r / 2;
        uint8_t bytes[4];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int flat = 64 * t0 + t1 + 16 * b + 8 * v1 + 256 * v2;
            int row = flat / MMA_K;
            int col = flat % MMA_K;
            bytes[b] = A_ptr[row * HEAD_DIM + col];
        }
        a_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }

    // Load B fragment - CORRECT FORMULA based on Test 4 routing discovery
    // Key insight from Test 4: n_idx = lane_id / 4 (NOT lane_id / 8 as CuTe suggests)
    // Each group of 4 lanes loads the SAME B column but different k values
    // 4 lanes * 8 bytes = 32 k values per B column ✓
    uint32_t b_frag[2];
    const uint8_t* B_ptr = smem_K;

    int n_idx = lane_id / 4;          // B column: 0-7 (one column per 4 lanes)
    int k_base = (lane_id % 4) * 8;   // Base k: each of 4 lanes handles 8 consecutive k values

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        uint8_t bytes[4];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int k_idx = k_base + r * 4 + b;  // k = 0-7 for lane%4=0, 8-15 for lane%4=1, etc.
            bytes[b] = B_ptr[n_idx * HEAD_DIM + k_idx];
        }
        b_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }

    // Execute MMA
    float d_frag[4] = {0, 0, 0, 0};
    mma_fp8_16x8x32(
        d_frag[0], d_frag[1], d_frag[2], d_frag[3],
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag[0], b_frag[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        scale_ue8m0, scale_ue8m0
    );

    // Store results
    for (int v = 0; v < 4; ++v) {
        output[lane_id * 4 + v] = d_frag[v];
    }

    // C fragment layout
    int row0 = lane_id / 4;
    int row1 = lane_id / 4 + 8;
    int col0 = (lane_id % 4) * 2;
    int col1 = (lane_id % 4) * 2 + 1;

    auto expected_val = [](int m, int n) -> float {
        (void)m;
        switch(n) {
            case 0: return 64.0f;
            case 1: return 48.0f;
            case 2: return 32.0f;
            case 3: return 24.0f;
            case 4: return 16.0f;
            case 5: return 12.0f;
            case 6: return 8.0f;
            case 7: return 6.0f;
            default: return 0.0f;
        }
    };

    expected[lane_id * 4 + 0] = expected_val(row0, col0);
    expected[lane_id * 4 + 1] = expected_val(row0, col1);
    expected[lane_id * 4 + 2] = expected_val(row1, col0);
    expected[lane_id * 4 + 3] = expected_val(row1, col1);
}

/**
 * Test 4: Empirical B-to-C routing discovery
 *
 * Each lane sets a unique value in b_frag[0] byte 0.
 * We then observe which unique values appear in which C outputs to
 * determine the MMA's internal routing.
 */
__global__ void test_mma_b_routing_kernel(
    float* __restrict__ output,
    float* __restrict__ debug
) {
    int lane_id = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    uint8_t scale_ue8m0 = 127;  // Scale = 1.0
    uint8_t fp8_one = 0x38;     // 1.0

    // Each lane has a unique value in b_frag[0] byte 0
    // Use values that are distinguishable: 1.0 + lane_id * 0.0625
    // FP8 E4M3 encoding: 0x38 = 1.0, each +0x01 is approximately +0.0625 at this scale
    // Actually, let's use values that are powers of 2 for cleaner math
    // Use: lane 0 = 2.0 (0x40), lane 1 = 1.0 (0x38), others = 0.5 (0x30)
    // This way we can identify which lanes' B values contribute to which C elements

    // A = all 1.0
    uint32_t one_reg = fp8_one | (fp8_one << 8) | (fp8_one << 16) | (fp8_one << 24);
    uint32_t a_frag[4] = {one_reg, one_reg, one_reg, one_reg};

    // B = all 1.0 initially
    uint32_t b_frag[2] = {one_reg, one_reg};

    // Now set unique markers for specific lanes
    // We want to find: which lane's b_frag[r] byte b contributes to which C[m, n]

    // Test: Set lane 0's b_frag[0] byte 0 to 2.0 (adds +1 to the sum)
    // Previous test showed this affects C column 0
    //
    // Now also set lane 1's b_frag[0] byte 0 to 1.5 (0x3C) to see if it affects C column 0 or a different column
    if (lane_id == 0) {
        uint8_t marker = 0x40;  // 2.0
        b_frag[0] = (b_frag[0] & 0xFFFFFF00) | marker;
    }
    if (lane_id == 1) {
        uint8_t marker = 0x3C;  // 1.5
        b_frag[0] = (b_frag[0] & 0xFFFFFF00) | marker;
    }
    if (lane_id == 2) {
        uint8_t marker = 0x34;  // 0.75
        b_frag[0] = (b_frag[0] & 0xFFFFFF00) | marker;
    }
    if (lane_id == 3) {
        uint8_t marker = 0x30;  // 0.5
        b_frag[0] = (b_frag[0] & 0xFFFFFF00) | marker;
    }
    // Lanes 4-7: modify b_frag[0] byte 1 instead of byte 0
    if (lane_id == 4) {
        uint8_t marker = 0x40;  // 2.0
        b_frag[0] = (b_frag[0] & 0xFFFF00FF) | (uint32_t(marker) << 8);
    }
    if (lane_id == 5) {
        uint8_t marker = 0x3C;  // 1.5
        b_frag[0] = (b_frag[0] & 0xFFFF00FF) | (uint32_t(marker) << 8);
    }
    // Lanes 8-15: modify b_frag[1] byte 0
    if (lane_id == 8) {
        uint8_t marker = 0x40;  // 2.0
        b_frag[1] = (b_frag[1] & 0xFFFFFF00) | marker;
    }
    if (lane_id == 9) {
        uint8_t marker = 0x3C;  // 1.5
        b_frag[1] = (b_frag[1] & 0xFFFFFF00) | marker;
    }

    // Execute MMA
    float d_frag[4] = {0, 0, 0, 0};
    mma_fp8_16x8x32(
        d_frag[0], d_frag[1], d_frag[2], d_frag[3],
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag[0], b_frag[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        scale_ue8m0, scale_ue8m0
    );

    // Store results
    for (int v = 0; v < 4; ++v) {
        output[lane_id * 4 + v] = d_frag[v];
    }

    // Store B fragments for debugging
    debug[lane_id * 2 + 0] = __uint_as_float(b_frag[0]);
    debug[lane_id * 2 + 1] = __uint_as_float(b_frag[1]);
}

// Forward declaration
inline void test_mma_full_pipeline();

/**
 * Run the direct MMA tests and analyze results.
 */
inline void test_mma_direct() {
    printf("=== Direct FP8 Block-Scale MMA Test ===\n\n");

    float* d_output;
    uint32_t* d_debug;
    cudaMalloc(&d_output, 32 * 4 * sizeof(float));
    cudaMalloc(&d_debug, 20 * sizeof(uint32_t));
    cudaMemset(d_debug, 0, 20 * sizeof(uint32_t));

    // Test 1: All ones
    printf("=== Test 1: All 1.0 inputs ===\n");
    printf("Expected: All outputs = 32.0 (sum of 32 products of 1.0*1.0)\n\n");

    test_mma_all_ones_kernel<<<1, 32>>>(d_output);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_debug);
        return;
    }

    float h_output[128];
    cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few lanes
    printf("Lane | d[0]    | d[1]    | d[2]    | d[3]\n");
    printf("-----+---------+---------+---------+---------\n");
    for (int lane = 0; lane < 8; ++lane) {
        printf("%4d | %7.2f | %7.2f | %7.2f | %7.2f\n",
               lane,
               h_output[lane * 4 + 0],
               h_output[lane * 4 + 1],
               h_output[lane * 4 + 2],
               h_output[lane * 4 + 3]);
    }

    // Check if any output is non-zero
    float max_val = 0.0f;
    for (int i = 0; i < 128; ++i) {
        if (fabsf(h_output[i]) > max_val) max_val = fabsf(h_output[i]);
    }
    printf("\nMax output value: %.4f\n", max_val);
    if (max_val < 0.01f) {
        printf("WARNING: All outputs are zero! MMA instruction may not be working.\n");
    } else if (fabsf(max_val - 32.0f) < 1.0f) {
        printf("SUCCESS: Output ~32.0 as expected!\n");
    } else {
        printf("UNEXPECTED: Output is non-zero but not 32.0\n");
    }

    // Test 2: Sequential with one modified element
    printf("\n=== Test 2: Modified B[0] for lane 0 ===\n");
    printf("All A=1.0, B=1.0 except lane 0's B has 2.0 in first byte\n\n");

    test_mma_sequential_kernel<<<1, 32>>>(d_output, d_debug);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_debug);
        return;
    }

    uint32_t h_debug[20];
    cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_debug, d_debug, 20 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Lane 0 B fragments: 0x%08x 0x%08x\n", h_debug[0], h_debug[1]);

    printf("\nLane | d[0]    | d[1]    | d[2]    | d[3]\n");
    printf("-----+---------+---------+---------+---------\n");
    for (int lane = 0; lane < 8; ++lane) {
        printf("%4d | %7.2f | %7.2f | %7.2f | %7.2f\n",
               lane,
               h_output[lane * 4 + 0],
               h_output[lane * 4 + 1],
               h_output[lane * 4 + 2],
               h_output[lane * 4 + 3]);
    }

    // Check for differences between columns
    float d0_sum = 0, d1_sum = 0;
    for (int lane = 0; lane < 32; ++lane) {
        d0_sum += h_output[lane * 4 + 0];
        d1_sum += h_output[lane * 4 + 1];
    }
    printf("\nSum of d[0] across all lanes: %.2f\n", d0_sum);
    printf("Sum of d[1] across all lanes: %.2f\n", d1_sum);
    if (fabsf(d0_sum - d1_sum) > 0.5f) {
        printf("Columns have different sums - fragment layout affects output!\n");
    } else {
        printf("Columns have same sums - may need different test to see layout effect.\n");
    }

    // Test 3: FA3 formula verification
    printf("\n=== Test 3: FA3 Fragment Loading Formula ===\n");
    printf("A: 16x32, B: 8x32 (transposed to 32x8)\n");
    printf("A[0,0] = 2.0, B[0,0] = 2.0, rest = 1.0\n");
    printf("Expected: C[0,0]=35, C[0,n>0]=33, C[m>0,0]=33, else=32\n\n");

    float* d_expected;
    cudaMalloc(&d_expected, 32 * 4 * sizeof(float));

    test_mma_fa3_formula_kernel<<<1, 32>>>(d_output, d_expected);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_debug);
        cudaFree(d_expected);
        return;
    }

    float h_expected[128];
    cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expected, d_expected, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Lane | d[0] exp | d[1] exp | d[2] exp | d[3] exp\n");
    printf("-----+----------+----------+----------+----------\n");
    for (int lane = 0; lane < 8; ++lane) {
        printf("%4d | %4.0f %4.0f | %4.0f %4.0f | %4.0f %4.0f | %4.0f %4.0f\n",
               lane,
               h_output[lane * 4 + 0], h_expected[lane * 4 + 0],
               h_output[lane * 4 + 1], h_expected[lane * 4 + 1],
               h_output[lane * 4 + 2], h_expected[lane * 4 + 2],
               h_output[lane * 4 + 3], h_expected[lane * 4 + 3]);
    }

    // Check correctness
    int num_errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < 128; ++i) {
        float err_val = fabsf(h_output[i] - h_expected[i]);
        if (err_val > 0.01f) {
            num_errors++;
            if (err_val > max_error) max_error = err_val;
        }
    }
    if (num_errors == 0) {
        printf("\nSUCCESS: FA3 formula produces correct results!\n");
    } else {
        printf("\nFAILURE: %d mismatches, max error = %.2f\n", num_errors, max_error);
        printf("The FA3 fragment loading formula is INCORRECT.\n");
    }

    // Test 4: B-to-C routing discovery
    printf("\n=== Test 4: B-to-C Routing Discovery ===\n");
    printf("Lanes 0-3: modify b_frag[0] byte 0 to 2.0, 1.5, 0.75, 0.5\n");
    printf("Lanes 4-5: modify b_frag[0] byte 1 to 2.0, 1.5\n");
    printf("Lanes 8-9: modify b_frag[1] byte 0 to 2.0, 1.5\n");
    printf("Base value: all 1.0, so sum = 32.0\n");
    printf("If a lane's marker affects output, it adds +1 (2.0-1.0), +0.5 (1.5-1.0), etc.\n\n");

    float* d_debug_f;
    cudaMalloc(&d_debug_f, 32 * 2 * sizeof(float));

    test_mma_b_routing_kernel<<<1, 32>>>(d_output, d_debug_f);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_debug);
        cudaFree(d_expected);
        cudaFree(d_debug_f);
        return;
    }

    cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results for all 32 lanes
    printf("Lane | d[0]    | d[1]    | d[2]    | d[3]   | C cols\n");
    printf("-----+---------+---------+---------+--------+--------\n");
    for (int lane = 0; lane < 32; ++lane) {
        int col0 = (lane % 4) * 2;
        int col1 = col0 + 1;
        printf("%4d | %7.2f | %7.2f | %7.2f | %7.2f | %d,%d\n",
               lane,
               h_output[lane * 4 + 0],
               h_output[lane * 4 + 1],
               h_output[lane * 4 + 2],
               h_output[lane * 4 + 3],
               col0, col1);
    }

    // Analysis: Look for patterns
    printf("\n=== Analysis ===\n");
    printf("If d[0] != 32, some marker affected it. Check which lane's marker.\n");
    printf("Expected: lane 0's marker (2.0) should add +1 to C column 0\n");
    printf("          lane 1's marker (1.5) should add +0.5 to C column 0 or 2\n");

    // Check which C columns were affected
    for (int col = 0; col < 8; ++col) {
        float col_sum = 0.0f;
        int count = 0;
        // Find all lanes that output this column
        for (int lane = 0; lane < 32; ++lane) {
            int col0 = (lane % 4) * 2;
            int col1 = col0 + 1;
            if (col0 == col) {
                col_sum += h_output[lane * 4 + 0] + h_output[lane * 4 + 2];
                count += 2;
            }
            if (col1 == col) {
                col_sum += h_output[lane * 4 + 1] + h_output[lane * 4 + 3];
                count += 2;
            }
        }
        float avg = col_sum / count;
        if (fabsf(avg - 32.0f) > 0.01f) {
            printf("C column %d: avg = %.2f (affected by some marker)\n", col, avg);
        }
    }

    cudaFree(d_output);
    cudaFree(d_debug);
    cudaFree(d_expected);
    cudaFree(d_debug_f);

    // Run extended pipeline tests
    test_mma_full_pipeline();
}

/**
 * Test 5: Full quantization + scale pipeline validation
 *
 * This test validates the complete FP8 pipeline:
 * 1. BF16 → FP8 quantization with per-head scale
 * 2. MMA with computed scales
 * 3. Scale application (should recover original values)
 *
 * Uses small known values to trace through the math.
 */
__global__ void test_mma_full_pipeline_kernel(
    float* __restrict__ output,
    float* __restrict__ debug
) {
    int lane_id = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 32;
    constexpr int HEAD_DIM = 32;

    // Shared memory for Q and K in FP8 format
    __shared__ uint8_t smem_Q[MMA_M * HEAD_DIM];
    __shared__ uint8_t smem_K[MMA_N * HEAD_DIM];
    __shared__ float s_debug[32];

    uint8_t fp8_one = 0x38;     // 1.0
    uint8_t fp8_two = 0x40;     // 2.0
    uint8_t fp8_half = 0x30;    // 0.5

    // Test case: Q = all 1.0, K = all 1.0
    // Expected raw MMA output (with scale=127, i.e., 1.0): 32.0 (sum of 32 products)
    //
    // Now simulate what happens with dynamic scales:
    // - Q absmax = 1.0, K absmax = 1.0
    // - Q scale = 1.0 / 448 ≈ 0.00223, so exp = ceil(log2(0.00223)) + 127 = ceil(-8.81) + 127 = -8 + 127 = 119
    // - K scale = same = 119
    // - inv_scale = 1 / 2^(119-127) = 1 / 2^(-8) = 256
    // - quantized Q = 1.0 * 256 = 256
    // - quantized K = 1.0 * 256 = 256
    // - MMA raw = 32 * 256 * 256 = 2,097,152
    // - MMA with scales: 2,097,152 * 2^(-8) * 2^(-8) = 2,097,152 * 2^(-16) = 32.0 ✓
    //
    // Let's test this by manually doing the quantization and using computed scales.

    // Compute scale for values with absmax = 1.0
    constexpr float FP8_E4M3_MAX = 448.0f;
    float absmax = 1.0f;
    float scale = absmax / FP8_E4M3_MAX;  // 0.00223
    int exp = static_cast<int>(ceilf(log2f(scale))) + 127;  // 119
    float inv_scale = 1.0f / exp2f(static_cast<float>(exp - 127));  // 256
    uint8_t scale_ue8m0 = static_cast<uint8_t>(exp);

    // Debug: print computed values
    if (lane_id == 0) {
        s_debug[0] = scale;
        s_debug[1] = (float)exp;
        s_debug[2] = inv_scale;
        s_debug[3] = (float)scale_ue8m0;
    }

    // Quantize 1.0 to FP8
    float q_val = 1.0f * inv_scale;  // 256.0
    q_val = fminf(fmaxf(q_val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    uint8_t q_fp8 = static_cast<uint8_t>(__nv_cvt_float_to_fp8(q_val, __NV_SATFINITE, __NV_E4M3));

    if (lane_id == 0) {
        s_debug[4] = q_val;  // Should be 256.0 (clamped to 448 if >448)
        s_debug[5] = (float)q_fp8;  // FP8 encoding

        // Initialize shared memory with quantized values
        for (int i = 0; i < MMA_M * HEAD_DIM; ++i) {
            smem_Q[i] = q_fp8;
        }
        for (int i = 0; i < MMA_N * HEAD_DIM; ++i) {
            smem_K[i] = q_fp8;
        }
    }
    __syncthreads();

    // Load A fragment using FA3 formula
    int t0 = lane_id / 8;
    int t1 = lane_id % 8;

    uint32_t a_frag[4];
    const uint8_t* A_ptr = smem_Q;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int v1 = r % 2;
        int v2 = r / 2;
        uint8_t bytes[4];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int flat = 64 * t0 + t1 + 16 * b + 8 * v1 + 256 * v2;
            int row = flat / MMA_K;
            int col = flat % MMA_K;
            bytes[b] = A_ptr[row * HEAD_DIM + col];
        }
        a_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }

    // Load B fragment using corrected formula
    uint32_t b_frag[2];
    const uint8_t* B_ptr = smem_K;
    int n_idx = lane_id / 4;
    int k_base = (lane_id % 4) * 8;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        uint8_t bytes[4];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int k_idx = k_base + r * 4 + b;
            bytes[b] = B_ptr[n_idx * HEAD_DIM + k_idx];
        }
        b_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }

    // Execute MMA WITH the computed scales
    float d_frag[4] = {0, 0, 0, 0};
    mma_fp8_16x8x32(
        d_frag[0], d_frag[1], d_frag[2], d_frag[3],
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag[0], b_frag[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        scale_ue8m0, scale_ue8m0  // Use computed scales, NOT 127
    );

    // Store results
    for (int v = 0; v < 4; ++v) {
        output[lane_id * 4 + v] = d_frag[v];
    }

    // Store debug info
    if (lane_id == 0) {
        // Dequantize one FP8 value to verify
        __nv_fp8_e4m3 fp8_struct;
        fp8_struct.__x = q_fp8;
        float dequant = float(fp8_struct);
        s_debug[6] = dequant;  // Should be ~256.0 (or close)

        // Expected output: 32.0 (since 1.0 * 1.0 * 32 = 32)
        s_debug[7] = 32.0f;

        for (int i = 0; i < 8; ++i) {
            debug[i] = s_debug[i];
        }
    }
}

/**
 * Test 6: Verify quantization produces correct FP8 representation
 *
 * This is a simple test to check if our quantization formula is correct.
 */
__global__ void test_quantization_kernel(
    float* __restrict__ output,
    float* __restrict__ debug
) {
    int tid = threadIdx.x;
    if (tid >= 8) return;

    // Test values: 0.5, 1.0, 1.5, 2.0, 0.25, 0.125, 3.0, 4.0
    float test_values[8] = {0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.125f, 3.0f, 4.0f};
    float val = test_values[tid];

    // Find absmax (assume 4.0 for this test)
    constexpr float FP8_E4M3_MAX = 448.0f;
    float absmax = 4.0f;
    float scale = absmax / FP8_E4M3_MAX;
    int exp = static_cast<int>(ceilf(log2f(scale))) + 127;
    float inv_scale = 1.0f / exp2f(static_cast<float>(exp - 127));
    uint8_t scale_ue8m0 = static_cast<uint8_t>(exp);

    // Quantize
    float scaled_val = val * inv_scale;
    scaled_val = fminf(fmaxf(scaled_val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    uint8_t fp8_val = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled_val, __NV_SATFINITE, __NV_E4M3));

    // Dequantize
    __nv_fp8_e4m3 fp8_struct;
    fp8_struct.__x = fp8_val;
    float dequant = float(fp8_struct);

    // Apply scale back
    float scale_factor = exp2f(static_cast<float>(exp - 127));
    float reconstructed = dequant * scale_factor;

    // Store results
    output[tid * 4 + 0] = val;            // Original value
    output[tid * 4 + 1] = scaled_val;     // Scaled value (before FP8)
    output[tid * 4 + 2] = dequant;        // Dequantized (FP8 -> float)
    output[tid * 4 + 3] = reconstructed;  // Reconstructed (should ≈ original)

    // Debug
    if (tid == 0) {
        debug[0] = absmax;
        debug[1] = scale;
        debug[2] = (float)exp;
        debug[3] = inv_scale;
        debug[4] = scale_factor;
        debug[5] = (float)scale_ue8m0;
    }
}

// Add Test 5 and Test 6 to the main test function
inline void test_mma_full_pipeline() {
    printf("\n=== Test 5: Full Quantization + Scale Pipeline ===\n");
    printf("Test: Q=all 1.0, K=all 1.0, with computed scales\n");
    printf("Expected output: 32.0 (sum of 32 products of 1.0*1.0)\n\n");

    float* d_output;
    float* d_debug;
    cudaMalloc(&d_output, 32 * 4 * sizeof(float));
    cudaMalloc(&d_debug, 32 * sizeof(float));

    test_mma_full_pipeline_kernel<<<1, 32>>>(d_output, d_debug);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_debug);
        return;
    }

    float h_output[128];
    float h_debug[32];
    cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_debug, d_debug, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Debug values:\n");
    printf("  scale = %.6f\n", h_debug[0]);
    printf("  exp = %.0f\n", h_debug[1]);
    printf("  inv_scale = %.2f\n", h_debug[2]);
    printf("  scale_ue8m0 = %.0f\n", h_debug[3]);
    printf("  q_val (scaled) = %.2f\n", h_debug[4]);
    printf("  q_fp8 (encoding) = %.0f (0x%02X)\n", h_debug[5], (int)h_debug[5]);
    printf("  dequant = %.2f\n", h_debug[6]);
    printf("  expected = %.2f\n\n", h_debug[7]);

    printf("Lane | d[0]    | d[1]    | d[2]    | d[3]\n");
    printf("-----+---------+---------+---------+---------\n");
    for (int lane = 0; lane < 8; ++lane) {
        printf("%4d | %7.2f | %7.2f | %7.2f | %7.2f\n",
               lane,
               h_output[lane * 4 + 0],
               h_output[lane * 4 + 1],
               h_output[lane * 4 + 2],
               h_output[lane * 4 + 3]);
    }

    // Check if output is close to expected 32.0
    float avg = 0.0f;
    for (int i = 0; i < 128; ++i) avg += h_output[i];
    avg /= 128.0f;
    printf("\nAverage output: %.2f (expected: 32.0)\n", avg);
    if (fabsf(avg - 32.0f) < 1.0f) {
        printf("SUCCESS: Output ≈ 32.0 as expected!\n");
    } else {
        printf("FAILURE: Output significantly differs from expected 32.0\n");
        printf("Ratio: %.2fx (output/expected)\n", avg / 32.0f);
    }

    // Test 6: Quantization verification
    printf("\n=== Test 6: Quantization Verification ===\n");
    printf("Testing quantization for values: 0.5, 1.0, 1.5, 2.0, 0.25, 0.125, 3.0, 4.0\n");
    printf("absmax = 4.0\n\n");

    cudaMemset(d_debug, 0, 32 * sizeof(float));
    test_quantization_kernel<<<1, 8>>>(d_output, d_debug);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_debug);
        return;
    }

    cudaMemcpy(h_output, d_output, 8 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_debug, d_debug, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Quantization params:\n");
    printf("  absmax = %.2f\n", h_debug[0]);
    printf("  scale = %.6f\n", h_debug[1]);
    printf("  exp = %.0f\n", h_debug[2]);
    printf("  inv_scale = %.2f\n", h_debug[3]);
    printf("  scale_factor = %.6f\n", h_debug[4]);
    printf("  scale_ue8m0 = %.0f\n\n", h_debug[5]);

    printf("Value | Scaled   | FP8->float | Reconstructed | Error%%\n");
    printf("------+----------+------------+---------------+--------\n");
    for (int i = 0; i < 8; ++i) {
        float original = h_output[i * 4 + 0];
        float scaled = h_output[i * 4 + 1];
        float dequant = h_output[i * 4 + 2];
        float recon = h_output[i * 4 + 3];
        float err_pct = fabsf(recon - original) / fabsf(original) * 100.0f;
        printf("%5.3f | %8.2f | %10.2f | %13.4f | %6.2f%%\n",
               original, scaled, dequant, recon, err_pct);
    }

    cudaFree(d_output);
    cudaFree(d_debug);
}

}  // namespace fp8_mma_test
}  // namespace matmul
}  // namespace ops
}  // namespace pygpukit

#endif  // __CUDACC_VER_MAJOR__ >= 13
