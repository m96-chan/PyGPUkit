/**
 * Test FP8 GEMM with BF16 I/O on SM120
 *
 * Build (from native/ops/matmul directory):
 *   nvcc -o test_fp8_bf16_sm120.exe test_fp8_bf16_sm120.cu ^
 *     -arch=sm_120a ^
 *     -I ../../../third_party/cutlass/include ^
 *     -I ../../../third_party/cutlass/examples/common ^
 *     -DCUTLASS_ARCH_MMA_SM120_SUPPORTED ^
 *     --expt-relaxed-constexpr ^
 *     /Zc:preprocessor ^
 *     -std=c++17
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Include the FP8 BF16 GEMM implementation
#include "matmul_fp8_bf16_sm120.cu"

// ============================================================================
// CPU Reference (BF16 -> FP32 for computation -> BF16)
// ============================================================================

void gemm_cpu_reference_bf16(
    const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C,
    int M, int N, int K,
    float alpha, float beta)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = __bfloat162float(A[m * K + k]);
                float b_val = __bfloat162float(B[k * N + n]);
                sum += a_val * b_val;
            }
            float c_val = beta != 0.0f ? __bfloat162float(C[m * N + n]) : 0.0f;
            float result = alpha * sum + beta * c_val;
            C[m * N + n] = __float2bfloat16(result);
        }
    }
}

void fill_random_bf16(nv_bfloat16* data, int64_t size, float scale = 1.0f) {
    for (int64_t i = 0; i < size; i++) {
        float val = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
        data[i] = __float2bfloat16(val);
    }
}

float compute_relative_error_bf16(const nv_bfloat16* ref, const nv_bfloat16* test, int64_t size) {
    float sum_err = 0.0f;
    float sum_ref = 0.0f;
    for (int64_t i = 0; i < size; i++) {
        float r = __bfloat162float(ref[i]);
        float t = __bfloat162float(test[i]);
        sum_err += fabsf(r - t);
        sum_ref += fabsf(r);
    }
    return sum_ref > 0 ? sum_err / sum_ref : sum_err;
}

// ============================================================================
// FP8 Quantization Simulation (for fair comparison)
// ============================================================================

nv_bfloat16 simulate_fp8_e4m3_bf16(nv_bfloat16 val_bf16) {
    float val = __bfloat162float(val_bf16);

    if (fabsf(val) < 1e-7f) return __float2bfloat16(0.0f);

    constexpr float FP8_MAX = 448.0f;
    constexpr float FP8_MIN_NORMAL = 0.015625f;  // 2^-6

    val = fminf(fmaxf(val, -FP8_MAX), FP8_MAX);
    if (fabsf(val) < FP8_MIN_NORMAL) return __float2bfloat16(0.0f);

    float sign = (val < 0) ? -1.0f : 1.0f;
    float abs_val = fabsf(val);

    int exp = static_cast<int>(floorf(log2f(abs_val)));
    float mantissa = abs_val / powf(2.0f, static_cast<float>(exp));
    mantissa = roundf(mantissa * 8.0f) / 8.0f;

    return __float2bfloat16(sign * mantissa * powf(2.0f, static_cast<float>(exp)));
}

void quantize_to_fp8_bf16(nv_bfloat16* data, int64_t size) {
    for (int64_t i = 0; i < size; i++) {
        data[i] = simulate_fp8_e4m3_bf16(data[i]);
    }
}

// ============================================================================
// Test
// ============================================================================

bool test_fp8_bf16_gemm(int M, int N, int K) {
    printf("Testing FP8 BF16 GEMM: M=%d, N=%d, K=%d\n", M, N, K);

    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_C = static_cast<int64_t>(M) * N;

    // Host memory
    nv_bfloat16* h_A = new nv_bfloat16[size_A];
    nv_bfloat16* h_B = new nv_bfloat16[size_B];
    nv_bfloat16* h_C_ref = new nv_bfloat16[size_C];
    nv_bfloat16* h_C_test = new nv_bfloat16[size_C];

    // Use range [-2, 2] to stay in FP8 normal range
    fill_random_bf16(h_A, size_A, 2.0f);
    fill_random_bf16(h_B, size_B, 2.0f);

    // Zero output buffers
    for (int64_t i = 0; i < size_C; i++) {
        h_C_ref[i] = __float2bfloat16(0.0f);
        h_C_test[i] = __float2bfloat16(0.0f);
    }

    // Quantize inputs to FP8 precision for fair comparison
    quantize_to_fp8_bf16(h_A, size_A);
    quantize_to_fp8_bf16(h_B, size_B);

    // CPU reference (using FP8-quantized inputs)
    gemm_cpu_reference_bf16(h_A, h_B, h_C_ref, M, N, K, 1.0f, 0.0f);

    // Device memory
    nv_bfloat16* d_A;
    nv_bfloat16* d_B;
    nv_bfloat16* d_C;
    cudaMalloc(&d_A, size_A * sizeof(nv_bfloat16));
    cudaMalloc(&d_B, size_B * sizeof(nv_bfloat16));
    cudaMalloc(&d_C, size_C * sizeof(nv_bfloat16));

    cudaMemcpy(d_A, h_A, size_A * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C * sizeof(nv_bfloat16));

    // Run FP8 BF16 GEMM
    printf("  Launching FP8 BF16 GEMM kernel...\n");
    cudaError_t err = pygpukit::ops::fp8_bf16_gemm_sm120::gemm_fp8_bf16(
        d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);

    if (err != cudaSuccess) {
        printf("  ERROR: FP8 BF16 GEMM failed: %s\n", cudaGetErrorString(err));
        delete[] h_A; delete[] h_B; delete[] h_C_ref; delete[] h_C_test;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return false;
    }
    printf("  FP8 BF16 GEMM kernel completed without error!\n");

    // Copy result
    cudaMemcpy(h_C_test, d_C, size_C * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compare
    float rel_err = compute_relative_error_bf16(h_C_ref, h_C_test, size_C);
    printf("  Relative error: %.6f\n", rel_err);

    // FP8 has limited precision, allow 10% tolerance
    bool pass = rel_err < 0.10f;
    printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C_ref; delete[] h_C_test;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== FP8 BF16 GEMM Test (SM120) ===\n");
    printf("Data flow: BF16 -> FP8 quantize -> GEMM -> BF16\n\n");

    // Check GPU
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Device: %s (SM %d.%d)\n\n", props.name, props.major, props.minor);

    int sm = props.major * 10 + props.minor;
    if (sm < 120) {
        printf("ERROR: This test requires SM120 (RTX 5090)\n");
        printf("Current device is SM %d\n", sm);
        return 1;
    }

    srand(42);  // Reproducible
    bool all_pass = true;

    // Test various sizes
    all_pass &= test_fp8_bf16_gemm(128, 128, 128);
    all_pass &= test_fp8_bf16_gemm(256, 256, 256);
    all_pass &= test_fp8_bf16_gemm(512, 512, 512);

    printf("=== SUMMARY ===\n");
    if (all_pass) {
        printf("All tests PASSED!\n");
        printf("FP8 BF16 GEMM works correctly on SM120.\n");
    } else {
        printf("Some tests FAILED.\n");
    }

    return all_pass ? 0 : 1;
}
