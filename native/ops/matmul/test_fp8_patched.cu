/**
 * Test FP8 GEMM on SM120 with CUTLASS alignment patch
 *
 * This tests whether the CUTLASS Issue #2902 alignment fix works.
 *
 * Build (from native/ops/matmul directory):
 *   Use build_fp8_test.bat which sets up all required paths.
 *
 *   Key flags:
 *   - arch=sm_120a  (enables __CUDA_ARCH_FEAT_SM120_ALL for kernel selection)
 *   - CUTLASS_ARCH_MMA_SM120_SUPPORTED
 *   - --expt-relaxed-constexpr
 *   - /Zc:preprocessor (MSVC conformant preprocessor)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Include the FP8 GEMM implementation (which includes patched CUTLASS)
#include "matmul_fp8_sm120.cu"

// ============================================================================
// CPU-side FP8 E4M3 simulation
// ============================================================================

// Simulate FP8 E4M3 quantization on CPU
float simulate_fp8_e4m3(float val) {
    if (fabsf(val) < 1e-7f) return 0.0f;

    // FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa
    // Range: ~0.0156 to 448
    constexpr float FP8_MAX = 448.0f;
    constexpr float FP8_MIN_NORMAL = 0.015625f;  // 2^-6

    // Clamp to range
    val = fminf(fmaxf(val, -FP8_MAX), FP8_MAX);

    // Handle subnormals (just zero them like GPU does)
    if (fabsf(val) < FP8_MIN_NORMAL) return 0.0f;

    // Quantize to 3-bit mantissa precision
    // FP8 has 3 mantissa bits = 8 levels per octave
    float sign = (val < 0) ? -1.0f : 1.0f;
    float abs_val = fabsf(val);

    // Find the exponent
    int exp = static_cast<int>(floorf(log2f(abs_val)));
    float mantissa = abs_val / powf(2.0f, static_cast<float>(exp));

    // Quantize mantissa to 3 bits (8 levels from 1.0 to 2.0)
    // mantissa is in [1.0, 2.0), quantize to nearest 1/8
    mantissa = roundf(mantissa * 8.0f) / 8.0f;

    return sign * mantissa * powf(2.0f, static_cast<float>(exp));
}

// Quantize an array to FP8 precision
void quantize_to_fp8(float* data, int64_t size) {
    for (int64_t i = 0; i < size; i++) {
        data[i] = simulate_fp8_e4m3(data[i]);
    }
}

// ============================================================================
// CPU Reference
// ============================================================================

void gemm_cpu_reference(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

void fill_random(float* data, int64_t size, float scale = 1.0f) {
    for (int64_t i = 0; i < size; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

float compute_relative_error(const float* ref, const float* test, int64_t size) {
    float sum_err = 0.0f;
    float sum_ref = 0.0f;
    for (int64_t i = 0; i < size; i++) {
        sum_err += fabsf(ref[i] - test[i]);
        sum_ref += fabsf(ref[i]);
    }
    return sum_ref > 0 ? sum_err / sum_ref : sum_err;
}

// ============================================================================
// Test
// ============================================================================

bool test_fp8_gemm(int M, int N, int K) {
    printf("Testing FP8 GEMM: M=%d, N=%d, K=%d\n", M, N, K);

    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_C = static_cast<int64_t>(M) * N;

    // Host memory
    float* h_A = new float[size_A];
    float* h_B = new float[size_B];
    float* h_C_ref = new float[size_C];
    float* h_C_test = new float[size_C];

    // Use range [-2, 2] like Example 87a to stay in FP8 normal range
    // FP8 E4M3 smallest normal is ~0.0156, so we need values > 0.0156
    fill_random(h_A, size_A, 2.0f);
    fill_random(h_B, size_B, 2.0f);
    memset(h_C_ref, 0, size_C * sizeof(float));
    memset(h_C_test, 0, size_C * sizeof(float));

    // Quantize inputs to FP8 precision for fair comparison
    // This simulates what the GPU does during FP32->FP8 conversion
    quantize_to_fp8(h_A, size_A);
    quantize_to_fp8(h_B, size_B);

    // CPU reference (using FP8-quantized inputs)
    gemm_cpu_reference(h_A, h_B, h_C_ref, M, N, K, 1.0f, 0.0f);

    // Device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C * sizeof(float));

    // Run FP8 GEMM
    printf("  Launching FP8 GEMM kernel...\n");
    cudaError_t err = pygpukit::ops::fp8_gemm_sm120::gemm_fp8(
        d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);

    if (err != cudaSuccess) {
        printf("  ERROR: FP8 GEMM failed: %s\n", cudaGetErrorString(err));
        delete[] h_A; delete[] h_B; delete[] h_C_ref; delete[] h_C_test;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return false;
    }
    printf("  FP8 GEMM kernel completed without error!\n");

    // Copy result
    cudaMemcpy(h_C_test, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float rel_err = compute_relative_error(h_C_ref, h_C_test, size_C);
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
    printf("=== FP8 GEMM Test with CUTLASS Alignment Patch ===\n");
    printf("Testing CUTLASS Issue #2902 workaround\n\n");

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
    all_pass &= test_fp8_gemm(128, 128, 128);
    all_pass &= test_fp8_gemm(256, 256, 256);
    all_pass &= test_fp8_gemm(512, 512, 512);

    printf("=== SUMMARY ===\n");
    if (all_pass) {
        printf("All tests PASSED!\n");
        printf("CUTLASS alignment fix works - FP8 GEMM is functional on SM120.\n");
    } else {
        printf("Some tests FAILED.\n");
    }

    return all_pass ? 0 : 1;
}
