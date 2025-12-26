/**
 * GEMV Correctness Test
 *
 * Verifies CUTLASS GEMV against CPU reference implementation.
 * No cuBLASLt dependency.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -arch=sm_86 test_gemv.cu -o test_gemv
 *
 * Usage:
 *   ./test_gemv
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "gemv_cutlass.cuh"

// ============================================================================
// CPU Reference Implementation
// ============================================================================

void gemv_cpu_reference(
    const float* A,  // [1, K]
    const float* B,  // [K, N]
    float* C,        // [1, N]
    int K, int N,
    float alpha, float beta
) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[k] * B[k * N + n];
        }
        if (beta != 0.0f) {
            C[n] = alpha * acc + beta * C[n];
        } else {
            C[n] = alpha * acc;
        }
    }
}

// ============================================================================
// Test Functions
// ============================================================================

bool test_gemv_bf16(int K, int N, float tolerance = 0.01f) {
    printf("Testing BF16 GEMV: K=%d, N=%d ... ", K, N);

    // Host allocations
    std::vector<float> h_A(K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(N, 0.0f);
    std::vector<__nv_bfloat16> h_A_bf16(K);
    std::vector<__nv_bfloat16> h_B_bf16(K * N);
    std::vector<__nv_bfloat16> h_C_bf16(N);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < K; ++i) {
        h_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        h_A_bf16[i] = __float2bfloat16(h_A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        h_B_bf16[i] = __float2bfloat16(h_B[i]);
    }

    // CPU reference (using BF16-rounded values for fair comparison)
    std::vector<float> h_A_rounded(K);
    std::vector<float> h_B_rounded(K * N);
    for (int i = 0; i < K; ++i) {
        h_A_rounded[i] = __bfloat162float(h_A_bf16[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_rounded[i] = __bfloat162float(h_B_bf16[i]);
    }
    gemv_cpu_reference(h_A_rounded.data(), h_B_rounded.data(), h_C_ref.data(), K, N, 1.0f, 0.0f);

    // Device allocations
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, N * sizeof(__nv_bfloat16));

    cudaMemcpy(d_A, h_A_bf16.data(), K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, N * sizeof(__nv_bfloat16));

    // Run GPU kernel
    cudaError_t err = pygpukit::ops::gemv::launch_gemv_bf16(d_A, d_B, d_C, K, N);
    if (err != cudaSuccess) {
        printf("FAILED (kernel launch error: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(h_C_bf16.data(), d_C, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compare results
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    int max_err_idx = 0;
    for (int i = 0; i < N; ++i) {
        float gpu_val = __bfloat162float(h_C_bf16[i]);
        float ref_val = h_C_ref[i];
        float err = std::abs(gpu_val - ref_val);
        float rel_err = err / (std::abs(ref_val) + 1e-6f);
        if (err > max_err) {
            max_err = err;
            max_err_idx = i;
        }
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (max_rel_err < tolerance) {
        printf("PASS (max_rel_err=%.6f at idx=%d)\n", max_rel_err, max_err_idx);
        return true;
    } else {
        printf("FAILED (max_rel_err=%.6f at idx=%d, ref=%.6f, gpu=%.6f)\n",
               max_rel_err, max_err_idx, h_C_ref[max_err_idx],
               __bfloat162float(h_C_bf16[max_err_idx]));
        return false;
    }
}

bool test_gemv_fp16(int K, int N, float tolerance = 0.005f) {
    printf("Testing FP16 GEMV: K=%d, N=%d ... ", K, N);

    // Host allocations
    std::vector<float> h_A(K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(N, 0.0f);
    std::vector<__half> h_A_fp16(K);
    std::vector<__half> h_B_fp16(K * N);
    std::vector<__half> h_C_fp16(N);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < K; ++i) {
        h_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        h_A_fp16[i] = __float2half(h_A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        h_B_fp16[i] = __float2half(h_B[i]);
    }

    // CPU reference (using FP16-rounded values)
    std::vector<float> h_A_rounded(K);
    std::vector<float> h_B_rounded(K * N);
    for (int i = 0; i < K; ++i) {
        h_A_rounded[i] = __half2float(h_A_fp16[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_rounded[i] = __half2float(h_B_fp16[i]);
    }
    gemv_cpu_reference(h_A_rounded.data(), h_B_rounded.data(), h_C_ref.data(), K, N, 1.0f, 0.0f);

    // Device allocations
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, N * sizeof(__half));

    cudaMemcpy(d_A, h_A_fp16.data(), K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, N * sizeof(__half));

    // Run GPU kernel
    cudaError_t err = pygpukit::ops::gemv::launch_gemv_fp16(d_A, d_B, d_C, K, N);
    if (err != cudaSuccess) {
        printf("FAILED (kernel launch error: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(h_C_fp16.data(), d_C, N * sizeof(__half), cudaMemcpyDeviceToHost);

    // Compare results
    float max_rel_err = 0.0f;
    int max_err_idx = 0;
    for (int i = 0; i < N; ++i) {
        float gpu_val = __half2float(h_C_fp16[i]);
        float ref_val = h_C_ref[i];
        float err = std::abs(gpu_val - ref_val);
        float rel_err = err / (std::abs(ref_val) + 1e-6f);
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            max_err_idx = i;
        }
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (max_rel_err < tolerance) {
        printf("PASS (max_rel_err=%.6f)\n", max_rel_err);
        return true;
    } else {
        printf("FAILED (max_rel_err=%.6f)\n", max_rel_err);
        return false;
    }
}

bool test_gemv_fp32(int K, int N, float tolerance = 0.002f) {
    printf("Testing FP32 GEMV: K=%d, N=%d ... ", K, N);

    // Host allocations
    std::vector<float> h_A(K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(N, 0.0f);
    std::vector<float> h_C_gpu(N, 0.0f);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < K; ++i) {
        h_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }

    // CPU reference
    gemv_cpu_reference(h_A.data(), h_B.data(), h_C_ref.data(), K, N, 1.0f, 0.0f);

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, N * sizeof(float));

    // Run GPU kernel
    cudaError_t err = pygpukit::ops::gemv::launch_gemv_fp32(d_A, d_B, d_C, K, N);
    if (err != cudaSuccess) {
        printf("FAILED (kernel launch error: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(h_C_gpu.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    float max_rel_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::abs(h_C_gpu[i] - h_C_ref[i]);
        float rel_err = err / (std::abs(h_C_ref[i]) + 1e-6f);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (max_rel_err < tolerance) {
        printf("PASS (max_rel_err=%.6f)\n", max_rel_err);
        return true;
    } else {
        printf("FAILED (max_rel_err=%.6f)\n", max_rel_err);
        return false;
    }
}

bool test_gemv_batched_bf16(int batch, int K, int N, float tolerance = 0.01f) {
    printf("Testing Batched BF16 GEMV: batch=%d, K=%d, N=%d ... ", batch, K, N);

    // Host allocations
    std::vector<float> h_A(batch * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(batch * N, 0.0f);
    std::vector<__nv_bfloat16> h_A_bf16(batch * K);
    std::vector<__nv_bfloat16> h_B_bf16(K * N);
    std::vector<__nv_bfloat16> h_C_bf16(batch * N);

    // Initialize
    srand(42);
    for (int i = 0; i < batch * K; ++i) {
        h_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        h_A_bf16[i] = __float2bfloat16(h_A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        h_B_bf16[i] = __float2bfloat16(h_B[i]);
    }

    // CPU reference (per batch)
    for (int b = 0; b < batch; ++b) {
        std::vector<float> h_A_rounded(K);
        std::vector<float> h_B_rounded(K * N);
        for (int i = 0; i < K; ++i) {
            h_A_rounded[i] = __bfloat162float(h_A_bf16[b * K + i]);
        }
        for (int i = 0; i < K * N; ++i) {
            h_B_rounded[i] = __bfloat162float(h_B_bf16[i]);
        }
        gemv_cpu_reference(h_A_rounded.data(), h_B_rounded.data(),
                          h_C_ref.data() + b * N, K, N, 1.0f, 0.0f);
    }

    // Device allocations
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, batch * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, batch * N * sizeof(__nv_bfloat16));

    cudaMemcpy(d_A, h_A_bf16.data(), batch * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, batch * N * sizeof(__nv_bfloat16));

    // Run GPU kernel
    cudaError_t err = pygpukit::ops::gemv::launch_gemv_bf16_batched(
        d_A, d_B, d_C, K, N, batch);
    if (err != cudaSuccess) {
        printf("FAILED (kernel launch error: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(h_C_bf16.data(), d_C, batch * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compare results
    float max_rel_err = 0.0f;
    for (int i = 0; i < batch * N; ++i) {
        float gpu_val = __bfloat162float(h_C_bf16[i]);
        float ref_val = h_C_ref[i];
        float err = std::abs(gpu_val - ref_val);
        float rel_err = err / (std::abs(ref_val) + 1e-6f);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (max_rel_err < tolerance) {
        printf("PASS (max_rel_err=%.6f)\n", max_rel_err);
        return true;
    } else {
        printf("FAILED (max_rel_err=%.6f)\n", max_rel_err);
        return false;
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Print device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("Device: %s (SM %d%d)\n", props.name, props.major, props.minor);
    printf("\n");

    printf("=== GEMV Correctness Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    // BF16 tests
    printf("--- BF16 GEMV ---\n");
    if (test_gemv_bf16(256, 256)) passed++; else failed++;
    if (test_gemv_bf16(512, 512)) passed++; else failed++;
    if (test_gemv_bf16(1024, 1024)) passed++; else failed++;
    if (test_gemv_bf16(4096, 4096)) passed++; else failed++;
    if (test_gemv_bf16(4096, 11008)) passed++; else failed++;  // LLaMA MLP
    if (test_gemv_bf16(8192, 28672)) passed++; else failed++;  // LLaMA-70B MLP
    printf("\n");

    // FP16 tests
    printf("--- FP16 GEMV ---\n");
    if (test_gemv_fp16(256, 256)) passed++; else failed++;
    if (test_gemv_fp16(1024, 1024)) passed++; else failed++;
    if (test_gemv_fp16(4096, 4096)) passed++; else failed++;
    printf("\n");

    // FP32 tests
    printf("--- FP32 GEMV ---\n");
    if (test_gemv_fp32(256, 256)) passed++; else failed++;
    if (test_gemv_fp32(1024, 1024)) passed++; else failed++;
    if (test_gemv_fp32(4096, 4096)) passed++; else failed++;
    printf("\n");

    // Batched BF16 tests
    printf("--- Batched BF16 GEMV ---\n");
    if (test_gemv_batched_bf16(4, 1024, 1024)) passed++; else failed++;
    if (test_gemv_batched_bf16(8, 4096, 4096)) passed++; else failed++;
    if (test_gemv_batched_bf16(16, 4096, 11008)) passed++; else failed++;
    printf("\n");

    // Summary
    printf("=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
