/**
 * GEMV Benchmark: CUTLASS vs cuBLASLt
 *
 * Compares our CUTLASS-based GEMV with cuBLASLt GEMV under identical conditions.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -arch=sm_86 benchmark_gemv.cu -lcublasLt -o benchmark_gemv
 *
 * Usage:
 *   ./benchmark_gemv [K] [N]
 *   Default: K=4096, N=4096 (typical LLM hidden size)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "gemv_cutlass.cuh"

// ============================================================================
// Benchmark Configuration
// ============================================================================

constexpr int WARMUP_ITERATIONS = 20;
constexpr int BENCHMARK_ITERATIONS = 100;

// Common LLM hidden sizes for benchmarking
struct BenchmarkCase {
    int K;
    int N;
    const char* name;
};

const BenchmarkCase BENCHMARK_CASES[] = {
    // Small models (< 1B params)
    {768, 768, "768x768 (BERT-base)"},
    {1024, 1024, "1024x1024 (GPT-small)"},
    {2048, 2048, "2048x2048 (GPT-medium)"},

    // Medium models (1-7B params)
    {4096, 4096, "4096x4096 (LLaMA-7B hidden)"},
    {4096, 11008, "4096x11008 (LLaMA-7B MLP)"},
    {4096, 14336, "4096x14336 (Qwen-7B MLP)"},

    // Large models (7-70B params)
    {5120, 5120, "5120x5120 (LLaMA-13B)"},
    {8192, 8192, "8192x8192 (LLaMA-70B hidden)"},
    {8192, 28672, "8192x28672 (LLaMA-70B MLP)"},

    // Extreme cases
    {16384, 16384, "16384x16384 (large)"},
    {4096, 32768, "4096x32768 (wide)"},
    {32768, 4096, "32768x4096 (tall)"},
};

// ============================================================================
// cuBLASLt GEMV Wrapper
// ============================================================================

class CuBLASLtGemv {
public:
    CuBLASLtGemv() {
        cublasLtCreate(&handle_);
    }

    ~CuBLASLtGemv() {
        cublasLtDestroy(handle_);
    }

    // BF16 GEMV using cuBLASLt
    // C[1,N] = A[1,K] @ B[K,N]
    cudaError_t gemv_bf16(
        const __nv_bfloat16* A,  // [1, K]
        const __nv_bfloat16* B,  // [K, N]
        __nv_bfloat16* C,        // [1, N]
        int K, int N,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        // cuBLASLt uses column-major, so we compute C^T = B^T @ A^T
        // For row-major: C[1,N] = A[1,K] @ B[K,N]
        // In col-major view: C^T[N,1] = B^T[N,K] @ A^T[K,1]
        //
        // However, for M=1, it's simpler to just call GEMM with M=1
        // cuBLASLt GEMM: D = alpha * A @ B + beta * C
        // With m=1, n=N, k=K in column-major terms

        cublasLtMatmulDesc_t operationDesc;
        cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
        cublasLtMatmulPreference_t preference;
        cublasLtMatmulHeuristicResult_t heuristicResult;
        int returnedResults = 0;

        cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
        cudaDataType_t scaleType = CUDA_R_32F;
        cudaDataType_t dataType = CUDA_R_16BF;

        // Create operation descriptor
        cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);

        // Set transpose operations for row-major inputs
        // For row-major C = A @ B:
        // Use CUBLAS_OP_N for both since we're treating row-major as transposed col-major
        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

        // Matrix layouts (column-major perspective)
        // A: [K, 1] in col-major = [1, K] row-major
        // B: [K, N] in col-major = [N, K] row-major, but we have [K, N] row-major
        // Need to swap and transpose

        // Actually, let's use the standard row-major approach:
        // For row-major C[M,N] = A[M,K] @ B[K,N]:
        // Compute as: C^T[N,M] = B^T[N,K] @ A^T[K,M]
        // In cuBLASLt terms with ColumnMajor default:
        // D[N,M] = B[N,K] @ A[K,M] where matrices are stored as their transposes

        // For M=1:
        // D[N,1] = B[N,K] @ A[K,1]
        // m=N, n=1, k=K

        int m = N;
        int n = 1;
        int k = K;

        int lda = K;  // Leading dim of A (row-major A[1,K])
        int ldb = N;  // Leading dim of B (row-major B[K,N])
        int ldc = N;  // Leading dim of C (row-major C[1,N])

        // Create matrix layouts
        // A as [K, 1] column-major (which is A^T of our row-major [1, K])
        cublasLtMatrixLayoutCreate(&Adesc, dataType, k, n, lda);

        // B as [N, K] column-major (which is B^T of our row-major [K, N])
        cublasLtMatrixLayoutCreate(&Bdesc, dataType, m, k, ldb);

        // C/D as [N, 1] column-major (which is C^T of our row-major [1, N])
        cublasLtMatrixLayoutCreate(&Cdesc, dataType, m, n, ldc);
        cublasLtMatrixLayoutCreate(&Ddesc, dataType, m, n, ldc);

        // Create preference
        cublasLtMatmulPreferenceCreate(&preference);
        size_t workspaceSize = 0;
        cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

        // Get heuristic
        cublasLtMatmulAlgoGetHeuristic(handle_, operationDesc, Bdesc, Adesc, Cdesc, Ddesc,
            preference, 1, &heuristicResult, &returnedResults);

        if (returnedResults == 0) {
            // Cleanup
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(Ddesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatmulDescDestroy(operationDesc);
            return cudaErrorNotSupported;
        }

        // Execute GEMM
        // Note: For row-major, we swap A and B pointers
        cublasStatus_t status = cublasLtMatmul(handle_,
            operationDesc,
            &alpha,
            B, Bdesc,  // First operand (was A in col-major)
            A, Adesc,  // Second operand (was B in col-major)
            &beta,
            C, Cdesc,
            C, Ddesc,  // Output
            &heuristicResult.algo,
            nullptr, 0,
            stream);

        // Cleanup
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(Ddesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatmulDescDestroy(operationDesc);

        return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    }

private:
    cublasLtHandle_t handle_;
};

// ============================================================================
// Benchmark Utilities
// ============================================================================

void initialize_random_bf16(__nv_bfloat16* data, size_t count) {
    std::vector<float> host(count);
    for (size_t i = 0; i < count; ++i) {
        host[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }
    std::vector<__nv_bfloat16> host_bf16(count);
    for (size_t i = 0; i < count; ++i) {
        host_bf16[i] = __float2bfloat16(host[i]);
    }
    cudaMemcpy(data, host_bf16.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
}

float compute_max_error_bf16(__nv_bfloat16* A, __nv_bfloat16* B, size_t count) {
    std::vector<__nv_bfloat16> host_A(count), host_B(count);
    cudaMemcpy(host_A.data(), A, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B.data(), B, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float a = __bfloat162float(host_A[i]);
        float b = __bfloat162float(host_B[i]);
        float err = std::abs(a - b);
        max_err = std::max(max_err, err);
    }
    return max_err;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

struct BenchmarkResult {
    double cutlass_us;
    double cublaslt_us;
    float speedup;
    float max_error;
};

BenchmarkResult run_benchmark(int K, int N, CuBLASLtGemv& cublas) {
    BenchmarkResult result;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C_cutlass, *d_C_cublas;
    cudaMalloc(&d_A, 1 * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_cutlass, 1 * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_cublas, 1 * N * sizeof(__nv_bfloat16));

    // Initialize with random data
    initialize_random_bf16(d_A, K);
    initialize_random_bf16(d_B, K * N);
    cudaMemset(d_C_cutlass, 0, N * sizeof(__nv_bfloat16));
    cudaMemset(d_C_cublas, 0, N * sizeof(__nv_bfloat16));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========================================================================
    // Benchmark CUTLASS GEMV
    // ========================================================================

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        pygpukit::ops::gemv::launch_gemv_bf16(d_A, d_B, d_C_cutlass, K, N);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        pygpukit::ops::gemv::launch_gemv_bf16(d_A, d_B, d_C_cutlass, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cutlass_ms;
    cudaEventElapsedTime(&cutlass_ms, start, stop);
    result.cutlass_us = (cutlass_ms * 1000.0) / BENCHMARK_ITERATIONS;

    // ========================================================================
    // Benchmark cuBLASLt GEMV
    // ========================================================================

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        cublas.gemv_bf16(d_A, d_B, d_C_cublas, K, N, 1.0f, 0.0f, nullptr);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        cublas.gemv_bf16(d_A, d_B, d_C_cublas, K, N, 1.0f, 0.0f, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublaslt_ms;
    cudaEventElapsedTime(&cublaslt_ms, start, stop);
    result.cublaslt_us = (cublaslt_ms * 1000.0) / BENCHMARK_ITERATIONS;

    // ========================================================================
    // Compute error
    // ========================================================================

    result.max_error = compute_max_error_bf16(d_C_cutlass, d_C_cublas, N);
    result.speedup = result.cublaslt_us / result.cutlass_us;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_cutlass);
    cudaFree(d_C_cublas);

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Print device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("Device: %s (SM %d%d)\n", props.name, props.major, props.minor);
    printf("Memory: %.1f GB\n", props.totalGlobalMem / 1e9);
    printf("\n");

    // Initialize cuBLASLt
    CuBLASLtGemv cublas;

    // Print header
    printf("GEMV Benchmark: CUTLASS vs cuBLASLt (BF16, M=1)\n");
    printf("Warmup: %d iterations, Benchmark: %d iterations\n", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    printf("\n");
    printf("%-30s %10s %10s %10s %10s %10s\n",
           "Case", "K", "N", "CUTLASS", "cuBLASLt", "Speedup");
    printf("%-30s %10s %10s %10s %10s %10s\n",
           "", "", "", "(us)", "(us)", "");
    printf("--------------------------------------------------------------------------------\n");

    // Run benchmarks
    for (const auto& test : BENCHMARK_CASES) {
        BenchmarkResult result = run_benchmark(test.K, test.N, cublas);

        printf("%-30s %10d %10d %10.2f %10.2f %9.2fx %s\n",
               test.name,
               test.K, test.N,
               result.cutlass_us,
               result.cublaslt_us,
               result.speedup,
               result.speedup >= 1.0f ? "(CUTLASS wins)" : "(cuBLASLt wins)");

        if (result.max_error > 0.01f) {
            printf("  WARNING: Max error = %.6f\n", result.max_error);
        }
    }

    printf("\n");
    printf("================================================================================\n");
    printf("Analysis:\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Performance gap causes (when cuBLASLt wins):\n");
    printf("1. cuBLASLt uses hand-tuned PTX/SASS assembly\n");
    printf("2. cuBLASLt may use specialized M=1 kernel paths\n");
    printf("3. cuBLASLt may use different memory access patterns (texture cache)\n");
    printf("4. Our UNROLL_K=8 may not be optimal for all K sizes\n");
    printf("\n");
    printf("Improvement opportunities for CUTLASS GEMV:\n");
    printf("1. Tune BLOCK_SIZE and UNROLL_K per (K, N) range\n");
    printf("2. Add shared memory tiling for A (reduces L2 pressure)\n");
    printf("3. Use vectorized BF16x2 or BF16x4 loads where aligned\n");
    printf("4. Add software pipelining (async copy + compute overlap)\n");
    printf("5. Consider warp specialization for very large K\n");
    printf("\n");
    printf("Future FP8/SM120 considerations:\n");
    printf("1. FP8 E4M3/E5M2 would require custom quantization\n");
    printf("2. SM120 lacks native FP8 GEMV support in CUTLASS 4.x\n");
    printf("3. BF16 fallback is the current solution for SM120\n");
    printf("4. When CUTLASS SM120 FP8 is fixed, add FP8 path\n");

    return 0;
}
