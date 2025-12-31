// Dynamic cuBLAS Loader Header
// Loads cuBLAS at runtime using LoadLibrary (Windows) or dlopen (Linux)
// This enables driver-only deployment without CUDA Toolkit
//
// PyGPUkit v0.2.19+

#pragma once

#include <cstdint>
#include <tuple>
#include <string>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace cublas {

// cuBLAS type definitions (matching cublas_v2.h)
// We define these ourselves to avoid requiring the header at compile time

using cublasHandle_t = void*;

// Status codes (same as cuBLASLt)
enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
};

// Operation types
enum cublasOperation_t {
    CUBLAS_OP_N = 0,  // Non-transpose
    CUBLAS_OP_T = 1,  // Transpose
    CUBLAS_OP_C = 2   // Conjugate transpose
};

// Math mode for TensorCore usage
enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,           // Deprecated in CUDA 11+
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,      // TF32 TensorCore
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
};

// Compute type for GemmEx
enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73
};

// CUDA data types (for GemmEx)
enum cudaDataType_cublas {
    CUDA_R_16F = 2,   // FP16
    CUDA_R_32F = 0,   // FP32
    CUDA_R_64F = 1,   // FP64
    CUDA_R_16BF = 14, // BF16
    CUDA_R_8I = 3,    // INT8
    CUDA_R_32I = 10,  // INT32
    CUDA_R_8F_E4M3 = 28,  // FP8 E4M3
    CUDA_R_8F_E5M2 = 29   // FP8 E5M2
};

// GemmAlgo for cublasGemmEx
enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,
    CUBLAS_GEMM_ALGO19 = 19,
    CUBLAS_GEMM_ALGO20 = 20,
    CUBLAS_GEMM_ALGO21 = 21,
    CUBLAS_GEMM_ALGO22 = 22,
    CUBLAS_GEMM_ALGO23 = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115
};

// ============================================================================
// Initialization and status
// ============================================================================

// Initialize the dynamic loader
// Returns true if cuBLAS was found and loaded successfully
bool initialize();

// Check if cuBLAS is available
bool is_available();

// Get the path to the loaded library
std::string get_library_path();

// Get cuBLAS version as (major, minor, patch)
std::tuple<int, int, int> get_version();

// ============================================================================
// Handle management
// ============================================================================

// Create a cuBLAS handle
cublasStatus_t create(cublasHandle_t* handle);

// Destroy a cuBLAS handle
cublasStatus_t destroy(cublasHandle_t handle);

// Set stream for a handle
cublasStatus_t set_stream(cublasHandle_t handle, CUstream stream);

// Set math mode (for TensorCore)
cublasStatus_t set_math_mode(cublasHandle_t handle, cublasMath_t mode);

// Get singleton handle (auto-initializes)
cublasHandle_t get_handle();

// ============================================================================
// GEMM operations
// ============================================================================

// FP32 GEMM: C = alpha * op(A) * op(B) + beta * C
cublasStatus_t sgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
);

// FP64 GEMM: C = alpha * op(A) * op(B) + beta * C
cublasStatus_t dgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
);

// FP16 GEMM (half precision)
cublasStatus_t hgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __half* alpha,
    const __half* A, int lda,
    const __half* B, int ldb,
    const __half* beta,
    __half* C, int ldc
);

// Mixed-precision GEMM (GemmEx)
cublasStatus_t gemm_ex(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType_cublas Atype, int lda,
    const void* B, cudaDataType_cublas Btype, int ldb,
    const void* beta,
    void* C, cudaDataType_cublas Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);

// Strided batched FP32 GEMM
cublasStatus_t sgemm_strided_batched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    const float* beta,
    float* C, int ldc, long long strideC,
    int batchCount
);

// ============================================================================
// GEMV operations
// ============================================================================

// FP32 GEMV: y = alpha * op(A) * x + beta * y
cublasStatus_t sgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float* alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy
);

// FP64 GEMV: y = alpha * op(A) * x + beta * y
cublasStatus_t dgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double* alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double* beta,
    double* y, int incy
);

// ============================================================================
// Convenience functions (row-major, using singleton handle)
// ============================================================================

// FP32 GEMM: C = A @ B (row-major)
cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    CUstream stream = nullptr
);

// FP64 GEMM: C = A @ B (row-major)
cudaError_t gemm_fp64(
    const double* A, const double* B, double* C,
    int M, int N, int K,
    CUstream stream = nullptr
);

// FP16 GEMM: C = A @ B (row-major)
cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    CUstream stream = nullptr
);

// BF16 GEMM: C = A @ B (row-major, via GemmEx)
cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    CUstream stream = nullptr
);

// FP32 GEMV: y = A @ x (row-major)
cudaError_t gemv_fp32(
    const float* A, const float* x, float* y,
    int M, int N,
    CUstream stream = nullptr
);

// ============================================================================
// Debug functions
// ============================================================================

// Get last cuBLAS error code
int get_last_error();

// Get error string
const char* get_status_string(cublasStatus_t status);

}  // namespace cublas
}  // namespace pygpukit
