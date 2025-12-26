// Dynamic cuBLASLt Loader Header
// Loads cuBLASLt at runtime using LoadLibrary (Windows) or dlopen (Linux)
// This enables driver-only deployment without CUDA Toolkit

#pragma once

#include <cstdint>
#include <tuple>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace cublaslt {

// cuBLASLt type definitions (matching cublasLt.h)
// We define these ourselves to avoid requiring the header at runtime

using cublasLtHandle_t = void*;
using cublasLtMatmulDesc_t = void*;
using cublasLtMatrixLayout_t = void*;
using cublasLtMatmulPreference_t = void*;
using cublasLtMatmulHeuristicResult_t = void*;

// Status codes
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

// Compute types
enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77
};

// Data types (matching cudaDataType)
enum cudaDataType_t_local {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_16BF = 14
};

// Operation types
enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
};

// Matmul desc attributes
enum cublasLtMatmulDescAttributes_t {
    CUBLASLT_MATMUL_DESC_TRANSA = 0,
    CUBLASLT_MATMUL_DESC_TRANSB = 1,
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 2
};

// Matmul preference attributes
enum cublasLtMatmulPreferenceAttributes_t {
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
};

// Matrix layout attributes for batched GEMM
enum cublasLtMatrixLayoutAttribute_t {
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6
};

// Matrix order
enum cublasLtOrder_t {
    CUBLASLT_ORDER_COL = 0,
    CUBLASLT_ORDER_ROW = 1
};

// Algorithm structure (64 bytes as per cuBLAS documentation)
struct cublasLtMatmulAlgo_t {
    uint64_t data[8];
};

// Heuristic result structure
struct cublasLtMatmulHeuristicResult_struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
};

// Initialize the dynamic loader
// Returns true if cuBLASLt was found and loaded successfully
bool initialize();

// Check if cuBLASLt is available
bool is_available();

// Get the path to the loaded library
std::string get_library_path();

// Get cuBLASLt version
std::tuple<int, int, int> get_version();

// ============================================================================
// cuBLASLt API wrappers
// ============================================================================

cublasStatus_t create(cublasLtHandle_t* handle);
cublasStatus_t destroy(cublasLtHandle_t handle);

cublasStatus_t matmul_desc_create(
    cublasLtMatmulDesc_t* matmulDesc,
    cublasComputeType_t computeType,
    int scaleType  // cudaDataType
);

cublasStatus_t matmul_desc_destroy(cublasLtMatmulDesc_t matmulDesc);

cublasStatus_t matmul_desc_set_attribute(
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes
);

cublasStatus_t matrix_layout_create(
    cublasLtMatrixLayout_t* matLayout,
    int type,  // cudaDataType
    uint64_t rows,
    uint64_t cols,
    int64_t ld
);

cublasStatus_t matrix_layout_destroy(cublasLtMatrixLayout_t matLayout);

cublasStatus_t matrix_layout_set_attribute(
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* buf,
    size_t sizeInBytes
);

cublasStatus_t matmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void* alpha,
    const void* A,
    cublasLtMatrixLayout_t Adesc,
    const void* B,
    cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C,
    cublasLtMatrixLayout_t Cdesc,
    void* D,
    cublasLtMatrixLayout_t Ddesc,
    const void* algo,  // cublasLtMatmulAlgo_t*
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream
);

// ============================================================================
// Convenience GEMM functions
// ============================================================================

// Get singleton handle (auto-initializes)
cublasLtHandle_t get_handle();

// FP16 GEMM: C = A @ B (row-major)
cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
);

// FP32 GEMM: C = A @ B (row-major)
cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
);

// BF16 GEMM: C = A @ B (row-major)
cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
);

// Strided Batched FP32 GEMM: C[b] = A[b] @ B[b] for b in [0, batch_count)
// A: [batch_count, M, K], B: [batch_count, K, N], C: [batch_count, M, N]
cudaError_t gemm_strided_batched_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_count,
    int64_t strideA, int64_t strideB, int64_t strideC,
    cudaStream_t stream = nullptr
);

// Debug functions
int get_last_cublaslt_error();  // Returns last cuBLASLt status code
int get_last_cublaslt_step();   // Returns which step failed (1-6)

}  // namespace cublaslt
}  // namespace pygpukit
