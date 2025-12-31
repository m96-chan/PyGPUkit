// Dynamic cuBLAS Loader Implementation
// Loads cuBLAS at runtime using LoadLibrary (Windows) or dlopen (Linux)
//
// PyGPUkit v0.2.19+

#include "cublas_loader.hpp"
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pygpukit {
namespace cublas {

namespace {

// Platform-specific library handle type
#ifdef _WIN32
using LibHandle = HMODULE;
#define LOAD_LIBRARY(path) LoadLibraryA(path)
#define GET_PROC(handle, name) GetProcAddress(handle, name)
#define FREE_LIBRARY(handle) FreeLibrary(handle)
#else
using LibHandle = void*;
#define LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
#define GET_PROC(handle, name) dlsym(handle, name)
#define FREE_LIBRARY(handle) dlclose(handle)
#endif

// Function pointer types
// Note: On Windows, cuBLAS uses __stdcall calling convention
#ifdef _WIN32
#define CUBLASAPI __stdcall
#else
#define CUBLASAPI
#endif

// Handle management
using PFN_cublasCreate = cublasStatus_t (CUBLASAPI *)(cublasHandle_t*);
using PFN_cublasDestroy = cublasStatus_t (CUBLASAPI *)(cublasHandle_t);
using PFN_cublasGetVersion = cublasStatus_t (CUBLASAPI *)(cublasHandle_t, int*);
using PFN_cublasSetStream = cublasStatus_t (CUBLASAPI *)(cublasHandle_t, CUstream);
using PFN_cublasSetMathMode = cublasStatus_t (CUBLASAPI *)(cublasHandle_t, cublasMath_t);

// GEMM
using PFN_cublasSgemm = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int
);

using PFN_cublasDgemm = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const double*, const double*, int,
    const double*, int,
    const double*, double*, int
);

using PFN_cublasHgemm = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const __half*, const __half*, int,
    const __half*, int,
    const __half*, __half*, int
);

using PFN_cublasGemmEx = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int  // computeType, algo
);

using PFN_cublasSgemmStridedBatched = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int, long long,
    const float*, int, long long,
    const float*, float*, int, long long,
    int
);

// GEMV
using PFN_cublasSgemv = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t,
    int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int
);

using PFN_cublasDgemv = cublasStatus_t (CUBLASAPI *)(
    cublasHandle_t, cublasOperation_t,
    int, int,
    const double*, const double*, int,
    const double*, int,
    const double*, double*, int
);

// Global state
struct CublasState {
    std::atomic<bool> initialized{false};
    std::atomic<bool> available{false};
    std::mutex init_mutex;
    LibHandle handle{nullptr};
    std::string library_path;
    int version{0};

    // Singleton handle
    cublasHandle_t cublas_handle{nullptr};
    std::mutex handle_mutex;

    // Last error
    std::atomic<int> last_error{0};

    // Function pointers
    PFN_cublasCreate pfn_create{nullptr};
    PFN_cublasDestroy pfn_destroy{nullptr};
    PFN_cublasGetVersion pfn_get_version{nullptr};
    PFN_cublasSetStream pfn_set_stream{nullptr};
    PFN_cublasSetMathMode pfn_set_math_mode{nullptr};
    PFN_cublasSgemm pfn_sgemm{nullptr};
    PFN_cublasDgemm pfn_dgemm{nullptr};
    PFN_cublasHgemm pfn_hgemm{nullptr};
    PFN_cublasGemmEx pfn_gemm_ex{nullptr};
    PFN_cublasSgemmStridedBatched pfn_sgemm_strided_batched{nullptr};
    PFN_cublasSgemv pfn_sgemv{nullptr};
    PFN_cublasDgemv pfn_dgemv{nullptr};
};

CublasState g_state;

// Get CUDA runtime major version
int get_cuda_major_version() {
    int version = 0;
    CUresult err = cuDriverGetVersion(&version);
    if (err != CUDA_SUCCESS) {
        return 12;  // Default to 12 if query fails
    }
    // version is encoded as major * 1000 + minor * 10
    return version / 1000;
}

// Search for cuBLAS library in various locations
std::vector<std::string> get_search_paths() {
    std::vector<std::string> paths;

    int cuda_major = get_cuda_major_version();
    fprintf(stderr, "[cuBLAS] CUDA driver major version: %d\n", cuda_major);

#ifdef _WIN32
    // Windows: Search for cublas64_*.dll

    if (cuda_major >= 13) {
        // CUDA 13.x: bin/x64 subdirectory
        paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64");
        paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64");
    }

    // CUDA 12.x: bin directly
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin");

    // Check CUDA_PATH
    const char* cuda_path = std::getenv("CUDA_PATH");
    if (cuda_path) {
        if (cuda_major >= 13) {
            paths.push_back(std::string(cuda_path) + "\\bin\\x64");
        }
        paths.push_back(std::string(cuda_path) + "\\bin");
    }

    // Check PATH directories
    const char* path_env = std::getenv("PATH");
    if (path_env) {
        std::string path_str(path_env);
        size_t pos = 0;
        while (pos < path_str.size()) {
            size_t end = path_str.find(';', pos);
            if (end == std::string::npos) end = path_str.size();
            if (end > pos) {
                paths.push_back(path_str.substr(pos, end - pos));
            }
            pos = end + 1;
        }
    }

#else
    // Linux/macOS: Search for libcublas.so

    // 1. Check LD_LIBRARY_PATH
    const char* ld_path = std::getenv("LD_LIBRARY_PATH");
    if (ld_path) {
        std::string path_str(ld_path);
        size_t pos = 0;
        while (pos < path_str.size()) {
            size_t end = path_str.find(':', pos);
            if (end == std::string::npos) end = path_str.size();
            if (end > pos) {
                paths.push_back(path_str.substr(pos, end - pos));
            }
            pos = end + 1;
        }
    }

    // 2. Check CUDA_PATH
    const char* cuda_path = std::getenv("CUDA_PATH");
    if (cuda_path) {
        paths.push_back(std::string(cuda_path) + "/lib64");
        paths.push_back(std::string(cuda_path) + "/lib");
    }

    // 3. Common installation paths
    paths.push_back("/usr/local/cuda/lib64");
    paths.push_back("/usr/local/cuda/lib");
    paths.push_back("/usr/lib/x86_64-linux-gnu");
    paths.push_back("/usr/lib64");

#endif

    return paths;
}

// Try to load cuBLAS from a specific path
bool try_load_library(const std::string& dir) {
#ifdef _WIN32
    // Windows DLL names
    std::vector<std::string> dll_names = {
        "cublas64_13.dll",
        "cublas64_12.dll",
        "cublas64_11.dll"
    };

    for (const auto& dll_name : dll_names) {
        std::string full_path = dir + "\\" + dll_name;

        // Set DLL directory to help load dependencies
        SetDllDirectoryA(dir.c_str());
        fprintf(stderr, "[cuBLAS] Trying to load: %s\n", full_path.c_str());

        LibHandle h = LOAD_LIBRARY(full_path.c_str());
        if (h) {
            g_state.handle = h;
            g_state.library_path = full_path;
            fprintf(stderr, "[cuBLAS] SUCCESS! Loaded from: %s\n", full_path.c_str());
            return true;
        }
    }

#else
    // Linux SO names
    std::vector<std::string> so_names = {
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so"
    };

    for (const auto& so_name : so_names) {
        std::string full_path = dir + "/" + so_name;
        fprintf(stderr, "[cuBLAS] Trying to load: %s\n", full_path.c_str());

        LibHandle h = LOAD_LIBRARY(full_path.c_str());
        if (h) {
            g_state.handle = h;
            g_state.library_path = full_path;
            fprintf(stderr, "[cuBLAS] SUCCESS! Loaded from: %s\n", full_path.c_str());
            return true;
        }
    }

#endif

    return false;
}

// Load function pointers from the library
bool load_functions() {
    if (!g_state.handle) return false;

#define LOAD_FUNC(name, suffix) \
    g_state.pfn_##name = (PFN_cublas##suffix)GET_PROC(g_state.handle, "cublas" #suffix "_v2"); \
    if (!g_state.pfn_##name) { \
        g_state.pfn_##name = (PFN_cublas##suffix)GET_PROC(g_state.handle, "cublas" #suffix); \
    } \
    if (!g_state.pfn_##name) { \
        fprintf(stderr, "[cuBLAS] Failed to load cublas%s\n", #suffix); \
        return false; \
    }

    // Handle management (always _v2)
    g_state.pfn_create = (PFN_cublasCreate)GET_PROC(g_state.handle, "cublasCreate_v2");
    if (!g_state.pfn_create) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasCreate_v2\n");
        return false;
    }

    g_state.pfn_destroy = (PFN_cublasDestroy)GET_PROC(g_state.handle, "cublasDestroy_v2");
    if (!g_state.pfn_destroy) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasDestroy_v2\n");
        return false;
    }

    g_state.pfn_get_version = (PFN_cublasGetVersion)GET_PROC(g_state.handle, "cublasGetVersion_v2");
    if (!g_state.pfn_get_version) {
        fprintf(stderr, "[cuBLAS] Warning: cublasGetVersion_v2 not found\n");
    }

    g_state.pfn_set_stream = (PFN_cublasSetStream)GET_PROC(g_state.handle, "cublasSetStream_v2");
    if (!g_state.pfn_set_stream) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasSetStream_v2\n");
        return false;
    }

    g_state.pfn_set_math_mode = (PFN_cublasSetMathMode)GET_PROC(g_state.handle, "cublasSetMathMode");
    // Math mode is optional (older cuBLAS versions may not have it)

    // GEMM functions
    g_state.pfn_sgemm = (PFN_cublasSgemm)GET_PROC(g_state.handle, "cublasSgemm_v2");
    if (!g_state.pfn_sgemm) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasSgemm_v2\n");
        return false;
    }

    g_state.pfn_dgemm = (PFN_cublasDgemm)GET_PROC(g_state.handle, "cublasDgemm_v2");
    if (!g_state.pfn_dgemm) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasDgemm_v2\n");
        return false;
    }

    g_state.pfn_hgemm = (PFN_cublasHgemm)GET_PROC(g_state.handle, "cublasHgemm");
    // Hgemm is optional (may not be available on older GPUs)

    g_state.pfn_gemm_ex = (PFN_cublasGemmEx)GET_PROC(g_state.handle, "cublasGemmEx");
    // GemmEx is optional (CUDA 8.0+)

    g_state.pfn_sgemm_strided_batched = (PFN_cublasSgemmStridedBatched)GET_PROC(
        g_state.handle, "cublasSgemmStridedBatched");
    // Strided batched is optional

    // GEMV functions
    g_state.pfn_sgemv = (PFN_cublasSgemv)GET_PROC(g_state.handle, "cublasSgemv_v2");
    if (!g_state.pfn_sgemv) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasSgemv_v2\n");
        return false;
    }

    g_state.pfn_dgemv = (PFN_cublasDgemv)GET_PROC(g_state.handle, "cublasDgemv_v2");
    if (!g_state.pfn_dgemv) {
        fprintf(stderr, "[cuBLAS] Failed to load cublasDgemv_v2\n");
        return false;
    }

#undef LOAD_FUNC

    return true;
}

}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

bool initialize() {
    if (g_state.initialized.load()) {
        return g_state.available.load();
    }

    std::lock_guard<std::mutex> lock(g_state.init_mutex);

    // Double-check after acquiring lock
    if (g_state.initialized.load()) {
        return g_state.available.load();
    }

    // Try to load library from search paths
    auto paths = get_search_paths();
    for (const auto& path : paths) {
        if (try_load_library(path)) {
            break;
        }
    }

    if (!g_state.handle) {
        fprintf(stderr, "[cuBLAS] Library not found in any search path\n");
        g_state.initialized.store(true);
        g_state.available.store(false);
        return false;
    }

    // Load function pointers
    if (!load_functions()) {
        fprintf(stderr, "[cuBLAS] Failed to load required functions\n");
        FREE_LIBRARY(g_state.handle);
        g_state.handle = nullptr;
        g_state.initialized.store(true);
        g_state.available.store(false);
        return false;
    }

    // Get version if possible
    if (g_state.pfn_get_version) {
        cublasHandle_t temp_handle = nullptr;
        if (g_state.pfn_create(&temp_handle) == CUBLAS_STATUS_SUCCESS) {
            g_state.pfn_get_version(temp_handle, &g_state.version);
            g_state.pfn_destroy(temp_handle);
            fprintf(stderr, "[cuBLAS] Version: %d\n", g_state.version);
        }
    }

    g_state.initialized.store(true);
    g_state.available.store(true);
    return true;
}

bool is_available() {
    if (!g_state.initialized.load()) {
        initialize();
    }
    return g_state.available.load();
}

std::string get_library_path() {
    return g_state.library_path;
}

std::tuple<int, int, int> get_version() {
    if (!is_available()) {
        return {0, 0, 0};
    }
    // Version is encoded as major * 10000 + minor * 100 + patch
    int v = g_state.version;
    return {v / 10000, (v / 100) % 100, v % 100};
}

// ============================================================================
// Handle management
// ============================================================================

cublasStatus_t create(cublasHandle_t* handle) {
    if (!is_available() || !g_state.pfn_create) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    return g_state.pfn_create(handle);
}

cublasStatus_t destroy(cublasHandle_t handle) {
    if (!is_available() || !g_state.pfn_destroy) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    return g_state.pfn_destroy(handle);
}

cublasStatus_t set_stream(cublasHandle_t handle, CUstream stream) {
    if (!is_available() || !g_state.pfn_set_stream) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    return g_state.pfn_set_stream(handle, stream);
}

cublasStatus_t set_math_mode(cublasHandle_t handle, cublasMath_t mode) {
    if (!is_available() || !g_state.pfn_set_math_mode) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return g_state.pfn_set_math_mode(handle, mode);
}

cublasHandle_t get_handle() {
    if (!is_available()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_state.handle_mutex);
    if (!g_state.cublas_handle) {
        cublasStatus_t status = g_state.pfn_create(&g_state.cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[cuBLAS] Failed to create handle: %d\n", status);
            return nullptr;
        }
    }
    return g_state.cublas_handle;
}

// ============================================================================
// GEMM operations
// ============================================================================

cublasStatus_t sgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
) {
    if (!is_available() || !g_state.pfn_sgemm) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasStatus_t status = g_state.pfn_sgemm(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
    g_state.last_error.store(status);
    return status;
}

cublasStatus_t dgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
) {
    if (!is_available() || !g_state.pfn_dgemm) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasStatus_t status = g_state.pfn_dgemm(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
    g_state.last_error.store(status);
    return status;
}

cublasStatus_t hgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __half* alpha,
    const __half* A, int lda,
    const __half* B, int ldb,
    const __half* beta,
    __half* C, int ldc
) {
    if (!is_available() || !g_state.pfn_hgemm) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    cublasStatus_t status = g_state.pfn_hgemm(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
    g_state.last_error.store(status);
    return status;
}

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
) {
    if (!is_available() || !g_state.pfn_gemm_ex) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    cublasStatus_t status = g_state.pfn_gemm_ex(
        handle, transa, transb, m, n, k,
        alpha, A, lda, static_cast<int>(Atype),
        B, ldb, static_cast<int>(Btype),
        beta, C, ldc, static_cast<int>(Ctype),
        static_cast<int>(computeType), static_cast<int>(algo)
    );
    g_state.last_error.store(status);
    return status;
}

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
) {
    if (!is_available() || !g_state.pfn_sgemm_strided_batched) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    cublasStatus_t status = g_state.pfn_sgemm_strided_batched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, strideA, B, ldb, strideB,
        beta, C, ldc, strideC, batchCount
    );
    g_state.last_error.store(status);
    return status;
}

// ============================================================================
// GEMV operations
// ============================================================================

cublasStatus_t sgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float* alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy
) {
    if (!is_available() || !g_state.pfn_sgemv) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasStatus_t status = g_state.pfn_sgemv(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    );
    g_state.last_error.store(status);
    return status;
}

cublasStatus_t dgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double* alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double* beta,
    double* y, int incy
) {
    if (!is_available() || !g_state.pfn_dgemv) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasStatus_t status = g_state.pfn_dgemv(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    );
    g_state.last_error.store(status);
    return status;
}

// ============================================================================
// Convenience functions (row-major)
// ============================================================================

cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    CUstream stream
) {
    cublasHandle_t handle = get_handle();
    if (!handle) {
        return cudaErrorNotReady;
    }

    if (stream) {
        set_stream(handle, stream);
    }

    // Row-major: C = A @ B
    // cuBLAS is column-major, so we compute: C^T = B^T @ A^T
    // This gives us C in row-major layout
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = sgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose (for col-major interpretation)
        N, M, K,                    // Swapped M,N for row-major
        &alpha,
        B, N,                       // B^T in col-major = B in row-major
        A, K,                       // A^T in col-major = A in row-major
        &beta,
        C, N                        // C^T in col-major = C in row-major
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t gemm_fp64(
    const double* A, const double* B, double* C,
    int M, int N, int K,
    CUstream stream
) {
    cublasHandle_t handle = get_handle();
    if (!handle) {
        return cudaErrorNotReady;
    }

    if (stream) {
        set_stream(handle, stream);
    }

    double alpha = 1.0;
    double beta = 0.0;

    cublasStatus_t status = dgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    CUstream stream
) {
    cublasHandle_t handle = get_handle();
    if (!handle || !g_state.pfn_hgemm) {
        return cudaErrorNotReady;
    }

    if (stream) {
        set_stream(handle, stream);
    }

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasStatus_t status = hgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    CUstream stream
) {
    cublasHandle_t handle = get_handle();
    if (!handle || !g_state.pfn_gemm_ex) {
        return cudaErrorNotReady;
    }

    if (stream) {
        set_stream(handle, stream);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = gemm_ex(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t gemv_fp32(
    const float* A, const float* x, float* y,
    int M, int N,
    CUstream stream
) {
    cublasHandle_t handle = get_handle();
    if (!handle) {
        return cudaErrorNotReady;
    }

    if (stream) {
        set_stream(handle, stream);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Row-major: y = A @ x
    // cuBLAS col-major: y = A^T @ x (with CUBLAS_OP_T)
    cublasStatus_t status = sgemv(
        handle,
        CUBLAS_OP_T,  // Transpose for row-major
        N, M,         // Swapped dimensions
        &alpha,
        A, N,         // Leading dimension is N for row-major
        x, 1,
        &beta,
        y, 1
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// ============================================================================
// Debug functions
// ============================================================================

int get_last_error() {
    return g_state.last_error.load();
}

const char* get_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

}  // namespace cublas
}  // namespace pygpukit
