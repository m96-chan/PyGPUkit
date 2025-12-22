// Dynamic cuBLASLt Loader Implementation
// Loads cuBLASLt at runtime using LoadLibrary (Windows) or dlopen (Linux)

#include "cublaslt_loader.hpp"
#include <cuda.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#endif

namespace pygpukit {
namespace cublaslt {

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
// Note: On Windows, cuBLAS uses __stdcall calling convention (CUBLASWINAPI)
#ifdef _WIN32
#define CUBLASAPI __stdcall
#else
#define CUBLASAPI
#endif

using PFN_cublasLtCreate = cublasStatus_t (CUBLASAPI *)(cublasLtHandle_t*);
using PFN_cublasLtDestroy = cublasStatus_t (CUBLASAPI *)(cublasLtHandle_t);
using PFN_cublasLtGetVersion = size_t (CUBLASAPI *)();
using PFN_cublasLtMatmulDescCreate = cublasStatus_t (CUBLASAPI *)(cublasLtMatmulDesc_t*, cublasComputeType_t, int);
using PFN_cublasLtMatmulDescDestroy = cublasStatus_t (CUBLASAPI *)(cublasLtMatmulDesc_t);
using PFN_cublasLtMatmulDescSetAttribute = cublasStatus_t (CUBLASAPI *)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*, size_t);
using PFN_cublasLtMatrixLayoutCreate = cublasStatus_t (CUBLASAPI *)(cublasLtMatrixLayout_t*, int, uint64_t, uint64_t, int64_t);
using PFN_cublasLtMatrixLayoutDestroy = cublasStatus_t (CUBLASAPI *)(cublasLtMatrixLayout_t);
using PFN_cublasLtMatmul = cublasStatus_t (CUBLASAPI *)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, cudaStream_t
);

// Preference and heuristic function pointers (for CUDA Graph compatibility)
using PFN_cublasLtMatmulPreferenceCreate = cublasStatus_t (CUBLASAPI *)(cublasLtMatmulPreference_t*);
using PFN_cublasLtMatmulPreferenceDestroy = cublasStatus_t (CUBLASAPI *)(cublasLtMatmulPreference_t);
using PFN_cublasLtMatmulPreferenceSetAttribute = cublasStatus_t (CUBLASAPI *)(
    cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t
);
using PFN_cublasLtMatmulAlgoGetHeuristic = cublasStatus_t (CUBLASAPI *)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_struct*, int*
);

// Global state
struct CublasLtState {
    std::atomic<bool> initialized{false};
    std::atomic<bool> available{false};
    std::mutex init_mutex;
    LibHandle handle{nullptr};
    std::string library_path;
    size_t version{0};

    // Singleton handle
    cublasLtHandle_t lt_handle{nullptr};
    std::mutex handle_mutex;

    // Function pointers
    PFN_cublasLtCreate pfn_create{nullptr};
    PFN_cublasLtDestroy pfn_destroy{nullptr};
    PFN_cublasLtGetVersion pfn_get_version{nullptr};
    PFN_cublasLtMatmulDescCreate pfn_matmul_desc_create{nullptr};
    PFN_cublasLtMatmulDescDestroy pfn_matmul_desc_destroy{nullptr};
    PFN_cublasLtMatmulDescSetAttribute pfn_matmul_desc_set_attr{nullptr};
    PFN_cublasLtMatrixLayoutCreate pfn_matrix_layout_create{nullptr};
    PFN_cublasLtMatrixLayoutDestroy pfn_matrix_layout_destroy{nullptr};
    PFN_cublasLtMatmul pfn_matmul{nullptr};

    // Preference and heuristic function pointers (for CUDA Graph compatibility)
    PFN_cublasLtMatmulPreferenceCreate pfn_pref_create{nullptr};
    PFN_cublasLtMatmulPreferenceDestroy pfn_pref_destroy{nullptr};
    PFN_cublasLtMatmulPreferenceSetAttribute pfn_pref_set_attr{nullptr};
    PFN_cublasLtMatmulAlgoGetHeuristic pfn_algo_get_heuristic{nullptr};
};

CublasLtState g_state;

// Search for cuBLASLt library in various locations
std::vector<std::string> get_search_paths() {
    std::vector<std::string> paths;

#ifdef _WIN32
    // Windows: Search for cublasLt64_*.dll
    // Note: CUDA 13.x puts DLLs in bin/x64/ subdirectory

    // 1. Check CUDA_PATH environment variable
    const char* cuda_path = std::getenv("CUDA_PATH");
    if (cuda_path) {
        paths.push_back(std::string(cuda_path) + "\\bin\\x64");  // CUDA 13.x
        paths.push_back(std::string(cuda_path) + "\\bin");       // CUDA 12.x and earlier
    }

    // 2. Check PATH directories
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

    // 3. Common installation paths (CUDA 13.x uses bin/x64)
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64");
    // CUDA 12.x uses bin directly
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin");

#else
    // Linux/macOS: Search for libcublasLt.so

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

#ifdef _WIN32
// Find cuBLASLt DLL in a directory (Windows)
std::string find_cublaslt_in_dir(const std::string& dir) {
    // Search for cublasLt64_*.dll pattern (e.g., cublasLt64_12.dll, cublasLt64_13.dll)
    WIN32_FIND_DATAA find_data;
    std::string pattern = dir + "\\cublasLt64_*.dll";
    HANDLE find_handle = FindFirstFileA(pattern.c_str(), &find_data);

    if (find_handle != INVALID_HANDLE_VALUE) {
        std::string result = dir + "\\" + find_data.cFileName;
        FindClose(find_handle);
        return result;
    }

    // Also try exact name cublasLt64.dll (older versions)
    std::string exact_path = dir + "\\cublasLt64.dll";
    if (GetFileAttributesA(exact_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
        return exact_path;
    }

    // Try specific version patterns for CUDA 13.x
    for (int ver = 13; ver >= 11; --ver) {
        std::string versioned_path = dir + "\\cublasLt64_" + std::to_string(ver) + ".dll";
        if (GetFileAttributesA(versioned_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
            return versioned_path;
        }
    }

    return "";
}
#else
// Find cuBLASLt shared library in a directory (Linux)
std::string find_cublaslt_in_dir(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return "";

    std::string result;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        // Match libcublasLt.so or libcublasLt.so.*
        if (name.find("libcublasLt.so") == 0) {
            result = dir + "/" + name;
            break;
        }
    }
    closedir(d);
    return result;
}
#endif

// Try to load cuBLASLt from a specific path
bool try_load(const std::string& path) {
    fprintf(stderr, "[cuBLASLt] Trying to load: %s\n", path.c_str());
#ifdef _WIN32
    // On Windows, we need to add the DLL directory to the search path
    // so that dependent DLLs (like cublas64_*.dll) can be found
    size_t last_slash = path.find_last_of("\\/");
    if (last_slash != std::string::npos) {
        std::string dir = path.substr(0, last_slash);
        fprintf(stderr, "[cuBLASLt] Setting DLL directory: %s\n", dir.c_str());
        SetDllDirectoryA(dir.c_str());
    }
#endif

    LibHandle handle = LOAD_LIBRARY(path.c_str());

#ifdef _WIN32
    // Reset DLL directory to default
    SetDllDirectoryA(nullptr);
#endif

    if (!handle) {
        return false;
    }

    // Resolve all required functions
    auto pfn_create = (PFN_cublasLtCreate)GET_PROC(handle, "cublasLtCreate");
    auto pfn_destroy = (PFN_cublasLtDestroy)GET_PROC(handle, "cublasLtDestroy");
    auto pfn_get_version = (PFN_cublasLtGetVersion)GET_PROC(handle, "cublasLtGetVersion");
    auto pfn_matmul_desc_create = (PFN_cublasLtMatmulDescCreate)GET_PROC(handle, "cublasLtMatmulDescCreate");
    auto pfn_matmul_desc_destroy = (PFN_cublasLtMatmulDescDestroy)GET_PROC(handle, "cublasLtMatmulDescDestroy");
    auto pfn_matmul_desc_set_attr = (PFN_cublasLtMatmulDescSetAttribute)GET_PROC(handle, "cublasLtMatmulDescSetAttribute");
    auto pfn_matrix_layout_create = (PFN_cublasLtMatrixLayoutCreate)GET_PROC(handle, "cublasLtMatrixLayoutCreate");
    auto pfn_matrix_layout_destroy = (PFN_cublasLtMatrixLayoutDestroy)GET_PROC(handle, "cublasLtMatrixLayoutDestroy");
    auto pfn_matmul = (PFN_cublasLtMatmul)GET_PROC(handle, "cublasLtMatmul");

    // Preference and heuristic functions (for CUDA Graph compatibility)
    auto pfn_pref_create = (PFN_cublasLtMatmulPreferenceCreate)GET_PROC(handle, "cublasLtMatmulPreferenceCreate");
    auto pfn_pref_destroy = (PFN_cublasLtMatmulPreferenceDestroy)GET_PROC(handle, "cublasLtMatmulPreferenceDestroy");
    auto pfn_pref_set_attr = (PFN_cublasLtMatmulPreferenceSetAttribute)GET_PROC(handle, "cublasLtMatmulPreferenceSetAttribute");
    auto pfn_algo_get_heuristic = (PFN_cublasLtMatmulAlgoGetHeuristic)GET_PROC(handle, "cublasLtMatmulAlgoGetHeuristic");

    // All core functions must be present
    if (!pfn_create || !pfn_destroy || !pfn_matmul_desc_create ||
        !pfn_matmul_desc_destroy || !pfn_matmul_desc_set_attr ||
        !pfn_matrix_layout_create || !pfn_matrix_layout_destroy || !pfn_matmul) {
        FREE_LIBRARY(handle);
        return false;
    }

    // Heuristic functions are required for CUDA Graph compatibility
    if (!pfn_pref_create || !pfn_pref_destroy || !pfn_pref_set_attr || !pfn_algo_get_heuristic) {
        fprintf(stderr, "[cuBLASLt] WARNING: Heuristic functions not found, CUDA Graph may not work\n");
    }

    // Get version (optional, may fail on old versions)
    size_t version = 0;
    if (pfn_get_version) {
        version = pfn_get_version();
    }

    // Success! Store everything
    fprintf(stderr, "[cuBLASLt] SUCCESS! Loaded from: %s (version: %zu)\n", path.c_str(), version);
    g_state.handle = handle;
    g_state.library_path = path;
    g_state.version = version;
    g_state.pfn_create = pfn_create;
    g_state.pfn_destroy = pfn_destroy;
    g_state.pfn_get_version = pfn_get_version;
    g_state.pfn_matmul_desc_create = pfn_matmul_desc_create;
    g_state.pfn_matmul_desc_destroy = pfn_matmul_desc_destroy;
    g_state.pfn_matmul_desc_set_attr = pfn_matmul_desc_set_attr;
    g_state.pfn_matrix_layout_create = pfn_matrix_layout_create;
    g_state.pfn_matrix_layout_destroy = pfn_matrix_layout_destroy;
    g_state.pfn_matmul = pfn_matmul;

    // Preference and heuristic function pointers
    g_state.pfn_pref_create = pfn_pref_create;
    g_state.pfn_pref_destroy = pfn_pref_destroy;
    g_state.pfn_pref_set_attr = pfn_pref_set_attr;
    g_state.pfn_algo_get_heuristic = pfn_algo_get_heuristic;

    return true;
}

}  // anonymous namespace

bool initialize() {
    // Fast path: already initialized
    if (g_state.initialized.load(std::memory_order_acquire)) {
        return g_state.available.load(std::memory_order_relaxed);
    }

    // Slow path: initialize with lock
    std::lock_guard<std::mutex> lock(g_state.init_mutex);

    // Double-check after acquiring lock
    if (g_state.initialized.load(std::memory_order_relaxed)) {
        return g_state.available.load(std::memory_order_relaxed);
    }

    // Search for cuBLASLt
    auto search_paths = get_search_paths();

    for (const auto& dir : search_paths) {
        std::string cublaslt_path = find_cublaslt_in_dir(dir);
        if (!cublaslt_path.empty() && try_load(cublaslt_path)) {
            g_state.available.store(true, std::memory_order_relaxed);
            g_state.initialized.store(true, std::memory_order_release);
            return true;
        }
    }

    // Not found
    g_state.available.store(false, std::memory_order_relaxed);
    g_state.initialized.store(true, std::memory_order_release);
    return false;
}

bool is_available() {
    // Ultra-fast path: just check the cached flag
    // After initialization, this is just a single memory read
    if (g_state.initialized.load(std::memory_order_acquire)) {
        return g_state.available.load(std::memory_order_relaxed);
    }
    // First call: do full initialization
    initialize();
    return g_state.available.load(std::memory_order_relaxed);
}

std::string get_library_path() {
    initialize();
    return g_state.library_path;
}

std::tuple<int, int, int> get_version() {
    initialize();
    // cuBLASLt version is encoded as major * 10000 + minor * 100 + patch
    int major = static_cast<int>(g_state.version / 10000);
    int minor = static_cast<int>((g_state.version / 100) % 100);
    int patch = static_cast<int>(g_state.version % 100);
    return {major, minor, patch};
}

// ============================================================================
// API Wrappers
// ============================================================================

cublasStatus_t create(cublasLtHandle_t* handle) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_create(handle);
}

cublasStatus_t destroy(cublasLtHandle_t handle) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_destroy(handle);
}

cublasStatus_t matmul_desc_create(
    cublasLtMatmulDesc_t* matmulDesc,
    cublasComputeType_t computeType,
    int scaleType
) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_matmul_desc_create(matmulDesc, computeType, scaleType);
}

cublasStatus_t matmul_desc_destroy(cublasLtMatmulDesc_t matmulDesc) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_matmul_desc_destroy(matmulDesc);
}

cublasStatus_t matmul_desc_set_attribute(
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes
) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_matmul_desc_set_attr(matmulDesc, attr, buf, sizeInBytes);
}

cublasStatus_t matrix_layout_create(
    cublasLtMatrixLayout_t* matLayout,
    int type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld
) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_matrix_layout_create(matLayout, type, rows, cols, ld);
}

cublasStatus_t matrix_layout_destroy(cublasLtMatrixLayout_t matLayout) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_matrix_layout_destroy(matLayout);
}

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
    const void* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream
) {
    if (!is_available()) return CUBLAS_STATUS_NOT_INITIALIZED;
    return g_state.pfn_matmul(
        lightHandle, computeDesc, alpha,
        A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        algo, workspace, workspaceSizeInBytes, stream
    );
}

// ============================================================================
// Singleton Handle Management
// ============================================================================

cublasLtHandle_t get_handle() {
    if (!is_available()) {
        return nullptr;
    }

    // Fast path: already created
    if (g_state.lt_handle) {
        return g_state.lt_handle;
    }

    // Slow path: create with lock
    std::lock_guard<std::mutex> lock(g_state.handle_mutex);

    if (g_state.lt_handle) {
        return g_state.lt_handle;
    }

    cublasLtHandle_t handle = nullptr;
    cublasStatus_t status = g_state.pfn_create(&handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
        g_state.lt_handle = handle;
    }

    return g_state.lt_handle;
}

// ============================================================================
// GEMM Convenience Functions
// ============================================================================

// Thread-local variable to store last cuBLASLt error for debugging
thread_local int g_last_cublaslt_error = 0;
thread_local int g_last_cublaslt_step = 0;

int get_last_cublaslt_error() { return g_last_cublaslt_error; }
int get_last_cublaslt_step() { return g_last_cublaslt_step; }

// ============================================================================
// Descriptor Cache for Performance
// ============================================================================

namespace {

// Cache key for GEMM descriptors
struct GemmCacheKey {
    int M, N, K;
    int dtype;  // CUDA_R_16F, CUDA_R_32F, CUDA_R_16BF

    bool operator==(const GemmCacheKey& other) const {
        return M == other.M && N == other.N && K == other.K && dtype == other.dtype;
    }
};

struct GemmCacheKeyHash {
    size_t operator()(const GemmCacheKey& k) const {
        // Simple hash combining
        size_t h = static_cast<size_t>(k.M);
        h ^= static_cast<size_t>(k.N) << 16;
        h ^= static_cast<size_t>(k.K) << 32;
        h ^= static_cast<size_t>(k.dtype) << 48;
        return h;
    }
};

// Cached GEMM configuration with fixed algo + workspace for CUDA Graph compatibility
struct GemmCachedDesc {
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;

    // Fixed algorithm and workspace for CUDA Graph compatibility
    cublasLtMatmulAlgo_t algo;
    void* workspace = nullptr;
    size_t workspaceSize = 0;
    bool hasAlgo = false;

    bool valid = false;
};

// Global descriptor cache
std::unordered_map<GemmCacheKey, GemmCachedDesc, GemmCacheKeyHash> g_gemm_cache;
std::mutex g_cache_mutex;

// Thread-safe cache using atomic flag for fast path
std::atomic<bool> g_cache_initialized{false};

// Get or create cached descriptors for a GEMM configuration
GemmCachedDesc* get_cached_desc(int M, int N, int K, int dtype, cublasComputeType_t computeType, int scaleType) {
    GemmCacheKey key{M, N, K, dtype};

    // Fast path: if cache is initialized, do lock-free lookup
    // Note: unordered_map iterators are stable, so we can safely read
    // while holding the lock briefly just for the find operation
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_gemm_cache.find(key);
        if (it != g_gemm_cache.end() && it->second.valid) {
            return &it->second;
        }
    }

    // Slow path: create new descriptors with lock held
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    // Double-check after acquiring lock
    auto it = g_gemm_cache.find(key);
    if (it != g_gemm_cache.end() && it->second.valid) {
        return &it->second;
    }

    // Create new cached entry
    GemmCachedDesc& cached = g_gemm_cache[key];

    cublasStatus_t status;

    // Create matmul descriptor
    status = matmul_desc_create(&cached.operationDesc, computeType, scaleType);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cached.valid = false;
        return nullptr;
    }

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    matmul_desc_set_attribute(cached.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    matmul_desc_set_attribute(cached.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Create matrix layouts (swapped for row-major)
    status = matrix_layout_create(&cached.Bdesc, dtype, N, K, N);
    if (status != CUBLAS_STATUS_SUCCESS) { cached.valid = false; return nullptr; }

    status = matrix_layout_create(&cached.Adesc, dtype, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) { cached.valid = false; return nullptr; }

    status = matrix_layout_create(&cached.Cdesc, dtype, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) { cached.valid = false; return nullptr; }

    // =========================================================================
    // Select algorithm and allocate workspace for CUDA Graph compatibility
    // =========================================================================
    cublasLtHandle_t handle = get_handle();
    if (handle && g_state.pfn_pref_create && g_state.pfn_algo_get_heuristic) {
        // Create preference
        cublasLtMatmulPreference_t preference = nullptr;
        status = g_state.pfn_pref_create(&preference);
        if (status == CUBLAS_STATUS_SUCCESS && preference) {
            // Set maximum workspace size (32MB should be enough for most cases)
            constexpr size_t MAX_WORKSPACE = 32 * 1024 * 1024;
            g_state.pfn_pref_set_attr(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                      &MAX_WORKSPACE, sizeof(MAX_WORKSPACE));

            // Get best algorithm
            cublasLtMatmulHeuristicResult_struct heuristicResult;
            int returnedResults = 0;

            status = g_state.pfn_algo_get_heuristic(
                handle, cached.operationDesc,
                cached.Bdesc, cached.Adesc,  // Swapped for row-major
                cached.Cdesc, cached.Cdesc,
                preference, 1, &heuristicResult, &returnedResults
            );

            fprintf(stderr, "[cuBLASLt] AlgoGetHeuristic: status=%d, returnedResults=%d\n",
                    static_cast<int>(status), returnedResults);

            if (status == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
                // Store the selected algorithm
                cached.algo = heuristicResult.algo;
                cached.workspaceSize = heuristicResult.workspaceSize;
                cached.hasAlgo = true;

                // Allocate fixed workspace if needed (using Driver API)
                if (cached.workspaceSize > 0) {
                    CUdeviceptr dptr = 0;
                    CUresult err = cuMemAlloc(&dptr, cached.workspaceSize);
                    if (err != CUDA_SUCCESS) {
                        cached.workspace = nullptr;
                        cached.workspaceSize = 0;
                        // Still valid, just without workspace
                    } else {
                        cached.workspace = reinterpret_cast<void*>(dptr);
                    }
                }

                fprintf(stderr, "[cuBLASLt] Cached algo for M=%d N=%d K=%d, workspace=%zu bytes\n",
                        M, N, K, cached.workspaceSize);
            }

            g_state.pfn_pref_destroy(preference);
        }
    }

    cached.valid = true;
    return &cached;
}

}  // anonymous namespace

cudaError_t gemm_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    g_last_cublaslt_error = 0;
    g_last_cublaslt_step = 0;

    cublasLtHandle_t handle = get_handle();
    if (!handle) {
        g_last_cublaslt_step = 1;
        g_last_cublaslt_error = -1;
        return cudaErrorNotReady;
    }

    // Get cached descriptors (creates if needed)
    GemmCachedDesc* cached = get_cached_desc(M, N, K, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if (!cached || !cached->valid) {
        g_last_cublaslt_step = 2;
        g_last_cublaslt_error = -2;
        return cudaErrorUnknown;
    }

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    // Use cached algorithm and workspace for CUDA Graph compatibility
    // If no algorithm was cached, pass nullptr (cuBLASLt will pick one)
    const cublasLtMatmulAlgo_t* algo_ptr = cached->hasAlgo ? &cached->algo : nullptr;
    void* workspace = cached->workspace;
    size_t workspaceSize = cached->workspaceSize;

    // Direct function pointer call for maximum performance
    cublasStatus_t status = g_state.pfn_matmul(
        handle, cached->operationDesc,
        &alpha,
        B, cached->Bdesc,
        A, cached->Adesc,
        &beta,
        C, cached->Cdesc,
        C, cached->Cdesc,
        algo_ptr, workspace, workspaceSize, stream
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        g_last_cublaslt_step = 6;
        g_last_cublaslt_error = static_cast<int>(status);
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

cudaError_t gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    g_last_cublaslt_error = 0;
    g_last_cublaslt_step = 0;

    cublasLtHandle_t handle = get_handle();
    if (!handle) {
        g_last_cublaslt_step = 1;
        g_last_cublaslt_error = -1;
        return cudaErrorNotReady;
    }

    // Get cached descriptors (creates if needed)
    GemmCachedDesc* cached = get_cached_desc(M, N, K, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (!cached || !cached->valid) {
        g_last_cublaslt_step = 2;
        g_last_cublaslt_error = -2;
        return cudaErrorUnknown;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Use cached algorithm and workspace for CUDA Graph compatibility
    const cublasLtMatmulAlgo_t* algo_ptr = cached->hasAlgo ? &cached->algo : nullptr;
    void* workspace = cached->workspace;
    size_t workspaceSize = cached->workspaceSize;

    // Direct function pointer call for maximum performance
    cublasStatus_t status = g_state.pfn_matmul(
        handle, cached->operationDesc,
        &alpha,
        B, cached->Bdesc,
        A, cached->Adesc,
        &beta,
        C, cached->Cdesc,
        C, cached->Cdesc,
        algo_ptr, workspace, workspaceSize, stream
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        g_last_cublaslt_step = 6;
        g_last_cublaslt_error = static_cast<int>(status);
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

cudaError_t gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    g_last_cublaslt_error = 0;
    g_last_cublaslt_step = 0;

    cublasLtHandle_t handle = get_handle();
    if (!handle) {
        g_last_cublaslt_step = 1;
        g_last_cublaslt_error = -1;
        return cudaErrorNotReady;
    }

    // Get cached descriptors (creates if needed)
    GemmCachedDesc* cached = get_cached_desc(M, N, K, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (!cached || !cached->valid) {
        g_last_cublaslt_step = 2;
        g_last_cublaslt_error = -2;
        return cudaErrorUnknown;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Use cached algorithm and workspace for CUDA Graph compatibility
    const cublasLtMatmulAlgo_t* algo_ptr = cached->hasAlgo ? &cached->algo : nullptr;
    void* workspace = cached->workspace;
    size_t workspaceSize = cached->workspaceSize;

    // Direct function pointer call for maximum performance
    cublasStatus_t status = g_state.pfn_matmul(
        handle, cached->operationDesc,
        &alpha,
        B, cached->Bdesc,
        A, cached->Adesc,
        &beta,
        C, cached->Cdesc,
        C, cached->Cdesc,
        algo_ptr, workspace, workspaceSize, stream
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        g_last_cublaslt_step = 6;
        g_last_cublaslt_error = static_cast<int>(status);
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

}  // namespace cublaslt
}  // namespace pygpukit
