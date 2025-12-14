// Dynamic NVRTC Loader Implementation
// Loads NVRTC at runtime using LoadLibrary (Windows) or dlopen (Linux)

#include "nvrtc_loader.hpp"
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#endif

namespace pygpukit {
namespace nvrtc {

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

// NVRTC function pointer types (matching nvrtc.h)
using nvrtcResult = int;
using nvrtcProgram = void*;

using PFN_nvrtcVersion = nvrtcResult (*)(int*, int*);
using PFN_nvrtcCreateProgram = nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
using PFN_nvrtcDestroyProgram = nvrtcResult (*)(nvrtcProgram*);
using PFN_nvrtcCompileProgram = nvrtcResult (*)(nvrtcProgram, int, const char* const*);
using PFN_nvrtcGetPTXSize = nvrtcResult (*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetPTX = nvrtcResult (*)(nvrtcProgram, char*);
using PFN_nvrtcGetProgramLogSize = nvrtcResult (*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetProgramLog = nvrtcResult (*)(nvrtcProgram, char*);
using PFN_nvrtcGetErrorString = const char* (*)(nvrtcResult);

// Global state
struct NvrtcState {
    std::atomic<bool> initialized{false};
    std::atomic<bool> available{false};
    std::mutex init_mutex;
    LibHandle handle{nullptr};
    std::string library_path;
    int version_major{0};
    int version_minor{0};

    // Function pointers
    PFN_nvrtcVersion pfn_version{nullptr};
    PFN_nvrtcCreateProgram pfn_create_program{nullptr};
    PFN_nvrtcDestroyProgram pfn_destroy_program{nullptr};
    PFN_nvrtcCompileProgram pfn_compile_program{nullptr};
    PFN_nvrtcGetPTXSize pfn_get_ptx_size{nullptr};
    PFN_nvrtcGetPTX pfn_get_ptx{nullptr};
    PFN_nvrtcGetProgramLogSize pfn_get_program_log_size{nullptr};
    PFN_nvrtcGetProgramLog pfn_get_program_log{nullptr};
    PFN_nvrtcGetErrorString pfn_get_error_string{nullptr};
};

NvrtcState g_state;

// Search for NVRTC library in various locations
std::vector<std::string> get_search_paths() {
    std::vector<std::string> paths;

#ifdef _WIN32
    // Windows: Search for nvrtc64_*.dll

    // 1. Check CUDA_PATH environment variable
    const char* cuda_path = std::getenv("CUDA_PATH");
    if (cuda_path) {
        paths.push_back(std::string(cuda_path) + "\\bin");
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

    // 3. Common installation paths
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin");
    paths.push_back("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin");

#else
    // Linux/macOS: Search for libnvrtc.so

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
// Find NVRTC DLL in a directory (Windows)
// Returns full path if found, empty string otherwise
std::string find_nvrtc_in_dir(const std::string& dir) {
    // Search for nvrtc64_*.dll pattern
    WIN32_FIND_DATAA find_data;
    std::string pattern = dir + "\\nvrtc64_*.dll";
    HANDLE find_handle = FindFirstFileA(pattern.c_str(), &find_data);

    if (find_handle != INVALID_HANDLE_VALUE) {
        std::string result = dir + "\\" + find_data.cFileName;
        FindClose(find_handle);
        return result;
    }

    // Also try exact name nvrtc64.dll (older versions)
    std::string exact_path = dir + "\\nvrtc64.dll";
    if (GetFileAttributesA(exact_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
        return exact_path;
    }

    return "";
}
#else
// Find NVRTC shared library in a directory (Linux)
std::string find_nvrtc_in_dir(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return "";

    std::string result;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        // Match libnvrtc.so or libnvrtc.so.*
        if (name.find("libnvrtc.so") == 0) {
            result = dir + "/" + name;
            break;
        }
    }
    closedir(d);
    return result;
}
#endif

// Try to load NVRTC from a specific path
bool try_load(const std::string& path) {
    LibHandle handle = LOAD_LIBRARY(path.c_str());
    if (!handle) {
        return false;
    }

    // Resolve all required functions
    auto pfn_version = (PFN_nvrtcVersion)GET_PROC(handle, "nvrtcVersion");
    auto pfn_create = (PFN_nvrtcCreateProgram)GET_PROC(handle, "nvrtcCreateProgram");
    auto pfn_destroy = (PFN_nvrtcDestroyProgram)GET_PROC(handle, "nvrtcDestroyProgram");
    auto pfn_compile = (PFN_nvrtcCompileProgram)GET_PROC(handle, "nvrtcCompileProgram");
    auto pfn_ptx_size = (PFN_nvrtcGetPTXSize)GET_PROC(handle, "nvrtcGetPTXSize");
    auto pfn_ptx = (PFN_nvrtcGetPTX)GET_PROC(handle, "nvrtcGetPTX");
    auto pfn_log_size = (PFN_nvrtcGetProgramLogSize)GET_PROC(handle, "nvrtcGetProgramLogSize");
    auto pfn_log = (PFN_nvrtcGetProgramLog)GET_PROC(handle, "nvrtcGetProgramLog");
    auto pfn_error = (PFN_nvrtcGetErrorString)GET_PROC(handle, "nvrtcGetErrorString");

    // All core functions must be present
    if (!pfn_version || !pfn_create || !pfn_destroy || !pfn_compile ||
        !pfn_ptx_size || !pfn_ptx || !pfn_log_size || !pfn_log) {
        FREE_LIBRARY(handle);
        return false;
    }

    // Verify it works by getting version
    int major = 0, minor = 0;
    if (pfn_version(&major, &minor) != 0 || major == 0) {
        FREE_LIBRARY(handle);
        return false;
    }

    // Success! Store everything
    g_state.handle = handle;
    g_state.library_path = path;
    g_state.version_major = major;
    g_state.version_minor = minor;
    g_state.pfn_version = pfn_version;
    g_state.pfn_create_program = pfn_create;
    g_state.pfn_destroy_program = pfn_destroy;
    g_state.pfn_compile_program = pfn_compile;
    g_state.pfn_get_ptx_size = pfn_ptx_size;
    g_state.pfn_get_ptx = pfn_ptx;
    g_state.pfn_get_program_log_size = pfn_log_size;
    g_state.pfn_get_program_log = pfn_log;
    g_state.pfn_get_error_string = pfn_error;  // May be null (optional)

    return true;
}

} // anonymous namespace

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

    // Search for NVRTC
    auto search_paths = get_search_paths();

    for (const auto& dir : search_paths) {
        std::string nvrtc_path = find_nvrtc_in_dir(dir);
        if (!nvrtc_path.empty() && try_load(nvrtc_path)) {
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
    initialize();
    return g_state.available.load(std::memory_order_relaxed);
}

std::string get_library_path() {
    initialize();
    return g_state.library_path;
}

std::tuple<int, int> get_version() {
    initialize();
    return {g_state.version_major, g_state.version_minor};
}

Result create_program(
    Program* prog,
    const char* src,
    const char* name,
    int num_headers,
    const char* const* headers,
    const char* const* include_names
) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(
        g_state.pfn_create_program(prog, src, name, num_headers, headers, include_names)
    );
}

Result destroy_program(Program* prog) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(g_state.pfn_destroy_program(prog));
}

Result compile_program(
    Program prog,
    int num_options,
    const char* const* options
) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(g_state.pfn_compile_program(prog, num_options, options));
}

Result get_ptx_size(Program prog, size_t* ptx_size) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(g_state.pfn_get_ptx_size(prog, ptx_size));
}

Result get_ptx(Program prog, char* ptx) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(g_state.pfn_get_ptx(prog, ptx));
}

Result get_program_log_size(Program prog, size_t* log_size) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(g_state.pfn_get_program_log_size(prog, log_size));
}

Result get_program_log(Program prog, char* log) {
    if (!is_available()) return Result::NotLoaded;
    return static_cast<Result>(g_state.pfn_get_program_log(prog, log));
}

const char* get_error_string(Result result) {
    if (result == Result::NotLoaded) {
        return "NVRTC not loaded";
    }
    if (!is_available() || !g_state.pfn_get_error_string) {
        return "Unknown error";
    }
    return g_state.pfn_get_error_string(static_cast<int>(result));
}

} // namespace nvrtc
} // namespace pygpukit
