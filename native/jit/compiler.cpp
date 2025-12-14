#include "compiler.hpp"
#include <nvrtc.h>
#include <vector>
#include <atomic>

namespace pygpukit {

namespace {

// Cached NVRTC availability check result
// -1 = not checked, 0 = not available, 1 = available
std::atomic<int> g_nvrtc_available{-1};

void check_nvrtc_error(nvrtcResult result, const char* msg) {
    if (result != NVRTC_SUCCESS) {
        throw NvrtcError(std::string(msg) + ": " + nvrtcGetErrorString(result));
    }
}

void ensure_nvrtc_available() {
    if (!is_nvrtc_available()) {
        throw NvrtcError(
            "NVRTC is not available. JIT compilation requires CUDA Toolkit installation. "
            "Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads "
            "or use pre-compiled kernels."
        );
    }
}

} // anonymous namespace

bool is_nvrtc_available() {
    int cached = g_nvrtc_available.load(std::memory_order_relaxed);
    if (cached >= 0) {
        return cached == 1;
    }

    // Try to call nvrtcVersion to check if NVRTC is functional
    try {
        int major = 0, minor = 0;
        nvrtcResult result = nvrtcVersion(&major, &minor);
        bool available = (result == NVRTC_SUCCESS && major > 0);
        g_nvrtc_available.store(available ? 1 : 0, std::memory_order_relaxed);
        return available;
    } catch (...) {
        // DLL not loaded or other error
        g_nvrtc_available.store(0, std::memory_order_relaxed);
        return false;
    }
}

CompiledPTX compile_to_ptx(
    const std::string& source,
    const std::string& name,
    const std::vector<std::string>& options
) {
    ensure_nvrtc_available();

    nvrtcProgram prog;
    nvrtcResult result;

    // Create program
    result = nvrtcCreateProgram(
        &prog,
        source.c_str(),
        name.c_str(),
        0,      // numHeaders
        nullptr, // headers
        nullptr  // includeNames
    );
    check_nvrtc_error(result, "Failed to create NVRTC program");

    // Convert options to char**
    std::vector<const char*> opt_ptrs;
    for (const auto& opt : options) {
        opt_ptrs.push_back(opt.c_str());
    }

    // Compile
    result = nvrtcCompileProgram(
        prog,
        static_cast<int>(opt_ptrs.size()),
        opt_ptrs.empty() ? nullptr : opt_ptrs.data()
    );

    // Get log regardless of success/failure
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 1) {
        nvrtcGetProgramLog(prog, &log[0]);
    }

    if (result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        throw NvrtcError("Compilation failed: " + log);
    }

    // Get PTX
    size_t ptx_size;
    result = nvrtcGetPTXSize(prog, &ptx_size);
    check_nvrtc_error(result, "Failed to get PTX size");

    std::string ptx(ptx_size, '\0');
    result = nvrtcGetPTX(prog, &ptx[0]);
    check_nvrtc_error(result, "Failed to get PTX");

    nvrtcDestroyProgram(&prog);

    CompiledPTX compiled;
    compiled.ptx = std::move(ptx);
    compiled.log = std::move(log);
    return compiled;
}

void get_nvrtc_version(int* major, int* minor) {
    ensure_nvrtc_available();
    nvrtcResult result = nvrtcVersion(major, minor);
    check_nvrtc_error(result, "Failed to get NVRTC version");
}

} // namespace pygpukit
