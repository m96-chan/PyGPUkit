#include "compiler.hpp"
#include "nvrtc_loader.hpp"
#include <vector>

namespace pygpukit {

namespace {

void check_nvrtc_error(nvrtc::Result result, const char* msg) {
    if (result != nvrtc::Result::Success) {
        throw NvrtcError(std::string(msg) + ": " + nvrtc::get_error_string(result));
    }
}

void ensure_nvrtc_available() {
    if (!is_nvrtc_available()) {
        throw NvrtcError(
            "NVRTC is not available. JIT compilation of custom kernels requires NVRTC. "
            "Pre-compiled GPU operations (matmul, add, mul) work without NVRTC. "
            "For custom kernels, see: https://developer.nvidia.com/cuda-downloads"
        );
    }
}

} // anonymous namespace

bool is_nvrtc_available() {
    return nvrtc::is_available();
}

std::string get_nvrtc_library_path() {
    return nvrtc::get_library_path();
}

CompiledPTX compile_to_ptx(
    const std::string& source,
    const std::string& name,
    const std::vector<std::string>& options
) {
    ensure_nvrtc_available();

    nvrtc::Program prog = nullptr;
    nvrtc::Result result;

    // Create program
    result = nvrtc::create_program(
        &prog,
        source.c_str(),
        name.c_str(),
        0,       // numHeaders
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
    result = nvrtc::compile_program(
        prog,
        static_cast<int>(opt_ptrs.size()),
        opt_ptrs.empty() ? nullptr : opt_ptrs.data()
    );

    // Get log regardless of success/failure
    size_t log_size = 0;
    nvrtc::get_program_log_size(prog, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 1) {
        nvrtc::get_program_log(prog, &log[0]);
    }

    if (result != nvrtc::Result::Success) {
        nvrtc::destroy_program(&prog);
        throw NvrtcError("Compilation failed: " + log);
    }

    // Get PTX
    size_t ptx_size = 0;
    result = nvrtc::get_ptx_size(prog, &ptx_size);
    check_nvrtc_error(result, "Failed to get PTX size");

    std::string ptx(ptx_size, '\0');
    result = nvrtc::get_ptx(prog, &ptx[0]);
    check_nvrtc_error(result, "Failed to get PTX");

    nvrtc::destroy_program(&prog);

    CompiledPTX compiled;
    compiled.ptx = std::move(ptx);
    compiled.log = std::move(log);
    return compiled;
}

void get_nvrtc_version(int* major, int* minor) {
    ensure_nvrtc_available();
    auto [maj, min] = nvrtc::get_version();
    *major = maj;
    *minor = min;
}

} // namespace pygpukit
