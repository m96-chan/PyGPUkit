#include "compiler.hpp"
#include <nvrtc.h>
#include <vector>

namespace pygpukit {

namespace {

void check_nvrtc_error(nvrtcResult result, const char* msg) {
    if (result != NVRTC_SUCCESS) {
        throw NvrtcError(std::string(msg) + ": " + nvrtcGetErrorString(result));
    }
}

} // anonymous namespace

CompiledPTX compile_to_ptx(
    const std::string& source,
    const std::string& name,
    const std::vector<std::string>& options
) {
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
    nvrtcResult result = nvrtcVersion(major, minor);
    check_nvrtc_error(result, "Failed to get NVRTC version");
}

} // namespace pygpukit
