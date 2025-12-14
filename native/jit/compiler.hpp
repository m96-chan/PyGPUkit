#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace pygpukit {

// Compiled PTX code
struct CompiledPTX {
    std::string ptx;
    std::string log;
};

// Check if NVRTC is available at runtime
// Returns true if NVRTC DLL/so is loaded and functional
bool is_nvrtc_available();

// Get the path to the loaded NVRTC library
// Returns empty string if NVRTC is not loaded
std::string get_nvrtc_library_path();

// Compile CUDA source to PTX using NVRTC
// Throws NvrtcError if NVRTC is not available
CompiledPTX compile_to_ptx(
    const std::string& source,
    const std::string& name = "kernel.cu",
    const std::vector<std::string>& options = {}
);

// Get NVRTC version
// Throws NvrtcError if NVRTC is not available
void get_nvrtc_version(int* major, int* minor);

} // namespace pygpukit
