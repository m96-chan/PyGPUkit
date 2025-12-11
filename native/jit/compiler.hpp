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

// Compile CUDA source to PTX using NVRTC
CompiledPTX compile_to_ptx(
    const std::string& source,
    const std::string& name = "kernel.cu",
    const std::vector<std::string>& options = {}
);

// Get NVRTC version
void get_nvrtc_version(int* major, int* minor);

} // namespace pygpukit
