// JIT kernel management using CUDA Driver API
// PyGPUkit v0.2.4+: Single-binary distribution (driver-only mode)

#pragma once

#include "../core/types.hpp"
#include "../core/stream.hpp"
#include <cuda.h>
#include <string>
#include <vector>
#include <memory>

// Driver-only mode: define our own Dim3 struct
struct Dim3 {
    unsigned int x, y, z;
    Dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

namespace pygpukit {

// Forward declaration
class JITKernel;

// Kernel launch configuration
struct LaunchConfig {
    Dim3 grid;
    Dim3 block;
    size_t shared_mem;
    StreamHandle stream;

    LaunchConfig()
        : grid(1), block(256), shared_mem(0), stream(nullptr) {}

    LaunchConfig(unsigned int grid_x, unsigned int block_x)
        : grid(grid_x), block(block_x), shared_mem(0), stream(nullptr) {}

    LaunchConfig(Dim3 g, Dim3 b, size_t smem = 0, StreamHandle s = nullptr)
        : grid(g), block(b), shared_mem(smem), stream(s) {}
};

// JIT-compiled CUDA kernel
class JITKernel {
public:
    JITKernel(const std::string& source,
              const std::string& func_name,
              const std::vector<std::string>& options = {});
    ~JITKernel();

    // Disable copy
    JITKernel(const JITKernel&) = delete;
    JITKernel& operator=(const JITKernel&) = delete;

    // Enable move
    JITKernel(JITKernel&& other) noexcept;
    JITKernel& operator=(JITKernel&& other) noexcept;

    // Accessors
    const std::string& name() const { return func_name_; }
    const std::string& ptx() const { return ptx_; }
    bool is_compiled() const { return function_ != nullptr; }

    // Launch kernel with raw arguments
    void launch(const LaunchConfig& config, void** args);

    // Get suggested block size for this kernel
    int get_suggested_block_size(size_t dynamic_smem = 0) const;

private:
    std::string source_;
    std::string func_name_;
    std::string ptx_;
    CUmodule module_;
    CUfunction function_;

    void compile(const std::vector<std::string>& options);
    void cleanup();
};

} // namespace pygpukit
