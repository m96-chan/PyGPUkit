#include "kernel.hpp"
#include "compiler.hpp"
#include "../core/driver_context.hpp"
#include <cuda.h>

namespace pygpukit {

namespace {

void check_cuda_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw CudaError(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

// Initialize CUDA driver API and set context (called once per thread)
void ensure_cuda_initialized() {
    // Always use unified context manager for proper context setup
    // This ensures cuModuleLoadData has an active context
    driver::DriverContext::instance().set_current();
}

} // anonymous namespace

JITKernel::JITKernel(const std::string& source,
                     const std::string& func_name,
                     const std::vector<std::string>& options)
    : source_(source),
      func_name_(func_name),
      module_(nullptr),
      function_(nullptr) {
    compile(options);
}

JITKernel::~JITKernel() {
    cleanup();
}

JITKernel::JITKernel(JITKernel&& other) noexcept
    : source_(std::move(other.source_)),
      func_name_(std::move(other.func_name_)),
      ptx_(std::move(other.ptx_)),
      module_(other.module_),
      function_(other.function_) {
    other.module_ = nullptr;
    other.function_ = nullptr;
}

JITKernel& JITKernel::operator=(JITKernel&& other) noexcept {
    if (this != &other) {
        cleanup();
        source_ = std::move(other.source_);
        func_name_ = std::move(other.func_name_);
        ptx_ = std::move(other.ptx_);
        module_ = other.module_;
        function_ = other.function_;
        other.module_ = nullptr;
        other.function_ = nullptr;
    }
    return *this;
}

void JITKernel::compile(const std::vector<std::string>& options) {
    ensure_cuda_initialized();

    // Compile source to PTX
    CompiledPTX compiled = compile_to_ptx(source_, "kernel.cu", options);
    ptx_ = std::move(compiled.ptx);

    // Load module from PTX
    CUresult result = cuModuleLoadData(&module_, ptx_.c_str());
    check_cuda_driver_error(result, "Failed to load module from PTX");

    // Get function handle
    result = cuModuleGetFunction(&function_, module_, func_name_.c_str());
    check_cuda_driver_error(result, "Failed to get function from module");
}

void JITKernel::cleanup() {
    if (module_ != nullptr) {
        cuModuleUnload(module_);
        module_ = nullptr;
        function_ = nullptr;
    }
}

void JITKernel::launch(const LaunchConfig& config, void** args) {
    if (function_ == nullptr) {
        throw CudaError("Kernel not compiled");
    }

    CUresult result = cuLaunchKernel(
        function_,
        config.grid.x, config.grid.y, config.grid.z,
        config.block.x, config.block.y, config.block.z,
        config.shared_mem,
        config.stream,
        args,
        nullptr  // extra
    );
    check_cuda_driver_error(result, "Failed to launch kernel");
}

int JITKernel::get_suggested_block_size(size_t dynamic_smem) const {
    if (function_ == nullptr) {
        return 256; // Default
    }

    int min_grid_size, block_size;
    CUresult result = cuOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        function_,
        nullptr,  // block size to shared mem mapping
        dynamic_smem,
        0  // block size limit
    );

    if (result != CUDA_SUCCESS) {
        return 256; // Default on error
    }

    return block_size;
}

} // namespace pygpukit
