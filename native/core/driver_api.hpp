#pragma once
/**
 * CUDA Driver API wrapper for PyGPUkit
 *
 * This header provides abstractions for Driver API calls that can replace
 * the Runtime API (cudart) calls for driver-only mode.
 *
 * Driver-Only Mode Benefits:
 * - No CUDA Toolkit installation required
 * - Only NVIDIA GPU driver needed
 * - NVRTC DLL bundled with wheel
 *
 * Migration Status:
 * - [x] Infrastructure created
 * - [ ] Memory management (cuMemAlloc, cuMemFree, cuMemcpy*)
 * - [ ] Device management (cuDeviceGet, cuDeviceGetAttribute)
 * - [ ] Stream management (cuStreamCreate, cuStreamSynchronize)
 * - [ ] Kernel launch (cuLaunchKernel)
 * - [ ] Error handling (cuGetErrorString)
 */

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace pygpukit {
namespace driver {

/**
 * Initialize CUDA Driver API
 * Must be called before any other driver API calls
 */
inline void init() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error(std::string("Failed to initialize CUDA Driver API: ") + error_str);
    }
}

/**
 * Check CUDA Driver API result and throw on error
 */
inline void check_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

/**
 * Get device count using Driver API
 */
inline int get_device_count() {
    int count = 0;
    CUresult result = cuDeviceGetCount(&count);
    if (result != CUDA_SUCCESS) {
        return 0;
    }
    return count;
}

/**
 * Get device handle
 */
inline CUdevice get_device(int device_id) {
    CUdevice device;
    check_driver_error(cuDeviceGet(&device, device_id), "Failed to get device");
    return device;
}

/**
 * Create CUDA context for a device
 */
inline CUcontext create_context(CUdevice device, unsigned int flags = 0) {
    CUcontext context;
    check_driver_error(cuCtxCreate(&context, flags, device), "Failed to create context");
    return context;
}

/**
 * Allocate device memory using Driver API
 */
inline CUdeviceptr mem_alloc(size_t size) {
    CUdeviceptr ptr;
    check_driver_error(cuMemAlloc(&ptr, size), "Failed to allocate device memory");
    return ptr;
}

/**
 * Free device memory
 */
inline void mem_free(CUdeviceptr ptr) {
    check_driver_error(cuMemFree(ptr), "Failed to free device memory");
}

/**
 * Copy host to device
 */
inline void mem_copy_htod(CUdeviceptr dst, const void* src, size_t size) {
    check_driver_error(cuMemcpyHtoD(dst, src, size), "Failed to copy H2D");
}

/**
 * Copy device to host
 */
inline void mem_copy_dtoh(void* dst, CUdeviceptr src, size_t size) {
    check_driver_error(cuMemcpyDtoH(dst, src, size), "Failed to copy D2H");
}

/**
 * Copy device to device
 */
inline void mem_copy_dtod(CUdeviceptr dst, CUdeviceptr src, size_t size) {
    check_driver_error(cuMemcpyDtoD(dst, src, size), "Failed to copy D2D");
}

/**
 * Create stream
 */
inline CUstream create_stream(unsigned int flags = CU_STREAM_NON_BLOCKING) {
    CUstream stream;
    check_driver_error(cuStreamCreate(&stream, flags), "Failed to create stream");
    return stream;
}

/**
 * Destroy stream
 */
inline void destroy_stream(CUstream stream) {
    check_driver_error(cuStreamDestroy(stream), "Failed to destroy stream");
}

/**
 * Synchronize stream
 */
inline void sync_stream(CUstream stream) {
    check_driver_error(cuStreamSynchronize(stream), "Failed to synchronize stream");
}

/**
 * Synchronize device (all streams)
 */
inline void sync_device() {
    check_driver_error(cuCtxSynchronize(), "Failed to synchronize device");
}

/**
 * Get device attribute
 */
inline int get_device_attribute(CUdevice_attribute attrib, CUdevice device) {
    int value;
    check_driver_error(cuDeviceGetAttribute(&value, attrib, device), "Failed to get device attribute");
    return value;
}

/**
 * Get device name
 */
inline std::string get_device_name(CUdevice device) {
    char name[256];
    check_driver_error(cuDeviceGetName(name, sizeof(name), device), "Failed to get device name");
    return std::string(name);
}

/**
 * Get device total memory
 */
inline size_t get_device_total_memory(CUdevice device) {
    size_t total;
    check_driver_error(cuDeviceTotalMem(&total, device), "Failed to get device memory");
    return total;
}

/**
 * Kernel launch configuration for Driver API
 */
struct DriverLaunchConfig {
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    unsigned int shared_mem;
    CUstream stream;

    DriverLaunchConfig()
        : grid_x(1), grid_y(1), grid_z(1)
        , block_x(256), block_y(1), block_z(1)
        , shared_mem(0), stream(nullptr) {}
};

/**
 * Launch kernel using Driver API
 *
 * @param func CUfunction handle from NVRTC compilation
 * @param config Launch configuration
 * @param args Kernel arguments (array of pointers to argument values)
 * @param n_args Number of arguments
 */
inline void launch_kernel(
    CUfunction func,
    const DriverLaunchConfig& config,
    void** args,
    size_t n_args
) {
    check_driver_error(
        cuLaunchKernel(
            func,
            config.grid_x, config.grid_y, config.grid_z,
            config.block_x, config.block_y, config.block_z,
            config.shared_mem,
            config.stream,
            args,
            nullptr  // extra params
        ),
        "Failed to launch kernel"
    );
}

} // namespace driver
} // namespace pygpukit
