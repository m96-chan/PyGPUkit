#include "device.hpp"
#include "types.hpp"

#ifdef PYGPUKIT_DRIVER_ONLY
// Driver-only mode: Use CUDA Driver API
#include "driver_context.hpp"
#include <cuda.h>

namespace pygpukit {

namespace {

void check_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        throw CudaError(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

} // anonymous namespace

bool is_cuda_available() {
    return driver::DriverContext::instance().is_available();
}

int get_driver_version() {
    int version = 0;
    check_driver_error(cuDriverGetVersion(&version), "Failed to get driver version");
    return version;
}

int get_runtime_version() {
    // No runtime in driver-only mode
    return 0;
}

int get_device_count() {
    return driver::DriverContext::instance().device_count();
}

DeviceProperties get_device_properties(int device_id) {
    auto& ctx = driver::DriverContext::instance();
    CUdevice device = ctx.get_device(device_id);

    DeviceProperties result;

    // Get device name
    char name[256];
    check_driver_error(cuDeviceGetName(name, sizeof(name), device), "Failed to get device name");
    result.name = name;

    // Get total memory
    size_t total_mem;
    check_driver_error(cuDeviceTotalMem(&total_mem, device), "Failed to get device memory");
    result.total_memory = total_mem;

    // Get compute capability
    int major, minor;
    check_driver_error(
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "Failed to get compute capability major"
    );
    check_driver_error(
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "Failed to get compute capability minor"
    );
    result.compute_capability_major = major;
    result.compute_capability_minor = minor;

    // Get multiprocessor count
    int mp_count;
    check_driver_error(
        cuDeviceGetAttribute(&mp_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
        "Failed to get multiprocessor count"
    );
    result.multiprocessor_count = mp_count;

    // Get max threads per block
    int max_threads;
    check_driver_error(
        cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device),
        "Failed to get max threads per block"
    );
    result.max_threads_per_block = max_threads;

    // Get warp size
    int warp_size;
    check_driver_error(
        cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device),
        "Failed to get warp size"
    );
    result.warp_size = warp_size;

    return result;
}

void set_device(int device_id) {
    driver::DriverContext::instance().set_current(device_id);
}

int get_current_device() {
    return driver::DriverContext::instance().current_device();
}

void device_synchronize() {
    driver::DriverContext::instance().synchronize();
}

int get_sm_version(int device_id) {
    auto& ctx = driver::DriverContext::instance();
    CUdevice device = ctx.get_device(device_id);

    int major, minor;
    check_driver_error(
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "Failed to get compute capability major"
    );
    check_driver_error(
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "Failed to get compute capability minor"
    );
    return major * 10 + minor;
}

void validate_compute_capability(int device_id) {
    int sm = get_sm_version(device_id);
    if (sm < 80) {
        DeviceProperties props = get_device_properties(device_id);
        throw std::runtime_error(
            "PyGPUkit requires SM >= 80 (Ampere or newer). "
            "Found: " + props.name + " with SM " +
            std::to_string(props.compute_capability_major) + "." +
            std::to_string(props.compute_capability_minor) +
            ". Older GPUs (Pascal, Turing, etc.) are not supported."
        );
    }
}

} // namespace pygpukit

#else
// Standard mode: Use CUDA Runtime API
#include <cuda_runtime.h>

namespace pygpukit {

namespace {

// Check CUDA error and throw if failed
void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw CudaError(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

} // anonymous namespace

bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int get_driver_version() {
    int version = 0;
    cudaError_t err = cudaDriverGetVersion(&version);
    check_cuda_error(err, "Failed to get driver version");
    return version;
}

int get_runtime_version() {
    int version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&version);
    check_cuda_error(err, "Failed to get runtime version");
    return version;
}

int get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    check_cuda_error(err, "Failed to get device count");
    return count;
}

DeviceProperties get_device_properties(int device_id) {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    check_cuda_error(err, "Failed to get device properties");

    DeviceProperties result;
    result.name = props.name;
    result.total_memory = props.totalGlobalMem;
    result.compute_capability_major = props.major;
    result.compute_capability_minor = props.minor;
    result.multiprocessor_count = props.multiProcessorCount;
    result.max_threads_per_block = props.maxThreadsPerBlock;
    result.warp_size = props.warpSize;

    return result;
}

void set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    check_cuda_error(err, "Failed to set device");
}

int get_current_device() {
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    check_cuda_error(err, "Failed to get current device");
    return device_id;
}

void device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    check_cuda_error(err, "Failed to synchronize device");
}

int get_sm_version(int device_id) {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    check_cuda_error(err, "Failed to get device properties");
    return props.major * 10 + props.minor;
}

void validate_compute_capability(int device_id) {
    int sm = get_sm_version(device_id);
    if (sm < 80) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        throw std::runtime_error(
            "PyGPUkit requires SM >= 80 (Ampere or newer). "
            "Found: " + std::string(props.name) + " with SM " +
            std::to_string(props.major) + "." + std::to_string(props.minor) +
            ". Older GPUs (Pascal, Turing, etc.) are not supported."
        );
    }
}

} // namespace pygpukit

#endif // PYGPUKIT_DRIVER_ONLY
