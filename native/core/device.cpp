#include "device.hpp"
#include "types.hpp"
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
