// Device management using CUDA Driver API
// PyGPUkit v0.2.4+: Single-binary distribution (driver-only mode)

#include "device.hpp"
#include "types.hpp"
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

std::string get_recommended_arch(int device_id) {
    int sm = get_sm_version(device_id);
    int driver_version = get_driver_version();

    // Driver version is MAJOR*1000 + MINOR*10
    // e.g., CUDA 12.4 = 12040, CUDA 11.8 = 11080

    // Clamp SM to what the driver supports
    // CUDA 12.x supports SM 90 (Hopper)
    // CUDA 11.8+ supports SM 90
    // CUDA 11.1-11.7 supports SM 86
    // CUDA 11.0 supports SM 80
    int max_supported_sm = 80;

    if (driver_version >= 12000) {
        max_supported_sm = 90;  // Hopper
    } else if (driver_version >= 11080) {
        max_supported_sm = 90;  // 11.8 added SM 90
    } else if (driver_version >= 11010) {
        max_supported_sm = 86;  // 11.1 added SM 86
    } else {
        max_supported_sm = 80;  // SM 80 baseline
    }

    // Use the minimum of actual SM and max supported
    int target_sm = std::min(sm, max_supported_sm);

    // Ensure minimum SM 80 for PyGPUkit
    if (target_sm < 80) {
        target_sm = 80;
    }

    return "sm_" + std::to_string(target_sm);
}

std::vector<std::string> get_fallback_archs(int device_id) {
    int sm = get_sm_version(device_id);
    std::vector<std::string> archs;

    // Start with the actual SM, then add fallbacks
    // Prefer SM versions, then compute versions (PTX only)

    // Add SM versions from current down to 80
    for (int target = sm; target >= 80; target -= (target > 86 ? 4 : 6)) {
        archs.push_back("sm_" + std::to_string(target));
        // Add specific versions
        if (target == 90) {
            // After 90, try 89 (Ada), then 86, then 80
            archs.push_back("sm_89");
        }
        if (target == 89 || target == 90) {
            archs.push_back("sm_86");
        }
    }

    // Finally add compute_80 as ultimate fallback (PTX only, JIT compiled by driver)
    if (archs.empty() || archs.back() != "sm_80") {
        archs.push_back("sm_80");
    }
    archs.push_back("compute_80");

    return archs;
}

bool is_arch_supported(const std::string& arch) {
    int driver_version = get_driver_version();

    // Parse SM version from arch string
    int sm_version = 0;
    if (arch.find("sm_") == 0 || arch.find("compute_") == 0) {
        size_t pos = arch.find('_');
        if (pos != std::string::npos) {
            try {
                sm_version = std::stoi(arch.substr(pos + 1));
            } catch (...) {
                return false;
            }
        }
    } else {
        return false;
    }

    // Check if driver supports this SM version
    int max_sm = 80;
    if (driver_version >= 12000) {
        max_sm = 90;
    } else if (driver_version >= 11080) {
        max_sm = 90;
    } else if (driver_version >= 11010) {
        max_sm = 86;
    }

    return sm_version <= max_sm && sm_version >= 80;
}

} // namespace pygpukit
