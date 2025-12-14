#pragma once

#include <string>
#include <optional>
#include <vector>

namespace pygpukit {

// Device properties structure
struct DeviceProperties {
    std::string name;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
};

// Check if CUDA is available
bool is_cuda_available();

// Get CUDA driver version
int get_driver_version();

// Get CUDA runtime version
int get_runtime_version();

// Get number of CUDA devices
int get_device_count();

// Get properties of a device
DeviceProperties get_device_properties(int device_id = 0);

// Set current device
void set_device(int device_id);

// Get current device
int get_current_device();

// Synchronize current device
void device_synchronize();

// Validate device compute capability (requires SM >= 80)
// Throws std::runtime_error if device is too old
void validate_compute_capability(int device_id = 0);

// Get SM version as integer (e.g., 86 for SM 8.6)
int get_sm_version(int device_id = 0);

// Get recommended -arch option for JIT compilation (e.g., "sm_86")
// Based on current GPU's compute capability
std::string get_recommended_arch(int device_id = 0);

// Get fallback -arch options for older drivers (in order of preference)
// Returns list like ["sm_80", "compute_80"] for fallback
std::vector<std::string> get_fallback_archs(int device_id = 0);

// Check if driver supports a given PTX architecture
// arch should be like "sm_86" or "compute_80"
bool is_arch_supported(const std::string& arch);

} // namespace pygpukit
