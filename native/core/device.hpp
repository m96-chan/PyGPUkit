#pragma once

#include <string>
#include <optional>

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

} // namespace pygpukit
