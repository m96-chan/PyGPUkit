/**
 * Device capability helpers
 */
#pragma once

#include <cuda.h>
#include "../../core/driver_context.hpp"

namespace pygpukit {
namespace ops {

// Minimum supported SM version (SM80 Ampere and newer)
constexpr int MIN_SM_VERSION = 80;

// Get SM version (e.g., 86 for SM 8.6)
inline int get_sm_version() {
    auto& ctx = driver::DriverContext::instance();
    CUdevice device = ctx.get_device(ctx.current_device());
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    return major * 10 + minor;
}

// Check if current device meets minimum SM version requirement
inline bool is_sm_supported() {
    return get_sm_version() >= MIN_SM_VERSION;
}

} // namespace ops
} // namespace pygpukit
