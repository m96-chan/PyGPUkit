/**
 * Device capability helpers
 */
#pragma once

#include <cuda.h>
#include "../../core/driver_context.hpp"

namespace pygpukit {
namespace ops {

// Get SM version (e.g., 80 for SM 8.0)
inline int get_sm_version() {
    auto& ctx = driver::DriverContext::instance();
    CUdevice device = ctx.get_device(ctx.current_device());
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    return major * 10 + minor;
}

} // namespace ops
} // namespace pygpukit
