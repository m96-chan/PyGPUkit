#pragma once
/**
 * CUDA Driver Context Manager for PyGPUkit
 *
 * This header provides a singleton context manager for Driver API operations.
 * The context is lazily initialized on first use.
 *
 * In driver-only mode, all CUDA operations must be done through a context.
 * This manager ensures proper initialization and cleanup.
 */

#include <cuda.h>
#include <stdexcept>
#include <string>
#include <mutex>
#include <vector>
#include "driver_api.hpp"

namespace pygpukit {
namespace driver {

/**
 * Singleton context manager for CUDA Driver API
 *
 * Manages CUDA initialization, device selection, and context lifecycle.
 * Thread-safe initialization via std::call_once.
 */
class DriverContext {
public:
    /**
     * Get singleton instance
     */
    static DriverContext& instance() {
        static DriverContext ctx;
        return ctx;
    }

    /**
     * Ensure CUDA Driver API is initialized
     * Safe to call multiple times
     */
    void ensure_init() {
        std::call_once(init_flag_, [this]() {
            CUresult result = cuInit(0);
            if (result != CUDA_SUCCESS) {
                initialized_ = false;
                const char* error_str = nullptr;
                cuGetErrorString(result, &error_str);
                init_error_ = std::string("Failed to initialize CUDA Driver API: ") +
                              (error_str ? error_str : "unknown error");
                return;
            }
            initialized_ = true;

            // Get device count
            int count = 0;
            result = cuDeviceGetCount(&count);
            if (result != CUDA_SUCCESS || count == 0) {
                device_count_ = 0;
                has_device_ = false;
                return;
            }

            device_count_ = count;
            has_device_ = true;
        });

        if (!initialized_) {
            throw std::runtime_error(init_error_);
        }
    }

    /**
     * Check if CUDA is available (initialized and has devices)
     */
    bool is_available() {
        try {
            ensure_init();
            return has_device_;
        } catch (...) {
            return false;
        }
    }

    /**
     * Get device count
     */
    int device_count() {
        ensure_init();
        return device_count_;
    }

    /**
     * Get or create context for a device
     * Creates primary context on first call
     */
    CUcontext get_context(int device_id = 0) {
        ensure_init();

        if (device_id < 0 || device_id >= device_count_) {
            throw std::runtime_error("Invalid device ID: " + std::to_string(device_id));
        }

        std::lock_guard<std::mutex> lock(context_mutex_);

        // Check if context already exists for this device
        if (device_id < static_cast<int>(contexts_.size()) && contexts_[device_id] != nullptr) {
            return contexts_[device_id];
        }

        // Ensure vector is large enough
        if (static_cast<int>(contexts_.size()) <= device_id) {
            contexts_.resize(device_id + 1, nullptr);
            devices_.resize(device_id + 1);
        }

        // Get device handle
        CUdevice device;
        check_driver_error(cuDeviceGet(&device, device_id), "Failed to get device");
        devices_[device_id] = device;

        // Retain primary context (shared with runtime API if used together)
        CUcontext ctx;
        check_driver_error(cuDevicePrimaryCtxRetain(&ctx, device), "Failed to retain primary context");
        contexts_[device_id] = ctx;

        return ctx;
    }

    /**
     * Set current context for the calling thread
     */
    void set_current(int device_id = 0) {
        CUcontext ctx = get_context(device_id);
        check_driver_error(cuCtxSetCurrent(ctx), "Failed to set current context");
        current_device_ = device_id;
    }

    /**
     * Get current device ID
     */
    int current_device() const {
        return current_device_;
    }

    /**
     * Get device handle
     */
    CUdevice get_device(int device_id = 0) {
        ensure_init();
        if (device_id < 0 || device_id >= device_count_) {
            throw std::runtime_error("Invalid device ID");
        }

        // Ensure context is created (which populates devices_)
        get_context(device_id);

        return devices_[device_id];
    }

    /**
     * Synchronize current context
     */
    void synchronize() {
        check_driver_error(cuCtxSynchronize(), "Failed to synchronize context");
    }

    // Prevent copying
    DriverContext(const DriverContext&) = delete;
    DriverContext& operator=(const DriverContext&) = delete;

private:
    DriverContext() = default;

    ~DriverContext() {
        // Release all retained primary contexts
        for (size_t i = 0; i < contexts_.size(); ++i) {
            if (contexts_[i] != nullptr && i < devices_.size()) {
                cuDevicePrimaryCtxRelease(devices_[i]);
            }
        }
    }

    std::once_flag init_flag_;
    bool initialized_ = false;
    bool has_device_ = false;
    int device_count_ = 0;
    std::string init_error_;

    std::mutex context_mutex_;
    std::vector<CUcontext> contexts_;
    std::vector<CUdevice> devices_;
    int current_device_ = 0;
};

/**
 * RAII guard to ensure context is set for current thread
 */
class ContextGuard {
public:
    explicit ContextGuard(int device_id = 0) {
        DriverContext::instance().set_current(device_id);
    }
};

} // namespace driver
} // namespace pygpukit
