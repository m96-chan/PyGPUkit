// CUDA Event implementation using CUDA Driver API
// PyGPUkit v0.2.11+

#include "event.hpp"
#include "driver_context.hpp"

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

CudaEvent::CudaEvent(bool blocking_sync) : event_(nullptr) {
    // Ensure context is initialized
    driver::DriverContext::instance().set_current();

    unsigned int flags = CU_EVENT_DEFAULT;
    if (!blocking_sync) {
        // CU_EVENT_DISABLE_TIMING is NOT set - we need timing
        // CU_EVENT_BLOCKING_SYNC disabled for non-blocking CPU behavior
        flags = CU_EVENT_DEFAULT;
    } else {
        flags = CU_EVENT_BLOCKING_SYNC;
    }

    check_driver_error(cuEventCreate(&event_, flags), "Failed to create CUDA event");
}

CudaEvent::~CudaEvent() {
    if (event_ != nullptr) {
        cuEventDestroy(event_);
    }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        if (event_ != nullptr) {
            cuEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record(const Stream& stream) {
    check_driver_error(cuEventRecord(event_, stream.handle()), "Failed to record event");
}

void CudaEvent::record() {
    // Record on default stream (nullptr)
    check_driver_error(cuEventRecord(event_, nullptr), "Failed to record event on default stream");
}

void CudaEvent::synchronize() {
    check_driver_error(cuEventSynchronize(event_), "Failed to synchronize event");
}

bool CudaEvent::query() const {
    CUresult result = cuEventQuery(event_);
    if (result == CUDA_SUCCESS) {
        return true;
    } else if (result == CUDA_ERROR_NOT_READY) {
        return false;
    }
    check_driver_error(result, "Failed to query event");
    return false;  // unreachable
}

float event_elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float ms = 0.0f;
    check_driver_error(
        cuEventElapsedTime(&ms, start.handle(), stop.handle()),
        "Failed to get elapsed time between events"
    );
    return ms;
}

float event_elapsed_us(const CudaEvent& start, const CudaEvent& stop) {
    return event_elapsed_ms(start, stop) * 1000.0f;
}

} // namespace pygpukit
