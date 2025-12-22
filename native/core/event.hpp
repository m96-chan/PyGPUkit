// CUDA Event for GPU timing
// PyGPUkit v0.2.11+

#pragma once

#include "types.hpp"
#include "stream.hpp"
#include <cuda.h>

namespace pygpukit {

// CUDA Event wrapper for GPU-side timing
class CudaEvent {
public:
    // Create event with optional flags
    // Default: blocking sync disabled for better performance
    explicit CudaEvent(bool blocking_sync = false);
    ~CudaEvent();

    // Disable copy
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Enable move
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    // Record event in a stream
    void record(const Stream& stream);

    // Record event in default stream
    void record();

    // Synchronize (wait for event to complete)
    void synchronize();

    // Check if event has completed (non-blocking)
    bool query() const;

    // Get raw handle
    CUevent handle() const { return event_; }

private:
    CUevent event_;
};

// Calculate elapsed time between two events in milliseconds
// start must be recorded before stop
float event_elapsed_ms(const CudaEvent& start, const CudaEvent& stop);

// Calculate elapsed time between two events in microseconds
float event_elapsed_us(const CudaEvent& start, const CudaEvent& stop);

} // namespace pygpukit
