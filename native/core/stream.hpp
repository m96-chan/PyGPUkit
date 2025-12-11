#pragma once

#include "types.hpp"
#include <cuda_runtime.h>

namespace pygpukit {

// Stream priority levels
enum class StreamPriority {
    High = 0,
    Low = 1
};

// CUDA Stream wrapper
class Stream {
public:
    explicit Stream(StreamPriority priority = StreamPriority::Low);
    ~Stream();

    // Disable copy
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Enable move
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;

    // Synchronize this stream
    void synchronize();

    // Get raw CUDA stream handle
    cudaStream_t handle() const { return stream_; }

    // Get priority
    StreamPriority priority() const { return priority_; }

private:
    cudaStream_t stream_;
    StreamPriority priority_;
};

// Get priority range supported by device
void get_stream_priority_range(int* least_priority, int* greatest_priority);

} // namespace pygpukit
