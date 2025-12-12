#include "stream.hpp"

#ifdef PYGPUKIT_DRIVER_ONLY
// Driver-only mode: Use CUDA Driver API
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

Stream::Stream(StreamPriority priority)
    : stream_(nullptr), priority_(priority) {
    // Ensure context is initialized
    driver::DriverContext::instance().set_current();

    int cuda_priority = (priority == StreamPriority::High) ? -1 : 0;
    check_driver_error(
        cuStreamCreateWithPriority(&stream_, CU_STREAM_NON_BLOCKING, cuda_priority),
        "Failed to create stream"
    );
}

Stream::~Stream() {
    if (stream_ != nullptr) {
        cuStreamDestroy(stream_);
    }
}

Stream::Stream(Stream&& other) noexcept
    : stream_(other.stream_), priority_(other.priority_) {
    other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cuStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        priority_ = other.priority_;
        other.stream_ = nullptr;
    }
    return *this;
}

void Stream::synchronize() {
    check_driver_error(cuStreamSynchronize(stream_), "Failed to synchronize stream");
}

void get_stream_priority_range(int* least_priority, int* greatest_priority) {
    // Ensure context is initialized
    driver::DriverContext::instance().set_current();
    check_driver_error(
        cuCtxGetStreamPriorityRange(least_priority, greatest_priority),
        "Failed to get stream priority range"
    );
}

} // namespace pygpukit

#else
// Standard mode: Use CUDA Runtime API

namespace pygpukit {

namespace {

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw CudaError(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

} // anonymous namespace

Stream::Stream(StreamPriority priority)
    : stream_(nullptr), priority_(priority) {
    int cuda_priority = (priority == StreamPriority::High) ? -1 : 0;
    cudaError_t err = cudaStreamCreateWithPriority(
        &stream_, cudaStreamNonBlocking, cuda_priority);
    check_cuda_error(err, "Failed to create stream");
}

Stream::~Stream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

Stream::Stream(Stream&& other) noexcept
    : stream_(other.stream_), priority_(other.priority_) {
    other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        priority_ = other.priority_;
        other.stream_ = nullptr;
    }
    return *this;
}

void Stream::synchronize() {
    cudaError_t err = cudaStreamSynchronize(stream_);
    check_cuda_error(err, "Failed to synchronize stream");
}

void get_stream_priority_range(int* least_priority, int* greatest_priority) {
    cudaError_t err = cudaDeviceGetStreamPriorityRange(least_priority, greatest_priority);
    check_cuda_error(err, "Failed to get stream priority range");
}

} // namespace pygpukit

#endif // PYGPUKIT_DRIVER_ONLY
