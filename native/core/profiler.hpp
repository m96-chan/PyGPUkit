// GPU Kernel Profiler using CUDA Driver API
// PyGPUkit v0.2.19+
//
// Provides accurate kernel timing by recording CUDA events
// directly in C++ without Python overhead.

#pragma once

#include "event.hpp"
#include "stream.hpp"
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <mutex>

namespace pygpukit {

// Record of a single kernel execution
struct KernelRecord {
    std::string name;
    float elapsed_ms;
    float elapsed_us;
    int64_t flops;           // -1 if not specified
    int64_t bytes;           // -1 if not specified
    double timestamp;        // Unix timestamp when recorded

    // Calculate TFLOPS (returns -1 if flops not set or time is 0)
    double tflops() const {
        if (flops < 0 || elapsed_ms <= 0) return -1.0;
        return (static_cast<double>(flops) / 1e12) / (elapsed_ms / 1000.0);
    }

    // Calculate bandwidth in GB/s (returns -1 if bytes not set or time is 0)
    double bandwidth_gb_s() const {
        if (bytes < 0 || elapsed_ms <= 0) return -1.0;
        return (static_cast<double>(bytes) / 1e9) / (elapsed_ms / 1000.0);
    }
};

// Scoped timer for automatic timing of kernel execution
// Uses RAII to ensure stop event is always recorded
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, int64_t flops = -1, int64_t bytes = -1);
    ~ScopedTimer();

    // Disable copy
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    // Get elapsed time (only valid after destructor or explicit stop)
    float elapsed_ms() const { return elapsed_ms_; }
    float elapsed_us() const { return elapsed_us_; }

    // Explicit stop (called automatically by destructor)
    void stop();

    // Get the record (only valid after stop)
    KernelRecord get_record() const;

private:
    std::string name_;
    int64_t flops_;
    int64_t bytes_;
    CudaEvent start_;
    CudaEvent stop_;
    float elapsed_ms_;
    float elapsed_us_;
    double timestamp_;
    bool stopped_;
};

// Kernel profiler that accumulates timing records
class KernelProfiler {
public:
    KernelProfiler();
    ~KernelProfiler() = default;

    // Enable/disable profiling (disabled profiling has minimal overhead)
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    // Start timing a kernel (returns timer that auto-stops on destruction)
    std::unique_ptr<ScopedTimer> start_timer(
        const std::string& name,
        int64_t flops = -1,
        int64_t bytes = -1
    );

    // Manual timing API (for more control)
    void record_start(const std::string& name, int64_t flops = -1, int64_t bytes = -1);
    void record_stop();

    // Add a pre-recorded kernel record
    void add_record(const KernelRecord& record);

    // Get all records
    const std::vector<KernelRecord>& records() const { return records_; }

    // Get records count
    size_t record_count() const { return records_.size(); }

    // Clear all records
    void clear();

    // Get total time in milliseconds
    float total_time_ms() const;

    // Summary statistics
    struct KernelStats {
        std::string name;
        int count;
        float total_ms;
        float avg_ms;
        float min_ms;
        float max_ms;
    };

    // Get summary grouped by kernel name
    std::vector<KernelStats> summary_by_name() const;

private:
    bool enabled_;
    std::vector<KernelRecord> records_;
    mutable std::mutex mutex_;

    // For manual timing API
    std::unique_ptr<CudaEvent> pending_start_;
    std::string pending_name_;
    int64_t pending_flops_;
    int64_t pending_bytes_;
    double pending_timestamp_;
};

// Global profiler instance (optional convenience)
KernelProfiler& global_profiler();

// Convenience macro for profiling (disabled in release builds if PYGPUKIT_PROFILE=0)
#ifndef PYGPUKIT_PROFILE
#define PYGPUKIT_PROFILE 1
#endif

#if PYGPUKIT_PROFILE
#define PYGPUKIT_PROFILE_KERNEL(name) \
    auto _pygpukit_timer = pygpukit::global_profiler().start_timer(name)
#define PYGPUKIT_PROFILE_KERNEL_FLOPS(name, flops) \
    auto _pygpukit_timer = pygpukit::global_profiler().start_timer(name, flops)
#else
#define PYGPUKIT_PROFILE_KERNEL(name) ((void)0)
#define PYGPUKIT_PROFILE_KERNEL_FLOPS(name, flops) ((void)0)
#endif

} // namespace pygpukit
