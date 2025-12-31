// GPU Kernel Profiler implementation using CUDA Driver API
// PyGPUkit v0.2.19+

#include "profiler.hpp"
#include "driver_context.hpp"
#include <algorithm>
#include <unordered_map>

namespace pygpukit {

namespace {

double get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

} // anonymous namespace

// ============================================================================
// ScopedTimer implementation
// ============================================================================

ScopedTimer::ScopedTimer(const std::string& name, int64_t flops, int64_t bytes)
    : name_(name)
    , flops_(flops)
    , bytes_(bytes)
    , start_(false)  // non-blocking sync
    , stop_(false)
    , elapsed_ms_(0.0f)
    , elapsed_us_(0.0f)
    , timestamp_(get_timestamp())
    , stopped_(false)
{
    // Record start event immediately
    start_.record();
}

ScopedTimer::~ScopedTimer() {
    if (!stopped_) {
        stop();
    }
}

void ScopedTimer::stop() {
    if (stopped_) return;

    // Record stop event and synchronize
    stop_.record();
    stop_.synchronize();

    // Calculate elapsed time
    elapsed_ms_ = event_elapsed_ms(start_, stop_);
    elapsed_us_ = elapsed_ms_ * 1000.0f;
    stopped_ = true;
}

KernelRecord ScopedTimer::get_record() const {
    return KernelRecord{
        name_,
        elapsed_ms_,
        elapsed_us_,
        flops_,
        bytes_,
        timestamp_
    };
}

// ============================================================================
// KernelProfiler implementation
// ============================================================================

KernelProfiler::KernelProfiler()
    : enabled_(true)
    , pending_flops_(-1)
    , pending_bytes_(-1)
    , pending_timestamp_(0.0)
{
}

std::unique_ptr<ScopedTimer> KernelProfiler::start_timer(
    const std::string& name,
    int64_t flops,
    int64_t bytes
) {
    if (!enabled_) {
        return nullptr;
    }
    return std::make_unique<ScopedTimer>(name, flops, bytes);
}

void KernelProfiler::record_start(const std::string& name, int64_t flops, int64_t bytes) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Create start event
    pending_start_ = std::make_unique<CudaEvent>(false);
    pending_start_->record();
    pending_name_ = name;
    pending_flops_ = flops;
    pending_bytes_ = bytes;
    pending_timestamp_ = get_timestamp();
}

void KernelProfiler::record_stop() {
    if (!enabled_ || !pending_start_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Create and record stop event
    CudaEvent stop_event(false);
    stop_event.record();
    stop_event.synchronize();

    // Calculate elapsed time
    float elapsed_ms = event_elapsed_ms(*pending_start_, stop_event);

    // Add record
    records_.push_back(KernelRecord{
        pending_name_,
        elapsed_ms,
        elapsed_ms * 1000.0f,
        pending_flops_,
        pending_bytes_,
        pending_timestamp_
    });

    // Clear pending state
    pending_start_.reset();
    pending_name_.clear();
    pending_flops_ = -1;
    pending_bytes_ = -1;
}

void KernelProfiler::add_record(const KernelRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back(record);
}

void KernelProfiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
}

float KernelProfiler::total_time_ms() const {
    std::lock_guard<std::mutex> lock(mutex_);
    float total = 0.0f;
    for (const auto& record : records_) {
        total += record.elapsed_ms;
    }
    return total;
}

std::vector<KernelProfiler::KernelStats> KernelProfiler::summary_by_name() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Group by name
    std::unordered_map<std::string, std::vector<float>> by_name;
    for (const auto& record : records_) {
        by_name[record.name].push_back(record.elapsed_ms);
    }

    // Calculate statistics
    std::vector<KernelStats> result;
    result.reserve(by_name.size());

    for (const auto& [name, times] : by_name) {
        KernelStats stats;
        stats.name = name;
        stats.count = static_cast<int>(times.size());
        stats.total_ms = 0.0f;
        stats.min_ms = times[0];
        stats.max_ms = times[0];

        for (float t : times) {
            stats.total_ms += t;
            if (t < stats.min_ms) stats.min_ms = t;
            if (t > stats.max_ms) stats.max_ms = t;
        }
        stats.avg_ms = stats.total_ms / static_cast<float>(stats.count);

        result.push_back(stats);
    }

    // Sort by total time descending
    std::sort(result.begin(), result.end(),
        [](const KernelStats& a, const KernelStats& b) {
            return a.total_ms > b.total_ms;
        });

    return result;
}

// ============================================================================
// Global profiler
// ============================================================================

KernelProfiler& global_profiler() {
    static KernelProfiler instance;
    return instance;
}

} // namespace pygpukit
