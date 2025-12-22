/**
 * CUDA Graph implementation using CUDA Driver API
 *
 * Uses stream capture for automatic graph construction.
 * Public API hides all CUDA types behind pimpl.
 *
 * PyGPUkit v0.2.12+: Converted from Runtime API to Driver API
 */
#include "cuda_graph.hpp"
#include "driver_context.hpp"
#include <cuda.h>
#include <stdexcept>

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

// =============================================================================
// Implementation struct (hidden from public API)
// =============================================================================
struct CudaGraphImpl {
    CUgraph graph = nullptr;
    CUgraphExec graph_exec = nullptr;
    CUstream capture_stream = nullptr;
    bool capturing = false;

    CudaGraphImpl() {
        // Ensure context is initialized
        driver::DriverContext::instance().set_current();

        // Create non-blocking stream for capture
        CUresult err = cuStreamCreate(&capture_stream, CU_STREAM_NON_BLOCKING);
        if (err != CUDA_SUCCESS) {
            const char* error_str = nullptr;
            cuGetErrorString(err, &error_str);
            throw CudaError(std::string("Failed to create stream for CUDA Graph: ") +
                          (error_str ? error_str : "unknown error"));
        }
    }

    ~CudaGraphImpl() {
        reset();
        if (capture_stream != nullptr) {
            cuStreamDestroy(capture_stream);
        }
    }

    void reset() {
        if (capturing) {
            internal::set_capture_stream(nullptr);
            CUgraph dummy = nullptr;
            cuStreamEndCapture(capture_stream, &dummy);
            if (dummy) cuGraphDestroy(dummy);
            capturing = false;
        }

        if (graph_exec != nullptr) {
            cuGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }

        if (graph != nullptr) {
            cuGraphDestroy(graph);
            graph = nullptr;
        }
    }
};

// =============================================================================
// Thread-local capture stream tracking
// =============================================================================
namespace internal {

static thread_local CUstream g_capture_stream = nullptr;
static thread_local int g_operation_counter = 0;
static thread_local bool g_is_capturing = false;

CUstream get_capture_stream() {
    return g_capture_stream;
}

bool is_currently_capturing() {
    return g_is_capturing;
}

int get_operation_counter() {
    return g_operation_counter;
}

void increment_operation_counter() {
    g_operation_counter++;
}

void set_capture_stream(CUstream stream) {
    g_capture_stream = stream;
    g_is_capturing = (stream != nullptr);
}

} // namespace internal

// =============================================================================
// CudaGraph implementation
// =============================================================================

CudaGraph::CudaGraph() : impl_(new CudaGraphImpl()) {}

CudaGraph::~CudaGraph() {
    delete impl_;
}

CudaGraph::CudaGraph(CudaGraph&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void CudaGraph::begin_capture() {
    if (!impl_) {
        throw std::runtime_error("CudaGraph: invalid state (moved-from object)");
    }
    if (impl_->capturing) {
        throw std::runtime_error("Graph capture already in progress");
    }

    // Reset any existing graph
    impl_->reset();

    // Synchronize context before capture to ensure all previous operations complete
    check_driver_error(cuCtxSynchronize(), "Failed to synchronize before capture");

    // Begin stream capture
    check_driver_error(
        cuStreamBeginCapture(impl_->capture_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
        "Failed to begin stream capture"
    );

    // Set global capture stream for kernel launches
    internal::set_capture_stream(impl_->capture_stream);
    impl_->capturing = true;
}

void CudaGraph::end_capture() {
    if (!impl_) {
        throw std::runtime_error("CudaGraph: invalid state (moved-from object)");
    }
    if (!impl_->capturing) {
        throw std::runtime_error("No graph capture in progress");
    }

    // Clear global capture stream
    internal::set_capture_stream(nullptr);

    // End capture and get the graph
    CUresult err = cuStreamEndCapture(impl_->capture_stream, &impl_->graph);
    if (err != CUDA_SUCCESS) {
        impl_->capturing = false;
        const char* error_str = nullptr;
        cuGetErrorString(err, &error_str);
        throw CudaError(std::string("Failed to end stream capture: ") +
                       (error_str ? error_str : "unknown error"));
    }

    impl_->capturing = false;

    if (impl_->graph == nullptr) {
        throw std::runtime_error("Graph capture failed - no operations captured");
    }

    // Instantiate the graph for execution
    // Note: cuGraphInstantiate signature changed in CUDA 12.0
    // Use cuGraphInstantiateWithFlags for CUDA 12.0+
#if CUDA_VERSION >= 12000
    check_driver_error(
        cuGraphInstantiate(&impl_->graph_exec, impl_->graph, 0),
        "Failed to instantiate graph"
    );
#else
    check_driver_error(
        cuGraphInstantiate(&impl_->graph_exec, impl_->graph, nullptr, nullptr, 0),
        "Failed to instantiate graph"
    );
#endif
}

void CudaGraph::replay() {
    if (!impl_) {
        throw std::runtime_error("CudaGraph: invalid state (moved-from object)");
    }
    if (!is_ready()) {
        throw std::runtime_error("Graph not ready - call end_capture() first");
    }

    // Launch the graph (asynchronous - caller should sync if needed)
    check_driver_error(
        cuGraphLaunch(impl_->graph_exec, impl_->capture_stream),
        "Failed to launch graph"
    );
    // NOTE: No synchronization here - caller is responsible for syncing
    // Use stream.synchronize() or graph.synchronize() when results are needed
}

void CudaGraph::synchronize() {
    if (!impl_) {
        throw std::runtime_error("CudaGraph: invalid state (moved-from object)");
    }
    if (impl_->capture_stream == nullptr) {
        throw std::runtime_error("No stream to synchronize");
    }
    check_driver_error(
        cuStreamSynchronize(impl_->capture_stream),
        "Failed to synchronize graph stream"
    );
}

bool CudaGraph::is_ready() const {
    return impl_ && impl_->graph_exec != nullptr;
}

void CudaGraph::reset() {
    if (impl_) {
        impl_->reset();
    }
}

size_t CudaGraph::num_nodes() const {
    if (!impl_ || impl_->graph == nullptr) {
        return 0;
    }

    size_t num_nodes = 0;
    CUresult err = cuGraphGetNodes(impl_->graph, nullptr, &num_nodes);
    if (err != CUDA_SUCCESS) {
        return 0;
    }
    return num_nodes;
}

bool CudaGraph::is_capturing() const {
    return impl_ && impl_->capturing;
}

void* CudaGraph::get_stream_handle() const {
    if (!impl_) {
        return nullptr;
    }
    return static_cast<void*>(impl_->capture_stream);
}

} // namespace pygpukit
