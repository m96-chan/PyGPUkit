/**
 * CUDA Graph implementation using CUDA Runtime API
 *
 * Uses stream capture for automatic graph construction.
 * Public API hides all CUDA types behind pimpl.
 */
#include "cuda_graph.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace pygpukit {

// =============================================================================
// Implementation struct (hidden from public API)
// =============================================================================
struct CudaGraphImpl {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t capture_stream = nullptr;
    bool capturing = false;

    CudaGraphImpl() {
        cudaError_t err = cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw CudaError(std::string("Failed to create stream for CUDA Graph: ") + cudaGetErrorString(err));
        }
    }

    ~CudaGraphImpl() {
        reset();
        if (capture_stream != nullptr) {
            cudaStreamDestroy(capture_stream);
        }
    }

    void reset() {
        if (capturing) {
            internal::set_capture_stream(nullptr);
            cudaGraph_t dummy;
            cudaStreamEndCapture(capture_stream, &dummy);
            if (dummy) cudaGraphDestroy(dummy);
            capturing = false;
        }

        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }

        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
    }
};

// =============================================================================
// Thread-local capture stream tracking
// =============================================================================
namespace internal {

static thread_local cudaStream_t g_capture_stream = nullptr;

cudaStream_t get_capture_stream() {
    return g_capture_stream;
}

void set_capture_stream(cudaStream_t stream) {
    g_capture_stream = stream;
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

    // Synchronize device before capture to ensure all previous operations complete
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw CudaError(std::string("Failed to synchronize before capture: ") + cudaGetErrorString(sync_err));
    }

    // Begin stream capture
    cudaError_t err = cudaStreamBeginCapture(impl_->capture_stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        throw CudaError(std::string("Failed to begin stream capture: ") + cudaGetErrorString(err));
    }

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
    cudaError_t err = cudaStreamEndCapture(impl_->capture_stream, &impl_->graph);
    if (err != cudaSuccess) {
        impl_->capturing = false;
        throw CudaError(std::string("Failed to end stream capture: ") + cudaGetErrorString(err));
    }

    impl_->capturing = false;

    if (impl_->graph == nullptr) {
        throw std::runtime_error("Graph capture failed - no operations captured");
    }

    // Instantiate the graph for execution
    err = cudaGraphInstantiate(&impl_->graph_exec, impl_->graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        throw CudaError(std::string("Failed to instantiate graph: ") + cudaGetErrorString(err));
    }
}

void CudaGraph::replay() {
    if (!impl_) {
        throw std::runtime_error("CudaGraph: invalid state (moved-from object)");
    }
    if (!is_ready()) {
        throw std::runtime_error("Graph not ready - call end_capture() first");
    }

    // Launch the graph (asynchronous - caller should sync if needed)
    cudaError_t err = cudaGraphLaunch(impl_->graph_exec, impl_->capture_stream);
    if (err != cudaSuccess) {
        throw CudaError(std::string("Failed to launch graph: ") + cudaGetErrorString(err));
    }
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
    cudaError_t err = cudaStreamSynchronize(impl_->capture_stream);
    if (err != cudaSuccess) {
        throw CudaError(std::string("Failed to synchronize graph stream: ") + cudaGetErrorString(err));
    }
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
    cudaError_t err = cudaGraphGetNodes(impl_->graph, nullptr, &num_nodes);
    if (err != cudaSuccess) {
        return 0;
    }
    return num_nodes;
}

bool CudaGraph::is_capturing() const {
    return impl_ && impl_->capturing;
}

} // namespace pygpukit
