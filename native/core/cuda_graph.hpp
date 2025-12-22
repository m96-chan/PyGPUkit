/**
 * CUDA Graph support for PyGPUkit
 *
 * Provides CUDA Graph capture and replay for optimized decode performance.
 * CUDA Graphs reduce kernel launch overhead by capturing a sequence of
 * operations and replaying them with minimal CPU involvement.
 *
 * Usage:
 *   1. Create CudaGraph instance
 *   2. Call begin_capture() before the operations
 *   3. Execute operations (they will be captured, not executed)
 *   4. Call end_capture() to finalize the graph
 *   5. Call replay() to execute the captured operations
 *
 * Note: Memory allocations during capture are not supported.
 *       All buffers must be pre-allocated before capture.
 */
#pragma once

#include <cstddef>
#include "types.hpp"

namespace pygpukit {

// Forward declarations (opaque pointers - no CUDA Runtime in public API)
struct CudaGraphImpl;

/**
 * CUDA Graph wrapper for efficient kernel replay
 */
class CudaGraph {
public:
    CudaGraph();
    ~CudaGraph();

    // Disable copy
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    // Enable move
    CudaGraph(CudaGraph&& other) noexcept;
    CudaGraph& operator=(CudaGraph&& other) noexcept;

    /**
     * Begin capturing operations.
     * All subsequent CUDA operations will be recorded into the graph.
     */
    void begin_capture();

    /**
     * End capturing and create an executable graph.
     * After this call, the graph can be replayed.
     */
    void end_capture();

    /**
     * Replay the captured graph (asynchronous).
     * This executes all captured operations with minimal CPU overhead.
     * Call synchronize() after replay to wait for completion.
     */
    void replay();

    /**
     * Synchronize the graph's internal stream.
     * Call this after replay() to wait for the graph execution to complete.
     */
    void synchronize();

    /**
     * Check if the graph has been captured and is ready for replay.
     */
    bool is_ready() const;

    /**
     * Reset the graph, freeing all resources.
     * After reset, begin_capture() can be called again.
     */
    void reset();

    /**
     * Get the number of nodes in the captured graph.
     */
    size_t num_nodes() const;

    /**
     * Check if currently capturing.
     */
    bool is_capturing() const;

    /**
     * Get the internal stream handle.
     * Returns the capture/replay stream as an opaque pointer.
     * Used for explicit stream synchronization or async operations.
     */
    void* get_stream_handle() const;

private:
    CudaGraphImpl* impl_ = nullptr;
};

/**
 * RAII helper for graph capture scope.
 *
 * Usage:
 *   CudaGraph graph;
 *   {
 *       CudaGraphCapture capture(graph);
 *       // Operations here are captured
 *   }
 *   graph.replay();
 */
class CudaGraphCapture {
public:
    explicit CudaGraphCapture(CudaGraph& graph) : graph_(graph) {
        graph_.begin_capture();
    }

    ~CudaGraphCapture() {
        if (!ended_) {
            graph_.end_capture();
        }
    }

    void end() {
        if (!ended_) {
            graph_.end_capture();
            ended_ = true;
        }
    }

private:
    CudaGraph& graph_;
    bool ended_ = false;
};

} // namespace pygpukit

// =============================================================================
// Internal API for kernel implementations (requires CUDA Driver API)
// Include this section only in .cu files that need stream access
// =============================================================================
#ifdef __CUDACC__
#include <cuda.h>

namespace pygpukit {
namespace internal {

/**
 * Get the current graph capture stream (internal use only).
 * Returns the capture stream if graph capture is in progress, or nullptr otherwise.
 */
CUstream get_capture_stream();

/**
 * Set the current graph capture stream (internal use only).
 * Called internally by CudaGraph::begin_capture() and end_capture().
 */
void set_capture_stream(CUstream stream);

/**
 * Check if currently in CUDA Graph capture mode.
 */
bool is_currently_capturing();

/**
 * Get the current operation sequence counter (for debugging).
 */
int get_operation_counter();

/**
 * Increment the operation counter (for debugging).
 */
void increment_operation_counter();

} // namespace internal
} // namespace pygpukit

/**
 * Helper macro for kernel launch that uses capture stream when available.
 * Usage: kernel<<<grid, block, smem, PYGPUKIT_GET_LAUNCH_STREAM()>>>(args...)
 *
 * Note: CUstream is compatible with cudaStream_t in kernel launch syntax.
 * The driver and runtime use the same underlying stream handle type.
 */
#define PYGPUKIT_GET_LAUNCH_STREAM() \
    (pygpukit::internal::get_capture_stream() ? \
     pygpukit::internal::get_capture_stream() : \
     CUstream(0))

#endif // __CUDACC__
