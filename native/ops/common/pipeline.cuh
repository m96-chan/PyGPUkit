/**
 * Pipeline Utilities for TMA-based Kernels
 *
 * Provides multi-stage async pipeline management for overlapping
 * memory transfers and computation.
 *
 * Usage:
 *   - Flash Attention 3: Pipeline K/V tile loading with score computation
 *   - Persistent GEMM: Pipeline A/B tile loading with MMA
 *
 * Architecture:
 *   Producer warps issue TMA loads into pipeline stages
 *   Consumer warps wait for stages and compute
 *   mbarrier synchronizes between stages
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "tma_utils.cuh"

namespace pygpukit {
namespace ops {
namespace pipeline {

// =============================================================================
// Pipeline Stage State
// =============================================================================

/**
 * State for a single pipeline stage.
 * Tracks barrier and phase for TMA synchronization.
 */
struct alignas(8) StageState {
    uint64_t barrier;  // mbarrier for this stage
    int phase;         // Current phase (0 or 1)

    __device__ __forceinline__
    void init(int thread_count = 1) {
        tma::barrier_init(barrier, thread_count);
        phase = 0;
    }

    __device__ __forceinline__
    void arrive_expect(uint32_t tx_bytes) {
        tma::barrier_arrive_expect_tx(barrier, tx_bytes);
    }

    __device__ __forceinline__
    void wait() {
        tma::barrier_wait(barrier, phase);
    }

    __device__ __forceinline__
    bool try_wait() {
        return tma::barrier_try_wait(barrier, phase);
    }

    __device__ __forceinline__
    void advance_phase() {
        phase ^= 1;
    }
};

// =============================================================================
// Multi-Stage Pipeline
// =============================================================================

/**
 * Multi-stage async pipeline.
 *
 * @tparam NUM_STAGES  Number of pipeline stages (2-8 typical)
 */
template<int NUM_STAGES>
struct Pipeline {
    static_assert(NUM_STAGES >= 2, "Pipeline needs at least 2 stages");
    static_assert(NUM_STAGES <= 8, "Too many stages may hurt performance");

    StageState stages[NUM_STAGES];
    int producer_stage;  // Current stage for producer
    int consumer_stage;  // Current stage for consumer

    /**
     * Initialize all pipeline stages.
     * Call from a single thread (elected).
     */
    __device__ __forceinline__
    void init(int thread_count_per_stage = 1) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; ++i) {
            stages[i].init(thread_count_per_stage);
        }
        producer_stage = 0;
        consumer_stage = 0;
    }

    /**
     * Get current producer stage.
     */
    __device__ __forceinline__
    StageState& get_producer_stage() {
        return stages[producer_stage];
    }

    /**
     * Get current consumer stage.
     */
    __device__ __forceinline__
    StageState& get_consumer_stage() {
        return stages[consumer_stage];
    }

    /**
     * Advance producer to next stage.
     */
    __device__ __forceinline__
    void advance_producer() {
        producer_stage = (producer_stage + 1) % NUM_STAGES;
    }

    /**
     * Advance consumer to next stage.
     */
    __device__ __forceinline__
    void advance_consumer() {
        stages[consumer_stage].advance_phase();
        consumer_stage = (consumer_stage + 1) % NUM_STAGES;
    }

    /**
     * Get number of stages currently in flight.
     */
    __device__ __forceinline__
    int stages_in_flight() const {
        int diff = producer_stage - consumer_stage;
        return (diff >= 0) ? diff : (diff + NUM_STAGES);
    }

    /**
     * Check if pipeline is full (all stages have pending loads).
     */
    __device__ __forceinline__
    bool is_full() const {
        return stages_in_flight() >= NUM_STAGES - 1;
    }

    /**
     * Check if pipeline is empty (no pending loads).
     */
    __device__ __forceinline__
    bool is_empty() const {
        return producer_stage == consumer_stage;
    }

    /**
     * Producer: Issue TMA load and advance.
     * Call after TMA load is issued.
     */
    __device__ __forceinline__
    void producer_commit(uint32_t tx_bytes) {
        get_producer_stage().arrive_expect(tx_bytes);
        advance_producer();
    }

    /**
     * Consumer: Wait for current stage and advance.
     * Blocking wait.
     */
    __device__ __forceinline__
    void consumer_wait() {
        get_consumer_stage().wait();
    }

    /**
     * Consumer: Try to wait (non-blocking).
     * Returns true if stage is ready.
     */
    __device__ __forceinline__
    bool consumer_try_wait() {
        return get_consumer_stage().try_wait();
    }

    /**
     * Consumer: Done with current stage, advance.
     */
    __device__ __forceinline__
    void consumer_release() {
        advance_consumer();
    }
};

// =============================================================================
// Shared Memory Buffer Manager
// =============================================================================

/**
 * Manages shared memory buffers for pipeline stages.
 *
 * @tparam T           Element type
 * @tparam TILE_SIZE   Elements per tile
 * @tparam NUM_STAGES  Number of pipeline stages
 */
template<typename T, int TILE_SIZE, int NUM_STAGES>
struct PipelineBuffer {
    T data[NUM_STAGES][TILE_SIZE];

    __device__ __forceinline__
    T* get_stage_buffer(int stage_idx) {
        return data[stage_idx];
    }

    __device__ __forceinline__
    const T* get_stage_buffer(int stage_idx) const {
        return data[stage_idx];
    }

    static constexpr size_t size_bytes() {
        return sizeof(T) * TILE_SIZE * NUM_STAGES;
    }
};

// =============================================================================
// Dual-Buffer Pipeline (Optimized 2-stage)
// =============================================================================

/**
 * Optimized dual-buffer (pingpong) pipeline.
 * Simpler than N-stage for cases where 2 stages suffice.
 */
struct DualBufferPipeline {
    StageState stage_a;
    StageState stage_b;
    int current_read;   // 0 = A, 1 = B
    int current_write;  // 0 = A, 1 = B

    __device__ __forceinline__
    void init(int thread_count = 1) {
        stage_a.init(thread_count);
        stage_b.init(thread_count);
        current_read = 0;
        current_write = 0;
    }

    __device__ __forceinline__
    StageState& read_stage() {
        return (current_read == 0) ? stage_a : stage_b;
    }

    __device__ __forceinline__
    StageState& write_stage() {
        return (current_write == 0) ? stage_a : stage_b;
    }

    __device__ __forceinline__
    void flip_read() {
        read_stage().advance_phase();
        current_read ^= 1;
    }

    __device__ __forceinline__
    void flip_write() {
        current_write ^= 1;
    }

    __device__ __forceinline__
    void producer_commit(uint32_t tx_bytes) {
        write_stage().arrive_expect(tx_bytes);
        flip_write();
    }

    __device__ __forceinline__
    void consumer_wait() {
        read_stage().wait();
    }

    __device__ __forceinline__
    void consumer_release() {
        flip_read();
    }
};

// =============================================================================
// Convenience Aliases
// =============================================================================

// Common pipeline configurations
using Pipeline2 = Pipeline<2>;
using Pipeline3 = Pipeline<3>;
using Pipeline4 = Pipeline<4>;

// Flash Attention 3 uses 4-stage pipeline for K/V
using FA3KVPipeline = Pipeline<4>;

// GEMM typically uses 3-stage
using GemmPipeline = Pipeline<3>;

}  // namespace pipeline
}  // namespace ops
}  // namespace pygpukit
