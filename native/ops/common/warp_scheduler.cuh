/**
 * Warp Scheduler Utilities
 *
 * Provides warp specialization patterns for producer/consumer kernels.
 * Used with TMA for overlapping data loading and computation.
 *
 * Usage:
 *   - Flash Attention 3: Producer loads Q/K/V, Consumer computes attention
 *   - Persistent GEMM: Producer loads A/B tiles, Consumer computes MMA
 *
 * Requirements:
 *   - SM90+ for full TMA support
 *   - SM120+ for GeForce TMA (no cluster)
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace scheduler {

// =============================================================================
// Warp Role Detection
// =============================================================================

/**
 * Warp role in producer/consumer model.
 */
enum class WarpRole {
    Producer,   // Issues TMA loads
    Consumer    // Computes MMA operations
};

/**
 * Get warp ID within the CTA.
 */
__device__ __forceinline__
int get_warp_id() {
    return threadIdx.x / 32 + (threadIdx.y * blockDim.x / 32) +
           (threadIdx.z * blockDim.x * blockDim.y / 32);
}

/**
 * Get lane ID within the warp.
 */
__device__ __forceinline__
int get_lane_id() {
    return threadIdx.x % 32;
}

/**
 * Get total number of warps in the CTA.
 */
__device__ __forceinline__
int get_num_warps() {
    return (blockDim.x * blockDim.y * blockDim.z + 31) / 32;
}

/**
 * Determine warp role based on warp ID.
 *
 * @param num_producer_warps  Number of warps dedicated to loading
 * @return WarpRole::Producer or WarpRole::Consumer
 */
__device__ __forceinline__
WarpRole get_warp_role(int num_producer_warps) {
    return (get_warp_id() < num_producer_warps) ? WarpRole::Producer : WarpRole::Consumer;
}

/**
 * Check if current warp is a producer.
 */
__device__ __forceinline__
bool is_producer_warp(int num_producer_warps) {
    return get_warp_id() < num_producer_warps;
}

/**
 * Check if current warp is a consumer.
 */
__device__ __forceinline__
bool is_consumer_warp(int num_producer_warps) {
    return get_warp_id() >= num_producer_warps;
}

/**
 * Get producer warp index (0 to num_producer_warps-1).
 * Returns -1 if not a producer.
 */
__device__ __forceinline__
int get_producer_warp_idx(int num_producer_warps) {
    int warp_id = get_warp_id();
    return (warp_id < num_producer_warps) ? warp_id : -1;
}

/**
 * Get consumer warp index (0 to num_consumer_warps-1).
 * Returns -1 if not a consumer.
 */
__device__ __forceinline__
int get_consumer_warp_idx(int num_producer_warps) {
    int warp_id = get_warp_id();
    return (warp_id >= num_producer_warps) ? (warp_id - num_producer_warps) : -1;
}

// =============================================================================
// Warpgroup Utilities (for WGMMA)
// =============================================================================

/**
 * Get warpgroup ID (group of 4 consecutive warps for WGMMA).
 */
__device__ __forceinline__
int get_warpgroup_id() {
    return get_warp_id() / 4;
}

/**
 * Get warp index within warpgroup (0-3).
 */
__device__ __forceinline__
int get_warp_idx_in_warpgroup() {
    return get_warp_id() % 4;
}

// =============================================================================
// Elected Thread Pattern
// =============================================================================

/**
 * Check if current thread is the elected (first) thread in the warp.
 * Used for issuing TMA loads and barrier operations.
 */
__device__ __forceinline__
bool is_elected_one() {
    return get_lane_id() == 0;
}

/**
 * Check if current thread is elected within the CTA.
 * Only thread 0 of the entire CTA.
 */
__device__ __forceinline__
bool is_elected_cta() {
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

/**
 * Elect one thread per warp to perform an action.
 * Returns true for lane 0 of each warp.
 */
__device__ __forceinline__
bool elect_one_per_warp() {
    return get_lane_id() == 0;
}

// =============================================================================
// Synchronization Utilities
// =============================================================================

/**
 * Named barrier for producer/consumer synchronization.
 * SM90+ supports up to 16 named barriers (0-15).
 */
__device__ __forceinline__
void named_barrier_arrive(int barrier_id, int count) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "bar.arrive %0, %1;\n"
        :: "r"(barrier_id), "r"(count)
    );
#else
    __syncthreads();
#endif
}

__device__ __forceinline__
void named_barrier_sync(int barrier_id, int count) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "bar.sync %0, %1;\n"
        :: "r"(barrier_id), "r"(count)
    );
#else
    __syncthreads();
#endif
}

/**
 * Cluster barrier (for inter-CTA synchronization on SM90+).
 * Not available on SM120 GeForce.
 */
__device__ __forceinline__
void cluster_barrier_arrive() {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1200
    asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__
void cluster_barrier_wait() {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1200
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
#endif
}

// =============================================================================
// Producer/Consumer Scheduling Helpers
// =============================================================================

/**
 * Configuration for warp-specialized kernel.
 */
template<int NUM_PRODUCER_WARPS, int NUM_CONSUMER_WARPS>
struct WarpSchedulerConfig {
    static constexpr int kNumProducerWarps = NUM_PRODUCER_WARPS;
    static constexpr int kNumConsumerWarps = NUM_CONSUMER_WARPS;
    static constexpr int kNumWarps = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int kNumThreads = kNumWarps * 32;

    // Producer threads count
    static constexpr int kProducerThreads = NUM_PRODUCER_WARPS * 32;

    // Consumer threads count
    static constexpr int kConsumerThreads = NUM_CONSUMER_WARPS * 32;
};

/**
 * Standard configurations for Flash Attention 3.
 */
using FA3ConfigSm120 = WarpSchedulerConfig<4, 8>;  // 4 producer, 8 consumer (12 total)
using FA3ConfigSm90 = WarpSchedulerConfig<4, 8>;   // Same for Hopper

/**
 * Standard configurations for persistent GEMM.
 */
using GemmConfigSm120 = WarpSchedulerConfig<2, 6>; // 2 producer, 6 consumer (8 total)
using GemmConfigSm90 = WarpSchedulerConfig<2, 6>;

// =============================================================================
// Pingpong Scheduling
// =============================================================================

/**
 * Pingpong buffer index tracking.
 * Alternates between 0 and 1 for double-buffering.
 */
struct PingpongState {
    int read_idx;   // Buffer index for reading (consumer)
    int write_idx;  // Buffer index for writing (producer)
    int phase;      // Phase bit for barrier

    __device__ __forceinline__
    PingpongState() : read_idx(0), write_idx(0), phase(0) {}

    __device__ __forceinline__
    void advance_read() {
        read_idx ^= 1;
    }

    __device__ __forceinline__
    void advance_write() {
        write_idx ^= 1;
    }

    __device__ __forceinline__
    void advance_phase() {
        phase ^= 1;
    }

    __device__ __forceinline__
    void advance_all() {
        read_idx ^= 1;
        write_idx ^= 1;
        phase ^= 1;
    }
};

/**
 * Multi-stage buffer index tracking.
 * For N-stage pipelines (N > 2).
 */
template<int NUM_STAGES>
struct MultistageState {
    int read_stage;
    int write_stage;
    int phase;

    __device__ __forceinline__
    MultistageState() : read_stage(0), write_stage(0), phase(0) {}

    __device__ __forceinline__
    void advance_read() {
        read_stage = (read_stage + 1) % NUM_STAGES;
        if (read_stage == 0) phase ^= 1;
    }

    __device__ __forceinline__
    void advance_write() {
        write_stage = (write_stage + 1) % NUM_STAGES;
    }

    __device__ __forceinline__
    int stages_in_flight() const {
        int diff = write_stage - read_stage;
        return (diff >= 0) ? diff : (diff + NUM_STAGES);
    }
};

}  // namespace scheduler
}  // namespace ops
}  // namespace pygpukit
