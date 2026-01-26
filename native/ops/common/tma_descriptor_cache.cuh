/**
 * TMA Descriptor Cache
 *
 * Caches TMA descriptors to avoid per-call overhead:
 * - cuTensorMapEncodeTiled (CPU descriptor creation)
 * - cudaMalloc (device memory allocation)
 * - cudaMemcpy (host to device copy)
 * - cudaFree (device memory deallocation)
 *
 * Usage:
 *   auto& cache = TmaDescriptorCache::instance();
 *   CUtensorMap* d_desc = cache.get_or_create_3d_bf16(
 *       ptr, num_heads, seq_len, head_dim, tile_q, tile_kv, swizzle);
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstring>
#include "tma_utils.cuh"

namespace pygpukit {
namespace ops {
namespace tma {

// =============================================================================
// Cache Key for TMA Descriptors
// =============================================================================

struct TmaDescriptorKey {
    void* data_ptr;       // Base data pointer (changes if tensor reallocated)
    uint64_t dim0;        // head_dim
    uint64_t dim1;        // seq_len
    uint64_t dim2;        // num_heads (0 for 2D)
    uint64_t stride1;     // Stride between seq positions
    uint64_t stride2;     // Stride between heads (0 for 2D)
    uint32_t tile0;       // Tile size for head_dim
    uint32_t tile1;       // Tile size for seq
    int swizzle;          // Swizzle mode

    bool operator==(const TmaDescriptorKey& other) const {
        return data_ptr == other.data_ptr &&
               dim0 == other.dim0 &&
               dim1 == other.dim1 &&
               dim2 == other.dim2 &&
               stride1 == other.stride1 &&
               stride2 == other.stride2 &&
               tile0 == other.tile0 &&
               tile1 == other.tile1 &&
               swizzle == other.swizzle;
    }
};

struct TmaDescriptorKeyHash {
    size_t operator()(const TmaDescriptorKey& k) const {
        // Simple hash combining all fields
        size_t h = std::hash<void*>()(k.data_ptr);
        h ^= std::hash<uint64_t>()(k.dim0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>()(k.dim1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>()(k.dim2) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>()(k.stride1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>()(k.stride2) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(k.tile0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(k.tile1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.swizzle) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// =============================================================================
// Cached Descriptor Entry
// =============================================================================

struct CachedTmaDescriptor {
    TmaDescriptor host_desc;     // Host-side descriptor
    CUtensorMap* device_desc;    // Device-side pointer (allocated once)

    CachedTmaDescriptor() : device_desc(nullptr) {}

    ~CachedTmaDescriptor() {
        if (device_desc) {
            cudaFree(device_desc);
            device_desc = nullptr;
        }
    }

    // Disable copy (has device pointer)
    CachedTmaDescriptor(const CachedTmaDescriptor&) = delete;
    CachedTmaDescriptor& operator=(const CachedTmaDescriptor&) = delete;

    // Enable move
    CachedTmaDescriptor(CachedTmaDescriptor&& other) noexcept
        : host_desc(other.host_desc), device_desc(other.device_desc) {
        other.device_desc = nullptr;
    }
    CachedTmaDescriptor& operator=(CachedTmaDescriptor&& other) noexcept {
        if (this != &other) {
            if (device_desc) cudaFree(device_desc);
            host_desc = other.host_desc;
            device_desc = other.device_desc;
            other.device_desc = nullptr;
        }
        return *this;
    }
};

// =============================================================================
// TMA Descriptor Cache (Singleton)
// =============================================================================

class TmaDescriptorCache {
public:
    static TmaDescriptorCache& instance() {
        static TmaDescriptorCache cache;
        return cache;
    }

    /**
     * Get or create a 3D BF16 TMA descriptor.
     * Returns device pointer to CUtensorMap (cached).
     *
     * @param base_ptr    Base pointer to tensor data
     * @param dim0        Inner dimension (head_dim)
     * @param dim1        Middle dimension (seq_len)
     * @param dim2        Outer dimension (num_heads)
     * @param stride1     Stride between seq positions (in elements)
     * @param stride2     Stride between heads (in elements)
     * @param tile0       Tile size for head_dim
     * @param tile1       Tile size for seq
     * @param swizzle     Swizzle mode
     * @param stream      CUDA stream for async copy (nullptr = default)
     * @return            Device pointer to CUtensorMap, or nullptr on error
     */
    CUtensorMap* get_or_create_3d_bf16(
        void* base_ptr,
        uint64_t dim0,
        uint64_t dim1,
        uint64_t dim2,
        uint64_t stride1,
        uint64_t stride2,
        uint32_t tile0,
        uint32_t tile1,
        SwizzleMode swizzle,
        cudaStream_t stream = nullptr
    ) {
        TmaDescriptorKey key{
            base_ptr, dim0, dim1, dim2, stride1, stride2,
            tile0, tile1, static_cast<int>(swizzle)
        };

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Cache hit - return existing device pointer
            cache_hits_++;
            return it->second.device_desc;
        }

        // Cache miss - create new descriptor
        cache_misses_++;

        CachedTmaDescriptor entry;
        CUresult cu_result = create_tma_descriptor_3d_bf16(
            entry.host_desc,
            base_ptr,
            dim0, dim1, dim2,
            stride1, stride2,
            tile0, tile1,
            swizzle
        );

        if (cu_result != CUDA_SUCCESS) {
            fprintf(stderr, "[TMA Cache] Failed to create descriptor: %d\n", cu_result);
            return nullptr;
        }

        // Allocate device memory for descriptor
        cudaError_t err = cudaMalloc(&entry.device_desc, sizeof(CUtensorMap));
        if (err != cudaSuccess) {
            fprintf(stderr, "[TMA Cache] cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }

        // Copy to device (async if stream provided)
        if (stream) {
            err = cudaMemcpyAsync(entry.device_desc, &entry.host_desc.tensor_map,
                                  sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        } else {
            err = cudaMemcpy(entry.device_desc, &entry.host_desc.tensor_map,
                             sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        }

        if (err != cudaSuccess) {
            fprintf(stderr, "[TMA Cache] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            cudaFree(entry.device_desc);
            return nullptr;
        }

        CUtensorMap* result = entry.device_desc;
        cache_.emplace(key, std::move(entry));
        return result;
    }

    /**
     * Get Q, K, V descriptors for FA3 attention in one call.
     * Optimized for the common case of attention with same shapes.
     *
     * @param q_ptr, k_ptr, v_ptr  Tensor data pointers
     * @param num_heads            Number of attention heads
     * @param seq_q, seq_kv        Sequence lengths
     * @param head_dim             Head dimension
     * @param tile_q, tile_kv      Tile sizes
     * @param swizzle              Swizzle mode
     * @param stream               CUDA stream
     * @param d_q_desc, d_k_desc, d_v_desc  Output device pointers
     * @return                     true on success
     */
    bool get_fa3_descriptors(
        void* q_ptr, void* k_ptr, void* v_ptr,
        int num_heads, int seq_q, int seq_kv, int head_dim,
        int tile_q, int tile_kv,
        SwizzleMode swizzle,
        cudaStream_t stream,
        CUtensorMap*& d_q_desc,
        CUtensorMap*& d_k_desc,
        CUtensorMap*& d_v_desc
    ) {
        // Q: [num_heads, seq_q, head_dim]
        d_q_desc = get_or_create_3d_bf16(
            q_ptr,
            head_dim, seq_q, num_heads,
            head_dim, seq_q * head_dim,
            head_dim, tile_q,
            swizzle, stream
        );

        // K: [num_heads, seq_kv, head_dim]
        d_k_desc = get_or_create_3d_bf16(
            k_ptr,
            head_dim, seq_kv, num_heads,
            head_dim, seq_kv * head_dim,
            head_dim, tile_kv,
            swizzle, stream
        );

        // V: [num_heads, seq_kv, head_dim]
        d_v_desc = get_or_create_3d_bf16(
            v_ptr,
            head_dim, seq_kv, num_heads,
            head_dim, seq_kv * head_dim,
            head_dim, tile_kv,
            swizzle, stream
        );

        return (d_q_desc != nullptr && d_k_desc != nullptr && d_v_desc != nullptr);
    }

    /**
     * Clear all cached descriptors (for testing/benchmarking).
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        cache_hits_ = 0;
        cache_misses_ = 0;
    }

    /**
     * Invalidate cache entries for a specific data pointer.
     * Call when tensor is deallocated/reallocated.
     */
    void invalidate(void* data_ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = cache_.begin(); it != cache_.end(); ) {
            if (it->first.data_ptr == data_ptr) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }

    /**
     * Get cache statistics.
     */
    void get_stats(size_t& hits, size_t& misses, size_t& size) const {
        std::lock_guard<std::mutex> lock(mutex_);
        hits = cache_hits_;
        misses = cache_misses_;
        size = cache_.size();
    }

    /**
     * Print cache statistics.
     */
    void print_stats() const {
        size_t hits, misses, size;
        get_stats(hits, misses, size);
        fprintf(stderr, "[TMA Cache] hits=%zu misses=%zu size=%zu hit_rate=%.1f%%\n",
                hits, misses, size,
                (hits + misses > 0) ? 100.0 * hits / (hits + misses) : 0.0);
    }

private:
    TmaDescriptorCache() : cache_hits_(0), cache_misses_(0) {}

    ~TmaDescriptorCache() {
        // Cache entries will be cleaned up by their destructors
    }

    // Disable copy/move
    TmaDescriptorCache(const TmaDescriptorCache&) = delete;
    TmaDescriptorCache& operator=(const TmaDescriptorCache&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<TmaDescriptorKey, CachedTmaDescriptor, TmaDescriptorKeyHash> cache_;
    size_t cache_hits_;
    size_t cache_misses_;
};

}  // namespace tma
}  // namespace ops
}  // namespace pygpukit
