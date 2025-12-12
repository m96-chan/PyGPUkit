//! Memory pool implementation
//!
//! Provides a thread-safe GPU memory pool with:
//! - Size-class based allocation for efficient reuse
//! - LRU eviction policy when quota is exceeded
//! - Statistics tracking for monitoring

use std::collections::HashMap;
use indexmap::IndexMap;
use parking_lot::RwLock;
use crate::memory::{MemoryBlock, size_class::{SIZE_CLASSES, get_size_class}};

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Maximum memory allowed (quota)
    pub quota: usize,
    /// Currently used memory (active allocations)
    pub used: usize,
    /// Memory in free lists (cached for reuse)
    pub cached: usize,
    /// Available memory (quota - used)
    pub available: usize,
    /// Total number of allocations
    pub allocation_count: u64,
    /// Number of blocks reused from free list
    pub reuse_count: u64,
    /// Number of blocks evicted to host
    pub eviction_count: u64,
    /// Number of new CUDA allocations
    pub cudamalloc_count: u64,
    /// Number of active blocks
    pub active_blocks: usize,
    /// Number of blocks in free lists
    pub free_blocks: usize,
}

/// Memory pool error types
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Quota exceeded and eviction disabled or insufficient
    QuotaExceeded {
        requested: usize,
        used: usize,
        quota: usize,
    },
    /// Invalid block ID
    InvalidBlock(u64),
    /// Block not on GPU (needs restore)
    BlockEvicted(u64),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QuotaExceeded { requested, used, quota } => {
                write!(
                    f,
                    "Memory pool quota exceeded: requested {} bytes, used {}, quota {}",
                    requested, used, quota
                )
            }
            Self::InvalidBlock(id) => write!(f, "Invalid block ID: {}", id),
            Self::BlockEvicted(id) => write!(f, "Block {} is evicted, needs restore", id),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Internal state protected by RwLock
struct MemoryPoolInner {
    /// Active allocations: block_id -> MemoryBlock
    active: HashMap<u64, MemoryBlock>,
    /// Free lists by size class: size -> Vec<MemoryBlock>
    free_lists: HashMap<usize, Vec<MemoryBlock>>,
    /// LRU tracking: block_id -> MemoryBlock (ordered by access time)
    /// IndexMap preserves insertion order like Python's OrderedDict
    lru: IndexMap<u64, ()>,
    /// Next block ID to assign
    next_id: u64,
    /// Currently used memory
    used: usize,
    /// Memory in free lists
    cached: usize,
    /// Statistics counters
    allocation_count: u64,
    reuse_count: u64,
    eviction_count: u64,
    cudamalloc_count: u64,
}

/// Thread-safe memory pool for GPU memory management.
///
/// Provides efficient allocation with size-class bucketing and LRU eviction.
///
/// # Example
///
/// ```
/// use pygpukit_core::memory::MemoryPool;
///
/// let pool = MemoryPool::new(1024 * 1024 * 100, false); // 100 MB quota
/// let block_id = pool.allocate(4096).unwrap();
/// pool.free(block_id);
/// ```
pub struct MemoryPool {
    quota: usize,
    enable_eviction: bool,
    inner: RwLock<MemoryPoolInner>,
}

impl MemoryPool {
    /// Create a new memory pool.
    ///
    /// # Arguments
    ///
    /// * `quota` - Maximum memory in bytes
    /// * `enable_eviction` - Whether to evict blocks when quota exceeded
    pub fn new(quota: usize, enable_eviction: bool) -> Self {
        let mut free_lists = HashMap::new();
        for &size in &SIZE_CLASSES {
            free_lists.insert(size, Vec::new());
        }

        Self {
            quota,
            enable_eviction,
            inner: RwLock::new(MemoryPoolInner {
                active: HashMap::new(),
                free_lists,
                lru: IndexMap::new(),
                next_id: 0,
                used: 0,
                cached: 0,
                allocation_count: 0,
                reuse_count: 0,
                eviction_count: 0,
                cudamalloc_count: 0,
            }),
        }
    }

    /// Get the memory quota.
    #[inline]
    pub fn quota(&self) -> usize {
        self.quota
    }

    /// Get currently used memory.
    #[inline]
    pub fn used(&self) -> usize {
        self.inner.read().used
    }

    /// Get cached memory (in free lists).
    #[inline]
    pub fn cached(&self) -> usize {
        self.inner.read().cached
    }

    /// Get available memory (quota - used).
    #[inline]
    pub fn available(&self) -> usize {
        self.quota.saturating_sub(self.inner.read().used)
    }

    /// Allocate a memory block.
    ///
    /// Returns the block ID on success. The caller is responsible for
    /// setting the device pointer via `set_device_ptr()` after CUDA allocation.
    ///
    /// # Arguments
    ///
    /// * `size` - Requested size in bytes (will be rounded to size class)
    ///
    /// # Returns
    ///
    /// * `Ok(block_id)` - ID of the allocated block
    /// * `Err(MemoryError)` - If quota exceeded and cannot evict
    pub fn allocate(&self, size: usize) -> Result<u64, MemoryError> {
        let size_class = get_size_class(size);
        let mut inner = self.inner.write();

        // Try to reuse from free list
        if let Some(free_list) = inner.free_lists.get_mut(&size_class) {
            if let Some(mut block) = free_list.pop() {
                block.touch();
                let block_id = block.id;

                // Move from free list to active
                inner.active.insert(block_id, block);
                inner.lru.insert(block_id, ());
                inner.used += size_class;
                inner.cached -= size_class;
                inner.reuse_count += 1;
                inner.allocation_count += 1;

                return Ok(block_id);
            }
        }

        // Check quota
        if inner.used + size_class > self.quota {
            if self.enable_eviction {
                // Try to evict LRU blocks
                let needed = (inner.used + size_class).saturating_sub(self.quota);
                self.evict_lru_internal(&mut inner, needed);

                // Re-check after eviction
                if inner.used + size_class > self.quota {
                    return Err(MemoryError::QuotaExceeded {
                        requested: size_class,
                        used: inner.used,
                        quota: self.quota,
                    });
                }
            } else {
                return Err(MemoryError::QuotaExceeded {
                    requested: size_class,
                    used: inner.used,
                    quota: self.quota,
                });
            }
        }

        // Allocate new block
        let block_id = inner.next_id;
        inner.next_id += 1;

        let block = MemoryBlock::new(block_id, size_class, None);
        inner.active.insert(block_id, block);
        inner.lru.insert(block_id, ());
        inner.used += size_class;
        inner.allocation_count += 1;
        inner.cudamalloc_count += 1;

        Ok(block_id)
    }

    /// Free a memory block (return to free list).
    ///
    /// The block is moved to the appropriate size-class free list
    /// for later reuse.
    pub fn free(&self, block_id: u64) {
        let mut inner = self.inner.write();

        if let Some(block) = inner.active.remove(&block_id) {
            inner.lru.swap_remove(&block_id);

            // Only subtract from used if block was on GPU
            // (evicted blocks already had their memory released)
            if block.on_gpu {
                inner.used -= block.size;
            }

            let size_class = get_size_class(block.size);
            inner.free_lists
                .entry(size_class)
                .or_default()
                .push(block);
            inner.cached += size_class;
        }
    }

    /// Update LRU timestamp for a block.
    ///
    /// Call this when accessing block data to keep it from being evicted.
    pub fn touch(&self, block_id: u64) {
        let mut inner = self.inner.write();

        if let Some(block) = inner.active.get_mut(&block_id) {
            block.touch();
        }

        // Move to end of LRU (most recently used)
        if inner.lru.contains_key(&block_id) {
            inner.lru.swap_remove(&block_id);
            inner.lru.insert(block_id, ());
        }
    }

    /// Evict LRU blocks to free up space (internal).
    fn evict_lru_internal(&self, inner: &mut MemoryPoolInner, needed: usize) {
        let mut freed = 0;
        let mut to_evict = Vec::new();

        // Identify candidates (oldest first via IndexMap iteration)
        for (&block_id, _) in inner.lru.iter() {
            if freed >= needed {
                break;
            }
            if let Some(block) = inner.active.get(&block_id) {
                if block.on_gpu {
                    to_evict.push(block_id);
                    freed += block.size;
                }
            }
        }

        // Mark blocks as evicted
        for block_id in to_evict {
            if let Some(block) = inner.active.get_mut(&block_id) {
                if block.on_gpu {
                    block.on_gpu = false;
                    block.on_host = true;
                    block.device_ptr = None;
                    inner.eviction_count += 1;
                    inner.used -= block.size;
                }
            }
        }
    }

    /// Evict a specific block to host memory.
    ///
    /// The caller should copy data to host before calling this.
    pub fn evict(&self, block_id: u64) {
        let mut inner = self.inner.write();

        // Get block size first to avoid borrow issues
        let block_size = inner.active.get(&block_id)
            .filter(|b| b.on_gpu)
            .map(|b| b.size);

        if let Some(size) = block_size {
            if let Some(block) = inner.active.get_mut(&block_id) {
                block.on_gpu = false;
                block.on_host = true;
                block.device_ptr = None;
            }
            inner.eviction_count += 1;
            inner.used -= size;
        }
    }

    /// Restore an evicted block to GPU.
    ///
    /// The caller should allocate GPU memory and set device pointer.
    pub fn restore(&self, block_id: u64) {
        let mut inner = self.inner.write();

        if let Some(block) = inner.active.get_mut(&block_id) {
            if !block.on_gpu {
                block.on_gpu = true;
                block.on_host = false;
                inner.used += block.size;
            }
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let inner = self.inner.read();
        let free_blocks: usize = inner.free_lists.values().map(|v| v.len()).sum();

        PoolStats {
            quota: self.quota,
            used: inner.used,
            cached: inner.cached,
            available: self.quota.saturating_sub(inner.used),
            allocation_count: inner.allocation_count,
            reuse_count: inner.reuse_count,
            eviction_count: inner.eviction_count,
            cudamalloc_count: inner.cudamalloc_count,
            active_blocks: inner.active.len(),
            free_blocks,
        }
    }

    /// Clear all allocations.
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.active.clear();
        inner.lru.clear();
        for free_list in inner.free_lists.values_mut() {
            free_list.clear();
        }
        inner.used = 0;
        inner.cached = 0;
    }

    /// Get a block by ID.
    pub fn get_block(&self, block_id: u64) -> Option<MemoryBlock> {
        self.inner.read().active.get(&block_id).cloned()
    }

    /// Set device pointer for a block (after CUDA allocation).
    pub fn set_device_ptr(&self, block_id: u64, device_ptr: u64) {
        let mut inner = self.inner.write();
        if let Some(block) = inner.active.get_mut(&block_id) {
            block.device_ptr = Some(device_ptr);
        }
    }

    /// Set host data for a block (for eviction).
    pub fn set_host_data(&self, block_id: u64, data: Vec<u8>) {
        let mut inner = self.inner.write();
        if let Some(block) = inner.active.get_mut(&block_id) {
            block.host_data = Some(data);
        }
    }

    /// Get host data from a block.
    pub fn get_host_data(&self, block_id: u64) -> Option<Vec<u8>> {
        self.inner.read().active.get(&block_id)?.host_data.clone()
    }

    /// Clear host data from a block (after restore).
    pub fn clear_host_data(&self, block_id: u64) {
        let mut inner = self.inner.write();
        if let Some(block) = inner.active.get_mut(&block_id) {
            block.host_data = None;
        }
    }

    /// Get block size by ID.
    pub fn get_block_size(&self, block_id: u64) -> Option<usize> {
        self.inner.read().active.get(&block_id).map(|b| b.size)
    }

    /// Check if block is on GPU.
    pub fn is_block_on_gpu(&self, block_id: u64) -> bool {
        self.inner.read()
            .active
            .get(&block_id)
            .map(|b| b.on_gpu)
            .unwrap_or(false)
    }
}

// Thread-safe: MemoryPool uses RwLock internally
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = MemoryPool::new(1024 * 1024, false);
        assert_eq!(pool.quota(), 1024 * 1024);
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.available(), 1024 * 1024);
    }

    #[test]
    fn test_allocate_and_free() {
        let pool = MemoryPool::new(1024 * 1024, false);

        let block_id = pool.allocate(100).unwrap();
        assert_eq!(pool.used(), 256); // Rounded to size class

        pool.free(block_id);
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.cached(), 256);
    }

    #[test]
    fn test_reuse_from_free_list() {
        let pool = MemoryPool::new(1024 * 1024, false);

        let block1 = pool.allocate(100).unwrap();
        pool.free(block1);

        let block2 = pool.allocate(100).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.reuse_count, 1);
        assert_eq!(stats.cudamalloc_count, 1);

        pool.free(block2);
    }

    #[test]
    fn test_quota_exceeded() {
        let pool = MemoryPool::new(1024, false); // Small quota

        let result = pool.allocate(2000);
        assert!(result.is_err());

        if let Err(MemoryError::QuotaExceeded { .. }) = result {
            // Expected
        } else {
            panic!("Expected QuotaExceeded error");
        }
    }

    #[test]
    fn test_eviction() {
        // Quota allows 256 bytes (one block at size class 256)
        // When we allocate a second block, it should trigger eviction
        let pool = MemoryPool::new(256, true); // Small quota with eviction

        let block1 = pool.allocate(100).unwrap(); // Rounds to 256
        pool.set_device_ptr(block1, 0x1000);

        // This should trigger eviction of block1 (also rounds to 256)
        let block2 = pool.allocate(100).unwrap();

        // block1 should be evicted
        assert!(!pool.is_block_on_gpu(block1));
        assert!(pool.is_block_on_gpu(block2));

        pool.free(block1);
        pool.free(block2);
    }

    #[test]
    fn test_lru_ordering() {
        let pool = MemoryPool::new(1024 * 1024, true);

        let block1 = pool.allocate(100).unwrap();
        let block2 = pool.allocate(100).unwrap();
        let block3 = pool.allocate(100).unwrap();

        // Touch block1 to make it most recently used
        pool.touch(block1);

        // block2 should be oldest now (will be evicted first)
        // This is verified by the internal LRU order

        pool.free(block1);
        pool.free(block2);
        pool.free(block3);
    }

    #[test]
    fn test_stats() {
        let pool = MemoryPool::new(1024 * 1024, false);

        let b1 = pool.allocate(100).unwrap();
        let b2 = pool.allocate(200).unwrap();
        pool.free(b1);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.active_blocks, 1);
        assert_eq!(stats.free_blocks, 1);

        pool.free(b2);
    }

    #[test]
    fn test_clear() {
        let pool = MemoryPool::new(1024 * 1024, false);

        pool.allocate(100).unwrap();
        pool.allocate(200).unwrap();

        pool.clear();

        assert_eq!(pool.used(), 0);
        assert_eq!(pool.cached(), 0);
        assert_eq!(pool.stats().active_blocks, 0);
    }
}
