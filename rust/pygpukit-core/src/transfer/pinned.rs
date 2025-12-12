//! Pinned Memory Manager
//!
//! Manages page-locked (pinned) host memory for faster CPU-GPU transfers.
//! Pinned memory enables DMA transfers and can significantly improve
//! transfer bandwidth compared to pageable memory.

use std::collections::HashMap;

/// Pinned memory block information
#[derive(Debug, Clone)]
pub struct PinnedBlock {
    /// Block ID
    pub id: u64,
    /// Host pointer (virtual address)
    pub host_ptr: u64,
    /// Size in bytes
    pub size: usize,
    /// Whether the block is currently in use
    pub in_use: bool,
    /// Associated task ID
    pub task_id: Option<String>,
    /// Allocation timestamp
    pub allocated_at: f64,
    /// Last access timestamp
    pub last_access: f64,
}

impl PinnedBlock {
    /// Create a new pinned block
    pub fn new(id: u64, host_ptr: u64, size: usize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Self {
            id,
            host_ptr,
            size,
            in_use: true,
            task_id: None,
            allocated_at: now,
            last_access: now,
        }
    }

    /// Associate with a task
    pub fn with_task(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Touch to update last access time
    pub fn touch(&mut self) {
        self.last_access = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
    }
}

/// Pinned memory pool configuration
#[derive(Debug, Clone)]
pub struct PinnedPoolConfig {
    /// Maximum total pinned memory in bytes
    pub max_size: usize,
    /// Enable pooling for reuse
    pub enable_pooling: bool,
    /// Size classes for pooling (in bytes)
    pub size_classes: Vec<usize>,
    /// Default alignment
    pub alignment: usize,
}

impl Default for PinnedPoolConfig {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024 * 1024, // 1GB default
            enable_pooling: true,
            size_classes: vec![
                4096,           // 4KB
                65536,          // 64KB
                262144,         // 256KB
                1048576,        // 1MB
                4194304,        // 4MB
                16777216,       // 16MB
                67108864,       // 64MB
                268435456,      // 256MB
            ],
            alignment: 256, // CUDA alignment
        }
    }
}

impl PinnedPoolConfig {
    /// Create with max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size,
            ..Default::default()
        }
    }

    /// Set pooling enabled
    pub fn enable_pooling(mut self, enable: bool) -> Self {
        self.enable_pooling = enable;
        self
    }

    /// Get size class for a given size
    pub fn get_size_class(&self, size: usize) -> usize {
        for &class_size in &self.size_classes {
            if size <= class_size {
                return class_size;
            }
        }
        // Round up to alignment
        let aligned = (size + self.alignment - 1) / self.alignment * self.alignment;
        aligned
    }
}

/// Pinned memory manager statistics
#[derive(Debug, Clone, Default)]
pub struct PinnedStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Current bytes in use
    pub current_used: usize,
    /// Peak bytes used
    pub peak_used: usize,
    /// Total allocations
    pub total_allocations: usize,
    /// Total frees
    pub total_frees: usize,
    /// Pool hits (reused from pool)
    pub pool_hits: usize,
    /// Pool misses (new allocation)
    pub pool_misses: usize,
    /// Current pool size
    pub pool_size: usize,
    /// Blocks currently pooled
    pub pooled_blocks: usize,
}

/// Pinned memory manager
///
/// Manages pinned host memory allocations with optional pooling
/// for efficient reuse.
#[derive(Debug)]
pub struct PinnedMemoryManager {
    config: PinnedPoolConfig,
    /// Active allocations by ID
    active: HashMap<u64, PinnedBlock>,
    /// Free pool by size class
    pool: HashMap<usize, Vec<PinnedBlock>>,
    /// Next block ID
    next_id: u64,
    /// Statistics
    total_allocated: usize,
    current_used: usize,
    peak_used: usize,
    total_allocations: usize,
    total_frees: usize,
    pool_hits: usize,
    pool_misses: usize,
}

impl PinnedMemoryManager {
    /// Create a new pinned memory manager
    pub fn new(config: PinnedPoolConfig) -> Self {
        Self {
            config,
            active: HashMap::new(),
            pool: HashMap::new(),
            next_id: 1,
            total_allocated: 0,
            current_used: 0,
            peak_used: 0,
            total_allocations: 0,
            total_frees: 0,
            pool_hits: 0,
            pool_misses: 0,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(PinnedPoolConfig::default())
    }

    /// Create with max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self::new(PinnedPoolConfig::with_max_size(max_size))
    }

    /// Check if allocation would exceed quota
    pub fn can_allocate(&self, size: usize) -> bool {
        // If pooling is enabled and we have a suitable block, always allow
        if self.config.enable_pooling {
            let size_class = self.config.get_size_class(size);
            if let Some(blocks) = self.pool.get(&size_class) {
                if !blocks.is_empty() {
                    return true;
                }
            }
        }
        // Check quota
        self.current_used + size <= self.config.max_size
    }

    /// Allocate pinned memory
    ///
    /// If pooling is enabled, may return a previously freed block.
    /// Returns the block ID and size class (actual allocated size).
    ///
    /// The `host_ptr` should be provided by the caller after performing
    /// the actual cudaHostAlloc call if this is a new allocation.
    pub fn allocate(&mut self, size: usize) -> Result<(u64, usize, bool), PinnedError> {
        let size_class = self.config.get_size_class(size);

        // Try to get from pool first
        if self.config.enable_pooling {
            if let Some(blocks) = self.pool.get_mut(&size_class) {
                if let Some(mut block) = blocks.pop() {
                    block.in_use = true;
                    block.touch();
                    let id = block.id;
                    self.active.insert(id, block);
                    self.current_used += size_class;
                    self.peak_used = self.peak_used.max(self.current_used);
                    self.total_allocations += 1;
                    self.pool_hits += 1;
                    // Return (id, size_class, reused=true)
                    return Ok((id, size_class, true));
                }
            }
        }

        // Check quota
        if self.current_used + size_class > self.config.max_size {
            return Err(PinnedError::QuotaExceeded {
                requested: size_class,
                available: self.config.max_size.saturating_sub(self.current_used),
            });
        }

        // Need new allocation
        let id = self.next_id;
        self.next_id += 1;
        self.total_allocations += 1;
        self.total_allocated += size_class;
        self.current_used += size_class;
        self.peak_used = self.peak_used.max(self.current_used);
        self.pool_misses += 1;

        // Return (id, size_class, reused=false) - caller must allocate
        Ok((id, size_class, false))
    }

    /// Register an allocated block
    ///
    /// Called after cudaHostAlloc succeeds with the actual pointer.
    pub fn register(&mut self, id: u64, host_ptr: u64, size: usize) {
        let size_class = self.config.get_size_class(size);
        let block = PinnedBlock::new(id, host_ptr, size_class);
        self.active.insert(id, block);
    }

    /// Free a pinned block
    ///
    /// If pooling is enabled, the block is returned to the pool.
    /// Returns (should_free, host_ptr) - if should_free is true,
    /// the caller should call cudaFreeHost.
    pub fn free(&mut self, id: u64) -> Result<(bool, u64), PinnedError> {
        let block = self.active.remove(&id)
            .ok_or(PinnedError::InvalidBlock { id })?;

        self.total_frees += 1;
        self.current_used = self.current_used.saturating_sub(block.size);

        // Return to pool if enabled
        if self.config.enable_pooling {
            let size_class = block.size;
            let mut pooled_block = block.clone();
            pooled_block.in_use = false;
            pooled_block.task_id = None;

            self.pool
                .entry(size_class)
                .or_insert_with(Vec::new)
                .push(pooled_block);

            // Don't free, return to pool
            Ok((false, block.host_ptr))
        } else {
            // Free immediately
            Ok((true, block.host_ptr))
        }
    }

    /// Associate a block with a task
    pub fn associate_task(&mut self, id: u64, task_id: String) -> Result<(), PinnedError> {
        let block = self.active.get_mut(&id)
            .ok_or(PinnedError::InvalidBlock { id })?;
        block.task_id = Some(task_id);
        Ok(())
    }

    /// Get a block by ID
    pub fn get(&self, id: u64) -> Option<&PinnedBlock> {
        self.active.get(&id)
    }

    /// Touch a block to update access time
    pub fn touch(&mut self, id: u64) -> Result<(), PinnedError> {
        let block = self.active.get_mut(&id)
            .ok_or(PinnedError::InvalidBlock { id })?;
        block.touch();
        Ok(())
    }

    /// Get blocks for a task
    pub fn get_blocks_for_task(&self, task_id: &str) -> Vec<&PinnedBlock> {
        self.active.values()
            .filter(|b| b.task_id.as_deref() == Some(task_id))
            .collect()
    }

    /// Free all blocks for a task
    ///
    /// Returns list of (should_free, host_ptr) for blocks to potentially free.
    pub fn free_task_blocks(&mut self, task_id: &str) -> Vec<(bool, u64)> {
        let ids: Vec<u64> = self.active.values()
            .filter(|b| b.task_id.as_deref() == Some(task_id))
            .map(|b| b.id)
            .collect();

        ids.into_iter()
            .filter_map(|id| self.free(id).ok())
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> PinnedStats {
        let pool_size: usize = self.pool.values()
            .flat_map(|v| v.iter())
            .map(|b| b.size)
            .sum();
        let pooled_blocks: usize = self.pool.values()
            .map(|v| v.len())
            .sum();

        PinnedStats {
            total_allocated: self.total_allocated,
            current_used: self.current_used,
            peak_used: self.peak_used,
            total_allocations: self.total_allocations,
            total_frees: self.total_frees,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            pool_size,
            pooled_blocks,
        }
    }

    /// Clear the pool (free all pooled blocks)
    ///
    /// Returns list of host pointers to free.
    pub fn clear_pool(&mut self) -> Vec<u64> {
        let mut ptrs = Vec::new();
        for blocks in self.pool.values() {
            for block in blocks {
                ptrs.push(block.host_ptr);
            }
        }
        self.pool.clear();
        ptrs
    }

    /// Clear all state (active + pool)
    ///
    /// Returns list of all host pointers to free.
    pub fn clear(&mut self) -> Vec<u64> {
        let mut ptrs: Vec<u64> = self.active.values()
            .map(|b| b.host_ptr)
            .collect();

        ptrs.extend(self.clear_pool());

        self.active.clear();
        self.current_used = 0;

        ptrs
    }

    /// Get config
    pub fn config(&self) -> &PinnedPoolConfig {
        &self.config
    }

    /// Get number of active blocks
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

/// Pinned memory errors
#[derive(Debug, Clone)]
pub enum PinnedError {
    /// Quota exceeded
    QuotaExceeded {
        requested: usize,
        available: usize,
    },
    /// Invalid block ID
    InvalidBlock {
        id: u64,
    },
    /// Allocation failed
    AllocationFailed {
        reason: String,
    },
}

impl std::fmt::Display for PinnedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinnedError::QuotaExceeded { requested, available } => {
                write!(f, "Pinned memory quota exceeded: requested {} bytes, {} available", requested, available)
            }
            PinnedError::InvalidBlock { id } => {
                write!(f, "Invalid pinned block ID: {}", id)
            }
            PinnedError::AllocationFailed { reason } => {
                write!(f, "Pinned memory allocation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for PinnedError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_config() {
        let config = PinnedPoolConfig::with_max_size(1024 * 1024)
            .enable_pooling(true);

        assert_eq!(config.max_size, 1024 * 1024);
        assert!(config.enable_pooling);
    }

    #[test]
    fn test_size_class() {
        let config = PinnedPoolConfig::default();

        // Small allocation should use 4KB class
        assert_eq!(config.get_size_class(100), 4096);

        // 10KB should use 64KB class
        assert_eq!(config.get_size_class(10000), 65536);

        // 100KB should use 256KB class
        assert_eq!(config.get_size_class(100_000), 262144);
    }

    #[test]
    fn test_allocate_new() {
        let mut manager = PinnedMemoryManager::with_max_size(1024 * 1024);

        // Allocate 1KB
        let result = manager.allocate(1024);
        assert!(result.is_ok());

        let (id, size_class, reused) = result.unwrap();
        assert_eq!(id, 1);
        assert_eq!(size_class, 4096); // Rounded to 4KB
        assert!(!reused);

        // Register the block
        manager.register(id, 0x1000, 4096);
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_pool_reuse() {
        let mut manager = PinnedMemoryManager::with_max_size(1024 * 1024);

        // Allocate and register
        let (id1, size_class, _) = manager.allocate(1024).unwrap();
        manager.register(id1, 0x1000, size_class);

        // Free to pool
        let (should_free, _) = manager.free(id1).unwrap();
        assert!(!should_free); // Pooled, not freed

        // Allocate again - should reuse from pool
        let (id2, _, reused) = manager.allocate(1024).unwrap();
        assert!(reused);
        assert_eq!(manager.stats().pool_hits, 1);

        // Same ID since reused from pool
        let block = manager.get(id2).unwrap();
        assert_eq!(block.host_ptr, 0x1000);
    }

    #[test]
    fn test_quota_exceeded() {
        let mut manager = PinnedMemoryManager::with_max_size(4096); // Only 4KB

        // First allocation should succeed
        let result1 = manager.allocate(1024);
        assert!(result1.is_ok());
        let (id1, size_class, _) = result1.unwrap();
        manager.register(id1, 0x1000, size_class);

        // Second allocation should fail (quota exceeded)
        let result2 = manager.allocate(1024);
        assert!(matches!(result2, Err(PinnedError::QuotaExceeded { .. })));
    }

    #[test]
    fn test_task_association() {
        let mut manager = PinnedMemoryManager::with_max_size(1024 * 1024);

        let (id, size_class, _) = manager.allocate(1024).unwrap();
        manager.register(id, 0x1000, size_class);

        manager.associate_task(id, "task-1".into()).unwrap();

        let blocks = manager.get_blocks_for_task("task-1");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].id, id);
    }

    #[test]
    fn test_free_task_blocks() {
        let mut manager = PinnedMemoryManager::with_max_size(1024 * 1024);

        // Allocate two blocks for same task
        let (id1, sc1, _) = manager.allocate(1024).unwrap();
        manager.register(id1, 0x1000, sc1);
        manager.associate_task(id1, "task-1".into()).unwrap();

        let (id2, sc2, _) = manager.allocate(2048).unwrap();
        manager.register(id2, 0x2000, sc2);
        manager.associate_task(id2, "task-1".into()).unwrap();

        // Allocate one for different task
        let (id3, sc3, _) = manager.allocate(1024).unwrap();
        manager.register(id3, 0x3000, sc3);
        manager.associate_task(id3, "task-2".into()).unwrap();

        assert_eq!(manager.active_count(), 3);

        // Free task-1 blocks
        let freed = manager.free_task_blocks("task-1");
        assert_eq!(freed.len(), 2);

        // Only task-2 block remains active
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_stats() {
        let mut manager = PinnedMemoryManager::with_max_size(1024 * 1024);

        let (id1, sc1, _) = manager.allocate(1024).unwrap();
        manager.register(id1, 0x1000, sc1);

        let (id2, sc2, _) = manager.allocate(2048).unwrap();
        manager.register(id2, 0x2000, sc2);

        let stats = manager.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.pool_misses, 2);
        assert_eq!(stats.pool_hits, 0);

        // Free and reallocate
        manager.free(id1).unwrap();
        manager.allocate(512).unwrap();

        let stats2 = manager.stats();
        assert_eq!(stats2.total_allocations, 3);
        assert_eq!(stats2.pool_hits, 1);
    }

    #[test]
    fn test_clear() {
        let mut manager = PinnedMemoryManager::with_max_size(1024 * 1024);

        let (id1, sc1, _) = manager.allocate(1024).unwrap();
        manager.register(id1, 0x1000, sc1);

        let (id2, sc2, _) = manager.allocate(2048).unwrap();
        manager.register(id2, 0x2000, sc2);

        manager.free(id1).unwrap();

        // Clear returns all pointers
        let ptrs = manager.clear();
        assert_eq!(ptrs.len(), 2); // 1 active + 1 pooled

        assert_eq!(manager.active_count(), 0);
        assert_eq!(manager.stats().current_used, 0);
    }

    #[test]
    fn test_no_pooling() {
        let config = PinnedPoolConfig::with_max_size(1024 * 1024)
            .enable_pooling(false);
        let mut manager = PinnedMemoryManager::new(config);

        let (id, size_class, _) = manager.allocate(1024).unwrap();
        manager.register(id, 0x1000, size_class);

        // Free should indicate actual free needed
        let (should_free, _) = manager.free(id).unwrap();
        assert!(should_free); // Not pooled

        // Next allocation is not reused
        let (_, _, reused) = manager.allocate(1024).unwrap();
        assert!(!reused);
    }
}
