//! Memory module Python bindings

use pyo3::prelude::*;
use std::sync::Arc;
use pygpukit_core::memory::{MemoryPool, MemoryBlock, PoolStats};

use crate::errors::memory_error_to_py;

/// Python wrapper for MemoryBlock
#[pyclass(name = "MemoryBlock")]
#[derive(Clone)]
pub struct PyMemoryBlock {
    inner: MemoryBlock,
}

#[pymethods]
impl PyMemoryBlock {
    /// Block ID
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    /// Block size in bytes
    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    /// Device pointer (as int, or None if not on GPU)
    #[getter]
    fn device_ptr(&self) -> Option<u64> {
        self.inner.device_ptr
    }

    /// Whether block is on GPU
    #[getter]
    fn on_gpu(&self) -> bool {
        self.inner.on_gpu
    }

    /// Whether block is on host
    #[getter]
    fn on_host(&self) -> bool {
        self.inner.on_host
    }

    /// Last access timestamp
    #[getter]
    fn last_access(&self) -> f64 {
        self.inner.last_access
    }

    /// Check if block is available for use
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Check if block has been evicted
    fn is_evicted(&self) -> bool {
        self.inner.is_evicted()
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryBlock(id={}, size={}, on_gpu={}, on_host={})",
            self.inner.id, self.inner.size, self.inner.on_gpu, self.inner.on_host
        )
    }
}

/// Python wrapper for PoolStats
#[pyclass(name = "PoolStats")]
#[derive(Clone)]
pub struct PyPoolStats {
    inner: PoolStats,
}

#[pymethods]
impl PyPoolStats {
    /// Memory quota
    #[getter]
    fn quota(&self) -> usize {
        self.inner.quota
    }

    /// Used memory
    #[getter]
    fn used(&self) -> usize {
        self.inner.used
    }

    /// Cached memory (in free lists)
    #[getter]
    fn cached(&self) -> usize {
        self.inner.cached
    }

    /// Available memory
    #[getter]
    fn available(&self) -> usize {
        self.inner.available
    }

    /// Total allocations
    #[getter]
    fn allocation_count(&self) -> u64 {
        self.inner.allocation_count
    }

    /// Blocks reused from cache
    #[getter]
    fn reuse_count(&self) -> u64 {
        self.inner.reuse_count
    }

    /// Blocks evicted
    #[getter]
    fn eviction_count(&self) -> u64 {
        self.inner.eviction_count
    }

    /// New CUDA allocations
    #[getter]
    fn cudamalloc_count(&self) -> u64 {
        self.inner.cudamalloc_count
    }

    /// Active block count
    #[getter]
    fn active_blocks(&self) -> usize {
        self.inner.active_blocks
    }

    /// Free block count
    #[getter]
    fn free_blocks(&self) -> usize {
        self.inner.free_blocks
    }

    fn __repr__(&self) -> String {
        format!(
            "PoolStats(quota={}, used={}, cached={}, available={}, active_blocks={}, free_blocks={})",
            self.inner.quota, self.inner.used, self.inner.cached,
            self.inner.available, self.inner.active_blocks, self.inner.free_blocks
        )
    }
}

/// Thread-safe GPU memory pool.
///
/// Provides efficient memory allocation with size-class bucketing and LRU eviction.
///
/// Args:
///     quota: Maximum memory in bytes
///     enable_eviction: Whether to evict blocks when quota exceeded
///
/// Example:
///     pool = MemoryPool(100 * 1024 * 1024, False)  # 100 MB
///     block_id = pool.allocate(4096)
///     pool.free(block_id)
#[pyclass(name = "MemoryPool")]
pub struct PyMemoryPool {
    inner: Arc<MemoryPool>,
}

#[pymethods]
impl PyMemoryPool {
    /// Create a new memory pool.
    #[new]
    #[pyo3(signature = (quota, enable_eviction=false))]
    fn new(quota: usize, enable_eviction: bool) -> Self {
        Self {
            inner: Arc::new(MemoryPool::new(quota, enable_eviction)),
        }
    }

    /// Get memory quota.
    #[getter]
    fn quota(&self) -> usize {
        self.inner.quota()
    }

    /// Get used memory.
    #[getter]
    fn used(&self) -> usize {
        self.inner.used()
    }

    /// Get cached memory.
    #[getter]
    fn cached(&self) -> usize {
        self.inner.cached()
    }

    /// Get available memory.
    #[getter]
    fn available(&self) -> usize {
        self.inner.available()
    }

    /// Allocate a memory block.
    ///
    /// Args:
    ///     size: Requested size in bytes (will be rounded to size class)
    ///
    /// Returns:
    ///     Block ID for the allocated block
    ///
    /// Raises:
    ///     MemoryError: If quota exceeded and cannot evict
    fn allocate(&self, size: usize) -> PyResult<u64> {
        self.inner.allocate(size).map_err(memory_error_to_py)
    }

    /// Free a memory block (return to free list).
    ///
    /// Args:
    ///     block_id: ID of the block to free
    fn free(&self, block_id: u64) {
        self.inner.free(block_id);
    }

    /// Update LRU timestamp for a block.
    ///
    /// Call this when accessing block data to keep it from being evicted.
    fn touch(&self, block_id: u64) {
        self.inner.touch(block_id);
    }

    /// Evict a block to host memory.
    ///
    /// The caller should copy data to host before calling this.
    fn evict(&self, block_id: u64) {
        self.inner.evict(block_id);
    }

    /// Restore an evicted block to GPU.
    ///
    /// The caller should allocate GPU memory and set device pointer.
    fn restore(&self, block_id: u64) {
        self.inner.restore(block_id);
    }

    /// Get pool statistics.
    fn stats(&self) -> PyPoolStats {
        PyPoolStats {
            inner: self.inner.stats(),
        }
    }

    /// Clear all allocations.
    fn clear(&self) {
        self.inner.clear();
    }

    /// Get a block by ID.
    fn get_block(&self, block_id: u64) -> Option<PyMemoryBlock> {
        self.inner.get_block(block_id).map(|b| PyMemoryBlock { inner: b })
    }

    /// Set device pointer for a block.
    fn set_device_ptr(&self, block_id: u64, device_ptr: u64) {
        self.inner.set_device_ptr(block_id, device_ptr);
    }

    /// Set host data for a block.
    fn set_host_data(&self, block_id: u64, data: Vec<u8>) {
        self.inner.set_host_data(block_id, data);
    }

    /// Get host data from a block.
    fn get_host_data(&self, block_id: u64) -> Option<Vec<u8>> {
        self.inner.get_host_data(block_id)
    }

    /// Clear host data from a block.
    fn clear_host_data(&self, block_id: u64) {
        self.inner.clear_host_data(block_id);
    }

    /// Get block size by ID.
    fn get_block_size(&self, block_id: u64) -> Option<usize> {
        self.inner.get_block_size(block_id)
    }

    /// Check if block is on GPU.
    fn is_block_on_gpu(&self, block_id: u64) -> bool {
        self.inner.is_block_on_gpu(block_id)
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryPool(quota={}, used={}, cached={}, available={})",
            self.inner.quota(),
            self.inner.used(),
            self.inner.cached(),
            self.inner.available()
        )
    }
}

/// Register memory module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryPool>()?;
    m.add_class::<PyMemoryBlock>()?;
    m.add_class::<PyPoolStats>()?;
    Ok(())
}
