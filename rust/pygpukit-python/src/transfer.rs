//! Python bindings for the async transfer engine

use pyo3::prelude::*;
use pygpukit_core::transfer::{
    AsyncTransferEngine, TransferOp, TransferState, TransferStats, TransferType, StreamType,
    PinnedMemoryManager, PinnedPoolConfig, PinnedBlock, PinnedStats,
};

use crate::errors::pinned_error_to_py;

/// Python wrapper for TransferType enum
#[pyclass(name = "TransferType")]
#[derive(Clone)]
pub struct PyTransferType {
    inner: TransferType,
}

#[pymethods]
impl PyTransferType {
    /// Host to Device transfer
    #[classattr]
    fn H2D() -> Self {
        Self { inner: TransferType::H2D }
    }

    /// Device to Host transfer
    #[classattr]
    fn D2H() -> Self {
        Self { inner: TransferType::D2H }
    }

    /// Device to Device transfer
    #[classattr]
    fn D2D() -> Self {
        Self { inner: TransferType::D2D }
    }

    fn __repr__(&self) -> String {
        format!("TransferType.{}", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Python wrapper for TransferState enum
#[pyclass(name = "TransferState")]
#[derive(Clone)]
pub struct PyTransferState {
    inner: TransferState,
}

#[pymethods]
impl PyTransferState {
    #[classattr]
    fn Queued() -> Self {
        Self { inner: TransferState::Queued }
    }

    #[classattr]
    fn InProgress() -> Self {
        Self { inner: TransferState::InProgress }
    }

    #[classattr]
    fn Completed() -> Self {
        Self { inner: TransferState::Completed }
    }

    #[classattr]
    fn Failed() -> Self {
        Self { inner: TransferState::Failed }
    }

    #[classattr]
    fn Cancelled() -> Self {
        Self { inner: TransferState::Cancelled }
    }

    /// Check if this is a terminal state
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn __repr__(&self) -> String {
        let name = match self.inner {
            TransferState::Queued => "Queued",
            TransferState::InProgress => "InProgress",
            TransferState::Completed => "Completed",
            TransferState::Failed => "Failed",
            TransferState::Cancelled => "Cancelled",
        };
        format!("TransferState.{}", name)
    }
}

/// Python wrapper for StreamType enum
#[pyclass(name = "StreamType")]
#[derive(Clone)]
pub struct PyStreamType {
    inner: StreamType,
}

#[pymethods]
impl PyStreamType {
    #[classattr]
    fn Compute() -> Self {
        Self { inner: StreamType::Compute }
    }

    #[classattr]
    fn MemcpyH2D() -> Self {
        Self { inner: StreamType::MemcpyH2D }
    }

    #[classattr]
    fn MemcpyD2H() -> Self {
        Self { inner: StreamType::MemcpyD2H }
    }

    #[staticmethod]
    fn Custom(id: u32) -> Self {
        Self { inner: StreamType::Custom(id) }
    }

    /// Get stream ID
    fn to_id(&self) -> u32 {
        self.inner.to_id()
    }
}

/// Python wrapper for TransferOp
#[pyclass(name = "TransferOp")]
#[derive(Clone)]
pub struct PyTransferOp {
    inner: TransferOp,
}

#[pymethods]
impl PyTransferOp {
    /// Operation ID
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    /// Transfer type
    #[getter]
    fn transfer_type(&self) -> PyTransferType {
        PyTransferType { inner: self.inner.transfer_type }
    }

    /// Source pointer
    #[getter]
    fn src_ptr(&self) -> u64 {
        self.inner.src_ptr
    }

    /// Destination pointer
    #[getter]
    fn dst_ptr(&self) -> u64 {
        self.inner.dst_ptr
    }

    /// Size in bytes
    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    /// Current state
    #[getter]
    fn state(&self) -> PyTransferState {
        PyTransferState { inner: self.inner.state }
    }

    /// Stream ID
    #[getter]
    fn stream_id(&self) -> u32 {
        self.inner.stream_id
    }

    /// Timestamp when queued
    #[getter]
    fn queued_at(&self) -> f64 {
        self.inner.queued_at
    }

    /// Timestamp when started
    #[getter]
    fn started_at(&self) -> Option<f64> {
        self.inner.started_at
    }

    /// Timestamp when completed
    #[getter]
    fn completed_at(&self) -> Option<f64> {
        self.inner.completed_at
    }

    /// Priority
    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority
    }

    /// Error message if failed
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    /// Associated task ID
    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id.clone()
    }

    /// Get wait time (time in queue)
    fn wait_time(&self) -> f64 {
        self.inner.wait_time()
    }

    /// Get transfer duration
    fn duration(&self) -> Option<f64> {
        self.inner.duration()
    }

    /// Get bandwidth in GB/s
    fn bandwidth_gbps(&self) -> Option<f64> {
        self.inner.bandwidth_gbps()
    }

    fn __repr__(&self) -> String {
        format!(
            "TransferOp(id={}, type={}, size={}, state={:?})",
            self.inner.id, self.inner.transfer_type.name(), self.inner.size, self.inner.state
        )
    }
}

/// Python wrapper for TransferStats
#[pyclass(name = "TransferStats")]
#[derive(Clone)]
pub struct PyTransferStats {
    inner: TransferStats,
}

#[pymethods]
impl PyTransferStats {
    #[getter]
    fn total_queued(&self) -> usize {
        self.inner.total_queued
    }

    #[getter]
    fn completed_count(&self) -> usize {
        self.inner.completed_count
    }

    #[getter]
    fn failed_count(&self) -> usize {
        self.inner.failed_count
    }

    #[getter]
    fn total_bytes(&self) -> usize {
        self.inner.total_bytes
    }

    #[getter]
    fn h2d_bytes(&self) -> usize {
        self.inner.h2d_bytes
    }

    #[getter]
    fn d2h_bytes(&self) -> usize {
        self.inner.d2h_bytes
    }

    #[getter]
    fn avg_h2d_bandwidth(&self) -> f64 {
        self.inner.avg_h2d_bandwidth
    }

    #[getter]
    fn avg_d2h_bandwidth(&self) -> f64 {
        self.inner.avg_d2h_bandwidth
    }

    #[getter]
    fn pending_count(&self) -> usize {
        self.inner.pending_count
    }

    #[getter]
    fn in_progress_count(&self) -> usize {
        self.inner.in_progress_count
    }

    fn __repr__(&self) -> String {
        format!(
            "TransferStats(completed={}, pending={}, in_progress={}, h2d_bw={:.2} GB/s, d2h_bw={:.2} GB/s)",
            self.inner.completed_count,
            self.inner.pending_count,
            self.inner.in_progress_count,
            self.inner.avg_h2d_bandwidth,
            self.inner.avg_d2h_bandwidth,
        )
    }
}

/// Async Memory Transfer Engine
///
/// Manages asynchronous memory transfers between host and device with
/// separate streams for H2D and D2H to enable overlap.
#[pyclass(name = "AsyncTransferEngine")]
pub struct PyAsyncTransferEngine {
    inner: AsyncTransferEngine,
}

#[pymethods]
impl PyAsyncTransferEngine {
    /// Create a new transfer engine
    ///
    /// Args:
    ///     max_concurrent: Maximum concurrent transfers per stream (default: 4)
    #[new]
    #[pyo3(signature = (max_concurrent=4))]
    fn new(max_concurrent: usize) -> Self {
        Self {
            inner: AsyncTransferEngine::new(max_concurrent),
        }
    }

    /// Enqueue an H2D transfer
    ///
    /// Args:
    ///     host_ptr: Host memory address (as int)
    ///     device_ptr: Device memory address (as int)
    ///     size: Size in bytes
    ///
    /// Returns:
    ///     Operation ID
    fn enqueue_h2d(&self, host_ptr: u64, device_ptr: u64, size: usize) -> u64 {
        self.inner.enqueue_h2d(host_ptr, device_ptr, size)
    }

    /// Enqueue a D2H transfer
    fn enqueue_d2h(&self, device_ptr: u64, host_ptr: u64, size: usize) -> u64 {
        self.inner.enqueue_d2h(device_ptr, host_ptr, size)
    }

    /// Enqueue a D2D transfer
    fn enqueue_d2d(&self, src_ptr: u64, dst_ptr: u64, size: usize) -> u64 {
        self.inner.enqueue_d2d(src_ptr, dst_ptr, size)
    }

    /// Enqueue with priority
    ///
    /// Args:
    ///     transfer_type: "h2d", "d2h", or "d2d"
    ///     src_ptr: Source address
    ///     dst_ptr: Destination address
    ///     size: Size in bytes
    ///     priority: Priority (higher = more urgent)
    fn enqueue_with_priority(
        &self,
        transfer_type: &str,
        src_ptr: u64,
        dst_ptr: u64,
        size: usize,
        priority: i32,
    ) -> PyResult<u64> {
        let t_type = match transfer_type.to_lowercase().as_str() {
            "h2d" => TransferType::H2D,
            "d2h" => TransferType::D2H,
            "d2d" => TransferType::D2D,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid transfer type. Use 'h2d', 'd2h', or 'd2d'"
            )),
        };

        Ok(self.inner.enqueue_with_priority(t_type, src_ptr, dst_ptr, size, priority))
    }

    /// Get transfers ready to execute
    ///
    /// Args:
    ///     max_transfers: Maximum number of transfers to return
    ///
    /// Returns:
    ///     List of TransferOp objects ready to execute
    fn get_ready_transfers(&self, max_transfers: usize) -> Vec<PyTransferOp> {
        self.inner
            .get_ready_transfers(max_transfers)
            .into_iter()
            .map(|op| PyTransferOp { inner: op })
            .collect()
    }

    /// Mark a transfer as started
    fn start_transfer(&self, op_id: u64) -> bool {
        self.inner.start_transfer(op_id)
    }

    /// Mark a transfer as completed
    fn complete_transfer(&self, op_id: u64) -> bool {
        self.inner.complete_transfer(op_id)
    }

    /// Mark a transfer as failed
    fn fail_transfer(&self, op_id: u64, error: String) -> bool {
        self.inner.fail_transfer(op_id, error)
    }

    /// Cancel a pending transfer
    fn cancel_transfer(&self, op_id: u64) -> bool {
        self.inner.cancel_transfer(op_id)
    }

    /// Get operation by ID
    fn get_operation(&self, op_id: u64) -> Option<PyTransferOp> {
        self.inner.get_operation(op_id).map(|op| PyTransferOp { inner: op })
    }

    /// Get transfer statistics
    fn stats(&self) -> PyTransferStats {
        PyTransferStats { inner: self.inner.stats() }
    }

    /// Get in-progress transfer IDs for a stream
    fn get_in_progress_for_stream(&self, stream_id: u32) -> Vec<u64> {
        self.inner.get_in_progress_for_stream(stream_id)
    }

    /// Check if there's pending work
    fn has_pending_work(&self) -> bool {
        self.inner.has_pending_work()
    }

    /// Garbage collect completed operations
    fn gc(&self) {
        self.inner.gc()
    }

    /// Clear all operations
    fn clear(&self) {
        self.inner.clear()
    }
}

// =============================================================================
// Pinned Memory Types
// =============================================================================

/// Pinned memory pool configuration for Python
#[pyclass(name = "PinnedPoolConfig")]
#[derive(Clone)]
pub struct PyPinnedPoolConfig {
    inner: PinnedPoolConfig,
}

#[pymethods]
impl PyPinnedPoolConfig {
    #[new]
    #[pyo3(signature = (max_size=1073741824, enable_pooling=true, alignment=256))]
    fn new(max_size: usize, enable_pooling: bool, alignment: usize) -> Self {
        Self {
            inner: PinnedPoolConfig {
                max_size,
                enable_pooling,
                alignment,
                ..Default::default()
            },
        }
    }

    #[getter]
    fn max_size(&self) -> usize {
        self.inner.max_size
    }

    #[getter]
    fn enable_pooling(&self) -> bool {
        self.inner.enable_pooling
    }

    #[getter]
    fn alignment(&self) -> usize {
        self.inner.alignment
    }

    /// Get size class for a given size
    fn get_size_class(&self, size: usize) -> usize {
        self.inner.get_size_class(size)
    }

    fn __repr__(&self) -> String {
        format!(
            "PinnedPoolConfig(max_size={}, pooling={}, alignment={})",
            self.inner.max_size, self.inner.enable_pooling, self.inner.alignment
        )
    }
}

/// Pinned memory block for Python
#[pyclass(name = "PinnedBlock")]
#[derive(Clone)]
pub struct PyPinnedBlock {
    inner: PinnedBlock,
}

#[pymethods]
impl PyPinnedBlock {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    #[getter]
    fn host_ptr(&self) -> u64 {
        self.inner.host_ptr
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[getter]
    fn in_use(&self) -> bool {
        self.inner.in_use
    }

    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id.clone()
    }

    #[getter]
    fn allocated_at(&self) -> f64 {
        self.inner.allocated_at
    }

    #[getter]
    fn last_access(&self) -> f64 {
        self.inner.last_access
    }

    fn __repr__(&self) -> String {
        format!(
            "PinnedBlock(id={}, size={}, in_use={})",
            self.inner.id, self.inner.size, self.inner.in_use
        )
    }
}

/// Pinned memory statistics for Python
#[pyclass(name = "PinnedStats")]
#[derive(Clone)]
pub struct PyPinnedStats {
    inner: PinnedStats,
}

#[pymethods]
impl PyPinnedStats {
    #[getter]
    fn total_allocated(&self) -> usize {
        self.inner.total_allocated
    }

    #[getter]
    fn current_used(&self) -> usize {
        self.inner.current_used
    }

    #[getter]
    fn peak_used(&self) -> usize {
        self.inner.peak_used
    }

    #[getter]
    fn total_allocations(&self) -> usize {
        self.inner.total_allocations
    }

    #[getter]
    fn total_frees(&self) -> usize {
        self.inner.total_frees
    }

    #[getter]
    fn pool_hits(&self) -> usize {
        self.inner.pool_hits
    }

    #[getter]
    fn pool_misses(&self) -> usize {
        self.inner.pool_misses
    }

    #[getter]
    fn pool_size(&self) -> usize {
        self.inner.pool_size
    }

    #[getter]
    fn pooled_blocks(&self) -> usize {
        self.inner.pooled_blocks
    }

    /// Pool hit rate (0.0 - 1.0)
    fn hit_rate(&self) -> f64 {
        let total = self.inner.pool_hits + self.inner.pool_misses;
        if total > 0 {
            self.inner.pool_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PinnedStats(used={}/{}, hit_rate={:.1}%)",
            self.inner.current_used, self.inner.peak_used,
            self.hit_rate() * 100.0
        )
    }
}

/// Pinned memory manager for Python
///
/// Manages page-locked (pinned) host memory for faster
/// CPU-GPU transfers with optional pooling for reuse.
#[pyclass(name = "PinnedMemoryManager")]
pub struct PyPinnedMemoryManager {
    inner: PinnedMemoryManager,
}

#[pymethods]
impl PyPinnedMemoryManager {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPinnedPoolConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: PinnedMemoryManager::new(cfg),
        }
    }

    /// Create with max size
    #[staticmethod]
    fn with_max_size(max_size: usize) -> Self {
        Self {
            inner: PinnedMemoryManager::with_max_size(max_size),
        }
    }

    /// Check if allocation would succeed
    fn can_allocate(&self, size: usize) -> bool {
        self.inner.can_allocate(size)
    }

    /// Allocate pinned memory
    ///
    /// Returns (id, size_class, reused) tuple.
    /// If reused=False, caller must perform cudaHostAlloc
    /// and then call register().
    fn allocate(&mut self, size: usize) -> PyResult<(u64, usize, bool)> {
        self.inner.allocate(size).map_err(pinned_error_to_py)
    }

    /// Register an allocated block
    ///
    /// Call this after cudaHostAlloc succeeds with the actual pointer.
    fn register(&mut self, id: u64, host_ptr: u64, size: usize) {
        self.inner.register(id, host_ptr, size);
    }

    /// Free a pinned block
    ///
    /// Returns (should_free, host_ptr) tuple.
    /// If should_free=True, caller should call cudaFreeHost.
    fn free(&mut self, id: u64) -> PyResult<(bool, u64)> {
        self.inner.free(id).map_err(pinned_error_to_py)
    }

    /// Associate a block with a task
    fn associate_task(&mut self, id: u64, task_id: String) -> PyResult<()> {
        self.inner.associate_task(id, task_id).map_err(pinned_error_to_py)
    }

    /// Get a block by ID
    fn get(&self, id: u64) -> Option<PyPinnedBlock> {
        self.inner.get(id).map(|b| PyPinnedBlock { inner: b.clone() })
    }

    /// Touch a block to update access time
    fn touch(&mut self, id: u64) -> PyResult<()> {
        self.inner.touch(id).map_err(pinned_error_to_py)
    }

    /// Get blocks for a task
    fn get_blocks_for_task(&self, task_id: &str) -> Vec<PyPinnedBlock> {
        self.inner.get_blocks_for_task(task_id)
            .into_iter()
            .map(|b| PyPinnedBlock { inner: b.clone() })
            .collect()
    }

    /// Free all blocks for a task
    ///
    /// Returns list of (should_free, host_ptr) tuples.
    fn free_task_blocks(&mut self, task_id: &str) -> Vec<(bool, u64)> {
        self.inner.free_task_blocks(task_id)
    }

    /// Get statistics
    fn stats(&self) -> PyPinnedStats {
        PyPinnedStats {
            inner: self.inner.stats(),
        }
    }

    /// Clear the pool (free all pooled blocks)
    ///
    /// Returns list of host pointers to free.
    fn clear_pool(&mut self) -> Vec<u64> {
        self.inner.clear_pool()
    }

    /// Clear all state
    ///
    /// Returns list of all host pointers to free.
    fn clear(&mut self) -> Vec<u64> {
        self.inner.clear()
    }

    /// Get number of active blocks
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "PinnedMemoryManager(active={}, used={} bytes)",
            self.inner.active_count(), stats.current_used
        )
    }
}

/// Register transfer module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransferType>()?;
    m.add_class::<PyTransferState>()?;
    m.add_class::<PyStreamType>()?;
    m.add_class::<PyTransferOp>()?;
    m.add_class::<PyTransferStats>()?;
    m.add_class::<PyAsyncTransferEngine>()?;
    // Pinned memory
    m.add_class::<PyPinnedPoolConfig>()?;
    m.add_class::<PyPinnedBlock>()?;
    m.add_class::<PyPinnedStats>()?;
    m.add_class::<PyPinnedMemoryManager>()?;
    Ok(())
}
