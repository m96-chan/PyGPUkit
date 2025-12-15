//! Python bindings for the kernel dispatch controller

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;
use std::path::PathBuf;
use pygpukit_core::dispatch::{
    KernelDispatcher, KernelLaunchRequest, KernelState, DispatchStats, LaunchConfig,
    KernelPacingEngine, PacingConfig, PacingDecision, PacingStats, StreamPacingStats,
    SliceScheduler, SliceConfig, KernelSlice, SliceInfo, SliceStats,
    KernelCache, CacheConfig, CachedKernel, CompileOptions, CacheStats,
    PersistentCache, PersistentCacheConfig, PersistentCacheStats, PersistentEntry, ArchFingerprint,
};

/// Python wrapper for KernelState enum
#[pyclass(name = "KernelState")]
#[derive(Clone)]
pub struct PyKernelState {
    inner: KernelState,
}

#[pymethods]
impl PyKernelState {
    #[classattr]
    fn Queued() -> Self {
        Self { inner: KernelState::Queued }
    }

    #[classattr]
    fn Launched() -> Self {
        Self { inner: KernelState::Launched }
    }

    #[classattr]
    fn Completed() -> Self {
        Self { inner: KernelState::Completed }
    }

    #[classattr]
    fn Failed() -> Self {
        Self { inner: KernelState::Failed }
    }

    #[classattr]
    fn Cancelled() -> Self {
        Self { inner: KernelState::Cancelled }
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn __repr__(&self) -> String {
        let name = match self.inner {
            KernelState::Queued => "Queued",
            KernelState::Launched => "Launched",
            KernelState::Completed => "Completed",
            KernelState::Failed => "Failed",
            KernelState::Cancelled => "Cancelled",
        };
        format!("KernelState.{}", name)
    }
}

/// Python wrapper for LaunchConfig
#[pyclass(name = "LaunchConfig")]
#[derive(Clone)]
pub struct PyLaunchConfig {
    inner: LaunchConfig,
}

#[pymethods]
impl PyLaunchConfig {
    #[new]
    #[pyo3(signature = (grid=(1, 1, 1), block=(256, 1, 1), shared_mem=0, stream_id=0))]
    fn new(
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream_id: u32,
    ) -> Self {
        Self {
            inner: LaunchConfig {
                grid,
                block,
                shared_mem,
                stream_id,
            },
        }
    }

    /// Create a 1D linear launch config
    #[staticmethod]
    #[pyo3(signature = (n_elements, block_size=256))]
    fn linear(n_elements: usize, block_size: u32) -> Self {
        Self {
            inner: LaunchConfig::linear(n_elements, block_size),
        }
    }

    /// Create a 2D grid launch config
    #[staticmethod]
    fn grid_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            inner: LaunchConfig::grid_2d(grid_x, grid_y, block_x, block_y),
        }
    }

    #[getter]
    fn grid(&self) -> (u32, u32, u32) {
        self.inner.grid
    }

    #[getter]
    fn block(&self) -> (u32, u32, u32) {
        self.inner.block
    }

    #[getter]
    fn shared_mem(&self) -> u32 {
        self.inner.shared_mem
    }

    #[setter]
    fn set_shared_mem(&mut self, bytes: u32) {
        self.inner.shared_mem = bytes;
    }

    #[getter]
    fn stream_id(&self) -> u32 {
        self.inner.stream_id
    }

    #[setter]
    fn set_stream_id(&mut self, stream_id: u32) {
        self.inner.stream_id = stream_id;
    }

    fn __repr__(&self) -> String {
        format!(
            "LaunchConfig(grid={:?}, block={:?}, shared_mem={}, stream_id={})",
            self.inner.grid, self.inner.block, self.inner.shared_mem, self.inner.stream_id
        )
    }
}

/// Python wrapper for KernelLaunchRequest
#[pyclass(name = "KernelLaunchRequest")]
#[derive(Clone)]
pub struct PyKernelLaunchRequest {
    inner: KernelLaunchRequest,
}

#[pymethods]
impl PyKernelLaunchRequest {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    #[getter]
    fn kernel_handle(&self) -> u64 {
        self.inner.kernel_handle
    }

    #[getter]
    fn config(&self) -> PyLaunchConfig {
        PyLaunchConfig { inner: self.inner.config.clone() }
    }

    #[getter]
    fn args(&self) -> Vec<u64> {
        self.inner.args.clone()
    }

    #[getter]
    fn state(&self) -> PyKernelState {
        PyKernelState { inner: self.inner.state }
    }

    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id.clone()
    }

    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority
    }

    #[getter]
    fn queued_at(&self) -> f64 {
        self.inner.queued_at
    }

    #[getter]
    fn launched_at(&self) -> Option<f64> {
        self.inner.launched_at
    }

    #[getter]
    fn completed_at(&self) -> Option<f64> {
        self.inner.completed_at
    }

    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    fn duration(&self) -> Option<f64> {
        self.inner.duration()
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelLaunchRequest(id={}, state={:?}, kernel=0x{:x})",
            self.inner.id, self.inner.state, self.inner.kernel_handle
        )
    }
}

/// Python wrapper for DispatchStats
#[pyclass(name = "DispatchStats")]
#[derive(Clone)]
pub struct PyDispatchStats {
    inner: DispatchStats,
}

#[pymethods]
impl PyDispatchStats {
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
    fn pending_count(&self) -> usize {
        self.inner.pending_count
    }

    #[getter]
    fn in_flight_count(&self) -> usize {
        self.inner.in_flight_count
    }

    #[getter]
    fn avg_exec_time(&self) -> f64 {
        self.inner.avg_exec_time
    }

    #[getter]
    fn launches_per_stream(&self) -> HashMap<u32, usize> {
        self.inner.launches_per_stream.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DispatchStats(completed={}, pending={}, in_flight={}, avg_exec={:.4}s)",
            self.inner.completed_count,
            self.inner.pending_count,
            self.inner.in_flight_count,
            self.inner.avg_exec_time,
        )
    }
}

/// Kernel Dispatch Controller
///
/// Coordinates GPU kernel launches with stream management
/// and scheduler integration.
#[pyclass(name = "KernelDispatcher")]
pub struct PyKernelDispatcher {
    inner: KernelDispatcher,
}

#[pymethods]
impl PyKernelDispatcher {
    /// Create a new kernel dispatcher
    ///
    /// Args:
    ///     max_in_flight: Maximum concurrent kernels per stream (default: 4)
    #[new]
    #[pyo3(signature = (max_in_flight=4))]
    fn new(max_in_flight: usize) -> Self {
        Self {
            inner: KernelDispatcher::new(max_in_flight),
        }
    }

    /// Queue a kernel launch
    ///
    /// Args:
    ///     kernel_handle: CUfunction handle as int
    ///     config: LaunchConfig
    ///     args: Kernel arguments as list of int (pointers/values)
    ///     task_id: Optional scheduler task ID
    ///     priority: Priority (default: 0)
    ///
    /// Returns:
    ///     Request ID
    #[pyo3(signature = (kernel_handle, config, args=None, task_id=None, priority=0))]
    fn queue(
        &self,
        kernel_handle: u64,
        config: PyLaunchConfig,
        args: Option<Vec<u64>>,
        task_id: Option<String>,
        priority: i32,
    ) -> u64 {
        let mut request = KernelLaunchRequest::new(kernel_handle, config.inner)
            .with_priority(priority);

        if let Some(a) = args {
            request = request.with_args(a);
        }

        if let Some(tid) = task_id {
            request = request.with_task(tid);
        }

        self.inner.queue(request)
    }

    /// Queue a kernel for a scheduler task
    fn queue_for_task(
        &self,
        task_id: String,
        kernel_handle: u64,
        config: PyLaunchConfig,
        args: Vec<u64>,
    ) -> u64 {
        self.inner.queue_for_task(task_id, kernel_handle, config.inner, args)
    }

    /// Get launch requests ready to execute
    fn get_ready(&self, max_requests: usize) -> Vec<PyKernelLaunchRequest> {
        self.inner
            .get_ready(max_requests)
            .into_iter()
            .map(|r| PyKernelLaunchRequest { inner: r })
            .collect()
    }

    /// Mark a request as launched
    fn mark_launched(&self, req_id: u64) -> bool {
        self.inner.mark_launched(req_id)
    }

    /// Mark a request as completed
    fn mark_completed(&self, req_id: u64) -> bool {
        self.inner.mark_completed(req_id)
    }

    /// Mark a request as failed
    fn mark_failed(&self, req_id: u64, error: String) -> bool {
        self.inner.mark_failed(req_id, error)
    }

    /// Cancel a pending request
    fn cancel(&self, req_id: u64) -> bool {
        self.inner.cancel(req_id)
    }

    /// Get a request by ID
    fn get_request(&self, req_id: u64) -> Option<PyKernelLaunchRequest> {
        self.inner.get_request(req_id).map(|r| PyKernelLaunchRequest { inner: r })
    }

    /// Get in-flight request IDs for a stream
    fn get_in_flight(&self, stream_id: u32) -> Vec<u64> {
        self.inner.get_in_flight(stream_id)
    }

    /// Get requests linked to a scheduler task
    fn get_requests_for_task(&self, task_id: &str) -> Vec<PyKernelLaunchRequest> {
        self.inner
            .get_requests_for_task(task_id)
            .into_iter()
            .map(|r| PyKernelLaunchRequest { inner: r })
            .collect()
    }

    /// Check if there's pending work
    fn has_pending_work(&self) -> bool {
        self.inner.has_pending_work()
    }

    /// Get dispatch statistics
    fn stats(&self) -> PyDispatchStats {
        PyDispatchStats { inner: self.inner.stats() }
    }

    /// Garbage collect completed requests
    fn gc(&self) {
        self.inner.gc()
    }

    /// Clear all state
    fn clear(&self) {
        self.inner.clear()
    }
}

// =============================================================================
// Kernel Pacing Types
// =============================================================================

/// Pacing configuration for Python
#[pyclass(name = "PacingConfig")]
#[derive(Clone)]
pub struct PyPacingConfig {
    inner: PacingConfig,
}

#[pymethods]
impl PyPacingConfig {
    #[new]
    #[pyo3(signature = (total_bandwidth=1.0, window_ms=100.0, min_interval_ms=0.1, adaptive=true))]
    fn new(total_bandwidth: f64, window_ms: f64, min_interval_ms: f64, adaptive: bool) -> Self {
        Self {
            inner: PacingConfig {
                total_bandwidth,
                window_ms,
                min_interval_ms,
                adaptive,
            },
        }
    }

    #[getter]
    fn total_bandwidth(&self) -> f64 {
        self.inner.total_bandwidth
    }

    #[getter]
    fn window_ms(&self) -> f64 {
        self.inner.window_ms
    }

    #[getter]
    fn min_interval_ms(&self) -> f64 {
        self.inner.min_interval_ms
    }

    #[getter]
    fn adaptive(&self) -> bool {
        self.inner.adaptive
    }

    fn __repr__(&self) -> String {
        format!(
            "PacingConfig(bandwidth={:.2}, window={}ms, min_interval={}ms)",
            self.inner.total_bandwidth, self.inner.window_ms, self.inner.min_interval_ms
        )
    }
}

/// Pacing decision for Python
#[pyclass(name = "PacingDecision")]
#[derive(Clone)]
pub struct PyPacingDecision {
    inner: PacingDecision,
}

#[pymethods]
impl PyPacingDecision {
    /// Check if immediate launch is allowed
    fn can_launch(&self) -> bool {
        self.inner.can_launch()
    }

    /// Check if throttled
    fn is_throttled(&self) -> bool {
        self.inner.is_throttled()
    }

    /// Get wait time in milliseconds
    fn wait_ms(&self) -> f64 {
        self.inner.wait_ms()
    }

    #[getter]
    fn decision_type(&self) -> String {
        match &self.inner {
            PacingDecision::Launch => "Launch".into(),
            PacingDecision::Wait { .. } => "Wait".into(),
            PacingDecision::Throttle { .. } => "Throttle".into(),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PacingDecision::Launch => "PacingDecision(Launch)".into(),
            PacingDecision::Wait { delay_ms } => format!("PacingDecision(Wait, delay={:.2}ms)", delay_ms),
            PacingDecision::Throttle { reason } => format!("PacingDecision(Throttle, reason='{}')", reason),
        }
    }
}

/// Stream pacing statistics for Python
#[pyclass(name = "StreamPacingStats")]
#[derive(Clone)]
pub struct PyStreamPacingStats {
    inner: StreamPacingStats,
}

#[pymethods]
impl PyStreamPacingStats {
    #[getter]
    fn stream_id(&self) -> u64 {
        self.inner.stream_id
    }

    #[getter]
    fn bandwidth(&self) -> f64 {
        self.inner.bandwidth
    }

    #[getter]
    fn launches_in_window(&self) -> usize {
        self.inner.launches_in_window
    }

    #[getter]
    fn total_launches(&self) -> usize {
        self.inner.total_launches
    }

    #[getter]
    fn throttled_count(&self) -> usize {
        self.inner.throttled_count
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamPacingStats(stream={}, bandwidth={:.2}, launches={})",
            self.inner.stream_id, self.inner.bandwidth, self.inner.total_launches
        )
    }
}

/// Global pacing statistics for Python
#[pyclass(name = "PacingStats")]
#[derive(Clone)]
pub struct PyPacingStats {
    inner: PacingStats,
}

#[pymethods]
impl PyPacingStats {
    #[getter]
    fn stream_count(&self) -> usize {
        self.inner.stream_count
    }

    #[getter]
    fn used_bandwidth(&self) -> f64 {
        self.inner.used_bandwidth
    }

    #[getter]
    fn available_bandwidth(&self) -> f64 {
        self.inner.available_bandwidth
    }

    #[getter]
    fn total_launches(&self) -> usize {
        self.inner.total_launches
    }

    #[getter]
    fn total_throttled(&self) -> usize {
        self.inner.total_throttled
    }

    #[getter]
    fn total_waited(&self) -> usize {
        self.inner.total_waited
    }

    fn __repr__(&self) -> String {
        format!(
            "PacingStats(streams={}, launches={}, throttled={})",
            self.inner.stream_count, self.inner.total_launches, self.inner.total_throttled
        )
    }
}

/// Kernel pacing engine for Python
#[pyclass(name = "KernelPacingEngine")]
pub struct PyKernelPacingEngine {
    inner: KernelPacingEngine,
}

#[pymethods]
impl PyKernelPacingEngine {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPacingConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: KernelPacingEngine::new(cfg),
        }
    }

    /// Allocate bandwidth for a stream
    fn allocate_stream(&mut self, stream_id: u64, bandwidth: f64) -> bool {
        self.inner.allocate_stream(stream_id, bandwidth)
    }

    /// Release bandwidth for a stream
    fn release_stream(&mut self, stream_id: u64) {
        self.inner.release_stream(stream_id);
    }

    /// Check if a kernel launch should proceed
    fn should_launch(&self, stream_id: u64) -> PyPacingDecision {
        PyPacingDecision {
            inner: self.inner.should_launch(stream_id),
        }
    }

    /// Record a kernel launch
    fn record_launch(&mut self, stream_id: u64) {
        self.inner.record_launch(stream_id);
    }

    /// Record a throttled request
    fn record_throttle(&mut self, stream_id: u64) {
        self.inner.record_throttle(stream_id);
    }

    /// Record a waited request
    fn record_wait(&mut self) {
        self.inner.record_wait();
    }

    /// Get stream statistics
    fn stream_stats(&self, stream_id: u64) -> Option<PyStreamPacingStats> {
        self.inner.stream_stats(stream_id).map(|s| PyStreamPacingStats { inner: s })
    }

    /// Get global statistics
    fn stats(&self) -> PyPacingStats {
        PyPacingStats {
            inner: self.inner.stats(),
        }
    }

    /// Reset all pacing state
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "KernelPacingEngine(streams={}, available_bw={:.2})",
            stats.stream_count, stats.available_bandwidth
        )
    }
}

// =============================================================================
// Micro-Slicing Types
// =============================================================================

/// Slice configuration for Python
#[pyclass(name = "SliceConfig")]
#[derive(Clone)]
pub struct PySliceConfig {
    inner: SliceConfig,
}

#[pymethods]
impl PySliceConfig {
    #[new]
    #[pyo3(signature = (max_items_per_slice=65536, max_duration_ms=1.0, min_slices=1, max_slices=256, adaptive=true))]
    fn new(
        max_items_per_slice: usize,
        max_duration_ms: f64,
        min_slices: usize,
        max_slices: usize,
        adaptive: bool,
    ) -> Self {
        Self {
            inner: SliceConfig {
                max_items_per_slice,
                max_duration_ms,
                min_slices,
                max_slices,
                adaptive,
            },
        }
    }

    #[getter]
    fn max_items_per_slice(&self) -> usize {
        self.inner.max_items_per_slice
    }

    #[getter]
    fn max_duration_ms(&self) -> f64 {
        self.inner.max_duration_ms
    }

    #[getter]
    fn min_slices(&self) -> usize {
        self.inner.min_slices
    }

    #[getter]
    fn max_slices(&self) -> usize {
        self.inner.max_slices
    }

    #[getter]
    fn adaptive(&self) -> bool {
        self.inner.adaptive
    }

    fn __repr__(&self) -> String {
        format!(
            "SliceConfig(max_items={}, max_duration={}ms, slices=[{}, {}])",
            self.inner.max_items_per_slice,
            self.inner.max_duration_ms,
            self.inner.min_slices,
            self.inner.max_slices
        )
    }
}

/// Single kernel slice for Python
#[pyclass(name = "KernelSlice")]
#[derive(Clone)]
pub struct PyKernelSlice {
    inner: KernelSlice,
}

#[pymethods]
impl PyKernelSlice {
    #[getter]
    fn id(&self) -> usize {
        self.inner.id
    }

    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    #[getter]
    fn grid(&self) -> (u32, u32, u32) {
        self.inner.grid
    }

    #[getter]
    fn executed(&self) -> bool {
        self.inner.executed
    }

    #[getter]
    fn exec_time_ms(&self) -> Option<f64> {
        self.inner.exec_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelSlice(id={}, offset={}, count={}, executed={})",
            self.inner.id, self.inner.offset, self.inner.count, self.inner.executed
        )
    }
}

/// Information about a slice to execute for Python
#[pyclass(name = "SliceInfo")]
#[derive(Clone)]
pub struct PySliceInfo {
    inner: SliceInfo,
}

#[pymethods]
impl PySliceInfo {
    #[getter]
    fn kernel_handle(&self) -> u64 {
        self.inner.kernel_handle
    }

    #[getter]
    fn block(&self) -> (u32, u32, u32) {
        self.inner.block
    }

    #[getter]
    fn shared_mem(&self) -> u32 {
        self.inner.shared_mem
    }

    #[getter]
    fn slice_id(&self) -> usize {
        self.inner.slice_id
    }

    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    #[getter]
    fn grid(&self) -> (u32, u32, u32) {
        self.inner.grid
    }

    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id.clone()
    }

    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority
    }

    fn __repr__(&self) -> String {
        format!(
            "SliceInfo(kernel=0x{:x}, slice_id={}, offset={}, count={})",
            self.inner.kernel_handle, self.inner.slice_id, self.inner.offset, self.inner.count
        )
    }
}

/// Slice scheduler statistics for Python
#[pyclass(name = "SliceStats")]
#[derive(Clone)]
pub struct PySliceStats {
    inner: SliceStats,
}

#[pymethods]
impl PySliceStats {
    #[getter]
    fn total_slices(&self) -> usize {
        self.inner.total_slices
    }

    #[getter]
    fn completed_slices(&self) -> usize {
        self.inner.completed_slices
    }

    #[getter]
    fn pending_slices(&self) -> usize {
        self.inner.pending_slices
    }

    #[getter]
    fn total_kernels(&self) -> usize {
        self.inner.total_kernels
    }

    #[getter]
    fn completed_kernels(&self) -> usize {
        self.inner.completed_kernels
    }

    #[getter]
    fn pending_kernels(&self) -> usize {
        self.inner.pending_kernels
    }

    fn __repr__(&self) -> String {
        format!(
            "SliceStats(slices={}/{}, kernels={}/{})",
            self.inner.completed_slices, self.inner.total_slices,
            self.inner.completed_kernels, self.inner.total_kernels
        )
    }
}

/// Slice scheduler for Python
///
/// Splits kernels into smaller slices for fair scheduling
/// and better latency under QoS constraints.
#[pyclass(name = "SliceScheduler")]
pub struct PySliceScheduler {
    inner: SliceScheduler,
}

#[pymethods]
impl PySliceScheduler {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PySliceConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: SliceScheduler::new(cfg),
        }
    }

    /// Submit a kernel for slicing
    ///
    /// Args:
    ///     kernel_handle: CUfunction handle as int
    ///     total_items: Total work items to process
    ///     block: Block dimensions (x, y, z)
    ///     shared_mem: Shared memory per block
    ///
    /// Returns:
    ///     Number of slices created
    fn submit(
        &mut self,
        kernel_handle: u64,
        total_items: usize,
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> usize {
        self.inner.submit(kernel_handle, total_items, block, shared_mem)
    }

    /// Submit a kernel for a specific task
    ///
    /// Args:
    ///     task_id: Associated task ID
    ///     kernel_handle: CUfunction handle as int
    ///     total_items: Total work items to process
    ///     block: Block dimensions (x, y, z)
    ///     shared_mem: Shared memory per block
    ///     priority: Priority (higher = more important)
    ///
    /// Returns:
    ///     Number of slices created
    fn submit_for_task(
        &mut self,
        task_id: String,
        kernel_handle: u64,
        total_items: usize,
        block: (u32, u32, u32),
        shared_mem: u32,
        priority: i32,
    ) -> usize {
        self.inner.submit_for_task(task_id, kernel_handle, total_items, block, shared_mem, priority)
    }

    /// Get next slice to execute (round-robin fair scheduling)
    fn get_next_slice(&mut self) -> Option<PySliceInfo> {
        self.inner.get_next_slice().map(|s| PySliceInfo { inner: s })
    }

    /// Complete the current slice
    fn complete_slice(&mut self, exec_time_ms: f64) {
        self.inner.complete_slice(exec_time_ms);
    }

    /// Get number of pending slices
    fn pending_slices(&self) -> usize {
        self.inner.pending_slices()
    }

    /// Get number of pending kernels
    fn pending_kernels(&self) -> usize {
        self.inner.pending_kernels()
    }

    /// Get statistics
    fn stats(&self) -> PySliceStats {
        PySliceStats {
            inner: self.inner.stats(),
        }
    }

    /// Clear all state
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get configuration
    fn config(&self) -> PySliceConfig {
        PySliceConfig {
            inner: self.inner.config().clone(),
        }
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "SliceScheduler(pending_slices={}, pending_kernels={})",
            stats.pending_slices, stats.pending_kernels
        )
    }
}

// =============================================================================
// Kernel Cache Types
// =============================================================================

/// Compile options for Python
#[pyclass(name = "CompileOptions")]
#[derive(Clone)]
pub struct PyCompileOptions {
    inner: CompileOptions,
}

#[pymethods]
impl PyCompileOptions {
    #[new]
    #[pyo3(signature = (compute_capability="sm_75"))]
    fn new(compute_capability: &str) -> Self {
        Self {
            inner: CompileOptions::with_compute(compute_capability),
        }
    }

    /// Add a compiler flag
    fn flag(&self, flag: &str) -> Self {
        Self {
            inner: self.inner.clone().flag(flag),
        }
    }

    /// Add a define macro
    fn define(&self, name: &str, value: &str) -> Self {
        Self {
            inner: self.inner.clone().define(name, value),
        }
    }

    /// Add an include path
    fn include(&self, path: &str) -> Self {
        Self {
            inner: self.inner.clone().include(path),
        }
    }

    #[getter]
    fn compute_capability(&self) -> &str {
        &self.inner.compute_capability
    }

    #[getter]
    fn flags(&self) -> Vec<String> {
        self.inner.flags.clone()
    }

    #[getter]
    fn defines(&self) -> Vec<(String, String)> {
        self.inner.defines.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompileOptions(compute='{}', flags={:?})",
            self.inner.compute_capability, self.inner.flags
        )
    }
}

/// Cache configuration for Python
#[pyclass(name = "CacheConfig")]
#[derive(Clone)]
pub struct PyCacheConfig {
    inner: CacheConfig,
}

#[pymethods]
impl PyCacheConfig {
    #[new]
    #[pyo3(signature = (max_entries=1024, max_ptx_size=268435456, enable_eviction=true, ttl_seconds=0.0))]
    fn new(max_entries: usize, max_ptx_size: usize, enable_eviction: bool, ttl_seconds: f64) -> Self {
        Self {
            inner: CacheConfig {
                max_entries,
                max_ptx_size,
                enable_eviction,
                ttl_seconds,
            },
        }
    }

    #[getter]
    fn max_entries(&self) -> usize {
        self.inner.max_entries
    }

    #[getter]
    fn max_ptx_size(&self) -> usize {
        self.inner.max_ptx_size
    }

    #[getter]
    fn enable_eviction(&self) -> bool {
        self.inner.enable_eviction
    }

    #[getter]
    fn ttl_seconds(&self) -> f64 {
        self.inner.ttl_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheConfig(max_entries={}, max_ptx_size={}, eviction={})",
            self.inner.max_entries, self.inner.max_ptx_size, self.inner.enable_eviction
        )
    }
}

/// Cached kernel entry for Python
#[pyclass(name = "CachedKernel")]
#[derive(Clone)]
pub struct PyCachedKernel {
    inner: CachedKernel,
}

#[pymethods]
impl PyCachedKernel {
    #[getter]
    fn key(&self) -> u64 {
        self.inner.key
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn ptx(&self) -> &str {
        &self.inner.ptx
    }

    #[getter]
    fn module_handle(&self) -> Option<u64> {
        self.inner.module_handle
    }

    #[getter]
    fn function_handle(&self) -> Option<u64> {
        self.inner.function_handle
    }

    #[getter]
    fn created_at(&self) -> f64 {
        self.inner.created_at
    }

    #[getter]
    fn last_access(&self) -> f64 {
        self.inner.last_access
    }

    #[getter]
    fn access_count(&self) -> usize {
        self.inner.access_count
    }

    /// Check if kernel is loaded (has function handle)
    fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    fn __repr__(&self) -> String {
        format!(
            "CachedKernel(name='{}', loaded={}, accesses={})",
            self.inner.name, self.inner.is_loaded(), self.inner.access_count
        )
    }
}

/// Cache statistics for Python
#[pyclass(name = "CacheStats")]
#[derive(Clone)]
pub struct PyCacheStats {
    inner: CacheStats,
}

#[pymethods]
impl PyCacheStats {
    #[getter]
    fn hits(&self) -> usize {
        self.inner.hits
    }

    #[getter]
    fn misses(&self) -> usize {
        self.inner.misses
    }

    #[getter]
    fn entries(&self) -> usize {
        self.inner.entries
    }

    #[getter]
    fn ptx_size(&self) -> usize {
        self.inner.ptx_size
    }

    #[getter]
    fn evictions(&self) -> usize {
        self.inner.evictions
    }

    #[getter]
    fn ttl_evictions(&self) -> usize {
        self.inner.ttl_evictions
    }

    #[getter]
    fn loaded_count(&self) -> usize {
        self.inner.loaded_count
    }

    /// Calculate hit rate (0.0 - 1.0)
    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate()
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheStats(entries={}, hit_rate={:.1}%, ptx_size={})",
            self.inner.entries,
            self.inner.hit_rate() * 100.0,
            self.inner.ptx_size
        )
    }
}

/// Kernel cache for Python
///
/// Caches compiled CUDA kernels (PTX) to avoid repeated
/// NVRTC compilation.
#[pyclass(name = "KernelCache")]
pub struct PyKernelCache {
    inner: KernelCache,
}

#[pymethods]
impl PyKernelCache {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyCacheConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: KernelCache::new(cfg),
        }
    }

    /// Compute cache key from source and options
    #[staticmethod]
    fn compute_key(source: &str, name: &str, options: &PyCompileOptions) -> u64 {
        KernelCache::compute_key(source, name, &options.inner)
    }

    /// Compute source hash
    #[staticmethod]
    fn hash_source(source: &str) -> u64 {
        KernelCache::hash_source(source)
    }

    /// Get cached kernel by key
    fn get(&mut self, key: u64) -> Option<PyCachedKernel> {
        self.inner.get(key).map(|k| PyCachedKernel { inner: k.clone() })
    }

    /// Get cached kernel by name and options
    fn get_by_name(&mut self, name: &str, options: &PyCompileOptions) -> Option<PyCachedKernel> {
        self.inner.get_by_name(name, &options.inner).map(|k| PyCachedKernel { inner: k.clone() })
    }

    /// Insert a compiled kernel
    fn insert(&mut self, source: &str, name: &str, ptx: &str, options: PyCompileOptions) -> u64 {
        self.inner.insert(source, name, ptx.into(), options.inner)
    }

    /// Set module and function handles for a cached kernel
    fn set_handles(&mut self, key: u64, module: u64, function: u64) -> bool {
        self.inner.set_handles(key, module, function)
    }

    /// Remove a kernel from cache
    fn remove(&mut self, key: u64) -> Option<PyCachedKernel> {
        self.inner.remove(key).map(|k| PyCachedKernel { inner: k })
    }

    /// Check if kernel is cached
    fn contains(&self, key: u64) -> bool {
        self.inner.contains(key)
    }

    /// Get all cached kernel names
    fn kernel_names(&self) -> Vec<String> {
        self.inner.kernel_names().into_iter().map(|s| s.to_string()).collect()
    }

    /// Clear expired entries (TTL)
    fn clear_expired(&mut self) -> usize {
        self.inner.clear_expired()
    }

    /// Get statistics
    fn stats(&self) -> PyCacheStats {
        PyCacheStats {
            inner: self.inner.stats(),
        }
    }

    /// Get number of entries
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all cache
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "KernelCache(entries={}, hit_rate={:.1}%)",
            stats.entries, stats.hit_rate() * 100.0
        )
    }
}

// =============================================================================
// Persistent Kernel Cache Types
// =============================================================================

/// Architecture fingerprint for cache keys
///
/// Contains GPU characteristics that affect PTX validity.
#[pyclass(name = "ArchFingerprint")]
#[derive(Clone)]
pub struct PyArchFingerprint {
    inner: ArchFingerprint,
}

#[pymethods]
impl PyArchFingerprint {
    /// Create a new architecture fingerprint
    ///
    /// Args:
    ///     sm_version: SM version (e.g., 86 for RTX 3090)
    ///     global_memory: Global memory in bytes
    ///     shared_memory_per_sm: Shared memory per SM in bytes
    ///     max_registers_per_block: Max registers per block
    ///     l2_cache_size: L2 cache size in bytes
    ///     driver_version: CUDA driver version
    #[new]
    fn new(
        sm_version: u32,
        global_memory: u64,
        shared_memory_per_sm: u32,
        max_registers_per_block: u32,
        l2_cache_size: u32,
        driver_version: u32,
    ) -> Self {
        Self {
            inner: ArchFingerprint::new(
                sm_version,
                global_memory,
                shared_memory_per_sm,
                max_registers_per_block,
                l2_cache_size,
                driver_version,
            ),
        }
    }

    #[getter]
    fn sm_version(&self) -> u32 {
        self.inner.sm_version
    }

    #[getter]
    fn global_memory(&self) -> u64 {
        self.inner.global_memory
    }

    #[getter]
    fn shared_memory_per_sm(&self) -> u32 {
        self.inner.shared_memory_per_sm
    }

    #[getter]
    fn max_registers_per_block(&self) -> u32 {
        self.inner.max_registers_per_block
    }

    #[getter]
    fn l2_cache_size(&self) -> u32 {
        self.inner.l2_cache_size
    }

    #[getter]
    fn driver_version(&self) -> u32 {
        self.inner.driver_version
    }

    /// Compute hash for this fingerprint
    fn hash(&self) -> u64 {
        self.inner.hash()
    }

    /// Check if this fingerprint is compatible with another
    fn is_compatible(&self, other: &PyArchFingerprint) -> bool {
        self.inner.is_compatible(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ArchFingerprint(sm={}, memory={}GB, driver={})",
            self.inner.sm_version,
            self.inner.global_memory / (1024 * 1024 * 1024),
            self.inner.driver_version
        )
    }
}

/// Persistent cache configuration
#[pyclass(name = "PersistentCacheConfig")]
#[derive(Clone)]
pub struct PyPersistentCacheConfig {
    inner: PersistentCacheConfig,
}

#[pymethods]
impl PyPersistentCacheConfig {
    /// Create a new persistent cache configuration
    ///
    /// Args:
    ///     cache_dir: Directory path for cache storage (default: ~/.pygpukit/cache)
    ///     max_size: Maximum total cache size in bytes (default: 512MB)
    ///     max_entries: Maximum number of cached entries (default: 1000)
    ///     auto_cleanup: Enable automatic cleanup (default: True)
    ///     ttl_seconds: Time-to-live in seconds, 0.0 for unlimited (default: 0.0)
    #[new]
    #[pyo3(signature = (cache_dir=None, max_size=536870912, max_entries=1000, auto_cleanup=true, ttl_seconds=0.0))]
    fn new(
        cache_dir: Option<String>,
        max_size: usize,
        max_entries: usize,
        auto_cleanup: bool,
        ttl_seconds: f64,
    ) -> Self {
        let cache_dir = cache_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".pygpukit")
                    .join("cache")
            });

        Self {
            inner: PersistentCacheConfig {
                cache_dir,
                max_size,
                max_entries,
                auto_cleanup,
                ttl_seconds,
            },
        }
    }

    #[getter]
    fn cache_dir(&self) -> String {
        self.inner.cache_dir.to_string_lossy().to_string()
    }

    #[getter]
    fn max_size(&self) -> usize {
        self.inner.max_size
    }

    #[getter]
    fn max_entries(&self) -> usize {
        self.inner.max_entries
    }

    #[getter]
    fn auto_cleanup(&self) -> bool {
        self.inner.auto_cleanup
    }

    #[getter]
    fn ttl_seconds(&self) -> f64 {
        self.inner.ttl_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "PersistentCacheConfig(dir='{}', max_size={}MB, max_entries={})",
            self.inner.cache_dir.display(),
            self.inner.max_size / (1024 * 1024),
            self.inner.max_entries
        )
    }
}

/// Persistent cache statistics
#[pyclass(name = "PersistentCacheStats")]
#[derive(Clone)]
pub struct PyPersistentCacheStats {
    inner: PersistentCacheStats,
}

#[pymethods]
impl PyPersistentCacheStats {
    #[getter]
    fn entries(&self) -> usize {
        self.inner.entries
    }

    #[getter]
    fn total_size(&self) -> usize {
        self.inner.total_size
    }

    #[getter]
    fn hits(&self) -> usize {
        self.inner.hits
    }

    #[getter]
    fn misses(&self) -> usize {
        self.inner.misses
    }

    #[getter]
    fn evictions(&self) -> usize {
        self.inner.evictions
    }

    #[getter]
    fn load_errors(&self) -> usize {
        self.inner.load_errors
    }

    #[getter]
    fn save_errors(&self) -> usize {
        self.inner.save_errors
    }

    /// Calculate hit rate (0.0 - 1.0)
    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate()
    }

    fn __repr__(&self) -> String {
        format!(
            "PersistentCacheStats(entries={}, size={}KB, hit_rate={:.1}%)",
            self.inner.entries,
            self.inner.total_size / 1024,
            self.inner.hit_rate() * 100.0
        )
    }
}

/// Persistent cache entry
#[pyclass(name = "PersistentEntry")]
#[derive(Clone)]
pub struct PyPersistentEntry {
    inner: PersistentEntry,
}

#[pymethods]
impl PyPersistentEntry {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn ptx(&self) -> &str {
        &self.inner.ptx
    }

    #[getter]
    fn source_hash(&self) -> u64 {
        self.inner.source_hash
    }

    #[getter]
    fn options_hash(&self) -> u64 {
        self.inner.options_hash
    }

    #[getter]
    fn created_at(&self) -> f64 {
        self.inner.created_at
    }

    #[getter]
    fn last_access(&self) -> f64 {
        self.inner.last_access
    }

    #[getter]
    fn access_count(&self) -> usize {
        self.inner.access_count
    }

    /// Get PTX size in bytes
    fn ptx_size(&self) -> usize {
        self.inner.ptx_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "PersistentEntry(name='{}', ptx_size={}KB, accesses={})",
            self.inner.name,
            self.inner.ptx_size() / 1024,
            self.inner.access_count
        )
    }
}

/// Persistent kernel cache
///
/// Caches compiled PTX to disk for fast startup across sessions.
/// Uses GPU architecture fingerprinting to ensure cache validity.
#[pyclass(name = "PersistentCache")]
pub struct PyPersistentCache {
    inner: PersistentCache,
}

#[pymethods]
impl PyPersistentCache {
    /// Create a new persistent cache
    ///
    /// Args:
    ///     config: Cache configuration
    ///     arch: Architecture fingerprint for the current GPU
    #[new]
    fn new(config: PyPersistentCacheConfig, arch: PyArchFingerprint) -> Self {
        Self {
            inner: PersistentCache::new(config.inner, arch.inner),
        }
    }

    /// Initialize the cache (creates directories, loads index)
    fn initialize(&mut self) -> PyResult<()> {
        self.inner.initialize()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Compute cache key from components
    #[staticmethod]
    fn compute_key(source_hash: u64, name: &str, options_hash: u64, arch_hash: u64) -> u64 {
        PersistentCache::compute_key(source_hash, name, options_hash, arch_hash)
    }

    /// Compute hash for source code
    #[staticmethod]
    fn hash_source(source: &str) -> u64 {
        PersistentCache::hash_source(source)
    }

    /// Compute hash for compile options
    #[staticmethod]
    fn hash_options(options: Vec<String>) -> u64 {
        PersistentCache::hash_options(&options)
    }

    /// Get entry by key
    fn get(&mut self, key: u64) -> PyResult<Option<PyPersistentEntry>> {
        self.inner.get(key)
            .map(|opt| opt.map(|e| PyPersistentEntry { inner: e }))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Insert a new entry
    ///
    /// Args:
    ///     source: CUDA source code
    ///     name: Kernel name
    ///     options: Compile options
    ///     ptx: Compiled PTX code
    ///
    /// Returns:
    ///     Cache key for the entry
    fn insert(
        &mut self,
        source: &str,
        name: &str,
        options: Vec<String>,
        ptx: String,
    ) -> PyResult<u64> {
        self.inner.insert(source, name, &options, ptx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove an entry by key
    fn remove(&mut self, key: u64) -> PyResult<bool> {
        self.inner.remove(key)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Clear all entries
    fn clear(&mut self) -> PyResult<()> {
        self.inner.clear()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Run cleanup (remove expired entries)
    fn cleanup(&mut self) -> PyResult<usize> {
        self.inner.cleanup()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if key exists
    fn contains(&self, key: u64) -> bool {
        self.inner.contains(key)
    }

    /// Get number of entries
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get statistics
    fn stats(&self) -> PyPersistentCacheStats {
        PyPersistentCacheStats {
            inner: self.inner.stats().clone(),
        }
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "PersistentCache(entries={}, size={}KB, hit_rate={:.1}%)",
            stats.entries,
            stats.total_size / 1024,
            stats.hit_rate() * 100.0
        )
    }
}

/// Register dispatch module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernelState>()?;
    m.add_class::<PyLaunchConfig>()?;
    m.add_class::<PyKernelLaunchRequest>()?;
    m.add_class::<PyDispatchStats>()?;
    m.add_class::<PyKernelDispatcher>()?;
    // Pacing
    m.add_class::<PyPacingConfig>()?;
    m.add_class::<PyPacingDecision>()?;
    m.add_class::<PyStreamPacingStats>()?;
    m.add_class::<PyPacingStats>()?;
    m.add_class::<PyKernelPacingEngine>()?;
    // Slicing
    m.add_class::<PySliceConfig>()?;
    m.add_class::<PyKernelSlice>()?;
    m.add_class::<PySliceInfo>()?;
    m.add_class::<PySliceStats>()?;
    m.add_class::<PySliceScheduler>()?;
    // Cache (in-memory)
    m.add_class::<PyCompileOptions>()?;
    m.add_class::<PyCacheConfig>()?;
    m.add_class::<PyCachedKernel>()?;
    m.add_class::<PyCacheStats>()?;
    m.add_class::<PyKernelCache>()?;
    // Persistent cache (disk)
    m.add_class::<PyArchFingerprint>()?;
    m.add_class::<PyPersistentCacheConfig>()?;
    m.add_class::<PyPersistentCacheStats>()?;
    m.add_class::<PyPersistentEntry>()?;
    m.add_class::<PyPersistentCache>()?;
    Ok(())
}
