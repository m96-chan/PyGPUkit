//! Python bindings for the kernel dispatch controller

use pyo3::prelude::*;
use std::collections::HashMap;
use pygpukit_core::dispatch::{
    KernelDispatcher, KernelLaunchRequest, KernelState, DispatchStats, LaunchConfig,
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

/// Register dispatch module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernelState>()?;
    m.add_class::<PyLaunchConfig>()?;
    m.add_class::<PyKernelLaunchRequest>()?;
    m.add_class::<PyDispatchStats>()?;
    m.add_class::<PyKernelDispatcher>()?;
    Ok(())
}
