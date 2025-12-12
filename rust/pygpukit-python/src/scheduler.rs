//! Scheduler module Python bindings

use pyo3::prelude::*;
use std::sync::Arc;
use pygpukit_core::scheduler::{
    Scheduler, SchedulerStats, TaskMeta, TaskState, TaskPolicy, TaskStats,
};

/// Task state enum for Python
#[pyclass(name = "TaskState", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyTaskState {
    Pending = 0,
    Running = 1,
    Completed = 2,
    Failed = 3,
    Cancelled = 4,
}

impl From<TaskState> for PyTaskState {
    fn from(state: TaskState) -> Self {
        match state {
            TaskState::Pending => PyTaskState::Pending,
            TaskState::Running => PyTaskState::Running,
            TaskState::Completed => PyTaskState::Completed,
            TaskState::Failed => PyTaskState::Failed,
            TaskState::Cancelled => PyTaskState::Cancelled,
        }
    }
}

impl From<PyTaskState> for TaskState {
    fn from(state: PyTaskState) -> Self {
        match state {
            PyTaskState::Pending => TaskState::Pending,
            PyTaskState::Running => TaskState::Running,
            PyTaskState::Completed => TaskState::Completed,
            PyTaskState::Failed => TaskState::Failed,
            PyTaskState::Cancelled => TaskState::Cancelled,
        }
    }
}

/// Task policy enum for Python
#[pyclass(name = "TaskPolicy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyTaskPolicy {
    Fifo = 0,
    Sjf = 1,
    Priority = 2,
}

impl From<TaskPolicy> for PyTaskPolicy {
    fn from(policy: TaskPolicy) -> Self {
        match policy {
            TaskPolicy::Fifo => PyTaskPolicy::Fifo,
            TaskPolicy::Sjf => PyTaskPolicy::Sjf,
            TaskPolicy::Priority => PyTaskPolicy::Priority,
        }
    }
}

impl From<PyTaskPolicy> for TaskPolicy {
    fn from(policy: PyTaskPolicy) -> Self {
        match policy {
            PyTaskPolicy::Fifo => TaskPolicy::Fifo,
            PyTaskPolicy::Sjf => TaskPolicy::Sjf,
            PyTaskPolicy::Priority => TaskPolicy::Priority,
        }
    }
}

/// Python wrapper for TaskMeta
#[pyclass(name = "TaskMeta")]
#[derive(Clone)]
pub struct PyTaskMeta {
    inner: TaskMeta,
}

#[pymethods]
impl PyTaskMeta {
    /// Create a new task.
    #[new]
    #[pyo3(signature = (id, name, memory_estimate=0, priority=0, dependencies=None))]
    fn new(
        id: String,
        name: String,
        memory_estimate: usize,
        priority: i32,
        dependencies: Option<Vec<String>>,
    ) -> Self {
        let mut task = TaskMeta::with_memory(id, name, memory_estimate)
            .with_priority(priority);
        if let Some(deps) = dependencies {
            task = task.with_dependencies(deps);
        }
        Self { inner: task }
    }

    /// Task ID
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Task name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Task state
    #[getter]
    fn state(&self) -> PyTaskState {
        self.inner.state.into()
    }

    /// Task policy
    #[getter]
    fn policy(&self) -> PyTaskPolicy {
        self.inner.policy.into()
    }

    /// Task priority
    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority
    }

    /// Memory estimate
    #[getter]
    fn memory_estimate(&self) -> usize {
        self.inner.memory_estimate
    }

    /// Submission timestamp
    #[getter]
    fn submitted_at(&self) -> f64 {
        self.inner.submitted_at
    }

    /// Start timestamp
    #[getter]
    fn started_at(&self) -> Option<f64> {
        self.inner.started_at
    }

    /// Completion timestamp
    #[getter]
    fn completed_at(&self) -> Option<f64> {
        self.inner.completed_at
    }

    /// Error message
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    /// Dependencies
    #[getter]
    fn dependencies(&self) -> Vec<String> {
        self.inner.dependencies.clone()
    }

    /// Check if task is in terminal state
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Get elapsed time since submission
    fn elapsed(&self) -> f64 {
        self.inner.elapsed()
    }

    /// Get execution duration
    fn duration(&self) -> Option<f64> {
        self.inner.duration()
    }

    fn __repr__(&self) -> String {
        format!(
            "TaskMeta(id='{}', name='{}', state={:?}, memory={})",
            self.inner.id, self.inner.name, self.inner.state, self.inner.memory_estimate
        )
    }
}

/// Python wrapper for SchedulerStats
#[pyclass(name = "SchedulerStats")]
#[derive(Clone)]
pub struct PySchedulerStats {
    inner: SchedulerStats,
}

#[pymethods]
impl PySchedulerStats {
    /// Total tasks submitted
    #[getter]
    fn total_submitted(&self) -> usize {
        self.inner.total_submitted
    }

    /// Pending tasks
    #[getter]
    fn pending_count(&self) -> usize {
        self.inner.pending_count
    }

    /// Running tasks
    #[getter]
    fn running_count(&self) -> usize {
        self.inner.running_count
    }

    /// Completed tasks
    #[getter]
    fn completed_count(&self) -> usize {
        self.inner.completed_count
    }

    /// Failed tasks
    #[getter]
    fn failed_count(&self) -> usize {
        self.inner.failed_count
    }

    /// Cancelled tasks
    #[getter]
    fn cancelled_count(&self) -> usize {
        self.inner.cancelled_count
    }

    /// Reserved memory
    #[getter]
    fn reserved_memory(&self) -> usize {
        self.inner.reserved_memory
    }

    /// Available memory
    #[getter]
    fn available_memory(&self) -> usize {
        self.inner.available_memory
    }

    /// Average wait time
    #[getter]
    fn avg_wait_time(&self) -> f64 {
        self.inner.avg_wait_time
    }

    /// Average execution time
    #[getter]
    fn avg_exec_time(&self) -> f64 {
        self.inner.avg_exec_time
    }

    fn __repr__(&self) -> String {
        format!(
            "SchedulerStats(pending={}, running={}, completed={}, failed={})",
            self.inner.pending_count, self.inner.running_count,
            self.inner.completed_count, self.inner.failed_count
        )
    }
}

/// Python wrapper for TaskStats
#[pyclass(name = "TaskStats")]
#[derive(Clone)]
pub struct PyTaskStats {
    inner: TaskStats,
}

#[pymethods]
impl PyTaskStats {
    /// Task ID
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Task name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Task state
    #[getter]
    fn state(&self) -> PyTaskState {
        self.inner.state.into()
    }

    /// Wait time
    #[getter]
    fn wait_time(&self) -> f64 {
        self.inner.wait_time
    }

    /// Execution time
    #[getter]
    fn exec_time(&self) -> f64 {
        self.inner.exec_time
    }

    /// Memory used
    #[getter]
    fn memory_used(&self) -> usize {
        self.inner.memory_used
    }

    fn __repr__(&self) -> String {
        format!(
            "TaskStats(id='{}', state={:?}, wait={:.3}s, exec={:.3}s)",
            self.inner.id, self.inner.state, self.inner.wait_time, self.inner.exec_time
        )
    }
}

/// Thread-safe task scheduler with bandwidth pacing.
///
/// Args:
///     total_memory: Total GPU memory available (None for unlimited)
///     sched_tick_ms: Scheduling tick interval in milliseconds
///     window_ms: Bandwidth pacing window in milliseconds
///
/// Example:
///     scheduler = Scheduler(100 * 1024 * 1024, 10.0, 100.0)
///     task = TaskMeta("task-1", "Compute", 1024)
///     scheduler.submit(task)
///     runnable = scheduler.get_runnable_tasks(10)
#[pyclass(name = "Scheduler")]
pub struct PyScheduler {
    inner: Arc<Scheduler>,
}

#[pymethods]
impl PyScheduler {
    /// Create a new scheduler.
    #[new]
    #[pyo3(signature = (total_memory=None, sched_tick_ms=10.0, window_ms=100.0))]
    fn new(total_memory: Option<usize>, sched_tick_ms: f64, window_ms: f64) -> Self {
        Self {
            inner: Arc::new(Scheduler::new(total_memory, sched_tick_ms, window_ms)),
        }
    }

    /// Submit a task for scheduling.
    fn submit(&self, task: PyTaskMeta) -> String {
        self.inner.submit(task.inner)
    }

    /// Get tasks that are ready to run.
    #[pyo3(signature = (max_tasks=1))]
    fn get_runnable_tasks(&self, max_tasks: usize) -> Vec<String> {
        self.inner.get_runnable_tasks(max_tasks)
    }

    /// Check if a specific task should run now.
    fn should_run(&self, task_id: &str) -> bool {
        self.inner.should_run(task_id)
    }

    /// Mark a task as started.
    fn start_task(&self, task_id: &str) -> bool {
        self.inner.start_task(task_id)
    }

    /// Mark a task as completed.
    fn complete_task(&self, task_id: &str) -> bool {
        self.inner.complete_task(task_id)
    }

    /// Mark a task as failed.
    fn fail_task(&self, task_id: &str, error: String) -> bool {
        self.inner.fail_task(task_id, error)
    }

    /// Cancel a task.
    fn cancel_task(&self, task_id: &str) -> bool {
        self.inner.cancel_task(task_id)
    }

    /// Get task by ID.
    fn get_task(&self, task_id: &str) -> Option<PyTaskMeta> {
        self.inner.get_task(task_id).map(|t| PyTaskMeta { inner: t })
    }

    /// Get task state.
    fn get_task_state(&self, task_id: &str) -> Option<PyTaskState> {
        self.inner.get_task_state(task_id).map(|s| s.into())
    }

    /// Get scheduler statistics.
    fn stats(&self) -> PySchedulerStats {
        PySchedulerStats {
            inner: self.inner.stats(),
        }
    }

    /// Get task statistics.
    fn task_stats(&self, task_id: &str) -> Option<PyTaskStats> {
        self.inner.task_stats(task_id).map(|s| PyTaskStats { inner: s })
    }

    /// Clear all tasks.
    fn clear(&self) {
        self.inner.clear();
    }

    /// Get total memory.
    #[getter]
    fn total_memory(&self) -> Option<usize> {
        self.inner.total_memory()
    }

    /// Get scheduling tick interval.
    #[getter]
    fn sched_tick_ms(&self) -> f64 {
        self.inner.sched_tick_ms()
    }

    /// Get bandwidth window.
    #[getter]
    fn window_ms(&self) -> f64 {
        self.inner.window_ms()
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "Scheduler(pending={}, running={}, completed={})",
            stats.pending_count, stats.running_count, stats.completed_count
        )
    }
}

/// Register scheduler module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScheduler>()?;
    m.add_class::<PyTaskMeta>()?;
    m.add_class::<PyTaskState>()?;
    m.add_class::<PyTaskPolicy>()?;
    m.add_class::<PySchedulerStats>()?;
    m.add_class::<PyTaskStats>()?;
    Ok(())
}
