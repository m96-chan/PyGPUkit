//! Scheduler module Python bindings

use pyo3::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

use crate::errors::partition_error_to_py;

use pygpukit_core::scheduler::{
    Scheduler, SchedulerStats, TaskMeta, TaskState, TaskPolicy, TaskStats,
    AdmissionController, AdmissionConfig, AdmissionDecision, AdmissionStats, RejectReason,
    QosClass, QosPolicy, QosTaskMeta, QosEvaluation, QosPolicyEvaluator, QosStats,
    ResourceRequirements,
    PartitionManager, PartitionConfig, Partition, PartitionLimits, PartitionUsage, PartitionStats,
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

/// Rejection reason details for Python (provides detailed info about rejection)
#[pyclass(name = "RejectReasonDetails")]
#[derive(Clone)]
pub struct PyRejectReasonDetails {
    inner: RejectReason,
}

#[pymethods]
impl PyRejectReasonDetails {
    /// Get rejection type as string
    #[getter]
    fn reason_type(&self) -> String {
        match &self.inner {
            RejectReason::InsufficientMemory { .. } => "InsufficientMemory".into(),
            RejectReason::BandwidthExceeded { .. } => "BandwidthExceeded".into(),
            RejectReason::QueueFull { .. } => "QueueFull".into(),
            RejectReason::UnsatisfiableDependencies { .. } => "UnsatisfiableDependencies".into(),
            RejectReason::Custom(_) => "Custom".into(),
        }
    }

    /// Get requested memory (if InsufficientMemory)
    #[getter]
    fn requested_memory(&self) -> Option<usize> {
        match &self.inner {
            RejectReason::InsufficientMemory { requested, .. } => Some(*requested),
            _ => None,
        }
    }

    /// Get available memory (if InsufficientMemory)
    #[getter]
    fn available_memory(&self) -> Option<usize> {
        match &self.inner {
            RejectReason::InsufficientMemory { available, .. } => Some(*available),
            _ => None,
        }
    }

    /// Get message
    fn message(&self) -> String {
        match &self.inner {
            RejectReason::InsufficientMemory { requested, available } => {
                format!("Insufficient memory: requested {} bytes, available {} bytes", requested, available)
            }
            RejectReason::BandwidthExceeded { requested_bw, available_bw } => {
                format!("Bandwidth exceeded: requested {:.2}, available {:.2}", requested_bw, available_bw)
            }
            RejectReason::QueueFull { current, max } => {
                format!("Queue full: {} tasks pending, max {}", current, max)
            }
            RejectReason::UnsatisfiableDependencies { missing } => {
                format!("Unsatisfiable dependencies: {:?}", missing)
            }
            RejectReason::Custom(msg) => msg.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("RejectReasonDetails({})", self.message())
    }
}

/// Rejection reason enum for comparison in Python
#[pyclass(name = "RejectReason", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRejectReasonEnum {
    InsufficientMemory = 0,
    InsufficientBandwidth = 1,
    QueueFull = 2,
    UnsatisfiableDependencies = 3,
    Custom = 4,
}

/// Admission decision for Python
#[pyclass(name = "AdmissionDecision")]
#[derive(Clone)]
pub struct PyAdmissionDecision {
    inner: AdmissionDecision,
}

#[pymethods]
impl PyAdmissionDecision {
    /// Create an Admitted decision (for testing)
    #[staticmethod]
    #[pyo3(name = "Admitted")]
    fn admitted() -> Self {
        Self {
            inner: AdmissionDecision::Admit {
                reserved_memory: 0,
                reserved_bandwidth: 0.0,
            },
        }
    }

    /// Create a Queued decision (for testing)
    #[staticmethod]
    #[pyo3(name = "Queued")]
    fn queued() -> Self {
        Self {
            inner: AdmissionDecision::Queue {
                position: 0,
                estimated_wait_ms: 0.0,
            },
        }
    }

    /// Check if admitted
    fn is_admitted(&self) -> bool {
        self.inner.is_admitted()
    }

    /// Check if rejected
    fn is_rejected(&self) -> bool {
        self.inner.is_rejected()
    }

    /// Check if queued
    fn is_queued(&self) -> bool {
        self.inner.is_queued()
    }

    /// Get rejection reason type (for comparison)
    fn reject_reason(&self) -> Option<PyRejectReasonEnum> {
        match &self.inner {
            AdmissionDecision::Reject { reason } => {
                Some(match reason {
                    RejectReason::InsufficientMemory { .. } => PyRejectReasonEnum::InsufficientMemory,
                    RejectReason::BandwidthExceeded { .. } => PyRejectReasonEnum::InsufficientBandwidth,
                    RejectReason::QueueFull { .. } => PyRejectReasonEnum::QueueFull,
                    RejectReason::UnsatisfiableDependencies { .. } => PyRejectReasonEnum::UnsatisfiableDependencies,
                    RejectReason::Custom(_) => PyRejectReasonEnum::Custom,
                })
            }
            _ => None,
        }
    }

    /// Get decision type as string
    #[getter]
    fn decision_type(&self) -> String {
        match &self.inner {
            AdmissionDecision::Admit { .. } => "Admit".into(),
            AdmissionDecision::Reject { .. } => "Reject".into(),
            AdmissionDecision::Queue { .. } => "Queue".into(),
        }
    }

    /// Get reserved memory (if admitted)
    #[getter]
    fn reserved_memory(&self) -> Option<usize> {
        match &self.inner {
            AdmissionDecision::Admit { reserved_memory, .. } => Some(*reserved_memory),
            _ => None,
        }
    }

    /// Get reserved bandwidth (if admitted)
    #[getter]
    fn reserved_bandwidth(&self) -> Option<f64> {
        match &self.inner {
            AdmissionDecision::Admit { reserved_bandwidth, .. } => Some(*reserved_bandwidth),
            _ => None,
        }
    }

    /// Get queue position (if queued)
    #[getter]
    fn queue_position(&self) -> Option<usize> {
        match &self.inner {
            AdmissionDecision::Queue { position, .. } => Some(*position),
            _ => None,
        }
    }

    /// Get estimated wait time in ms (if queued)
    #[getter]
    fn estimated_wait_ms(&self) -> Option<f64> {
        match &self.inner {
            AdmissionDecision::Queue { estimated_wait_ms, .. } => Some(*estimated_wait_ms),
            _ => None,
        }
    }

    /// Get rejection reason details (if rejected)
    #[getter]
    fn rejection_reason(&self) -> Option<PyRejectReasonDetails> {
        match &self.inner {
            AdmissionDecision::Reject { reason } => Some(PyRejectReasonDetails { inner: reason.clone() }),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            AdmissionDecision::Admit { reserved_memory, reserved_bandwidth } => {
                format!("AdmissionDecision(Admit, memory={}, bandwidth={:.4})", reserved_memory, reserved_bandwidth)
            }
            AdmissionDecision::Reject { reason } => {
                format!("AdmissionDecision(Reject, reason={})", PyRejectReasonDetails { inner: reason.clone() }.message())
            }
            AdmissionDecision::Queue { position, estimated_wait_ms } => {
                format!("AdmissionDecision(Queue, position={}, wait={:.1}ms)", position, estimated_wait_ms)
            }
        }
    }
}

/// Admission statistics for Python
#[pyclass(name = "AdmissionStats")]
#[derive(Clone)]
pub struct PyAdmissionStats {
    inner: AdmissionStats,
}

#[pymethods]
impl PyAdmissionStats {
    /// Total admitted tasks
    #[getter]
    fn admitted_count(&self) -> usize {
        self.inner.admitted_count
    }

    /// Total rejected tasks
    #[getter]
    fn rejected_count(&self) -> usize {
        self.inner.rejected_count
    }

    /// Total queued tasks (best-effort)
    #[getter]
    fn queued_count(&self) -> usize {
        self.inner.queued_count
    }

    /// Current pending tasks
    #[getter]
    fn pending_count(&self) -> usize {
        self.inner.pending_count
    }

    /// Currently reserved memory
    #[getter]
    fn reserved_memory(&self) -> usize {
        self.inner.reserved_memory
    }

    /// Currently used memory (alias for reserved_memory)
    #[getter]
    fn used_memory(&self) -> usize {
        self.inner.reserved_memory
    }

    /// Currently reserved bandwidth
    #[getter]
    fn reserved_bandwidth(&self) -> f64 {
        self.inner.reserved_bandwidth
    }

    /// Currently used bandwidth (alias for reserved_bandwidth)
    #[getter]
    fn used_bandwidth(&self) -> f64 {
        self.inner.reserved_bandwidth
    }

    /// Available memory
    #[getter]
    fn available_memory(&self) -> usize {
        self.inner.available_memory
    }

    /// Available bandwidth
    #[getter]
    fn available_bandwidth(&self) -> f64 {
        self.inner.available_bandwidth
    }

    fn __repr__(&self) -> String {
        format!(
            "AdmissionStats(admitted={}, rejected={}, queued={}, pending={})",
            self.inner.admitted_count, self.inner.rejected_count,
            self.inner.queued_count, self.inner.pending_count
        )
    }
}

/// Admission configuration for Python
#[pyclass(name = "AdmissionConfig")]
#[derive(Clone)]
pub struct PyAdmissionConfig {
    inner: AdmissionConfig,
}

#[pymethods]
impl PyAdmissionConfig {
    /// Create a new AdmissionConfig
    #[new]
    #[pyo3(signature = (max_memory, max_bandwidth, max_pending_tasks=1000, enable_best_effort=true))]
    fn new(max_memory: usize, max_bandwidth: f64, max_pending_tasks: usize, enable_best_effort: bool) -> Self {
        Self {
            inner: AdmissionConfig {
                total_memory: max_memory,
                max_reserved_memory: max_memory,
                max_pending_tasks,
                total_bandwidth: max_bandwidth,
                enable_best_effort,
                memory_overcommit_ratio: 1.0,
            },
        }
    }

    #[getter]
    fn max_memory(&self) -> usize {
        self.inner.total_memory
    }

    #[getter]
    fn max_bandwidth(&self) -> f64 {
        self.inner.total_bandwidth
    }

    #[getter]
    fn max_pending_tasks(&self) -> usize {
        self.inner.max_pending_tasks
    }

    #[getter]
    fn enable_best_effort(&self) -> bool {
        self.inner.enable_best_effort
    }

    fn __repr__(&self) -> String {
        format!(
            "AdmissionConfig(max_memory={}, max_bandwidth={:.2}, max_pending={})",
            self.inner.total_memory, self.inner.total_bandwidth, self.inner.max_pending_tasks
        )
    }
}

/// Admission controller for Python with task tracking
#[pyclass(name = "AdmissionController")]
pub struct PyAdmissionController {
    inner: AdmissionController,
    // Track allocations by task_id for release()
    allocations: HashMap<String, (usize, f64)>,
}

#[pymethods]
impl PyAdmissionController {
    /// Create a new AdmissionController
    #[new]
    fn new(config: PyAdmissionConfig) -> Self {
        Self {
            inner: AdmissionController::new(config.inner),
            allocations: HashMap::new(),
        }
    }

    /// Try to admit a task with given resource requirements
    fn try_admit(&mut self, task_id: &str, memory: usize, bandwidth: f64) -> PyAdmissionDecision {
        // Create a temporary TaskMeta for admission
        let task = TaskMeta::with_memory(task_id.to_string(), task_id.to_string(), memory);
        let decision = self.inner.admit(&task);

        // Track allocation if admitted
        if decision.is_admitted() {
            self.allocations.insert(task_id.to_string(), (memory, bandwidth));
        }

        PyAdmissionDecision { inner: decision }
    }

    /// Release resources for a task
    fn release(&mut self, task_id: &str) {
        if let Some((memory, bandwidth)) = self.allocations.remove(task_id) {
            self.inner.release(memory, bandwidth);
        }
    }

    /// Get admission statistics
    fn stats(&self) -> PyAdmissionStats {
        PyAdmissionStats {
            inner: self.inner.stats(),
        }
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "AdmissionController(admitted={}, rejected={}, pending={})",
            stats.admitted_count, stats.rejected_count, stats.pending_count
        )
    }
}

// =============================================================================
// QoS Policy Types
// =============================================================================

/// QoS class enum for Python
#[pyclass(name = "QosClass", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyQosClass {
    Guaranteed = 0,
    Burstable = 1,
    BestEffort = 2,
}

impl From<QosClass> for PyQosClass {
    fn from(class: QosClass) -> Self {
        match class {
            QosClass::Guaranteed => PyQosClass::Guaranteed,
            QosClass::Burstable => PyQosClass::Burstable,
            QosClass::BestEffort => PyQosClass::BestEffort,
        }
    }
}

impl From<PyQosClass> for QosClass {
    fn from(class: PyQosClass) -> Self {
        match class {
            PyQosClass::Guaranteed => QosClass::Guaranteed,
            PyQosClass::Burstable => QosClass::Burstable,
            PyQosClass::BestEffort => QosClass::BestEffort,
        }
    }
}

#[pymethods]
impl PyQosClass {
    /// Get scheduling priority for this QoS class
    fn priority(&self) -> i32 {
        QosClass::from(*self).priority()
    }

    /// Check if this class can preempt another
    fn can_preempt(&self, other: PyQosClass) -> bool {
        QosClass::from(*self).can_preempt(&QosClass::from(other))
    }

    /// Get memory overcommit ratio
    fn memory_overcommit_ratio(&self) -> f64 {
        QosClass::from(*self).memory_overcommit_ratio()
    }

    /// Get bandwidth allocation ratio
    fn bandwidth_ratio(&self) -> f64 {
        QosClass::from(*self).bandwidth_ratio()
    }
}

/// Resource requirements for Python
#[pyclass(name = "ResourceRequirements")]
#[derive(Clone)]
pub struct PyResourceRequirements {
    inner: ResourceRequirements,
}

#[pymethods]
impl PyResourceRequirements {
    /// Create new resource requirements
    #[new]
    #[pyo3(signature = (memory_request, memory_limit=None, bandwidth_request=0.0, bandwidth_limit=1.0))]
    fn new(
        memory_request: usize,
        memory_limit: Option<usize>,
        bandwidth_request: f64,
        bandwidth_limit: f64,
    ) -> Self {
        Self {
            inner: ResourceRequirements {
                memory_request,
                memory_limit: memory_limit.unwrap_or(memory_request),
                bandwidth_request,
                bandwidth_limit,
            },
        }
    }

    /// Create guaranteed requirements
    #[staticmethod]
    fn guaranteed(memory: usize) -> Self {
        Self {
            inner: ResourceRequirements::guaranteed(memory),
        }
    }

    /// Create burstable requirements
    #[staticmethod]
    fn burstable(memory_request: usize, burst_ratio: f64) -> Self {
        Self {
            inner: ResourceRequirements::burstable(memory_request, burst_ratio),
        }
    }

    /// Create best-effort requirements
    #[staticmethod]
    fn best_effort() -> Self {
        Self {
            inner: ResourceRequirements::best_effort(),
        }
    }

    #[getter]
    fn memory_request(&self) -> usize {
        self.inner.memory_request
    }

    #[getter]
    fn memory_limit(&self) -> usize {
        self.inner.memory_limit
    }

    #[getter]
    fn bandwidth_request(&self) -> f64 {
        self.inner.bandwidth_request
    }

    #[getter]
    fn bandwidth_limit(&self) -> f64 {
        self.inner.bandwidth_limit
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourceRequirements(memory={}/{}, bandwidth={:.2}/{:.2})",
            self.inner.memory_request, self.inner.memory_limit,
            self.inner.bandwidth_request, self.inner.bandwidth_limit
        )
    }
}

/// QoS policy for Python
#[pyclass(name = "QosPolicy")]
#[derive(Clone)]
pub struct PyQosPolicy {
    inner: QosPolicy,
}

#[pymethods]
impl PyQosPolicy {
    /// Create a Guaranteed policy
    #[staticmethod]
    fn guaranteed(memory: usize) -> Self {
        Self {
            inner: QosPolicy::guaranteed(memory),
        }
    }

    /// Create a Burstable policy
    #[staticmethod]
    fn burstable(memory_request: usize, burst_ratio: f64) -> Self {
        Self {
            inner: QosPolicy::burstable(memory_request, burst_ratio),
        }
    }

    /// Create a BestEffort policy
    #[staticmethod]
    fn best_effort() -> Self {
        Self {
            inner: QosPolicy::best_effort(),
        }
    }

    #[getter]
    fn qos_class(&self) -> PyQosClass {
        self.inner.class.into()
    }

    /// Get memory to reserve
    fn memory_to_reserve(&self) -> usize {
        self.inner.memory_to_reserve()
    }

    /// Get bandwidth to reserve
    fn bandwidth_to_reserve(&self) -> f64 {
        self.inner.bandwidth_to_reserve()
    }

    fn __repr__(&self) -> String {
        format!("QosPolicy({:?})", self.inner.class)
    }
}

/// QoS task metadata for Python
#[pyclass(name = "QosTaskMeta")]
#[derive(Clone)]
pub struct PyQosTaskMeta {
    inner: QosTaskMeta,
}

#[pymethods]
impl PyQosTaskMeta {
    /// Create a Guaranteed task
    #[staticmethod]
    fn guaranteed(id: String, name: String, memory: usize) -> Self {
        Self {
            inner: QosTaskMeta::guaranteed(id, name, memory),
        }
    }

    /// Create a Burstable task
    #[staticmethod]
    fn burstable(id: String, name: String, memory_request: usize, burst_ratio: f64) -> Self {
        Self {
            inner: QosTaskMeta::burstable(id, name, memory_request, burst_ratio),
        }
    }

    /// Create a BestEffort task
    #[staticmethod]
    fn best_effort(id: String, name: String) -> Self {
        Self {
            inner: QosTaskMeta::best_effort(id, name),
        }
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.task.id.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.task.name.clone()
    }

    #[getter]
    fn qos_class(&self) -> PyQosClass {
        self.inner.qos.class.into()
    }

    /// Get memory request (bytes)
    #[getter]
    fn memory_request(&self) -> usize {
        self.inner.qos.resources.memory_request
    }

    /// Get memory limit (bytes)
    #[getter]
    fn memory_limit(&self) -> usize {
        self.inner.qos.resources.memory_limit
    }

    /// Get burst ratio (memory_limit / memory_request)
    #[getter]
    fn burst_ratio(&self) -> f64 {
        if self.inner.qos.resources.memory_request > 0 {
            self.inner.qos.resources.memory_limit as f64 / self.inner.qos.resources.memory_request as f64
        } else {
            1.0
        }
    }

    /// Get bandwidth request (0.0 - 1.0)
    #[getter]
    fn bandwidth_request(&self) -> f64 {
        self.inner.qos.resources.bandwidth_request
    }

    /// Get effective priority
    fn effective_priority(&self) -> i32 {
        self.inner.effective_priority()
    }

    fn __repr__(&self) -> String {
        format!(
            "QosTaskMeta(id='{}', class={:?}, memory={}, priority={})",
            self.inner.task.id, self.inner.qos.class, self.inner.qos.resources.memory_request, self.inner.effective_priority()
        )
    }
}

/// QoS evaluation result for Python
#[pyclass(name = "QosEvaluation")]
#[derive(Clone)]
pub struct PyQosEvaluation {
    inner: QosEvaluation,
}

#[pymethods]
impl PyQosEvaluation {
    fn is_admitted(&self) -> bool {
        self.inner.is_admitted()
    }

    fn is_throttled(&self) -> bool {
        self.inner.is_throttled()
    }

    fn is_queued(&self) -> bool {
        self.inner.is_queued()
    }

    fn is_rejected(&self) -> bool {
        self.inner.is_rejected()
    }

    #[getter]
    fn decision_type(&self) -> String {
        match &self.inner {
            QosEvaluation::Admit { .. } => "Admit".into(),
            QosEvaluation::Throttle { .. } => "Throttle".into(),
            QosEvaluation::Queue { .. } => "Queue".into(),
            QosEvaluation::Reject { .. } => "Reject".into(),
        }
    }

    #[getter]
    fn qos_class(&self) -> Option<PyQosClass> {
        match &self.inner {
            QosEvaluation::Admit { class, .. } => Some((*class).into()),
            QosEvaluation::Throttle { class, .. } => Some((*class).into()),
            _ => None,
        }
    }

    #[getter]
    fn reserved_memory(&self) -> Option<usize> {
        match &self.inner {
            QosEvaluation::Admit { reserved_memory, .. } => Some(*reserved_memory),
            _ => None,
        }
    }

    #[getter]
    fn reject_reason(&self) -> Option<String> {
        match &self.inner {
            QosEvaluation::Reject { reason } => Some(reason.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            QosEvaluation::Admit { class, reserved_memory, reserved_bandwidth } => {
                format!("QosEvaluation(Admit, class={:?}, memory={}, bw={:.2})",
                    class, reserved_memory, reserved_bandwidth)
            }
            QosEvaluation::Throttle { class, allowed_memory, allowed_bandwidth } => {
                format!("QosEvaluation(Throttle, class={:?}, allowed_mem={}, allowed_bw={:.2})",
                    class, allowed_memory, allowed_bandwidth)
            }
            QosEvaluation::Queue { position } => {
                format!("QosEvaluation(Queue, position={})", position)
            }
            QosEvaluation::Reject { reason } => {
                format!("QosEvaluation(Reject, reason='{}')", reason)
            }
        }
    }
}

/// QoS statistics for Python
#[pyclass(name = "QosStats")]
#[derive(Clone)]
pub struct PyQosStats {
    inner: QosStats,
}

#[pymethods]
impl PyQosStats {
    #[getter]
    fn total_memory(&self) -> usize {
        self.inner.total_memory
    }

    #[getter]
    fn total_bandwidth(&self) -> f64 {
        self.inner.total_bandwidth
    }

    #[getter]
    fn guaranteed_memory(&self) -> usize {
        self.inner.guaranteed_memory
    }

    #[getter]
    fn guaranteed_bandwidth(&self) -> f64 {
        self.inner.guaranteed_bandwidth
    }

    #[getter]
    fn burstable_memory(&self) -> usize {
        self.inner.burstable_memory
    }

    #[getter]
    fn best_effort_queue(&self) -> usize {
        self.inner.best_effort_queue
    }

    #[getter]
    fn available_memory(&self) -> usize {
        self.inner.available_memory
    }

    #[getter]
    fn available_bandwidth(&self) -> f64 {
        self.inner.available_bandwidth
    }

    fn __repr__(&self) -> String {
        format!(
            "QosStats(guaranteed_mem={}, burstable_mem={}, best_effort_queue={})",
            self.inner.guaranteed_memory, self.inner.burstable_memory,
            self.inner.best_effort_queue
        )
    }
}

/// QoS policy evaluator for Python
#[pyclass(name = "QosPolicyEvaluator")]
pub struct PyQosPolicyEvaluator {
    inner: QosPolicyEvaluator,
}

#[pymethods]
impl PyQosPolicyEvaluator {
    #[new]
    #[pyo3(signature = (total_memory, total_bandwidth=1.0))]
    fn new(total_memory: usize, total_bandwidth: f64) -> Self {
        Self {
            inner: QosPolicyEvaluator::new(total_memory, total_bandwidth),
        }
    }

    /// Evaluate QoS policy for a task
    fn evaluate(&self, task: &PyQosTaskMeta) -> PyQosEvaluation {
        PyQosEvaluation {
            inner: self.inner.evaluate(&task.inner),
        }
    }

    /// Reserve resources for an admitted task
    fn reserve(&mut self, evaluation: &PyQosEvaluation) {
        self.inner.reserve(&evaluation.inner);
    }

    /// Release resources when a task completes
    fn release(&mut self, qos_class: PyQosClass, memory: usize, bandwidth: f64) {
        self.inner.release(qos_class.into(), memory, bandwidth);
    }

    /// Get statistics
    fn stats(&self) -> PyQosStats {
        PyQosStats {
            inner: self.inner.stats(),
        }
    }

    /// Reset evaluator state
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "QosPolicyEvaluator(avail_mem={}, avail_bw={:.2})",
            stats.available_memory, stats.available_bandwidth
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

    /// Admit a task through admission control.
    ///
    /// Returns an AdmissionDecision indicating whether the task
    /// was admitted, queued, or rejected.
    fn admit(&self, task: PyTaskMeta) -> PyAdmissionDecision {
        PyAdmissionDecision {
            inner: self.inner.admit(task.inner),
        }
    }

    /// Evaluate admission for a task without submitting.
    fn evaluate_admission(&self, task: &PyTaskMeta) -> PyAdmissionDecision {
        PyAdmissionDecision {
            inner: self.inner.evaluate_admission(&task.inner),
        }
    }

    /// Get admission control statistics.
    fn admission_stats(&self) -> PyAdmissionStats {
        PyAdmissionStats {
            inner: self.inner.admission_stats(),
        }
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

// =============================================================================
// GPU Partitioning Types
// =============================================================================

/// Partition limits for Python
#[pyclass(name = "PartitionLimits")]
#[derive(Clone)]
pub struct PyPartitionLimits {
    inner: PartitionLimits,
}

#[pymethods]
impl PyPartitionLimits {
    #[new]
    #[pyo3(signature = (memory_quota=0, compute_share=1.0, bandwidth_share=1.0, max_streams=16))]
    fn new(memory_quota: usize, compute_share: f64, bandwidth_share: f64, max_streams: usize) -> Self {
        Self {
            inner: PartitionLimits {
                memory_quota,
                compute_share,
                bandwidth_share,
                max_streams,
                ..Default::default()
            },
        }
    }

    /// Create with memory quota
    #[staticmethod]
    fn with_memory(memory_quota: usize) -> Self {
        Self {
            inner: PartitionLimits::with_memory(memory_quota),
        }
    }

    /// Create with compute share
    #[staticmethod]
    fn with_compute(compute_share: f64) -> Self {
        Self {
            inner: PartitionLimits::with_compute(compute_share),
        }
    }

    /// Set memory quota
    fn memory(&self, quota: usize) -> Self {
        Self {
            inner: self.inner.clone().memory(quota),
        }
    }

    /// Set compute share
    fn compute(&self, share: f64) -> Self {
        Self {
            inner: self.inner.clone().compute(share),
        }
    }

    /// Set bandwidth share
    fn bandwidth(&self, share: f64) -> Self {
        Self {
            inner: self.inner.clone().bandwidth(share),
        }
    }

    #[getter]
    fn memory_quota(&self) -> usize {
        self.inner.memory_quota
    }

    #[getter]
    fn compute_share(&self) -> f64 {
        self.inner.compute_share
    }

    #[getter]
    fn bandwidth_share(&self) -> f64 {
        self.inner.bandwidth_share
    }

    #[getter]
    fn max_streams(&self) -> usize {
        self.inner.max_streams
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionLimits(memory={}, compute={:.1}%, bandwidth={:.1}%)",
            self.inner.memory_quota,
            self.inner.compute_share * 100.0,
            self.inner.bandwidth_share * 100.0
        )
    }
}

/// Partition usage for Python
#[pyclass(name = "PartitionUsage")]
#[derive(Clone)]
pub struct PyPartitionUsage {
    inner: PartitionUsage,
}

#[pymethods]
impl PyPartitionUsage {
    #[getter]
    fn memory_used(&self) -> usize {
        self.inner.memory_used
    }

    #[getter]
    fn active_streams(&self) -> usize {
        self.inner.active_streams
    }

    #[getter]
    fn pending_kernels(&self) -> usize {
        self.inner.pending_kernels
    }

    #[getter]
    fn pending_transfers(&self) -> usize {
        self.inner.pending_transfers
    }

    #[getter]
    fn total_kernels(&self) -> usize {
        self.inner.total_kernels
    }

    #[getter]
    fn total_transfers(&self) -> usize {
        self.inner.total_transfers
    }

    #[getter]
    fn compute_time_ms(&self) -> f64 {
        self.inner.compute_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionUsage(memory={}, streams={}, kernels={})",
            self.inner.memory_used, self.inner.active_streams, self.inner.total_kernels
        )
    }
}

/// Partition for Python
#[pyclass(name = "Partition")]
#[derive(Clone)]
pub struct PyPartition {
    inner: Partition,
}

#[pymethods]
impl PyPartition {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn limits(&self) -> PyPartitionLimits {
        PyPartitionLimits { inner: self.inner.limits.clone() }
    }

    #[getter]
    fn usage(&self) -> PyPartitionUsage {
        PyPartitionUsage { inner: self.inner.usage.clone() }
    }

    #[getter]
    fn tasks(&self) -> Vec<String> {
        self.inner.tasks.clone()
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }

    /// Get memory utilization (0.0 - 1.0)
    fn memory_utilization(&self) -> f64 {
        self.inner.memory_utilization()
    }

    fn __repr__(&self) -> String {
        format!(
            "Partition(id='{}', name='{}', enabled={})",
            self.inner.id, self.inner.name, self.inner.enabled
        )
    }
}

/// Partition config for Python
#[pyclass(name = "PartitionConfig")]
#[derive(Clone)]
pub struct PyPartitionConfig {
    inner: PartitionConfig,
}

#[pymethods]
impl PyPartitionConfig {
    #[new]
    #[pyo3(signature = (total_memory=8589934592, allow_overcommit=false, overcommit_ratio=1.0))]
    fn new(total_memory: usize, allow_overcommit: bool, overcommit_ratio: f64) -> Self {
        Self {
            inner: PartitionConfig {
                total_memory,
                allow_overcommit,
                overcommit_ratio,
                ..Default::default()
            },
        }
    }

    #[getter]
    fn total_memory(&self) -> usize {
        self.inner.total_memory
    }

    #[getter]
    fn allow_overcommit(&self) -> bool {
        self.inner.allow_overcommit
    }

    #[getter]
    fn overcommit_ratio(&self) -> f64 {
        self.inner.overcommit_ratio
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionConfig(memory={}, overcommit={})",
            self.inner.total_memory, self.inner.allow_overcommit
        )
    }
}

/// Partition statistics for Python
#[pyclass(name = "PartitionStats")]
#[derive(Clone)]
pub struct PyPartitionStats {
    inner: PartitionStats,
}

#[pymethods]
impl PyPartitionStats {
    #[getter]
    fn partition_count(&self) -> usize {
        self.inner.partition_count
    }

    #[getter]
    fn active_partitions(&self) -> usize {
        self.inner.active_partitions
    }

    #[getter]
    fn total_memory_allocated(&self) -> usize {
        self.inner.total_memory_allocated
    }

    #[getter]
    fn total_compute_allocated(&self) -> f64 {
        self.inner.total_compute_allocated
    }

    #[getter]
    fn total_bandwidth_allocated(&self) -> f64 {
        self.inner.total_bandwidth_allocated
    }

    #[getter]
    fn available_memory(&self) -> usize {
        self.inner.available_memory
    }

    #[getter]
    fn available_compute(&self) -> f64 {
        self.inner.available_compute
    }

    #[getter]
    fn available_bandwidth(&self) -> f64 {
        self.inner.available_bandwidth
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionStats(partitions={}, memory_alloc={}, compute_alloc={:.1}%)",
            self.inner.partition_count,
            self.inner.total_memory_allocated,
            self.inner.total_compute_allocated * 100.0
        )
    }
}

/// Partition manager for Python
///
/// Manages GPU resource partitions for multi-tenant or multi-task isolation.
#[pyclass(name = "PartitionManager")]
pub struct PyPartitionManager {
    inner: PartitionManager,
}

#[pymethods]
impl PyPartitionManager {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPartitionConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: PartitionManager::new(cfg),
        }
    }

    /// Create with total memory
    #[staticmethod]
    fn with_memory(total_memory: usize) -> Self {
        Self {
            inner: PartitionManager::with_memory(total_memory),
        }
    }

    /// Create a new partition
    fn create_partition(&mut self, id: &str, name: &str, limits: PyPartitionLimits) -> PyResult<()> {
        self.inner.create_partition(id, name, limits.inner).map_err(partition_error_to_py)
    }

    /// Delete a partition
    fn delete_partition(&mut self, id: &str) -> PyResult<PyPartition> {
        self.inner.delete_partition(id)
            .map(|p| PyPartition { inner: p })
            .map_err(partition_error_to_py)
    }

    /// Get a partition
    fn get(&self, id: &str) -> Option<PyPartition> {
        self.inner.get(id).map(|p| PyPartition { inner: p.clone() })
    }

    /// Assign a task to a partition
    fn assign_task(&mut self, task_id: &str, partition_id: &str) -> PyResult<()> {
        self.inner.assign_task(task_id, partition_id).map_err(partition_error_to_py)
    }

    /// Get partition for a task
    fn get_task_partition(&self, task_id: &str) -> Option<PyPartition> {
        self.inner.get_task_partition(task_id).map(|p| PyPartition { inner: p.clone() })
    }

    /// Unassign a task from its partition
    fn unassign_task(&mut self, task_id: &str) {
        self.inner.unassign_task(task_id);
    }

    /// Set default partition
    fn set_default(&mut self, id: &str) -> PyResult<()> {
        self.inner.set_default(id).map_err(partition_error_to_py)
    }

    /// Get default partition
    fn default_partition(&self) -> Option<PyPartition> {
        self.inner.default_partition().map(|p| PyPartition { inner: p.clone() })
    }

    /// List all partition IDs
    fn partition_ids(&self) -> Vec<String> {
        self.inner.partition_ids().into_iter().map(|s| s.to_string()).collect()
    }

    /// Get statistics
    fn stats(&self) -> PyPartitionStats {
        PyPartitionStats {
            inner: self.inner.stats(),
        }
    }

    /// Clear all partitions
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "PartitionManager(partitions={}, memory_alloc={})",
            stats.partition_count, stats.total_memory_allocated
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
    // Admission control
    m.add_class::<PyAdmissionConfig>()?;
    m.add_class::<PyAdmissionController>()?;
    m.add_class::<PyAdmissionDecision>()?;
    m.add_class::<PyAdmissionStats>()?;
    m.add_class::<PyRejectReasonEnum>()?;
    m.add_class::<PyRejectReasonDetails>()?;
    // QoS policy
    m.add_class::<PyQosClass>()?;
    m.add_class::<PyQosPolicy>()?;
    m.add_class::<PyQosTaskMeta>()?;
    m.add_class::<PyQosEvaluation>()?;
    m.add_class::<PyQosPolicyEvaluator>()?;
    m.add_class::<PyQosStats>()?;
    m.add_class::<PyResourceRequirements>()?;
    // Partitioning
    m.add_class::<PyPartitionLimits>()?;
    m.add_class::<PyPartitionUsage>()?;
    m.add_class::<PyPartition>()?;
    m.add_class::<PyPartitionConfig>()?;
    m.add_class::<PyPartitionStats>()?;
    m.add_class::<PyPartitionManager>()?;
    Ok(())
}
