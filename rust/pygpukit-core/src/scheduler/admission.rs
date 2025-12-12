//! Admission Control for GPU task scheduling
//!
//! Implements deterministic admission pipeline that enforces
//! memory quotas and bandwidth reservation before scheduling.

use crate::scheduler::task::TaskMeta;

/// Reason for admission rejection
#[derive(Debug, Clone, PartialEq)]
pub enum RejectReason {
    /// Insufficient memory quota
    InsufficientMemory {
        requested: usize,
        available: usize,
    },
    /// Bandwidth quota exceeded
    BandwidthExceeded {
        requested_bw: f64,
        available_bw: f64,
    },
    /// Too many pending tasks
    QueueFull {
        current: usize,
        max: usize,
    },
    /// Task dependencies not satisfiable
    UnsatisfiableDependencies {
        missing: Vec<String>,
    },
    /// Custom rejection reason
    Custom(String),
}

/// Result of admission control decision
#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionDecision {
    /// Task is admitted and can be scheduled
    Admit {
        /// Reserved memory for this task
        reserved_memory: usize,
        /// Reserved bandwidth fraction (0.0 - 1.0)
        reserved_bandwidth: f64,
    },
    /// Task is rejected
    Reject {
        reason: RejectReason,
    },
    /// Task is queued for later (best-effort)
    Queue {
        /// Position in queue
        position: usize,
        /// Estimated wait time in milliseconds
        estimated_wait_ms: f64,
    },
}

impl AdmissionDecision {
    /// Create an admit decision with memory reservation
    pub fn admit(reserved_memory: usize) -> Self {
        Self::Admit {
            reserved_memory,
            reserved_bandwidth: 0.0,
        }
    }

    /// Create an admit decision with memory and bandwidth reservation
    pub fn admit_with_bandwidth(reserved_memory: usize, reserved_bandwidth: f64) -> Self {
        Self::Admit {
            reserved_memory,
            reserved_bandwidth,
        }
    }

    /// Create a reject decision due to insufficient memory
    pub fn reject_memory(requested: usize, available: usize) -> Self {
        Self::Reject {
            reason: RejectReason::InsufficientMemory {
                requested,
                available,
            },
        }
    }

    /// Create a reject decision due to bandwidth exceeded
    pub fn reject_bandwidth(requested_bw: f64, available_bw: f64) -> Self {
        Self::Reject {
            reason: RejectReason::BandwidthExceeded {
                requested_bw,
                available_bw,
            },
        }
    }

    /// Create a reject decision due to queue full
    pub fn reject_queue_full(current: usize, max: usize) -> Self {
        Self::Reject {
            reason: RejectReason::QueueFull { current, max },
        }
    }

    /// Create a reject decision due to unsatisfiable dependencies
    pub fn reject_dependencies(missing: Vec<String>) -> Self {
        Self::Reject {
            reason: RejectReason::UnsatisfiableDependencies { missing },
        }
    }

    /// Create a queue decision for best-effort tasks
    pub fn queue(position: usize, estimated_wait_ms: f64) -> Self {
        Self::Queue {
            position,
            estimated_wait_ms,
        }
    }

    /// Check if the decision is an admission
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Admit { .. })
    }

    /// Check if the decision is a rejection
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Reject { .. })
    }

    /// Check if the decision is queued
    pub fn is_queued(&self) -> bool {
        matches!(self, Self::Queue { .. })
    }

    /// Get the rejection reason if rejected
    pub fn rejection_reason(&self) -> Option<&RejectReason> {
        match self {
            Self::Reject { reason } => Some(reason),
            _ => None,
        }
    }
}

/// Admission control configuration
#[derive(Debug, Clone)]
pub struct AdmissionConfig {
    /// Total GPU memory available (bytes)
    pub total_memory: usize,
    /// Maximum memory that can be reserved (bytes)
    pub max_reserved_memory: usize,
    /// Maximum number of pending tasks
    pub max_pending_tasks: usize,
    /// Total bandwidth available (0.0 - 1.0)
    pub total_bandwidth: f64,
    /// Enable best-effort queueing for tasks that exceed quotas
    pub enable_best_effort: bool,
    /// Memory overcommit ratio (1.0 = no overcommit)
    pub memory_overcommit_ratio: f64,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            total_memory: usize::MAX,
            max_reserved_memory: usize::MAX,
            max_pending_tasks: 10000,
            total_bandwidth: 1.0,
            enable_best_effort: true,
            memory_overcommit_ratio: 1.0,
        }
    }
}

impl AdmissionConfig {
    /// Create a new admission config with memory limit
    pub fn with_memory(total_memory: usize) -> Self {
        Self {
            total_memory,
            max_reserved_memory: total_memory,
            ..Default::default()
        }
    }

    /// Set maximum pending tasks
    pub fn max_pending(mut self, max: usize) -> Self {
        self.max_pending_tasks = max;
        self
    }

    /// Set bandwidth limit
    pub fn bandwidth(mut self, bw: f64) -> Self {
        self.total_bandwidth = bw;
        self
    }

    /// Enable/disable best-effort queueing
    pub fn best_effort(mut self, enable: bool) -> Self {
        self.enable_best_effort = enable;
        self
    }

    /// Set memory overcommit ratio
    pub fn overcommit(mut self, ratio: f64) -> Self {
        self.memory_overcommit_ratio = ratio;
        self
    }
}

/// Admission controller state
#[derive(Debug)]
pub struct AdmissionController {
    config: AdmissionConfig,
    /// Currently reserved memory
    reserved_memory: usize,
    /// Currently reserved bandwidth
    reserved_bandwidth: f64,
    /// Number of pending tasks
    pending_count: usize,
    /// Total admitted tasks
    admitted_count: usize,
    /// Total rejected tasks
    rejected_count: usize,
    /// Total queued tasks (best-effort)
    queued_count: usize,
}

impl AdmissionController {
    /// Create a new admission controller
    pub fn new(config: AdmissionConfig) -> Self {
        Self {
            config,
            reserved_memory: 0,
            reserved_bandwidth: 0.0,
            pending_count: 0,
            admitted_count: 0,
            rejected_count: 0,
            queued_count: 0,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(AdmissionConfig::default())
    }

    /// Create with memory limit
    pub fn with_memory(total_memory: usize) -> Self {
        Self::new(AdmissionConfig::with_memory(total_memory))
    }

    /// Evaluate admission for a task
    ///
    /// This is a deterministic evaluation that does not modify state.
    /// Call `reserve()` after admission to actually reserve resources.
    pub fn evaluate(&self, task: &TaskMeta) -> AdmissionDecision {
        // Check queue capacity
        if self.pending_count >= self.config.max_pending_tasks {
            return AdmissionDecision::reject_queue_full(
                self.pending_count,
                self.config.max_pending_tasks,
            );
        }

        // Calculate effective memory limit with overcommit
        let effective_memory_limit =
            (self.config.max_reserved_memory as f64 * self.config.memory_overcommit_ratio) as usize;
        let available_memory = effective_memory_limit.saturating_sub(self.reserved_memory);

        // Check memory quota
        if task.memory_estimate > available_memory {
            if self.config.enable_best_effort {
                // Queue for later
                let estimated_wait = self.estimate_wait_time(task.memory_estimate);
                return AdmissionDecision::queue(self.pending_count, estimated_wait);
            } else {
                return AdmissionDecision::reject_memory(task.memory_estimate, available_memory);
            }
        }

        // Calculate bandwidth requirement (based on memory estimate)
        // Simplified model: bandwidth proportional to memory (adjusted for overcommit)
        let bandwidth_estimate = if effective_memory_limit > 0 {
            task.memory_estimate as f64 / effective_memory_limit as f64
        } else {
            0.0
        };

        let available_bandwidth = self.config.total_bandwidth - self.reserved_bandwidth;

        // Check bandwidth quota
        if bandwidth_estimate > available_bandwidth {
            if self.config.enable_best_effort {
                let estimated_wait = self.estimate_wait_time(task.memory_estimate);
                return AdmissionDecision::queue(self.pending_count, estimated_wait);
            } else {
                return AdmissionDecision::reject_bandwidth(bandwidth_estimate, available_bandwidth);
            }
        }

        // Admit the task
        AdmissionDecision::admit_with_bandwidth(task.memory_estimate, bandwidth_estimate)
    }

    /// Reserve resources for an admitted task
    ///
    /// Call this after receiving an `Admit` decision.
    pub fn reserve(&mut self, decision: &AdmissionDecision) -> bool {
        match decision {
            AdmissionDecision::Admit {
                reserved_memory,
                reserved_bandwidth,
            } => {
                self.reserved_memory += reserved_memory;
                self.reserved_bandwidth += reserved_bandwidth;
                self.pending_count += 1;
                self.admitted_count += 1;
                true
            }
            AdmissionDecision::Queue { .. } => {
                self.pending_count += 1;
                self.queued_count += 1;
                true
            }
            AdmissionDecision::Reject { .. } => {
                self.rejected_count += 1;
                false
            }
        }
    }

    /// Release resources when a task completes
    pub fn release(&mut self, memory: usize, bandwidth: f64) {
        self.reserved_memory = self.reserved_memory.saturating_sub(memory);
        self.reserved_bandwidth = (self.reserved_bandwidth - bandwidth).max(0.0);
        self.pending_count = self.pending_count.saturating_sub(1);
    }

    /// Admit a task (evaluate + reserve in one call)
    pub fn admit(&mut self, task: &TaskMeta) -> AdmissionDecision {
        let decision = self.evaluate(task);
        self.reserve(&decision);
        decision
    }

    /// Estimate wait time for a task requiring given memory
    fn estimate_wait_time(&self, _memory_needed: usize) -> f64 {
        // Simple heuristic: assume tasks complete at ~100MB/s throughput
        let memory_throughput = 100.0 * 1024.0 * 1024.0; // 100 MB/s
        let wait_seconds = self.reserved_memory as f64 / memory_throughput;
        wait_seconds * 1000.0 // Convert to ms
    }

    /// Get current reserved memory
    pub fn reserved_memory(&self) -> usize {
        self.reserved_memory
    }

    /// Get current reserved bandwidth
    pub fn reserved_bandwidth(&self) -> f64 {
        self.reserved_bandwidth
    }

    /// Get available memory
    pub fn available_memory(&self) -> usize {
        let effective = (self.config.max_reserved_memory as f64
            * self.config.memory_overcommit_ratio) as usize;
        effective.saturating_sub(self.reserved_memory)
    }

    /// Get available bandwidth
    pub fn available_bandwidth(&self) -> f64 {
        (self.config.total_bandwidth - self.reserved_bandwidth).max(0.0)
    }

    /// Get admission statistics
    pub fn stats(&self) -> AdmissionStats {
        AdmissionStats {
            admitted_count: self.admitted_count,
            rejected_count: self.rejected_count,
            queued_count: self.queued_count,
            pending_count: self.pending_count,
            reserved_memory: self.reserved_memory,
            reserved_bandwidth: self.reserved_bandwidth,
            available_memory: self.available_memory(),
            available_bandwidth: self.available_bandwidth(),
        }
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.reserved_memory = 0;
        self.reserved_bandwidth = 0.0;
        self.pending_count = 0;
        self.admitted_count = 0;
        self.rejected_count = 0;
        self.queued_count = 0;
    }

    /// Get the config
    pub fn config(&self) -> &AdmissionConfig {
        &self.config
    }
}

/// Admission control statistics
#[derive(Debug, Clone, Default)]
pub struct AdmissionStats {
    /// Total admitted tasks
    pub admitted_count: usize,
    /// Total rejected tasks
    pub rejected_count: usize,
    /// Total queued tasks (best-effort)
    pub queued_count: usize,
    /// Current pending tasks
    pub pending_count: usize,
    /// Currently reserved memory
    pub reserved_memory: usize,
    /// Currently reserved bandwidth
    pub reserved_bandwidth: f64,
    /// Available memory
    pub available_memory: usize,
    /// Available bandwidth
    pub available_bandwidth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admission_simple() {
        let mut controller = AdmissionController::with_memory(1000);

        let task = TaskMeta::with_memory("task-1".into(), "Test".into(), 500);
        let decision = controller.admit(&task);

        assert!(decision.is_admitted());
        assert_eq!(controller.reserved_memory(), 500);
    }

    #[test]
    fn test_admission_reject_memory() {
        let config = AdmissionConfig::with_memory(1000).best_effort(false);
        let mut controller = AdmissionController::new(config);

        let task = TaskMeta::with_memory("task-1".into(), "Test".into(), 1500);
        let decision = controller.admit(&task);

        assert!(decision.is_rejected());
        match decision.rejection_reason() {
            Some(RejectReason::InsufficientMemory { requested, available }) => {
                assert_eq!(*requested, 1500);
                assert_eq!(*available, 1000);
            }
            _ => panic!("Expected InsufficientMemory rejection"),
        }
    }

    #[test]
    fn test_admission_queue_best_effort() {
        let config = AdmissionConfig::with_memory(1000).best_effort(true);
        let mut controller = AdmissionController::new(config);

        // Fill memory
        let task1 = TaskMeta::with_memory("task-1".into(), "Test".into(), 800);
        controller.admit(&task1);

        // This should be queued
        let task2 = TaskMeta::with_memory("task-2".into(), "Test".into(), 500);
        let decision = controller.admit(&task2);

        assert!(decision.is_queued());
    }

    #[test]
    fn test_admission_queue_full() {
        let config = AdmissionConfig::with_memory(10000).max_pending(2);
        let mut controller = AdmissionController::new(config);

        // Submit 2 tasks
        let task1 = TaskMeta::with_memory("task-1".into(), "Test".into(), 100);
        let task2 = TaskMeta::with_memory("task-2".into(), "Test".into(), 100);
        controller.admit(&task1);
        controller.admit(&task2);

        // Third should be rejected
        let task3 = TaskMeta::with_memory("task-3".into(), "Test".into(), 100);
        let decision = controller.admit(&task3);

        assert!(decision.is_rejected());
        match decision.rejection_reason() {
            Some(RejectReason::QueueFull { current, max }) => {
                assert_eq!(*current, 2);
                assert_eq!(*max, 2);
            }
            _ => panic!("Expected QueueFull rejection"),
        }
    }

    #[test]
    fn test_admission_release() {
        let mut controller = AdmissionController::with_memory(1000);

        let task = TaskMeta::with_memory("task-1".into(), "Test".into(), 500);
        let decision = controller.admit(&task);

        assert!(decision.is_admitted());
        assert_eq!(controller.reserved_memory(), 500);

        // Release
        controller.release(500, 0.0);
        assert_eq!(controller.reserved_memory(), 0);
        assert_eq!(controller.available_memory(), 1000);
    }

    #[test]
    fn test_admission_overcommit() {
        let config = AdmissionConfig::with_memory(1000).overcommit(1.5);
        let mut controller = AdmissionController::new(config);

        // Can admit up to 1500 bytes
        let task = TaskMeta::with_memory("task-1".into(), "Test".into(), 1200);
        let decision = controller.admit(&task);

        assert!(decision.is_admitted());
    }

    #[test]
    fn test_admission_stats() {
        let mut controller = AdmissionController::with_memory(1000);

        // Admit one
        let task1 = TaskMeta::with_memory("task-1".into(), "Test".into(), 500);
        controller.admit(&task1);

        let stats = controller.stats();
        assert_eq!(stats.admitted_count, 1);
        assert_eq!(stats.pending_count, 1);
        assert_eq!(stats.reserved_memory, 500);
        assert_eq!(stats.available_memory, 500);
    }
}
