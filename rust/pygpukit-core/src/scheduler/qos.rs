//! QoS (Quality of Service) Policy Framework
//!
//! Provides Kubernetes-style QoS tiers for GPU task scheduling:
//! - Guaranteed: Reserved resources, highest priority
//! - Burstable: Partial reservations, can use spare capacity
//! - BestEffort: No reservations, lowest priority

use crate::scheduler::task::TaskMeta;

/// QoS class for GPU tasks (Kubernetes-style)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QosClass {
    /// Guaranteed: Full resource reservation
    /// - Memory and bandwidth are fully reserved
    /// - Never preempted
    /// - Highest scheduling priority
    Guaranteed,

    /// Burstable: Partial resource reservation
    /// - Memory request is reserved, can burst to limit
    /// - May be throttled under contention
    /// - Medium scheduling priority
    #[default]
    Burstable,

    /// BestEffort: No resource reservation
    /// - No guaranteed resources
    /// - First to be evicted under pressure
    /// - Lowest scheduling priority
    BestEffort,
}

impl QosClass {
    /// Get scheduling priority for this QoS class
    pub fn priority(&self) -> i32 {
        match self {
            QosClass::Guaranteed => 100,
            QosClass::Burstable => 50,
            QosClass::BestEffort => 0,
        }
    }

    /// Check if this class can preempt another
    pub fn can_preempt(&self, other: &QosClass) -> bool {
        self.priority() > other.priority()
    }

    /// Get memory overcommit ratio for this class
    pub fn memory_overcommit_ratio(&self) -> f64 {
        match self {
            QosClass::Guaranteed => 1.0,  // No overcommit
            QosClass::Burstable => 1.5,   // 50% overcommit allowed
            QosClass::BestEffort => 2.0,  // 100% overcommit allowed
        }
    }

    /// Get bandwidth allocation ratio for this class
    pub fn bandwidth_ratio(&self) -> f64 {
        match self {
            QosClass::Guaranteed => 1.0,  // Full bandwidth
            QosClass::Burstable => 0.8,   // 80% bandwidth
            QosClass::BestEffort => 0.5,  // 50% bandwidth
        }
    }

    /// Check if resources should be reserved for this class
    pub fn reserves_resources(&self) -> bool {
        match self {
            QosClass::Guaranteed => true,
            QosClass::Burstable => true,
            QosClass::BestEffort => false,
        }
    }
}

/// Resource requirements for a task
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    /// Minimum memory required (request)
    pub memory_request: usize,
    /// Maximum memory allowed (limit)
    pub memory_limit: usize,
    /// Minimum bandwidth required (0.0 - 1.0)
    pub bandwidth_request: f64,
    /// Maximum bandwidth allowed (0.0 - 1.0)
    pub bandwidth_limit: f64,
}

impl ResourceRequirements {
    /// Create new resource requirements
    pub fn new(memory_request: usize, memory_limit: usize) -> Self {
        Self {
            memory_request,
            memory_limit,
            bandwidth_request: 0.0,
            bandwidth_limit: 1.0,
        }
    }

    /// Create requirements with just memory limit (request = limit)
    pub fn guaranteed(memory: usize) -> Self {
        Self {
            memory_request: memory,
            memory_limit: memory,
            bandwidth_request: 0.0,
            bandwidth_limit: 1.0,
        }
    }

    /// Create requirements with request/limit ratio
    pub fn burstable(memory_request: usize, burst_ratio: f64) -> Self {
        Self {
            memory_request,
            memory_limit: (memory_request as f64 * burst_ratio) as usize,
            bandwidth_request: 0.0,
            bandwidth_limit: 1.0,
        }
    }

    /// Create best-effort requirements (no limits)
    pub fn best_effort() -> Self {
        Self {
            memory_request: 0,
            memory_limit: usize::MAX,
            bandwidth_request: 0.0,
            bandwidth_limit: 1.0,
        }
    }

    /// Set bandwidth requirements
    pub fn with_bandwidth(mut self, request: f64, limit: f64) -> Self {
        self.bandwidth_request = request;
        self.bandwidth_limit = limit;
        self
    }
}

/// QoS policy configuration
#[derive(Debug, Clone)]
pub struct QosPolicy {
    /// QoS class
    pub class: QosClass,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for QosPolicy {
    fn default() -> Self {
        Self {
            class: QosClass::Burstable,
            resources: ResourceRequirements::default(),
        }
    }
}

impl QosPolicy {
    /// Create a Guaranteed policy
    pub fn guaranteed(memory: usize) -> Self {
        Self {
            class: QosClass::Guaranteed,
            resources: ResourceRequirements::guaranteed(memory),
        }
    }

    /// Create a Burstable policy
    pub fn burstable(memory_request: usize, burst_ratio: f64) -> Self {
        Self {
            class: QosClass::Burstable,
            resources: ResourceRequirements::burstable(memory_request, burst_ratio),
        }
    }

    /// Create a BestEffort policy
    pub fn best_effort() -> Self {
        Self {
            class: QosClass::BestEffort,
            resources: ResourceRequirements::best_effort(),
        }
    }

    /// Get effective scheduling priority
    pub fn effective_priority(&self, base_priority: i32) -> i32 {
        base_priority + self.class.priority()
    }

    /// Check if this policy can be satisfied with available resources
    pub fn can_satisfy(&self, available_memory: usize, available_bandwidth: f64) -> bool {
        self.resources.memory_request <= available_memory
            && self.resources.bandwidth_request <= available_bandwidth
    }

    /// Get amount of memory to reserve
    pub fn memory_to_reserve(&self) -> usize {
        match self.class {
            QosClass::Guaranteed => self.resources.memory_limit,
            QosClass::Burstable => self.resources.memory_request,
            QosClass::BestEffort => 0,
        }
    }

    /// Get amount of bandwidth to reserve
    pub fn bandwidth_to_reserve(&self) -> f64 {
        match self.class {
            QosClass::Guaranteed => self.resources.bandwidth_limit,
            QosClass::Burstable => self.resources.bandwidth_request,
            QosClass::BestEffort => 0.0,
        }
    }
}

/// QoS-aware task metadata
#[derive(Debug, Clone)]
pub struct QosTaskMeta {
    /// Base task metadata
    pub task: TaskMeta,
    /// QoS policy
    pub qos: QosPolicy,
}

impl QosTaskMeta {
    /// Create a new QoS task
    pub fn new(task: TaskMeta, qos: QosPolicy) -> Self {
        Self { task, qos }
    }

    /// Create a Guaranteed task
    pub fn guaranteed(id: String, name: String, memory: usize) -> Self {
        let task = TaskMeta::with_memory(id, name, memory);
        Self {
            task,
            qos: QosPolicy::guaranteed(memory),
        }
    }

    /// Create a Burstable task
    pub fn burstable(id: String, name: String, memory_request: usize, burst_ratio: f64) -> Self {
        let task = TaskMeta::with_memory(id, name, memory_request);
        Self {
            task,
            qos: QosPolicy::burstable(memory_request, burst_ratio),
        }
    }

    /// Create a BestEffort task
    pub fn best_effort(id: String, name: String) -> Self {
        let task = TaskMeta::new(id, name);
        Self {
            task,
            qos: QosPolicy::best_effort(),
        }
    }

    /// Get effective priority
    pub fn effective_priority(&self) -> i32 {
        self.qos.effective_priority(self.task.priority)
    }
}

/// QoS evaluation result
#[derive(Debug, Clone)]
pub enum QosEvaluation {
    /// Task should be admitted with the given QoS class
    Admit {
        class: QosClass,
        reserved_memory: usize,
        reserved_bandwidth: f64,
    },
    /// Task should be throttled (Burstable exceeding request)
    Throttle {
        class: QosClass,
        allowed_memory: usize,
        allowed_bandwidth: f64,
    },
    /// Task should be queued (BestEffort waiting)
    Queue {
        position: usize,
    },
    /// Task should be rejected
    Reject {
        reason: String,
    },
}

impl QosEvaluation {
    /// Check if admitted
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Admit { .. })
    }

    /// Check if throttled
    pub fn is_throttled(&self) -> bool {
        matches!(self, Self::Throttle { .. })
    }

    /// Check if queued
    pub fn is_queued(&self) -> bool {
        matches!(self, Self::Queue { .. })
    }

    /// Check if rejected
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Reject { .. })
    }
}

/// QoS policy evaluator
#[derive(Debug, Clone)]
pub struct QosPolicyEvaluator {
    /// Total system memory
    total_memory: usize,
    /// Total system bandwidth
    total_bandwidth: f64,
    /// Reserved memory for Guaranteed tasks
    guaranteed_memory: usize,
    /// Reserved bandwidth for Guaranteed tasks
    guaranteed_bandwidth: f64,
    /// Memory used by Burstable tasks
    burstable_memory: usize,
    /// Best-effort queue count
    best_effort_queue: usize,
}

impl QosPolicyEvaluator {
    /// Create a new evaluator
    pub fn new(total_memory: usize, total_bandwidth: f64) -> Self {
        Self {
            total_memory,
            total_bandwidth,
            guaranteed_memory: 0,
            guaranteed_bandwidth: 0.0,
            burstable_memory: 0,
            best_effort_queue: 0,
        }
    }

    /// Evaluate QoS policy for a task
    pub fn evaluate(&self, qos_task: &QosTaskMeta) -> QosEvaluation {
        let policy = &qos_task.qos;
        let available_memory = self.total_memory
            .saturating_sub(self.guaranteed_memory)
            .saturating_sub(self.burstable_memory);
        let available_bandwidth = (self.total_bandwidth - self.guaranteed_bandwidth).max(0.0);

        match policy.class {
            QosClass::Guaranteed => {
                // Guaranteed tasks need full resource reservation
                if policy.can_satisfy(available_memory, available_bandwidth) {
                    QosEvaluation::Admit {
                        class: QosClass::Guaranteed,
                        reserved_memory: policy.memory_to_reserve(),
                        reserved_bandwidth: policy.bandwidth_to_reserve(),
                    }
                } else {
                    QosEvaluation::Reject {
                        reason: format!(
                            "Insufficient resources for Guaranteed task: need {} bytes, {} bandwidth",
                            policy.resources.memory_request, policy.resources.bandwidth_request
                        ),
                    }
                }
            }
            QosClass::Burstable => {
                // Burstable tasks can be admitted with throttling
                if policy.resources.memory_request <= available_memory {
                    let allowed_memory = policy.resources.memory_limit.min(available_memory);
                    let allowed_bandwidth = policy.resources.bandwidth_limit.min(available_bandwidth);

                    if allowed_memory < policy.resources.memory_limit
                        || allowed_bandwidth < policy.resources.bandwidth_limit
                    {
                        QosEvaluation::Throttle {
                            class: QosClass::Burstable,
                            allowed_memory,
                            allowed_bandwidth,
                        }
                    } else {
                        QosEvaluation::Admit {
                            class: QosClass::Burstable,
                            reserved_memory: policy.memory_to_reserve(),
                            reserved_bandwidth: policy.bandwidth_to_reserve(),
                        }
                    }
                } else {
                    QosEvaluation::Reject {
                        reason: format!(
                            "Insufficient memory for Burstable task: need {} bytes",
                            policy.resources.memory_request
                        ),
                    }
                }
            }
            QosClass::BestEffort => {
                // BestEffort tasks are queued if resources unavailable
                if available_memory > 0 && available_bandwidth > 0.0 {
                    QosEvaluation::Admit {
                        class: QosClass::BestEffort,
                        reserved_memory: 0,
                        reserved_bandwidth: 0.0,
                    }
                } else {
                    QosEvaluation::Queue {
                        position: self.best_effort_queue,
                    }
                }
            }
        }
    }

    /// Reserve resources for an admitted task
    pub fn reserve(&mut self, evaluation: &QosEvaluation) {
        match evaluation {
            QosEvaluation::Admit {
                class,
                reserved_memory,
                reserved_bandwidth,
            } => match class {
                QosClass::Guaranteed => {
                    self.guaranteed_memory += reserved_memory;
                    self.guaranteed_bandwidth += reserved_bandwidth;
                }
                QosClass::Burstable => {
                    self.burstable_memory += reserved_memory;
                }
                QosClass::BestEffort => {
                    // No reservation for best-effort
                }
            },
            QosEvaluation::Throttle { .. } => {
                // Throttled tasks use limited resources
            }
            QosEvaluation::Queue { .. } => {
                self.best_effort_queue += 1;
            }
            QosEvaluation::Reject { .. } => {}
        }
    }

    /// Release resources when a task completes
    pub fn release(&mut self, class: QosClass, memory: usize, bandwidth: f64) {
        match class {
            QosClass::Guaranteed => {
                self.guaranteed_memory = self.guaranteed_memory.saturating_sub(memory);
                self.guaranteed_bandwidth = (self.guaranteed_bandwidth - bandwidth).max(0.0);
            }
            QosClass::Burstable => {
                self.burstable_memory = self.burstable_memory.saturating_sub(memory);
            }
            QosClass::BestEffort => {
                self.best_effort_queue = self.best_effort_queue.saturating_sub(1);
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> QosStats {
        QosStats {
            total_memory: self.total_memory,
            total_bandwidth: self.total_bandwidth,
            guaranteed_memory: self.guaranteed_memory,
            guaranteed_bandwidth: self.guaranteed_bandwidth,
            burstable_memory: self.burstable_memory,
            best_effort_queue: self.best_effort_queue,
            available_memory: self.total_memory
                .saturating_sub(self.guaranteed_memory)
                .saturating_sub(self.burstable_memory),
            available_bandwidth: (self.total_bandwidth - self.guaranteed_bandwidth).max(0.0),
        }
    }

    /// Reset evaluator state
    pub fn reset(&mut self) {
        self.guaranteed_memory = 0;
        self.guaranteed_bandwidth = 0.0;
        self.burstable_memory = 0;
        self.best_effort_queue = 0;
    }
}

/// QoS statistics
#[derive(Debug, Clone, Default)]
pub struct QosStats {
    /// Total system memory
    pub total_memory: usize,
    /// Total system bandwidth
    pub total_bandwidth: f64,
    /// Memory reserved for Guaranteed tasks
    pub guaranteed_memory: usize,
    /// Bandwidth reserved for Guaranteed tasks
    pub guaranteed_bandwidth: f64,
    /// Memory used by Burstable tasks
    pub burstable_memory: usize,
    /// Number of BestEffort tasks in queue
    pub best_effort_queue: usize,
    /// Available memory
    pub available_memory: usize,
    /// Available bandwidth
    pub available_bandwidth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qos_class_priority() {
        assert!(QosClass::Guaranteed.priority() > QosClass::Burstable.priority());
        assert!(QosClass::Burstable.priority() > QosClass::BestEffort.priority());
    }

    #[test]
    fn test_qos_class_preemption() {
        assert!(QosClass::Guaranteed.can_preempt(&QosClass::Burstable));
        assert!(QosClass::Guaranteed.can_preempt(&QosClass::BestEffort));
        assert!(QosClass::Burstable.can_preempt(&QosClass::BestEffort));
        assert!(!QosClass::BestEffort.can_preempt(&QosClass::Guaranteed));
    }

    #[test]
    fn test_guaranteed_task() {
        let mut evaluator = QosPolicyEvaluator::new(1000, 1.0);
        let task = QosTaskMeta::guaranteed("task-1".into(), "Test".into(), 500);

        let eval = evaluator.evaluate(&task);
        assert!(eval.is_admitted());

        if let QosEvaluation::Admit {
            class,
            reserved_memory,
            ..
        } = &eval
        {
            assert_eq!(*class, QosClass::Guaranteed);
            assert_eq!(*reserved_memory, 500);
        }

        evaluator.reserve(&eval);
        assert_eq!(evaluator.stats().guaranteed_memory, 500);
    }

    #[test]
    fn test_burstable_task() {
        let mut evaluator = QosPolicyEvaluator::new(1000, 1.0);
        let task = QosTaskMeta::burstable("task-1".into(), "Test".into(), 300, 2.0);

        let eval = evaluator.evaluate(&task);
        assert!(eval.is_admitted());

        evaluator.reserve(&eval);
        assert_eq!(evaluator.stats().burstable_memory, 300);
    }

    #[test]
    fn test_burstable_throttled() {
        let mut evaluator = QosPolicyEvaluator::new(500, 1.0);

        // First task uses some memory
        let task1 = QosTaskMeta::guaranteed("task-1".into(), "Test".into(), 200);
        let eval1 = evaluator.evaluate(&task1);
        evaluator.reserve(&eval1);

        // Second task requests 200 but can burst to 400
        let task2 = QosTaskMeta::burstable("task-2".into(), "Test".into(), 200, 2.0);
        let eval2 = evaluator.evaluate(&task2);

        // Should be throttled because burst limit (400) exceeds available (300)
        assert!(eval2.is_throttled());
    }

    #[test]
    fn test_best_effort_task() {
        let evaluator = QosPolicyEvaluator::new(1000, 1.0);
        let task = QosTaskMeta::best_effort("task-1".into(), "Test".into());

        let eval = evaluator.evaluate(&task);
        assert!(eval.is_admitted());

        if let QosEvaluation::Admit {
            reserved_memory, ..
        } = &eval
        {
            assert_eq!(*reserved_memory, 0); // No reservation
        }
    }

    #[test]
    fn test_best_effort_queued() {
        let evaluator = QosPolicyEvaluator::new(0, 0.0); // No resources
        let task = QosTaskMeta::best_effort("task-1".into(), "Test".into());

        let eval = evaluator.evaluate(&task);
        assert!(eval.is_queued());
    }

    #[test]
    fn test_guaranteed_reject() {
        let evaluator = QosPolicyEvaluator::new(100, 1.0);
        let task = QosTaskMeta::guaranteed("task-1".into(), "Test".into(), 500);

        let eval = evaluator.evaluate(&task);
        assert!(eval.is_rejected());
    }

    #[test]
    fn test_effective_priority() {
        let guaranteed = QosTaskMeta::guaranteed("g".into(), "G".into(), 100);
        let burstable = QosTaskMeta::burstable("b".into(), "B".into(), 100, 1.5);
        let best_effort = QosTaskMeta::best_effort("e".into(), "E".into());

        assert!(guaranteed.effective_priority() > burstable.effective_priority());
        assert!(burstable.effective_priority() > best_effort.effective_priority());
    }
}
