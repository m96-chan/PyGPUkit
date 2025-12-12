//! Task scheduler module
//!
//! Provides task scheduling with:
//! - Priority-based task execution
//! - Bandwidth pacing
//! - Memory reservation tracking
//! - Admission control
//! - QoS policy framework
//! - GPU resource partitioning

mod task;
mod core;
mod admission;
mod qos;
mod partition;

pub use task::{TaskState, TaskPolicy, TaskMeta, TaskStats};
pub use core::{Scheduler, SchedulerStats};
pub use admission::{
    AdmissionController, AdmissionConfig, AdmissionDecision,
    AdmissionStats, RejectReason,
};
pub use qos::{
    QosClass, QosPolicy, QosTaskMeta, QosEvaluation,
    QosPolicyEvaluator, QosStats, ResourceRequirements,
};
pub use partition::{
    PartitionManager, PartitionConfig, Partition, PartitionLimits,
    PartitionUsage, PartitionStats, PartitionError,
};
