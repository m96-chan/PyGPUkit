//! Task scheduler module
//!
//! Provides task scheduling with:
//! - Priority-based task execution
//! - Bandwidth pacing
//! - Memory reservation tracking
//! - Admission control
//! - QoS policy framework
//! - GPU resource partitioning
//! - Multi-LLM execution contexts

mod task;
mod core;
mod admission;
mod qos;
mod partition;
mod async_exec;
mod execution_context;
mod dispatch_controller;

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
pub use async_exec::{
    FutureState, KernelFuture, KernelResult, AsyncKernelRequest, AsyncExecStats, AsyncExecutor,
};
pub use execution_context::{ExecutionContext, ContextState, ContextStats};
pub use dispatch_controller::{MultiLLMController, ControllerStats};
