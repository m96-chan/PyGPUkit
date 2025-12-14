//! PyGPUkit Core - Rust implementation of memory pool, scheduler, transfer engine, and kernel dispatcher
//!
//! This crate provides the core data structures and algorithms for:
//! - GPU memory pool with LRU eviction
//! - Task scheduler with bandwidth pacing
//! - Async memory transfer engine with separate streams
//! - Kernel dispatch controller with stream management
//! - Kernel pacing engine with bandwidth control
//! - Device capabilities and kernel type selection

pub mod memory;
pub mod scheduler;
pub mod transfer;
pub mod dispatch;
pub mod device;

pub use memory::{MemoryBlock, MemoryPool, PoolStats, MemoryError};
pub use scheduler::{
    TaskState, TaskPolicy, TaskMeta, Scheduler, SchedulerStats, TaskStats,
    AdmissionController, AdmissionConfig, AdmissionDecision, AdmissionStats, RejectReason,
    QosClass, QosPolicy, QosTaskMeta, QosEvaluation, QosPolicyEvaluator, QosStats, ResourceRequirements,
    PartitionManager, PartitionConfig, Partition, PartitionLimits, PartitionUsage, PartitionStats, PartitionError,
};
pub use transfer::{
    TransferType, TransferOp, TransferState, AsyncTransferEngine, StreamType, TransferStats,
    PinnedMemoryManager, PinnedPoolConfig, PinnedBlock, PinnedStats, PinnedError,
};
pub use dispatch::{
    KernelDispatcher, KernelLaunchRequest, KernelState, DispatchStats, LaunchConfig,
    KernelPacingEngine, PacingConfig, PacingDecision, PacingStats, StreamPacingStats,
    SliceScheduler, SliceConfig, SlicedKernel, KernelSlice, SliceInfo, SliceStats,
    KernelCache, CacheConfig, CachedKernel, CompileOptions, CacheStats,
};
pub use device::{KernelType, DeviceCapabilities};
