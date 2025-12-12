//! PyGPUkit Core - Rust implementation of memory pool, scheduler, transfer engine, and kernel dispatcher
//!
//! This crate provides the core data structures and algorithms for:
//! - GPU memory pool with LRU eviction
//! - Task scheduler with bandwidth pacing
//! - Async memory transfer engine with separate streams
//! - Kernel dispatch controller with stream management

pub mod memory;
pub mod scheduler;
pub mod transfer;
pub mod dispatch;

pub use memory::{MemoryBlock, MemoryPool, PoolStats, MemoryError};
pub use scheduler::{TaskState, TaskPolicy, TaskMeta, Scheduler, SchedulerStats, TaskStats};
pub use transfer::{TransferType, TransferOp, TransferState, AsyncTransferEngine, StreamType, TransferStats};
pub use dispatch::{KernelDispatcher, KernelLaunchRequest, KernelState, DispatchStats, LaunchConfig};
