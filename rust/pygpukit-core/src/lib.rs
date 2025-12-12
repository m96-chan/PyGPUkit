//! PyGPUkit Core - Rust implementation of memory pool and scheduler
//!
//! This crate provides the core data structures and algorithms for:
//! - GPU memory pool with LRU eviction
//! - Task scheduler with bandwidth pacing

pub mod memory;
pub mod scheduler;

pub use memory::{MemoryBlock, MemoryPool, PoolStats, MemoryError};
pub use scheduler::{TaskState, TaskPolicy, TaskMeta, Scheduler, SchedulerStats, TaskStats};
