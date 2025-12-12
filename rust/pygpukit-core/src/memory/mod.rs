//! Memory management module
//!
//! Provides GPU memory pool with:
//! - Size-class based allocation
//! - LRU eviction policy
//! - Thread-safe operations

mod block;
mod pool;
mod size_class;

pub use block::MemoryBlock;
pub use pool::{MemoryPool, PoolStats, MemoryError};
pub use size_class::{SIZE_CLASSES, get_size_class};
