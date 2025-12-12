//! Async Memory Transfer Engine
//!
//! Provides asynchronous memory transfer operations with:
//! - Separate compute and memcpy streams
//! - H2D, D2H, and D2D transfer types
//! - Stream synchronization model
//! - Integration with the scheduler tick loop
//!
//! Note: This module provides the Rust-side coordination logic.
//! Actual CUDA stream operations are handled by the C++ backend via callbacks.

mod operation;
mod engine;

pub use operation::{TransferType, TransferOp, TransferState};
pub use engine::{AsyncTransferEngine, StreamType, TransferStats, TransferCallback};
