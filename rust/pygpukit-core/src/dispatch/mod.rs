//! Kernel Dispatch Controller
//!
//! Provides coordination for GPU kernel launches with:
//! - Per-task stream assignment
//! - Integration with the scheduler tick loop
//! - Kernel execution tracking
//!
//! Note: Actual CUDA Driver API calls (cuLaunchKernel) are handled by C++ backend.
//! This module provides the Rust-side coordination logic.

mod controller;

pub use controller::{KernelDispatcher, KernelLaunchRequest, KernelState, DispatchStats, LaunchConfig};
