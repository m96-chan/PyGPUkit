//! Kernel Dispatch Controller
//!
//! Provides coordination for GPU kernel launches with:
//! - Per-task stream assignment
//! - Integration with the scheduler tick loop
//! - Kernel execution tracking
//! - Bandwidth-based kernel pacing
//! - Micro-slicing for fairness and latency
//! - Kernel caching for compiled PTX
//!
//! Note: Actual CUDA Driver API calls (cuLaunchKernel) are handled by C++ backend.
//! This module provides the Rust-side coordination logic.

mod controller;
mod pacing;
mod slicing;
mod cache;

pub use controller::{KernelDispatcher, KernelLaunchRequest, KernelState, DispatchStats, LaunchConfig};
pub use pacing::{
    KernelPacingEngine, PacingConfig, PacingDecision, PacingStats, StreamPacingStats,
};
pub use slicing::{
    SliceScheduler, SliceConfig, SlicedKernel, KernelSlice, SliceInfo, SliceStats,
};
pub use cache::{
    KernelCache, CacheConfig, CachedKernel, CompileOptions, CacheStats,
};
