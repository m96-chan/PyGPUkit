//! Unified error handling for PyGPUkit Python bindings
//!
//! This module provides helper functions for consistent error conversion
//! from Rust errors to Python exceptions.
//!
//! Error convention:
//! - MemoryError: Resource exhaustion (quota exceeded, allocation failures)
//! - ValueError: Invalid arguments (bad IDs, not found)
//! - RuntimeError: Operation failures (eviction, state errors)

use pyo3::exceptions::{PyRuntimeError, PyValueError, PyMemoryError};
use pyo3::PyErr;
use pygpukit_core::memory::MemoryError;
use pygpukit_core::scheduler::PartitionError;
use pygpukit_core::transfer::PinnedError;

/// Convert MemoryError to PyErr
pub fn memory_error_to_py(err: MemoryError) -> PyErr {
    match err {
        MemoryError::QuotaExceeded { requested, used, quota } => {
            PyMemoryError::new_err(format!(
                "Memory quota exceeded: requested {} bytes, {} used, {} quota",
                requested, used, quota
            ))
        }
        MemoryError::InvalidBlock(id) => {
            PyValueError::new_err(format!("Invalid memory block ID: {}", id))
        }
        MemoryError::BlockEvicted(id) => {
            PyRuntimeError::new_err(format!("Memory block {} was evicted", id))
        }
    }
}

/// Convert PartitionError to PyErr
pub fn partition_error_to_py(err: PartitionError) -> PyErr {
    match err {
        PartitionError::NotFound { id } => {
            PyValueError::new_err(format!("Partition not found: {}", id))
        }
        PartitionError::AlreadyExists { id } => {
            PyValueError::new_err(format!("Partition already exists: {}", id))
        }
        PartitionError::InsufficientResources { resource, requested, available } => {
            PyRuntimeError::new_err(format!(
                "Insufficient {} for partition: requested {}, {} available",
                resource, requested, available
            ))
        }
        PartitionError::NotAllowed { reason } => {
            PyRuntimeError::new_err(format!("Operation not allowed: {}", reason))
        }
    }
}

/// Convert PinnedError to PyErr
pub fn pinned_error_to_py(err: PinnedError) -> PyErr {
    match err {
        PinnedError::QuotaExceeded { requested, available } => {
            PyMemoryError::new_err(format!(
                "Pinned memory quota exceeded: requested {} bytes, {} available",
                requested, available
            ))
        }
        PinnedError::InvalidBlock { id } => {
            PyValueError::new_err(format!("Pinned memory block not found: {}", id))
        }
        PinnedError::AllocationFailed { reason } => {
            PyMemoryError::new_err(format!("Pinned memory allocation failed: {}", reason))
        }
    }
}
