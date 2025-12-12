"""Scheduler module for PyGPUkit.

Provides Kubernetes-style GPU task scheduling with:
- Memory reservation
- Bandwidth pacing
- QoS policies (Guaranteed, Burstable, BestEffort)
"""

from pygpukit.scheduler.core import (
    Scheduler,
    Task,
    TaskPolicy,
    TaskState,
)

# Rust scheduler (v0.2+)
# Import Rust implementation if available
try:
    import _pygpukit_rust._pygpukit_rust as _rust

    RustScheduler = _rust.Scheduler
    RustTaskMeta = _rust.TaskMeta
    RustTaskState = _rust.scheduler.TaskState
    RustTaskPolicy = _rust.scheduler.TaskPolicy
    RustSchedulerStats = _rust.SchedulerStats
    RustTaskStats = _rust.TaskStats
    HAS_RUST_BACKEND = True
except ImportError:
    RustScheduler = None  # type: ignore
    RustTaskMeta = None  # type: ignore
    RustTaskState = None  # type: ignore
    RustTaskPolicy = None  # type: ignore
    RustSchedulerStats = None  # type: ignore
    RustTaskStats = None  # type: ignore
    HAS_RUST_BACKEND = False

__all__ = [
    "Scheduler",
    "Task",
    "TaskPolicy",
    "TaskState",
    # Rust backend (v0.2+)
    "RustScheduler",
    "RustTaskMeta",
    "RustTaskState",
    "RustTaskPolicy",
    "RustSchedulerStats",
    "RustTaskStats",
    "HAS_RUST_BACKEND",
]
