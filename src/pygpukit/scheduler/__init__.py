"""Scheduler module for PyGPUkit.

Provides Kubernetes-style GPU task scheduling with:
- Memory reservation
- Bandwidth pacing
- QoS policies (Guaranteed, Burstable, BestEffort)
- Multi-LLM execution contexts (v0.2.6+)
"""

from pygpukit.scheduler.core import (
    Scheduler,
    Task,
    TaskPolicy,
    TaskState,
)

# Multi-LLM execution context API (v0.2.6+)
from pygpukit.scheduler.execution import (
    GB,
    HAS_MULTI_LLM,
    KB,
    MB,
    # Async execution (v0.2.6+)
    AsyncKernelRequest,
    ContextStats,
    ExecutionContext,
    KernelFuture,
    KernelResult,
    SchedulerStats,
    context_session,
    create_context,
    destroy_context,
    get_context,
    initialize,
    is_session_active,
    list_contexts,
    reset,
    session,
    stats,
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
    # Multi-LLM execution context API (v0.2.6+)
    "KB",
    "MB",
    "GB",
    "initialize",
    "create_context",
    "get_context",
    "destroy_context",
    "list_contexts",
    "session",
    "context_session",
    "is_session_active",
    "stats",
    "reset",
    "ExecutionContext",
    "ContextStats",
    "SchedulerStats",
    # Async execution (v0.2.6+)
    "AsyncKernelRequest",
    "KernelFuture",
    "KernelResult",
    "HAS_MULTI_LLM",
]
