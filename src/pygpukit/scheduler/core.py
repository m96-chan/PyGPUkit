"""Scheduler Core implementation for PyGPUkit.

This module provides a Kubernetes-style GPU task scheduler with:
- Task registration and management
- Memory/bandwidth reservation
- Scheduling loop with pacing
- Task state management
- Rust backend for high-performance scheduling (v0.2+)

Note: CUDA does not provide native scheduling features. Everything is
implemented via host-side scheduling and kernel structuring.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

# Check if Rust backend should be used
_USE_RUST = os.environ.get("PYGPUKIT_USE_RUST", "1").lower() in ("1", "true", "yes")

# Try to import Rust backend
_RUST_AVAILABLE = False
_rust_module: Any = None
try:
    import _pygpukit_rust._pygpukit_rust as _rust_module  # noqa: F811

    _RUST_AVAILABLE = True
except ImportError:
    pass


class TaskState(Enum):
    """Task execution state."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class TaskPolicy(Enum):
    """QoS policy for task scheduling.

    - GUARANTEED: Hard memory and bandwidth guarantees (highest priority)
    - BURSTABLE: Hard memory, soft bandwidth (may throttle)
    - BEST_EFFORT: Soft memory and bandwidth (uses leftovers)
    """

    GUARANTEED = auto()
    BURSTABLE = auto()
    BEST_EFFORT = auto()


@dataclass
class Task:
    """Represents a GPU task in the scheduler.

    Attributes:
        fn: The function to execute
        memory: Memory reservation in bytes (None = no reservation)
        bandwidth: Bandwidth quota as fraction 0.0-1.0 (None = no limit)
        policy: QoS policy for scheduling
        state: Current task state
        id: Unique task identifier
    """

    fn: Callable[[], Any]
    memory: int | None = None
    bandwidth: float | None = None
    policy: TaskPolicy = TaskPolicy.BEST_EFFORT
    state: TaskState = TaskState.PENDING
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Execution tracking
    last_launch: float = 0.0
    execution_count: int = 0
    pacing_delay_count: int = 0

    def touch(self) -> None:
        """Update last launch time."""
        self.last_launch = time.time()


class Scheduler:
    """GPU Task Scheduler with Kubernetes-style resource control.

    Features:
    - Task submission and management
    - Memory reservation tracking
    - Bandwidth pacing (time-based throttling)
    - FIFO task execution
    - Thread-safe operations
    - Rust backend for high-performance scheduling (v0.2+)

    Attributes:
        sched_tick_ms: Scheduler tick interval in milliseconds
        window_ms: Scheduling window for bandwidth calculation
        total_memory: Total GPU memory available for scheduling

    Environment Variables:
        PYGPUKIT_USE_RUST: Set to "0" to disable Rust backend (default: "1")
    """

    def __init__(
        self,
        sched_tick_ms: float = 1.0,
        window_ms: float = 10.0,
        total_memory: int | None = None,
    ):
        """Initialize the scheduler.

        Args:
            sched_tick_ms: Scheduler tick interval (default 1ms)
            window_ms: Scheduling window for bandwidth (default 10ms)
            total_memory: Total memory available (None = unlimited)
        """
        self._sched_tick_ms = sched_tick_ms
        self._window_ms = window_ms
        self._total_memory = total_memory

        self._lock = threading.RLock()

        # Use Rust backend if available and enabled
        self._use_rust = _USE_RUST and _RUST_AVAILABLE
        self._rust_scheduler = None
        if self._use_rust:
            self._rust_scheduler = _rust_module.Scheduler(total_memory, sched_tick_ms, window_ms)

        # Python-side storage for callable functions (Rust doesn't store closures)
        # Maps task_id -> callable
        self._task_functions: dict[str, Callable[[], Any]] = {}

        # Task storage (Python Task objects with all metadata)
        self._tasks: dict[str, Task] = {}

        # Pure Python fallback storage (only used when Rust is disabled)
        if not self._use_rust:
            self._pending_queue: deque[str] = deque()

            # Resource tracking
            self._reserved_memory = 0
            self._completed_count = 0

    @property
    def task_count(self) -> int:
        """Total number of tasks."""
        with self._lock:
            if self._use_rust:
                return self._rust_scheduler.stats().total_submitted
            return len(self._tasks)

    @property
    def completed_count(self) -> int:
        """Number of completed tasks."""
        with self._lock:
            if self._use_rust:
                return self._rust_scheduler.stats().completed_count
            return self._completed_count

    @property
    def reserved_memory(self) -> int:
        """Currently reserved memory in bytes."""
        with self._lock:
            if self._use_rust:
                return self._rust_scheduler.stats().reserved_memory
            return self._reserved_memory

    @property
    def available_memory(self) -> int:
        """Available memory in bytes."""
        with self._lock:
            if self._use_rust:
                return self._rust_scheduler.stats().available_memory
            if self._total_memory is None:
                return 0
            return self._total_memory - self._reserved_memory

    def submit(
        self,
        fn: Callable[[], Any],
        memory: int | None = None,
        bandwidth: float | None = None,
        policy: str | TaskPolicy = "best_effort",
    ) -> str:
        """Submit a task to the scheduler.

        Args:
            fn: Function to execute
            memory: Memory reservation in bytes
            bandwidth: Bandwidth quota (0.0-1.0)
            policy: QoS policy ("guaranteed", "burstable", "best_effort")

        Returns:
            Task ID for tracking
        """
        # Convert string policy to enum
        if isinstance(policy, str):
            policy_map = {
                "guaranteed": TaskPolicy.GUARANTEED,
                "burstable": TaskPolicy.BURSTABLE,
                "best_effort": TaskPolicy.BEST_EFFORT,
            }
            policy = policy_map.get(policy.lower(), TaskPolicy.BEST_EFFORT)

        task = Task(
            fn=fn,
            memory=memory,
            bandwidth=bandwidth,
            policy=policy,
        )

        if self._use_rust:
            return self._submit_rust(task)
        return self._submit_python(task)

    def _submit_rust(self, task: Task) -> str:
        """Submit task using Rust backend."""
        with self._lock:
            # Create Rust TaskMeta
            rust_task = _rust_module.TaskMeta(
                task.id,
                f"Task-{task.id}",
                task.memory or 0,
            )

            # Submit to Rust scheduler
            self._rust_scheduler.submit(rust_task)

            # Store function on Python side
            self._task_functions[task.id] = task.fn
            self._tasks[task.id] = task

        return task.id

    def _submit_python(self, task: Task) -> str:
        """Submit task using pure Python backend."""
        with self._lock:
            self._tasks[task.id] = task
            self._pending_queue.append(task.id)

            # Reserve memory
            if task.memory is not None:
                self._reserved_memory += task.memory

        return task.id

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)

    def step(self) -> None:
        """Execute one scheduler tick.

        This method should be called repeatedly in the main loop.
        It processes pending tasks respecting pacing constraints.
        """
        if self._use_rust:
            self._step_rust()
        else:
            self._step_python()

    def _step_rust(self) -> None:
        """Execute scheduler tick using Rust backend."""
        with self._lock:
            # Get runnable tasks from Rust
            runnable_ids = self._rust_scheduler.get_runnable_tasks(10)

            # Start tasks in Rust
            for task_id in runnable_ids:
                self._rust_scheduler.start_task(task_id)

        # Execute tasks outside the lock
        for task_id in runnable_ids:
            self._execute_task_rust(task_id)

    def _step_python(self) -> None:
        """Execute scheduler tick using pure Python backend."""
        now = time.time()

        with self._lock:
            # Process pending tasks
            tasks_to_run: list[Task] = []

            for task_id in list(self._pending_queue):
                task = self._tasks.get(task_id)
                if task is None:
                    self._pending_queue.remove(task_id)
                    continue

                if task.state == TaskState.PENDING:
                    task.state = TaskState.RUNNING

                if task.state == TaskState.RUNNING:
                    if self.should_run(task, now):
                        tasks_to_run.append(task)

        # Execute tasks outside the lock
        for task in tasks_to_run:
            self._execute_task(task)

    def should_run(self, task: Task, now: float) -> bool:
        """Determine if a task should run based on pacing.

        Args:
            task: Task to check
            now: Current timestamp

        Returns:
            True if task should run, False if paced
        """
        if task.state != TaskState.RUNNING:
            return False

        if task.bandwidth is None:
            return True

        # Calculate pacing interval based on bandwidth
        # bandwidth = allowed_time / window
        # pacing_interval = window - allowed_time = window * (1 - bandwidth)
        window_sec = self._window_ms / 1000.0
        pacing_interval = window_sec * (1.0 - task.bandwidth)

        elapsed = now - task.last_launch
        if elapsed < pacing_interval:
            with self._lock:
                task.pacing_delay_count += 1
            return False

        return True

    def _execute_task_rust(self, task_id: str) -> None:
        """Execute a task using Rust backend.

        Args:
            task_id: ID of the task to execute
        """
        fn = self._task_functions.get(task_id)
        task = self._tasks.get(task_id)

        if fn is None or task is None:
            return

        try:
            task.touch()
            fn()
            task.execution_count += 1

            # Mark as completed in Rust
            with self._lock:
                self._rust_scheduler.complete_task(task_id)
                task.state = TaskState.COMPLETED

        except Exception as e:
            with self._lock:
                self._rust_scheduler.fail_task(task_id, str(e))
                task.state = TaskState.FAILED

    def _execute_task(self, task: Task) -> None:
        """Execute a task's function (pure Python backend).

        Args:
            task: Task to execute
        """
        try:
            task.touch()
            task.fn()
            task.execution_count += 1

            # Mark as completed (single-shot tasks)
            with self._lock:
                task.state = TaskState.COMPLETED
                self._completed_count += 1

                # Release memory
                if task.memory is not None:
                    self._reserved_memory -= task.memory

                # Remove from pending queue
                if task.id in self._pending_queue:
                    self._pending_queue.remove(task.id)

        except Exception:
            with self._lock:
                task.state = TaskState.FAILED
                if task.id in self._pending_queue:
                    self._pending_queue.remove(task.id)

    def stats(self, task_id: str) -> dict[str, Any]:
        """Get statistics for a task.

        Args:
            task_id: Task identifier

        Returns:
            Dictionary with task statistics
        """
        with self._lock:
            if self._use_rust:
                rust_stats = self._rust_scheduler.task_stats(task_id)
                if rust_stats is None:
                    return {}
                task = self._tasks.get(task_id)
                # Convert Rust state to lowercase string
                state = rust_stats.state
                if hasattr(state, "name"):
                    state_str = state.name.lower()
                else:
                    # Handle PyTaskState enum (e.g., TaskState.Completed -> "completed")
                    state_str = str(state).split(".")[-1].lower()
                return {
                    "id": rust_stats.id,
                    "state": state_str,
                    "memory": task.memory if task else rust_stats.memory_used,
                    "bandwidth": task.bandwidth if task else None,
                    "policy": task.policy.name.lower() if task else "best_effort",
                    "execution_count": task.execution_count if task else 0,
                    "pacing_delay_count": task.pacing_delay_count if task else 0,
                    "last_launch": task.last_launch if task else 0.0,
                    "wait_time": rust_stats.wait_time,
                    "exec_time": rust_stats.exec_time,
                }

            task = self._tasks.get(task_id)
            if task is None:
                return {}

            return {
                "id": task.id,
                "state": task.state.name.lower(),
                "memory": task.memory,
                "bandwidth": task.bandwidth,
                "policy": task.policy.name.lower(),
                "execution_count": task.execution_count,
                "pacing_delay_count": task.pacing_delay_count,
                "last_launch": task.last_launch,
            }

    def global_stats(self) -> dict[str, Any]:
        """Get global scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        with self._lock:
            if self._use_rust:
                rust_stats = self._rust_scheduler.stats()
                return {
                    "task_count": rust_stats.total_submitted,
                    "pending_count": rust_stats.pending_count,
                    "running_count": rust_stats.running_count,
                    "completed_count": rust_stats.completed_count,
                    "failed_count": rust_stats.failed_count,
                    "reserved_memory": rust_stats.reserved_memory,
                    "available_memory": rust_stats.available_memory,
                    "total_memory": self._total_memory,
                    "sched_tick_ms": self._sched_tick_ms,
                    "window_ms": self._window_ms,
                    "avg_wait_time": rust_stats.avg_wait_time,
                    "avg_exec_time": rust_stats.avg_exec_time,
                }

            pending = sum(1 for t in self._tasks.values() if t.state == TaskState.PENDING)
            running = sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING)
            completed = sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED)

            return {
                "task_count": len(self._tasks),
                "pending_count": pending,
                "running_count": running,
                "completed_count": completed,
                "reserved_memory": self._reserved_memory,
                "total_memory": self._total_memory,
                "sched_tick_ms": self._sched_tick_ms,
                "window_ms": self._window_ms,
            }
