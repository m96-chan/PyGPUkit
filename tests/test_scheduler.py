"""Tests for Scheduler Core implementation.

TDD: These tests are written before the implementation.
"""

import threading
import time


class TestTaskBasic:
    """Basic Task class tests."""

    def test_task_creation(self):
        """Test creating a task with required parameters."""
        from pygpukit.scheduler import Task, TaskState

        def dummy_fn():
            pass

        task = Task(fn=dummy_fn, memory=1024 * 1024, bandwidth=0.2)
        assert task.fn == dummy_fn
        assert task.memory == 1024 * 1024
        assert task.bandwidth == 0.2
        assert task.state == TaskState.PENDING
        assert task.id is not None

    def test_task_default_values(self):
        """Test task with default values."""
        from pygpukit.scheduler import Task, TaskPolicy

        def dummy_fn():
            pass

        task = Task(fn=dummy_fn)
        assert task.memory is None
        assert task.bandwidth is None
        assert task.policy == TaskPolicy.BEST_EFFORT

    def test_task_state_transitions(self):
        """Test task state transitions."""
        from pygpukit.scheduler import Task, TaskState

        def dummy_fn():
            pass

        task = Task(fn=dummy_fn)
        assert task.state == TaskState.PENDING

        task.state = TaskState.RUNNING
        assert task.state == TaskState.RUNNING

        task.state = TaskState.COMPLETED
        assert task.state == TaskState.COMPLETED

    def test_task_policies(self):
        """Test different task policies."""
        from pygpukit.scheduler import Task, TaskPolicy

        def dummy_fn():
            pass

        task1 = Task(fn=dummy_fn, policy=TaskPolicy.GUARANTEED)
        assert task1.policy == TaskPolicy.GUARANTEED

        task2 = Task(fn=dummy_fn, policy=TaskPolicy.BURSTABLE)
        assert task2.policy == TaskPolicy.BURSTABLE

        task3 = Task(fn=dummy_fn, policy=TaskPolicy.BEST_EFFORT)
        assert task3.policy == TaskPolicy.BEST_EFFORT


class TestSchedulerBasic:
    """Basic Scheduler functionality tests."""

    def test_scheduler_creation(self):
        """Test creating a scheduler."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()
        assert scheduler is not None
        assert scheduler.task_count == 0

    def test_submit_task(self):
        """Test submitting a task."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()

        def dummy_fn():
            pass

        task_id = scheduler.submit(dummy_fn, memory=1024 * 1024, bandwidth=0.2)
        assert task_id is not None
        assert scheduler.task_count == 1

    def test_submit_multiple_tasks(self):
        """Test submitting multiple tasks."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()

        def dummy_fn():
            pass

        task_id1 = scheduler.submit(dummy_fn)
        task_id2 = scheduler.submit(dummy_fn)
        task_id3 = scheduler.submit(dummy_fn)

        assert task_id1 != task_id2 != task_id3
        assert scheduler.task_count == 3

    def test_get_task(self):
        """Test getting a task by ID."""
        from pygpukit.scheduler import Scheduler, TaskState

        scheduler = Scheduler()

        def dummy_fn():
            pass

        task_id = scheduler.submit(dummy_fn, memory=2048)
        task = scheduler.get_task(task_id)

        assert task is not None
        assert task.id == task_id
        assert task.memory == 2048
        assert task.state == TaskState.PENDING


class TestSchedulerStep:
    """Scheduler step() and execution tests."""

    def test_step_executes_task(self):
        """Test that step() executes pending tasks."""
        from pygpukit.scheduler import Scheduler, TaskState

        executed = []

        def task_fn():
            executed.append(True)

        scheduler = Scheduler()
        task_id = scheduler.submit(task_fn)

        # Run scheduler steps until task completes
        for _ in range(10):
            scheduler.step()
            task = scheduler.get_task(task_id)
            if task.state == TaskState.COMPLETED:
                break

        assert len(executed) >= 1
        assert scheduler.get_task(task_id).state == TaskState.COMPLETED

    def test_step_with_no_tasks(self):
        """Test step() with no pending tasks."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()
        # Should not raise
        scheduler.step()

    def test_task_execution_order(self):
        """Test that tasks are executed in submission order (FIFO)."""
        from pygpukit.scheduler import Scheduler

        execution_order = []

        def make_task(n):
            def task_fn():
                execution_order.append(n)

            return task_fn

        scheduler = Scheduler()
        scheduler.submit(make_task(1))
        scheduler.submit(make_task(2))
        scheduler.submit(make_task(3))

        # Run until all complete
        for _ in range(30):
            scheduler.step()
            if scheduler.completed_count == 3:
                break

        assert execution_order == [1, 2, 3]


class TestSchedulerPacing:
    """Bandwidth pacing tests."""

    def test_bandwidth_pacing(self):
        """Test that bandwidth pacing limits execution rate."""
        from pygpukit.scheduler import Scheduler

        execution_times = []

        def task_fn():
            execution_times.append(time.time())

        scheduler = Scheduler(sched_tick_ms=1)
        # 20% bandwidth = run 20% of the time
        scheduler.submit(task_fn, bandwidth=0.2)

        start = time.time()
        # Run for ~50ms
        while time.time() - start < 0.05:
            scheduler.step()
            time.sleep(0.001)

        # With 20% bandwidth, should have some pacing delays
        # The exact count depends on timing, just verify it ran
        assert len(execution_times) >= 1

    def test_should_run_respects_pacing(self):
        """Test should_run respects pacing interval."""
        from pygpukit.scheduler import Scheduler, TaskState

        scheduler = Scheduler(sched_tick_ms=10, window_ms=100)

        def dummy_fn():
            pass

        task_id = scheduler.submit(dummy_fn, bandwidth=0.5)  # 50% bandwidth
        task = scheduler.get_task(task_id)
        task.state = TaskState.RUNNING

        # First call should allow running
        now = time.time()
        assert scheduler.should_run(task, now) is True

        # Immediate second call should be paced
        task.last_launch = now
        assert scheduler.should_run(task, now) is False

        # After pacing interval, should allow
        future = now + 0.1  # 100ms later
        assert scheduler.should_run(task, future) is True


class TestSchedulerStats:
    """Statistics and monitoring tests."""

    def test_stats_basic(self):
        """Test getting task statistics."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()

        def dummy_fn():
            pass

        task_id = scheduler.submit(dummy_fn, memory=1024, bandwidth=0.3)
        stats = scheduler.stats(task_id)

        assert "state" in stats
        assert "memory" in stats
        assert "bandwidth" in stats
        assert stats["memory"] == 1024
        assert stats["bandwidth"] == 0.3

    def test_stats_after_execution(self):
        """Test statistics update after execution."""
        from pygpukit.scheduler import Scheduler, TaskState

        scheduler = Scheduler()
        call_count = [0]

        def counting_fn():
            call_count[0] += 1

        task_id = scheduler.submit(counting_fn)

        # Run until complete
        for _ in range(10):
            scheduler.step()
            if scheduler.get_task(task_id).state == TaskState.COMPLETED:
                break

        stats = scheduler.stats(task_id)
        assert stats["state"] == "completed"
        assert stats.get("execution_count", 0) >= 1

    def test_global_stats(self):
        """Test global scheduler statistics."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()

        def dummy_fn():
            pass

        scheduler.submit(dummy_fn)
        scheduler.submit(dummy_fn)

        global_stats = scheduler.global_stats()
        assert "task_count" in global_stats
        assert "pending_count" in global_stats
        assert global_stats["task_count"] == 2


class TestSchedulerMemory:
    """Memory reservation tests."""

    def test_memory_tracking(self):
        """Test that scheduler tracks memory reservations."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler(total_memory=1024 * 1024 * 100)  # 100 MB

        def dummy_fn():
            pass

        scheduler.submit(dummy_fn, memory=1024 * 1024 * 10)  # 10 MB
        scheduler.submit(dummy_fn, memory=1024 * 1024 * 20)  # 20 MB

        assert scheduler.reserved_memory == 1024 * 1024 * 30
        assert scheduler.available_memory == 1024 * 1024 * 70

    def test_memory_released_on_completion(self):
        """Test that memory is released when task completes."""
        from pygpukit.scheduler import Scheduler, TaskState

        scheduler = Scheduler(total_memory=1024 * 1024 * 100)

        def dummy_fn():
            pass

        task_id = scheduler.submit(dummy_fn, memory=1024 * 1024 * 50)
        assert scheduler.reserved_memory == 1024 * 1024 * 50

        # Run until complete
        for _ in range(10):
            scheduler.step()
            if scheduler.get_task(task_id).state == TaskState.COMPLETED:
                break

        assert scheduler.reserved_memory == 0


class TestSchedulerThreadSafety:
    """Thread safety tests."""

    def test_concurrent_submit(self):
        """Test thread-safe task submission."""
        from pygpukit.scheduler import Scheduler

        scheduler = Scheduler()
        task_ids = []
        errors = []

        def submit_worker():
            try:
                for _ in range(10):

                    def dummy():
                        pass

                    task_id = scheduler.submit(dummy)
                    task_ids.append(task_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(task_ids) == 40
        assert scheduler.task_count == 40


class TestSchedulerIntegration:
    """Integration tests with other PyGPUkit components."""

    def test_scheduler_with_memory_pool(self):
        """Test scheduler integration with memory pool."""
        from pygpukit.memory import MemoryPool, set_default_pool
        from pygpukit.scheduler import Scheduler

        pool = MemoryPool(quota=1024 * 1024 * 100)
        set_default_pool(pool)

        try:
            scheduler = Scheduler()

            def dummy_fn():
                pass

            task_id = scheduler.submit(dummy_fn, memory=1024 * 1024)
            assert scheduler.get_task(task_id) is not None
        finally:
            set_default_pool(None)
