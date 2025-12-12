"""
PyGPUkit Stress Test Script for v0.2.1
Tests Rust backend components under sustained load.
Default: 5 minutes runtime.
"""
import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Skip if Rust module not available
try:
    import _pygpukit_rust as rust
except ImportError:
    print("Rust module not available. Skipping stress test.")
    exit(0)


class StressTestStats:
    """Thread-safe statistics collector."""
    def __init__(self):
        self.lock = threading.Lock()
        self.operations = 0
        self.errors = 0
        self.memory_ops = 0
        self.scheduler_ops = 0
        self.admission_ops = 0
        self.qos_ops = 0
        self.partition_ops = 0

    def inc(self, op_type: str):
        with self.lock:
            self.operations += 1
            if op_type == "memory":
                self.memory_ops += 1
            elif op_type == "scheduler":
                self.scheduler_ops += 1
            elif op_type == "admission":
                self.admission_ops += 1
            elif op_type == "qos":
                self.qos_ops += 1
            elif op_type == "partition":
                self.partition_ops += 1

    def inc_error(self):
        with self.lock:
            self.errors += 1

    def get_stats(self):
        with self.lock:
            return {
                "operations": self.operations,
                "errors": self.errors,
                "memory_ops": self.memory_ops,
                "scheduler_ops": self.scheduler_ops,
                "admission_ops": self.admission_ops,
                "qos_ops": self.qos_ops,
                "partition_ops": self.partition_ops,
            }


def stress_memory_pool(stats: StressTestStats, duration_sec: float):
    """Stress test memory pool with allocations and frees."""
    pool = rust.MemoryPool(quota=100 * 1024 * 1024, enable_eviction=True)
    end_time = time.time() + duration_sec
    blocks = []

    while time.time() < end_time:
        try:
            # Random allocation size (1KB - 1MB)
            size = random.randint(1024, 1024 * 1024)
            block = pool.allocate(size)
            blocks.append(block)
            stats.inc("memory")

            # Randomly free some blocks to keep memory bounded
            if len(blocks) > 50:
                idx = random.randint(0, len(blocks) - 1)
                b = blocks.pop(idx)
                pool.free(b.id)
                stats.inc("memory")

        except Exception:
            stats.inc_error()

    # Cleanup
    for b in blocks:
        try:
            pool.free(b.id)
        except Exception:
            pass


def stress_scheduler(stats: StressTestStats, duration_sec: float):
    """Stress test scheduler with task submissions."""
    scheduler = rust.Scheduler(total_memory=1024 * 1024 * 1024)
    end_time = time.time() + duration_sec
    task_counter = 0

    while time.time() < end_time:
        try:
            # Submit task
            task_id = f"task-{task_counter}"
            task = rust.TaskMeta(
                id=task_id,
                name=f"Stress Task {task_counter}",
                memory_estimate=random.randint(1024, 10 * 1024 * 1024),
                priority=random.randint(0, 10),
            )
            scheduler.submit(task)
            stats.inc("scheduler")
            task_counter += 1

            # Randomly complete some tasks
            if task_counter % 10 == 0:
                runnable = scheduler.get_runnable_tasks(5)
                for tid in runnable:
                    scheduler.start_task(tid)
                    scheduler.complete_task(tid)
                    stats.inc("scheduler")

        except Exception:
            stats.inc_error()


def stress_admission_controller(stats: StressTestStats, duration_sec: float):
    """Stress test admission controller."""
    config = rust.AdmissionConfig(
        max_memory=100 * 1024 * 1024,
        max_bandwidth=1.0,
    )
    controller = rust.AdmissionController(config)
    end_time = time.time() + duration_sec
    task_counter = 0
    admitted_tasks = []

    while time.time() < end_time:
        try:
            task_id = f"admit-task-{task_counter}"
            memory = random.randint(1024, 20 * 1024 * 1024)
            bandwidth = random.uniform(0.01, 0.3)

            decision = controller.try_admit(task_id, memory, bandwidth)
            stats.inc("admission")
            task_counter += 1

            if decision.is_admitted():
                admitted_tasks.append(task_id)

            # Release some tasks to free resources
            if len(admitted_tasks) > 20:
                release_id = admitted_tasks.pop(0)
                controller.release(release_id)
                stats.inc("admission")

        except Exception:
            stats.inc_error()


def stress_qos_evaluator(stats: StressTestStats, duration_sec: float):
    """Stress test QoS policy evaluator."""
    evaluator = rust.QosPolicyEvaluator(
        total_memory=1024 * 1024 * 1024,
        total_bandwidth=1.0,
    )
    end_time = time.time() + duration_sec
    task_counter = 0
    reservations = []

    while time.time() < end_time:
        try:
            # Create task with random QoS class
            qos_type = random.choice(["guaranteed", "burstable", "best_effort"])
            task_id = f"qos-task-{task_counter}"

            if qos_type == "guaranteed":
                task = rust.QosTaskMeta.guaranteed(
                    task_id, f"Guaranteed {task_counter}",
                    random.randint(1024, 50 * 1024 * 1024)
                )
            elif qos_type == "burstable":
                task = rust.QosTaskMeta.burstable(
                    task_id, f"Burstable {task_counter}",
                    random.randint(1024, 30 * 1024 * 1024),
                    random.uniform(1.5, 3.0)
                )
            else:
                task = rust.QosTaskMeta.best_effort(task_id, f"BestEffort {task_counter}")

            result = evaluator.evaluate(task)
            stats.inc("qos")
            task_counter += 1

            if result.is_admitted():
                evaluator.reserve(result)
                reservations.append((task.qos_class, task.memory_request, 0.0))
                stats.inc("qos")

            # Release some reservations
            if len(reservations) > 30:
                qos_class, mem, bw = reservations.pop(0)
                evaluator.release(qos_class, mem, bw)
                stats.inc("qos")

        except Exception:
            stats.inc_error()


def stress_partition_manager(stats: StressTestStats, duration_sec: float):
    """Stress test partition manager."""
    config = rust.PartitionConfig(total_memory=8 * 1024 * 1024 * 1024)
    manager = rust.PartitionManager(config)
    end_time = time.time() + duration_sec
    partition_counter = 0
    task_counter = 0
    partitions = []

    while time.time() < end_time:
        try:
            action = random.choice(["create", "assign", "stats", "delete"])

            if action == "create" and len(partitions) < 10:
                pid = f"partition-{partition_counter}"
                limits = rust.PartitionLimits().memory(
                    random.randint(100 * 1024 * 1024, 500 * 1024 * 1024)
                ).compute(random.uniform(0.05, 0.3))
                manager.create_partition(pid, f"Partition {partition_counter}", limits)
                partitions.append(pid)
                partition_counter += 1
                stats.inc("partition")

            elif action == "assign" and partitions:
                pid = random.choice(partitions)
                task_id = f"p-task-{task_counter}"
                try:
                    manager.assign_task(task_id, pid)
                    task_counter += 1
                    stats.inc("partition")
                except Exception:
                    pass  # Partition might not exist

            elif action == "stats":
                manager.stats()
                stats.inc("partition")

            elif action == "delete" and len(partitions) > 3:
                pid = partitions.pop(random.randint(0, len(partitions) - 1))
                try:
                    manager.delete_partition(pid)
                    stats.inc("partition")
                except Exception:
                    pass

        except Exception:
            stats.inc_error()


def run_stress_test(duration_minutes: float = 5.0, workers: int = 4):
    """Run all stress tests concurrently."""
    duration_sec = duration_minutes * 60
    stats = StressTestStats()

    print("Starting PyGPUkit Stress Test")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Workers per component: {workers}")
    print("-" * 50)

    start_time = time.time()

    test_functions = [
        stress_memory_pool,
        stress_scheduler,
        stress_admission_controller,
        stress_qos_evaluator,
        stress_partition_manager,
    ]

    # Start all stress tests in parallel
    with ThreadPoolExecutor(max_workers=len(test_functions) * workers) as executor:
        futures = []
        for test_func in test_functions:
            for _ in range(workers):
                futures.append(executor.submit(test_func, stats, duration_sec))

        # Progress reporting
        last_report = start_time
        while not all(f.done() for f in futures):
            time.sleep(1)
            now = time.time()
            if now - last_report >= 10:  # Report every 10 seconds
                elapsed = now - start_time
                current_stats = stats.get_stats()
                ops_per_sec = current_stats["operations"] / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:.0f}s] Ops: {current_stats['operations']:,} "
                      f"({ops_per_sec:.0f}/s) | Errors: {current_stats['errors']}")
                last_report = now

        # Wait for all to complete
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Worker error: {e}")

    elapsed = time.time() - start_time
    final_stats = stats.get_stats()

    print("-" * 50)
    print("Stress Test Complete")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Total Operations: {final_stats['operations']:,}")
    print(f"Operations/sec: {final_stats['operations'] / elapsed:.0f}")
    print(f"Errors: {final_stats['errors']}")
    print("-" * 50)
    print("Breakdown:")
    print(f"  Memory Pool:    {final_stats['memory_ops']:,}")
    print(f"  Scheduler:      {final_stats['scheduler_ops']:,}")
    print(f"  Admission:      {final_stats['admission_ops']:,}")
    print(f"  QoS:            {final_stats['qos_ops']:,}")
    print(f"  Partitioning:   {final_stats['partition_ops']:,}")
    print("-" * 50)

    if final_stats['errors'] > 0:
        print(f"WARNING: {final_stats['errors']} errors occurred during test")
        return 1
    else:
        print("SUCCESS: No errors detected")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyGPUkit Stress Test")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test duration in minutes (default: 5)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Workers per component (default: 4)")
    args = parser.parse_args()

    exit(run_stress_test(args.duration, args.workers))
