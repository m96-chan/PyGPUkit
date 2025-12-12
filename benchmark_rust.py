"""Benchmark Rust vs Python backend for PyGPUkit."""

import time


def benchmark_rust():
    """Benchmark Rust memory pool and scheduler."""
    import _pygpukit_rust._pygpukit_rust as rust

    print("=" * 60)
    print("PyGPUkit Rust Backend Benchmark")
    print("=" * 60)

    # Memory Pool Benchmark
    print("\n### Memory Pool Benchmark ###\n")

    pool = rust.MemoryPool(1024 * 1024 * 100, False)  # 100 MB

    # Allocation benchmark
    n_allocs = 10000
    start = time.perf_counter()
    block_ids = []
    for _ in range(n_allocs):
        block_id = pool.allocate(4096)
        block_ids.append(block_id)
    alloc_time = time.perf_counter() - start
    print(
        f"Allocate {n_allocs} blocks:  {alloc_time*1000:.2f} ms ({n_allocs/alloc_time:.0f} ops/sec)"
    )

    # Free benchmark
    start = time.perf_counter()
    for block_id in block_ids:
        pool.free(block_id)
    free_time = time.perf_counter() - start
    print(
        f"Free {n_allocs} blocks:      {free_time*1000:.2f} ms ({n_allocs/free_time:.0f} ops/sec)"
    )

    # Reuse benchmark (allocate from free list)
    start = time.perf_counter()
    block_ids = []
    for _ in range(n_allocs):
        block_id = pool.allocate(4096)
        block_ids.append(block_id)
    reuse_time = time.perf_counter() - start
    print(
        f"Reuse {n_allocs} blocks:     {reuse_time*1000:.2f} ms ({n_allocs/reuse_time:.0f} ops/sec)"
    )

    stats = pool.stats()
    print(
        f"\nPool stats: reuse_count={stats.reuse_count}, cudamalloc_count={stats.cudamalloc_count}"
    )

    # Cleanup
    for block_id in block_ids:
        pool.free(block_id)

    # Scheduler Benchmark
    print("\n### Scheduler Benchmark ###\n")

    sched = rust.Scheduler(1024 * 1024 * 1000, 10.0, 100.0)  # 1GB memory

    # Submit benchmark
    n_tasks = 10000
    start = time.perf_counter()
    for i in range(n_tasks):
        task = rust.TaskMeta(f"task-{i}", f"Task {i}", 1024)
        sched.submit(task)
    submit_time = time.perf_counter() - start
    print(
        f"Submit {n_tasks} tasks:      {submit_time*1000:.2f} ms ({n_tasks/submit_time:.0f} ops/sec)"
    )

    # Get runnable benchmark
    start = time.perf_counter()
    runnable = sched.get_runnable_tasks(n_tasks)
    get_runnable_time = time.perf_counter() - start
    print(f"Get runnable {len(runnable)} tasks: {get_runnable_time*1000:.2f} ms")

    # Complete benchmark
    start = time.perf_counter()
    for task_id in runnable:
        sched.complete_task(task_id)
    complete_time = time.perf_counter() - start
    print(
        f"Complete {len(runnable)} tasks:   {complete_time*1000:.2f} ms ({len(runnable)/complete_time:.0f} ops/sec)"
    )

    stats = sched.stats()
    print(f"\nScheduler stats: completed={stats.completed_count}")


def benchmark_python():
    """Benchmark Python memory pool and scheduler."""
    from pygpukit.memory.pool import MemoryPool
    from pygpukit.scheduler.core import Scheduler, Task

    print("\n" + "=" * 60)
    print("PyGPUkit Python Backend Benchmark")
    print("=" * 60)

    # Memory Pool Benchmark
    print("\n### Memory Pool Benchmark ###\n")

    pool = MemoryPool(1024 * 1024 * 100, False)  # 100 MB

    # Allocation benchmark
    n_allocs = 10000
    start = time.perf_counter()
    blocks = []
    for _ in range(n_allocs):
        block = pool.allocate(4096)
        blocks.append(block)
    alloc_time = time.perf_counter() - start
    print(
        f"Allocate {n_allocs} blocks:  {alloc_time*1000:.2f} ms ({n_allocs/alloc_time:.0f} ops/sec)"
    )

    # Free benchmark
    start = time.perf_counter()
    for block in blocks:
        pool.free(block)
    free_time = time.perf_counter() - start
    print(
        f"Free {n_allocs} blocks:      {free_time*1000:.2f} ms ({n_allocs/free_time:.0f} ops/sec)"
    )

    # Reuse benchmark (allocate from free list)
    start = time.perf_counter()
    blocks = []
    for _ in range(n_allocs):
        block = pool.allocate(4096)
        blocks.append(block)
    reuse_time = time.perf_counter() - start
    print(
        f"Reuse {n_allocs} blocks:     {reuse_time*1000:.2f} ms ({n_allocs/reuse_time:.0f} ops/sec)"
    )

    stats = pool.stats()
    print(
        f"\nPool stats: reuse_count={stats['reuse_count']}, cudamalloc_count={stats['cudamalloc_count']}"
    )

    # Cleanup
    for block in blocks:
        pool.free(block)

    # Scheduler Benchmark
    print("\n### Scheduler Benchmark ###\n")

    sched = Scheduler(total_memory=1024 * 1024 * 1000)  # 1GB memory

    # Submit benchmark
    n_tasks = 10000
    start = time.perf_counter()
    tasks = []
    for _ in range(n_tasks):
        task = Task(fn=lambda: None, memory=1024)
        sched.submit(task)
        tasks.append(task)
    submit_time = time.perf_counter() - start
    print(
        f"Submit {n_tasks} tasks:      {submit_time*1000:.2f} ms ({n_tasks/submit_time:.0f} ops/sec)"
    )

    # Note: Python scheduler has different API (run_once, etc.)
    print("(Python scheduler uses different API - skipping detailed benchmark)")


if __name__ == "__main__":
    benchmark_rust()
    benchmark_python()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
