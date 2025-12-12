#!/usr/bin/env python3
"""PyGPUkit GPU Scheduler Full Execution Simulation.

This demonstrates the Kubernetes-style GPU task scheduler with:
- Memory pool management
- Bandwidth pacing
- QoS policies
- Task admission and execution
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

# Add src to path for development
sys.path.insert(0, "src")


def log(msg: str, level: str = "INFO") -> None:
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level:8}] {msg}")


def separator(title: str = "") -> None:
    """Print separator line."""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("-" * 60)


def main() -> None:
    """Run full scheduler simulation."""

    separator("PyGPUkit GPU Scheduler Simulation")
    print("Version: 0.2.0-dev")
    print("Mode: Full Execution Simulation")
    print()

    # =========================================================================
    # PHASE 1: GPU Discovery & Initialization
    # =========================================================================
    separator("PHASE 1: GPU Discovery")

    log("Initializing PyGPUkit backend...")

    from pygpukit.core.backend import get_backend

    backend = get_backend()

    log(f"Backend type: {backend.__class__.__name__}")

    # Get device properties
    props = backend.get_device_properties()
    log(f"GPU Device: {props.name}")
    log(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
    log(f"Compute Capability: {props.compute_capability}")
    log(f"Multiprocessors: {props.multiprocessor_count} SMs")
    log(f"Max Threads/Block: {props.max_threads_per_block}")
    log(f"Warp Size: {props.warp_size}")

    separator()
    log("Resource Table:")
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  Resource           │  Total        │  Available       │
    ├─────────────────────────────────────────────────────────┤
    │  GPU Memory         │  {props.total_memory/(1024**3):6.2f} GB   │  {props.total_memory/(1024**3):6.2f} GB        │
    │  Streaming MPs      │  {props.multiprocessor_count:6d} SMs   │  {props.multiprocessor_count:6d} SMs        │
    │  Bandwidth          │  100.00 %     │  100.00 %        │
    │  CUDA Streams       │      32       │      32          │
    └─────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # PHASE 2: Memory Pool Creation
    # =========================================================================
    separator("PHASE 2: Memory Pool Initialization")

    from pygpukit.memory import MemoryPool, set_default_pool

    # Use 80% of GPU memory for the pool
    pool_size = int(props.total_memory * 0.8)
    log(f"Creating memory pool with quota: {pool_size / (1024**3):.2f} GB")

    pool = MemoryPool(quota=pool_size, enable_eviction=True)
    set_default_pool(pool)

    log("Memory Pool initialized")
    log(f"  Quota: {pool.quota / (1024**3):.2f} GB")
    log(f"  Used: {pool.used / (1024**3):.4f} GB")
    log(f"  Cached: {pool.cached / (1024**3):.4f} GB")
    log("  Eviction: ENABLED")

    print("""
    Size Classes:
    ┌────────────┬─────────────┬───────────┐
    │  Class     │  Size       │  Blocks   │
    ├────────────┼─────────────┼───────────┤
    │  tiny      │  256 B      │  0        │
    │  small     │  4 KB       │  0        │
    │  medium    │  64 KB      │  0        │
    │  large     │  1 MB       │  0        │
    │  xlarge    │  16 MB      │  0        │
    │  huge      │  256 MB     │  0        │
    └────────────┴─────────────┴───────────┘
    """)

    # =========================================================================
    # PHASE 3: Scheduler Creation
    # =========================================================================
    separator("PHASE 3: Scheduler Initialization")

    from pygpukit.scheduler import Scheduler, TaskPolicy

    scheduler = Scheduler(
        sched_tick_ms=1.0,  # 1ms tick
        window_ms=10.0,  # 10ms scheduling window
        total_memory=pool_size,
    )

    log("Scheduler initialized")
    log(f"  Tick interval: {scheduler._sched_tick_ms} ms")
    log(f"  Scheduling window: {scheduler._window_ms} ms")
    log(f"  Total memory: {scheduler._total_memory / (1024**3):.2f} GB")
    log("Scheduler state: READY")
    log("Listening for task submissions...")

    # =========================================================================
    # PHASE 4: Task Submission
    # =========================================================================
    separator("PHASE 4: Task Submission")

    # Track execution for simulation
    execution_log: list[tuple[str, str, float]] = []

    def make_workload(name: str, flops: int, duration_ms: float):
        """Create a simulated GPU workload."""

        def workload():
            start = time.time()
            execution_log.append((name, "START", start))
            log(f"[KERNEL] {name}: Launching kernel (est. {flops/1e9:.1f} GFLOPS)", "EXEC")
            # Simulate work
            time.sleep(duration_ms / 1000.0)
            end = time.time()
            execution_log.append((name, "END", end))
            log(f"[KERNEL] {name}: Completed in {(end-start)*1000:.2f} ms", "EXEC")

        return workload

    # Submit multiple tasks with different characteristics
    tasks_config = [
        # (name, memory_mb, bandwidth, policy, flops, duration_ms)
        ("matmul_large", 512, 0.40, "guaranteed", 2.1e12, 50),
        ("conv2d_batch", 256, 0.30, "guaranteed", 1.5e12, 40),
        ("attention_qkv", 128, 0.20, "burstable", 0.8e12, 30),
        ("softmax_norm", 64, 0.15, "burstable", 0.2e12, 15),
        ("embedding_lookup", 32, 0.10, "best_effort", 0.1e12, 10),
        ("loss_backward", 256, 0.25, "guaranteed", 1.2e12, 35),
    ]

    task_ids = []

    for name, mem_mb, bw, policy, flops, dur in tasks_config:
        log(f"Submitting task: {name}", "SUBMIT")

        mem_bytes = mem_mb * 1024 * 1024
        task_id = scheduler.submit(
            fn=make_workload(name, flops, dur),
            memory=mem_bytes,
            bandwidth=bw,
            policy=policy,
        )
        task_ids.append((task_id, name))

        task = scheduler.get_task(task_id)

        print(f"""
    Task Registered:
    ┌──────────────────────────────────────────────────────────┐
    │  ID: {task_id:<8}                                        │
    │  Name: {name:<20}                            │
    │  Memory Request: {mem_mb:>6} MB                              │
    │  Bandwidth: {bw*100:>5.1f} %                                   │
    │  Policy: {policy.upper():<12}                               │
    │  Est. FLOPs: {flops/1e12:.2f} TFLOPS                              │
    │  State: {task.state.name:<10}                                 │
    └──────────────────────────────────────────────────────────┘
        """)

    # Show queue state
    separator("Scheduler Queue State")

    global_stats = scheduler.global_stats()
    log(f"Total tasks: {global_stats['task_count']}")
    log(f"Pending: {global_stats['pending_count']}")
    log(f"Reserved memory: {global_stats['reserved_memory'] / (1024**2):.0f} MB")

    avail_mem = pool_size - global_stats["reserved_memory"]
    avail_pct = avail_mem / pool_size * 100

    print(f"""
    Resource Allocation After Submission:
    ┌─────────────────────────────────────────────────────────┐
    │  Resource           │  Reserved     │  Available       │
    ├─────────────────────────────────────────────────────────┤
    │  GPU Memory         │  {global_stats['reserved_memory']/(1024**2):6.0f} MB   │  {avail_mem/(1024**2):6.0f} MB ({avail_pct:.1f}%)  │
    │  Bandwidth          │   140.0 %     │   -40.0 % (!)    │
    └─────────────────────────────────────────────────────────┘

    WARNING: Total bandwidth request (140%) exceeds 100%.
             Pacing will throttle lower-priority tasks.
    """)

    # =========================================================================
    # PHASE 5: Admission Control Simulation
    # =========================================================================
    separator("PHASE 5: Admission Control Analysis")

    log("Analyzing admission decisions...")

    # Simulate admission logic
    guaranteed_tasks = [
        (tid, name)
        for tid, name in task_ids
        if scheduler.get_task(tid).policy == TaskPolicy.GUARANTEED
    ]
    burstable_tasks = [
        (tid, name)
        for tid, name in task_ids
        if scheduler.get_task(tid).policy == TaskPolicy.BURSTABLE
    ]
    besteffort_tasks = [
        (tid, name)
        for tid, name in task_ids
        if scheduler.get_task(tid).policy == TaskPolicy.BEST_EFFORT
    ]

    print("""
    Admission Decision Matrix:
    ┌────────────────────┬────────────┬──────────┬─────────────────────────────┐
    │  Task              │  Policy    │  Decision│  Reason                     │
    ├────────────────────┼────────────┼──────────┼─────────────────────────────┤""")

    for tid, name in guaranteed_tasks:
        task = scheduler.get_task(tid)
        print(f"    │  {name:<18}│  GUARANTEED│  ADMIT   │  Hard guarantee satisfied   │")

    for tid, name in burstable_tasks:
        task = scheduler.get_task(tid)
        print(f"    │  {name:<18}│  BURSTABLE │  ADMIT   │  Memory OK, BW may throttle │")

    for tid, name in besteffort_tasks:
        task = scheduler.get_task(tid)
        print(f"    │  {name:<18}│  BEST_EFFORT│ ADMIT   │  Uses leftover resources    │")

    print("    └────────────────────┴────────────┴──────────┴─────────────────────────────┘")

    log("All tasks ADMITTED (memory within quota)")
    log("Bandwidth overcommit will be handled by pacing")

    # =========================================================================
    # PHASE 6: Memory Pool Operations
    # =========================================================================
    separator("PHASE 6: Memory Pool Operations")

    log("Simulating memory block allocations...")

    # Allocate blocks to show pool behavior
    blocks = []
    alloc_sizes = [
        ("matmul_weights", 256 * 1024 * 1024),
        ("matmul_activations", 128 * 1024 * 1024),
        ("conv_filters", 64 * 1024 * 1024),
        ("attention_cache", 32 * 1024 * 1024),
    ]

    for name, size in alloc_sizes:
        log(f"pool.allocate({size // (1024*1024)} MB) for {name}", "ALLOC")
        try:
            block = pool.allocate(size)
            blocks.append((name, block))
            stats = pool.stats()
            log(f"  Block ID: {block.id}, Size class: {block.size // (1024*1024)} MB", "ALLOC")
            log(
                f"  Pool used: {stats['used'] // (1024**2)} MB, Cached: {stats['cached'] // (1024**2)} MB",
                "ALLOC",
            )
        except MemoryError as e:
            log(f"  FAILED: {e}", "ERROR")

    separator()
    log("Freeing matmul_activations block (simulating kernel completion)...")

    # Free one block
    _, block_to_free = blocks[1]
    pool.free(block_to_free)
    stats = pool.stats()
    log(f"Block {block_to_free.id} returned to free list", "FREE")
    log(
        f"Pool used: {stats['used'] // (1024**2)} MB, Cached: {stats['cached'] // (1024**2)} MB",
        "FREE",
    )

    separator()
    log("Allocating new block (should reuse from free list)...")

    new_block = pool.allocate(128 * 1024 * 1024)
    stats = pool.stats()
    log(f"Block ID: {new_block.id} (reuse_count: {stats['reuse_count']})", "REUSE")

    print(f"""
    Memory Pool Statistics:
    ┌────────────────────────────────────────┐
    │  Metric              │  Value          │
    ├────────────────────────────────────────┤
    │  cudaMalloc calls    │  {stats['cudamalloc_count']:>6}          │
    │  Reuse count         │  {stats['reuse_count']:>6}          │
    │  Eviction count      │  {stats['eviction_count']:>6}          │
    │  Active blocks       │  {stats['active_blocks']:>6}          │
    │  Free blocks         │  {stats['free_blocks']:>6}          │
    └────────────────────────────────────────┘
    """)

    # =========================================================================
    # PHASE 7: Execution Timeline
    # =========================================================================
    separator("PHASE 7: Execution Timeline")

    log("Starting scheduler main loop...")
    log("Bandwidth pacing ACTIVE")

    print("""
    Execution Plan:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Time    │  Task              │  Action    │  Notes                      │
    ├──────────────────────────────────────────────────────────────────────────┤
    │  T+0ms   │  matmul_large      │  START     │  GUARANTEED, highest BW     │
    │  T+0ms   │  conv2d_batch      │  START     │  GUARANTEED, parallel exec  │
    │  T+0ms   │  loss_backward     │  START     │  GUARANTEED, parallel exec  │
    │  T+5ms   │  attention_qkv     │  PACED     │  BURSTABLE, waiting slot    │
    │  T+10ms  │  attention_qkv     │  START     │  Pacing interval elapsed    │
    │  T+15ms  │  softmax_norm      │  START     │  BURSTABLE                  │
    │  T+20ms  │  embedding_lookup  │  PACED     │  BEST_EFFORT, low priority  │
    │  T+25ms  │  softmax_norm      │  COMPLETE  │  Duration: 15ms             │
    │  T+30ms  │  embedding_lookup  │  START     │  Resources available        │
    │  T+35ms  │  loss_backward     │  COMPLETE  │  Duration: 35ms             │
    │  T+40ms  │  attention_qkv     │  COMPLETE  │  Duration: 30ms             │
    │  T+40ms  │  conv2d_batch      │  COMPLETE  │  Duration: 40ms             │
    │  T+40ms  │  embedding_lookup  │  COMPLETE  │  Duration: 10ms             │
    │  T+50ms  │  matmul_large      │  COMPLETE  │  Duration: 50ms             │
    └──────────────────────────────────────────────────────────────────────────┘
    """)

    # Actually run the scheduler
    log("Executing scheduler loop (real execution)...")
    separator()

    start_time = time.time()
    iteration = 0
    max_iterations = 500  # Safety limit

    while scheduler.completed_count < len(task_ids) and iteration < max_iterations:
        scheduler.step()
        iteration += 1
        time.sleep(0.001)  # 1ms tick

        # Log progress every 50 iterations
        if iteration % 50 == 0:
            elapsed = (time.time() - start_time) * 1000
            log(
                f"Tick {iteration}: {scheduler.completed_count}/{len(task_ids)} tasks complete, elapsed: {elapsed:.1f}ms"
            )

    end_time = time.time()
    total_time = (end_time - start_time) * 1000

    separator()
    log(f"Scheduler loop completed in {total_time:.2f} ms")
    log(f"Total ticks: {iteration}")

    # =========================================================================
    # PHASE 8: Final Statistics
    # =========================================================================
    separator("PHASE 8: Final Statistics")

    final_stats = scheduler.global_stats()
    pool_stats = pool.stats()

    print(f"""
    Scheduler Summary:
    ┌────────────────────────────────────────┐
    │  Metric              │  Value          │
    ├────────────────────────────────────────┤
    │  Total tasks         │  {final_stats['task_count']:>6}          │
    │  Completed           │  {final_stats['completed_count']:>6}          │
    │  Total time          │  {total_time:>6.1f} ms      │
    │  Avg task time       │  {total_time/len(task_ids):>6.1f} ms      │
    └────────────────────────────────────────┘

    Per-Task Statistics:
    ┌────────────────────┬──────────┬────────────┬────────────────┐
    │  Task              │  State   │  Exec Count│  Pacing Delays │
    ├────────────────────┼──────────┼────────────┼────────────────┤""")

    for tid, name in task_ids:
        stats = scheduler.stats(tid)
        print(
            f"    │  {name:<18}│  {stats['state'].upper():<8}│  {stats['execution_count']:>10}│  {stats['pacing_delay_count']:>14}│"
        )

    print("""    └────────────────────┴──────────┴────────────┴────────────────┘

    Memory Pool Final State:
    ┌────────────────────────────────────────┐
    │  Metric              │  Value          │
    ├────────────────────────────────────────┤""")
    print(f"    │  cudaMalloc calls    │  {pool_stats['cudamalloc_count']:>6}          │")
    print(f"    │  Block reuse         │  {pool_stats['reuse_count']:>6}          │")
    print(f"    │  Evictions           │  {pool_stats['eviction_count']:>6}          │")
    print(f"    │  Active blocks       │  {pool_stats['active_blocks']:>6}          │")
    print(f"    │  Cached blocks       │  {pool_stats['free_blocks']:>6}          │")
    print("    └────────────────────────────────────────┘")

    # =========================================================================
    # PHASE 9: Cleanup
    # =========================================================================
    separator("PHASE 9: Cleanup")

    log("Clearing memory pool...")
    pool.clear()
    log("Releasing default pool reference...")
    set_default_pool(None)
    log("Scheduler simulation complete.")

    separator("END OF SIMULATION")
    print()


if __name__ == "__main__":
    main()
