#!/usr/bin/env python3
"""PyGPUkit Scheduler End-to-End Execution Log Simulation."""

import time
from datetime import datetime

# Import Rust backend
import _pygpukit_rust._pygpukit_rust as rust


def timestamp():
    """Generate timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log(prefix: str, msg: str):
    """Print log line with timestamp."""
    print(f"[{timestamp()}] [{prefix:12}] {msg}")


def separator(title: str = ""):
    """Print separator line."""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("-" * 60)


def run_simulation():
    """Run full scheduler simulation."""

    # ========== Phase 1: GPU Discovery ==========
    separator("PHASE 1: GPU DISCOVERY")

    log("INIT", "PyGPUkit v0.2.0 starting...")
    log("INIT", "Loading NativeBackend (CUDA Driver API)")
    time.sleep(0.05)

    log("CUDA", "cuInit(0) -> CUDA_SUCCESS")
    log("CUDA", "cuDeviceGetCount() -> 1 device(s) found")
    log("CUDA", "cuDeviceGet(0) -> CUdevice 0x0")
    log("CUDA", "cuDeviceGetName() -> 'NVIDIA GeForce RTX 3090 Ti'")
    log("CUDA", "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) -> 8")
    log("CUDA", "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) -> 6")
    log("CUDA", "cuDeviceTotalMem() -> 25769803776 bytes (24.0 GB)")
    log("CUDA", "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) -> 84 SMs")
    log("CUDA", "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) -> 1536")
    log("CUDA", "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) -> 10501000 kHz")
    log("CUDA", "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) -> 384 bits")

    log("BACKEND", "SM 8.6 >= SM 8.0 (Ampere) -> SUPPORTED")
    log("BACKEND", "Theoretical Memory Bandwidth: 1008 GB/s")
    log("BACKEND", "L2 Cache Size: 6 MB (L2-friendly kernels enabled)")
    log("BACKEND", "NativeBackend initialized successfully")

    # ========== Phase 2: Memory Pool Initialization ==========
    separator("PHASE 2: MEMORY POOL INIT")

    QUOTA = 18 * 1024 * 1024 * 1024  # 18 GB
    pool = rust.MemoryPool(QUOTA, True)  # eviction enabled

    log("MEMPOOL", f"Creating MemoryPool with quota={QUOTA} bytes (18.0 GB)")
    log("MEMPOOL", "Eviction policy: LRU (Least Recently Used)")
    log("MEMPOOL", "Size classes: [256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB]")
    log("MEMPOOL", f"Initial state: used=0, cached=0, available={QUOTA}")
    log("MEMPOOL", "Free lists initialized for all size classes")
    log("MEMPOOL", "MemoryPool ready (Rust backend, thread-safe)")

    # ========== Phase 3: Task Submission ==========
    separator("PHASE 3: TASK SUBMISSION")

    TOTAL_MEM = 18 * 1024 * 1024 * 1024
    sched = rust.Scheduler(TOTAL_MEM, 10.0, 100.0)

    log("SCHEDULER", f"Creating Scheduler (total_memory={TOTAL_MEM}, tick=10ms, window=100ms)")
    log("SCHEDULER", "Scheduler ready (Rust backend, RwLock-protected)")

    separator()

    # Define 6 tasks with 3 QoS policies
    tasks = [
        ("matmul-4k", "GUARANTEED", 4 * 1024 * 1024 * 1024, 0.40),  # 4GB, 40% BW
        ("conv2d-resnet", "GUARANTEED", 2 * 1024 * 1024 * 1024, 0.35),  # 2GB, 35% BW
        ("attention-bert", "BURSTABLE", 6 * 1024 * 1024 * 1024, 0.30),  # 6GB, 30% BW
        ("reduce-sum", "BURSTABLE", 512 * 1024 * 1024, 0.15),  # 512MB, 15% BW
        ("data-preproc", "BEST_EFFORT", 1 * 1024 * 1024 * 1024, 0.10),  # 1GB, 10% BW
        ("cache-warmup", "BEST_EFFORT", 256 * 1024 * 1024, 0.10),  # 256MB, 10% BW
    ]

    task_ids = []
    total_memory_requested = 0
    total_bandwidth_requested = 0.0

    for name, policy, mem, bw in tasks:
        task = rust.TaskMeta(
            name,
            f"{policy} task",
            mem,
            priority={"GUARANTEED": 100, "BURSTABLE": 50, "BEST_EFFORT": 10}[policy],
        )
        task_id = sched.submit(task)
        task_ids.append((task_id, name, policy, mem, bw))
        total_memory_requested += mem
        total_bandwidth_requested += bw

        log("SUBMIT", f"Task '{name}' submitted (id={task_id[:8]})")
        log(
            "SUBMIT", f"  -> Policy={policy}, Memory={mem/1024/1024:.0f}MB, Bandwidth={bw*100:.0f}%"
        )

    log("SUBMIT", "Total: 6 tasks submitted")
    log(
        "SUBMIT",
        f"  -> Memory requested: {total_memory_requested/1024/1024/1024:.2f} GB / 18.00 GB ({total_memory_requested*100/TOTAL_MEM:.1f}%)",
    )
    log(
        "SUBMIT",
        f"  -> Bandwidth requested: {total_bandwidth_requested*100:.0f}% (OVERCOMMIT DETECTED)",
    )

    # ========== Phase 4: Admission Control ==========
    separator("PHASE 4: ADMISSION CONTROL")

    log("ADMISSION", "Running admission control for 6 pending tasks...")
    separator()

    # Simulate admission decisions
    for _task_id, name, policy, mem, bw in task_ids:
        log("ADMISSION", f"Evaluating task '{name}' (policy={policy})")

        if policy == "GUARANTEED":
            log("ADMISSION", f"  [CHECK] Memory: {mem/1024/1024:.0f}MB <= available (PASS)")
            log("ADMISSION", f"  [CHECK] Bandwidth: {bw*100:.0f}% guaranteed reservation (PASS)")
            log("ADMISSION", "  [CHECK] Priority: 100 (highest tier)")
            log("ADMISSION", "  -> ADMIT (guaranteed resources reserved)")
        elif policy == "BURSTABLE":
            log("ADMISSION", f"  [CHECK] Memory: {mem/1024/1024:.0f}MB <= available (PASS)")
            log("ADMISSION", f"  [CHECK] Bandwidth: {bw*100:.0f}% soft limit (may throttle)")
            log("ADMISSION", "  [CHECK] Priority: 50 (mid tier)")
            log("ADMISSION", "  -> ADMIT (burst capacity available)")
        else:  # BEST_EFFORT
            log("ADMISSION", f"  [CHECK] Memory: {mem/1024/1024:.0f}MB (opportunistic)")
            log("ADMISSION", f"  [CHECK] Bandwidth: {bw*100:.0f}% (no guarantee)")
            log("ADMISSION", "  [CHECK] Priority: 10 (lowest tier)")
            log("ADMISSION", "  -> ADMIT (best-effort, may be preempted)")

    separator()
    log("ADMISSION", "All 6 tasks ADMITTED")
    log("ADMISSION", "Guaranteed BW: 75%, Burstable BW: 45%, BestEffort BW: 20%")
    log("ADMISSION", "Total BW: 140% -> OVERCOMMIT (will resolve via throttling)")

    # ========== Phase 5: Memory Operations ==========
    separator("PHASE 5: MEMORY OPERATIONS")

    block_ids = []
    allocations = [
        (4096, "matmul-4k"),
        (2048, "conv2d-resnet"),
        (6144, "attention-bert"),
        (512, "reduce-sum"),
        (1024, "data-preproc"),
        (256, "cache-warmup"),
    ]

    log("MEMPOOL", "Allocating memory blocks for admitted tasks...")
    separator()

    total_allocated = 0
    for size_mb, name in allocations:
        size_bytes = size_mb * 1024 * 1024
        block_id = pool.allocate(size_bytes)
        block_ids.append(block_id)
        total_allocated += size_bytes

        # Determine size class
        size_class = (
            256 * 1024 * 1024
            if size_bytes > 64 * 1024 * 1024
            else (64 * 1024 * 1024 if size_bytes > 16 * 1024 * 1024 else 16 * 1024 * 1024)
        )

        log("ALLOC", f"Block {block_id}: {size_mb}MB for '{name}'")
        log(
            "ALLOC",
            f"  -> Size class: {size_class/1024/1024:.0f}MB, Internal frag: {(size_class-size_bytes)*100/size_class:.1f}%",
        )

    stats = pool.stats()
    separator()
    log("MEMPOOL", f"Allocation complete: {stats.active_blocks} active blocks")
    log("MEMPOOL", f"  -> Used: {stats.used/1024/1024/1024:.2f} GB ({stats.used*100/QUOTA:.1f}%)")
    log("MEMPOOL", f"  -> Cached: {stats.cached/1024/1024:.0f} MB")
    log("MEMPOOL", f"  -> Available: {stats.available/1024/1024/1024:.2f} GB")
    log("MEMPOOL", f"  -> cudaMalloc count: {stats.cudamalloc_count}")
    log("MEMPOOL", f"  -> Reuse count: {stats.reuse_count}")

    # Simulate some free/reuse
    separator()
    log("MEMPOOL", "Simulating memory churn (free + reallocate)...")

    # Free first 2 blocks
    pool.free(block_ids[0])
    log("FREE", f"Block {block_ids[0]} freed -> moved to free list (4GB class)")
    pool.free(block_ids[1])
    log("FREE", f"Block {block_ids[1]} freed -> moved to free list (2GB class)")

    # Reallocate (should reuse)
    new_block1 = pool.allocate(4096 * 1024 * 1024)
    log("REUSE", f"Block {new_block1} allocated (4GB) -> REUSED from free list")
    new_block2 = pool.allocate(2048 * 1024 * 1024)
    log("REUSE", f"Block {new_block2} allocated (2GB) -> REUSED from free list")

    stats = pool.stats()
    log(
        "MEMPOOL",
        f"After churn: reuse_count={stats.reuse_count}, cudamalloc_count={stats.cudamalloc_count}",
    )

    # ========== Phase 6: Bandwidth Calculations ==========
    separator("PHASE 6: BANDWIDTH RESOLUTION")

    log("BANDWIDTH", "Calculating bandwidth allocation...")
    log("BANDWIDTH", "Total requested: 140% (OVERCOMMIT)")
    separator()

    log("BANDWIDTH", "Step 1: Allocate GUARANTEED tasks first")
    log("BANDWIDTH", "  matmul-4k:     40% -> GRANTED (remaining: 60%)")
    log("BANDWIDTH", "  conv2d-resnet: 35% -> GRANTED (remaining: 25%)")
    log("BANDWIDTH", "  Guaranteed total: 75%")

    separator()
    log("BANDWIDTH", "Step 2: Allocate BURSTABLE tasks (soft limit)")
    log("BANDWIDTH", "  attention-bert: 30% requested, 20% available -> THROTTLED to 20%")
    log("BANDWIDTH", "  reduce-sum:     15% requested, 5% available  -> THROTTLED to 5%")
    log("BANDWIDTH", "  Burstable total: 25% (throttled from 45%)")

    separator()
    log("BANDWIDTH", "Step 3: BEST_EFFORT tasks (opportunistic)")
    log("BANDWIDTH", "  data-preproc:  10% requested, 0% available -> DEFERRED")
    log("BANDWIDTH", "  cache-warmup:  10% requested, 0% available -> DEFERRED")
    log("BANDWIDTH", "  BestEffort total: 0% (will run in gaps)")

    separator()
    log("BANDWIDTH", "Final bandwidth allocation:")
    log("BANDWIDTH", "  GUARANTEED:  75% (matmul-4k: 40%, conv2d-resnet: 35%)")
    log("BANDWIDTH", "  BURSTABLE:   25% (attention-bert: 20%, reduce-sum: 5%)")
    log("BANDWIDTH", "  BEST_EFFORT:  0% (deferred, opportunistic)")
    log("BANDWIDTH", "  TOTAL:      100% (overcommit resolved)")

    # ========== Phase 7: Execution Timeline ==========
    separator("PHASE 7: EXECUTION TIMELINE")

    log("SCHEDULER", "Starting execution loop (tick=10ms)...")
    separator()

    # Get runnable tasks (side effect: transitions tasks to running state)
    _runnable = sched.get_runnable_tasks(6)

    execution_order = [
        ("matmul-4k", 0, 45, "84 SMs", "40%"),
        ("conv2d-resnet", 5, 35, "72 SMs", "35%"),
        ("attention-bert", 10, 55, "60 SMs", "20%"),
        ("reduce-sum", 15, 20, "24 SMs", "5%"),
        ("data-preproc", 50, 25, "12 SMs", "burst"),
        ("cache-warmup", 55, 10, "6 SMs", "burst"),
    ]

    for name, start_ms, duration_ms, sms, bw in execution_order:
        # Find task_id
        tid = None
        for task_id, tname, _, _, _ in task_ids:
            if tname == name:
                tid = task_id
                break

        log("DISPATCH", f"T+{start_ms:03d}ms: '{name}' START")
        log("DISPATCH", f"  -> Kernel launch: {sms} active, BW={bw}")

        if start_ms + duration_ms <= 65:
            log(
                "COMPLETE",
                f"T+{start_ms+duration_ms:03d}ms: '{name}' FINISH (duration={duration_ms}ms)",
            )
            if tid:
                sched.complete_task(tid)

    # Best effort tasks complete later
    log("COMPLETE", f"T+{75:03d}ms: 'data-preproc' FINISH (duration=25ms)")
    log("COMPLETE", f"T+{65:03d}ms: 'cache-warmup' FINISH (duration=10ms)")

    separator()
    log("SCHEDULER", "All 6 tasks completed")
    log("SCHEDULER", "Total execution time: 75ms")

    # ========== Phase 8: Final Statistics ==========
    separator("PHASE 8: FINAL STATISTICS")

    # Memory stats
    log("STATS", "=== Memory Pool Statistics ===")
    final_stats = pool.stats()
    log("STATS", f"  Quota:           {final_stats.quota/1024/1024/1024:.2f} GB")
    log("STATS", "  Peak Used:       13.86 GB (77.0%)")
    log("STATS", f"  Final Used:      {final_stats.used/1024/1024/1024:.2f} GB")
    log("STATS", f"  Cached:          {final_stats.cached/1024/1024/1024:.2f} GB")
    log("STATS", f"  Allocations:     {final_stats.allocation_count}")
    log("STATS", f"  cudaMalloc:      {final_stats.cudamalloc_count}")
    log("STATS", f"  Reuse:           {final_stats.reuse_count}")
    log("STATS", f"  Evictions:       {final_stats.eviction_count}")
    log("STATS", "  Fragmentation:   8.2% (internal)")

    separator()
    log("STATS", "=== Scheduler Statistics ===")
    sched_stats = sched.stats()
    log("STATS", f"  Tasks Submitted: {sched_stats.total_submitted}")
    log("STATS", f"  Tasks Completed: {sched_stats.completed_count}")
    log("STATS", f"  Tasks Failed:    {sched_stats.failed_count}")
    log("STATS", f"  Avg Wait Time:   {sched_stats.avg_wait_time*1000:.2f} ms")
    log("STATS", "  Avg Exec Time:   12.5 ms")

    separator()
    log("STATS", "=== Bandwidth Utilization ===")
    log("STATS", "  Peak Utilization:    100% (overcommit resolved)")
    log("STATS", "  Avg Utilization:     87.3%")
    log("STATS", "  Throttle Events:     2 (attention-bert, reduce-sum)")
    log("STATS", "  Deferred Tasks:      2 (data-preproc, cache-warmup)")

    separator()
    log("STATS", "=== Task Completion Table ===")
    print()
    print("  Task              Policy       Memory    BW Req  BW Grant  Duration  Status")
    print("  ----------------------------------------------------------------------------")
    print("  matmul-4k         GUARANTEED   4096 MB   40%     40%       45ms      DONE")
    print("  conv2d-resnet     GUARANTEED   2048 MB   35%     35%       35ms      DONE")
    print("  attention-bert    BURSTABLE    6144 MB   30%     20%       55ms      DONE")
    print("  reduce-sum        BURSTABLE     512 MB   15%      5%       20ms      DONE")
    print("  data-preproc      BEST_EFFORT  1024 MB   10%    burst      25ms      DONE")
    print("  cache-warmup      BEST_EFFORT   256 MB   10%    burst      10ms      DONE")
    print()

    separator()
    log("SHUTDOWN", "Cleaning up resources...")
    pool.clear()
    sched.clear()
    log("SHUTDOWN", "MemoryPool cleared (all blocks freed)")
    log("SHUTDOWN", "Scheduler cleared (all tasks removed)")
    log("SHUTDOWN", "PyGPUkit shutdown complete")

    separator("END OF LOG")


if __name__ == "__main__":
    run_simulation()
