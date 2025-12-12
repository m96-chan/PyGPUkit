"""
PyGPUkit v0.2 Full Feature Demo & Benchmark

This demo showcases all v0.2 features:
1. Tiled Matmul Kernel (C++/CUDA) - Shared memory + double buffering
2. Rust Memory Pool - LRU eviction, size classes
3. Rust Scheduler - Task management, bandwidth pacing
4. Rust Async Transfer Engine - Separate H2D/D2H streams
5. Rust Kernel Dispatch Controller - Per-stream launch management
"""

import os
import sys
import time

# Add CUDA DLLs to PATH
cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
cuda_bin = os.path.join(cuda_path, "bin")
if cuda_bin not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_bin)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# =============================================================================
# Header
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_section(title: str):
    print(f"\n--- {title} ---")

# =============================================================================
# Main Demo
# =============================================================================

def main():
    print_header("PyGPUkit v0.2 Full Feature Demo")

    # Import modules
    try:
        import pygpukit
        native = pygpukit._pygpukit_native
        import _pygpukit_rust as rust
        print(f"PyGPUkit version: {pygpukit.__version__}")
        print("Native module loaded: OK")
        print("Rust module loaded: OK")
    except ImportError as e:
        print(f"Import error: {e}")
        return 1

    # Environment info
    print_section("Environment")
    print(f"CUDA available: {native.is_cuda_available()}")

    if not native.is_cuda_available():
        print("GPU not available, exiting")
        return 1

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute Capability: {props.compute_capability_major}.{props.compute_capability_minor}")
    print(f"SMs: {props.multiprocessor_count}")

    # =========================================================================
    # 1. Rust Memory Pool Demo
    # =========================================================================
    print_header("1. Rust Memory Pool")

    pool = rust.MemoryPool(
        quota=100 * 1024 * 1024,  # 100 MB quota
        enable_eviction=True
    )
    print(f"Created pool with 100 MB quota, eviction enabled")

    # Allocate blocks
    block_ids = []
    for i, size in enumerate([1024, 4096, 16384, 65536]):
        block_id = pool.allocate(size)
        block_ids.append(block_id)
        block = pool.get_block(block_id)
        print(f"  Allocated block {block.id}: {block.size} bytes")

    stats = pool.stats()
    print(f"\nPool stats:")
    print(f"  Active: {stats.active_blocks} blocks, {stats.used} bytes")
    print(f"  Allocations: {stats.allocation_count}")
    print(f"  Quota usage: {stats.used / stats.quota:.1%}")

    # Free and reuse
    pool.free(block_ids[0])
    print(f"\nFreed block {block_ids[0]}")

    new_block_id = pool.allocate(1024)
    print(f"Allocated new 1024-byte block: {new_block_id} (should reuse free list)")

    stats = pool.stats()
    print(f"Reuse count: {stats.reuse_count}")

    # =========================================================================
    # 2. Rust Scheduler Demo
    # =========================================================================
    print_header("2. Rust Scheduler")

    scheduler = rust.Scheduler(
        sched_tick_ms=10.0,
        window_ms=100.0,
        total_memory=1024 * 1024 * 1024  # 1 GB
    )
    print("Created scheduler (10ms tick, 100ms window, 1GB memory)")

    # Submit tasks
    task_ids = []
    for i in range(5):
        task = rust.TaskMeta(
            id=f"task_{i}",
            name=f"Layer {i}",
            memory_estimate=100 * 1024 * 1024,  # 100 MB
            priority=i % 3
        )
        task_id = scheduler.submit(task)
        task_ids.append(task_id)
        print(f"  Submitted: {task_id} (priority: {i % 3})")

    sched_stats = scheduler.stats()
    print(f"\nPending tasks: {sched_stats.pending_count}")

    # Run tasks (returns list of task_ids)
    runnable_ids = scheduler.get_runnable_tasks(max_tasks=3)
    print(f"Runnable tasks: {len(runnable_ids)}")
    for task_id in runnable_ids:
        task = scheduler.get_task(task_id)
        print(f"  {task.id}: state={task.state}")

    # Start and complete a task
    if runnable_ids:
        scheduler.start_task(runnable_ids[0])
        scheduler.complete_task(runnable_ids[0])
        print(f"\nCompleted: {runnable_ids[0]}")

    sched_stats = scheduler.stats()
    print(f"\nScheduler stats:")
    print(f"  Total submitted: {sched_stats.total_submitted}")
    print(f"  Completed: {sched_stats.completed_count}")
    print(f"  Pending: {sched_stats.pending_count}")
    print(f"  Reserved memory: {sched_stats.reserved_memory / 1024 / 1024:.0f} MB")

    # =========================================================================
    # 3. Rust Async Transfer Engine Demo
    # =========================================================================
    print_header("3. Rust Async Transfer Engine")

    transfer_engine = rust.AsyncTransferEngine(max_concurrent=4)
    print("Created transfer engine (max 4 concurrent)")

    # Queue transfers (using enqueue_with_priority, type is string: "h2d", "d2h", "d2d")
    transfer_ids = []
    for i in range(6):
        type_name = "h2d" if i % 2 == 0 else "d2h"
        op_id = transfer_engine.enqueue_with_priority(
            transfer_type=type_name,
            src_ptr=0x1000 + i * 0x1000,
            dst_ptr=0x2000 + i * 0x1000,
            size=1024 * 1024,  # 1 MB
            priority=i % 3
        )
        transfer_ids.append(op_id)
        print(f"  Queued transfer {op_id}: {type_name.upper()}, priority={i % 3}")

    # Get ready transfers
    ready = transfer_engine.get_ready_transfers(max_transfers=4)
    print(f"\nReady transfers: {len(ready)}")
    for op in ready:
        print(f"  {op.id}: {op.size} bytes")

    # Simulate completion
    for op in ready[:2]:
        transfer_engine.start_transfer(op.id)
        transfer_engine.complete_transfer(op.id)
        print(f"  Completed transfer {op.id}")

    transfer_stats = transfer_engine.stats()
    print(f"\nTransfer stats:")
    print(f"  Total queued: {transfer_stats.total_queued}")
    print(f"  Completed: {transfer_stats.completed_count}")
    print(f"  Pending: {transfer_stats.pending_count}")

    # =========================================================================
    # 4. Rust Kernel Dispatch Controller Demo
    # =========================================================================
    print_header("4. Rust Kernel Dispatch Controller")

    dispatcher = rust.KernelDispatcher(max_in_flight=4)
    print("Created dispatcher (max 4 in-flight per stream)")

    # Queue kernel launches
    for i in range(8):
        config = rust.LaunchConfig(
            grid=(128, 1, 1),
            block=(256, 1, 1),
            shared_mem=0,
            stream_id=i % 2  # Alternate between stream 0 and 1
        )
        req_id = dispatcher.queue(
            kernel_handle=0xDEADBEEF + i,
            config=config,
            priority=i % 3
        )
        print(f"  Queued kernel {req_id}: stream={i % 2}, priority={i % 3}")

    # Get ready kernels
    ready_kernels = dispatcher.get_ready(max_requests=6)
    print(f"\nReady kernels: {len(ready_kernels)}")
    for req in ready_kernels:
        print(f"  {req.id}: kernel=0x{req.kernel_handle:X}, stream={req.config.stream_id}")

    # Simulate launch and completion
    for req in ready_kernels[:4]:
        dispatcher.mark_launched(req.id)
        dispatcher.mark_completed(req.id)

    dispatch_stats = dispatcher.stats()
    print(f"\nDispatch stats:")
    print(f"  Total queued: {dispatch_stats.total_queued}")
    print(f"  Completed: {dispatch_stats.completed_count}")
    print(f"  Pending: {dispatch_stats.pending_count}")
    print(f"  In-flight: {dispatch_stats.in_flight_count}")

    # =========================================================================
    # 5. Tiled Matmul Benchmark
    # =========================================================================
    print_header("5. Tiled Matmul Benchmark")

    print("\nMatrix Size | Kernel    | Time (ms) | GFLOPS  | vs NumPy")
    print("-" * 60)

    sizes = [512, 1024, 2048, 4096]
    results = []

    for size in sizes:
        M, N, K = size, size, size

        # Create test matrices
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        # Warmup
        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)
        _ = native.matmul(A_gpu, B_gpu)

        # Benchmark GPU
        iterations = 5 if size >= 2048 else 10
        times = []
        for _ in range(iterations):
            A_gpu = native.from_numpy(A_np)
            B_gpu = native.from_numpy(B_np)
            start = time.perf_counter()
            C_gpu = native.matmul(A_gpu, B_gpu)
            gpu_time = time.perf_counter() - start
            times.append(gpu_time)

        avg_time = np.median(times)
        gflops = 2 * M * N * K / avg_time / 1e9

        # Kernel type (threshold is 2048)
        kernel = "Tiled" if size >= 2048 else "L2-opt"

        # NumPy reference
        start = time.perf_counter()
        C_cpu = np.matmul(A_np, B_np)
        cpu_time = time.perf_counter() - start

        speedup = cpu_time / avg_time

        # Verify
        C_result = C_gpu.to_numpy()
        rel_error = np.max(np.abs(C_result - C_cpu)) / np.max(np.abs(C_cpu))

        results.append({
            'size': size,
            'kernel': kernel,
            'time_ms': avg_time * 1000,
            'gflops': gflops,
            'speedup': speedup,
            'error': rel_error
        })

        status = "OK" if rel_error < 1e-3 else f"ERR:{rel_error:.1e}"
        print(f"{size:>5}x{size:<5} | {kernel:<9} | {avg_time*1000:>8.2f} | {gflops:>7.1f} | {speedup:>5.1f}x ({status})")

    print("-" * 60)

    # Peak performance
    peak = max(results, key=lambda x: x['gflops'])
    print(f"\nPeak: {peak['gflops']:.1f} GFLOPS at {peak['size']}x{peak['size']} ({peak['kernel']})")

    # =========================================================================
    # 6. Integrated Demo - Full Pipeline
    # =========================================================================
    print_header("6. Integrated Demo - Full Pipeline")

    print("\nSimulating a deep learning inference pipeline:")
    print("  1. Scheduler manages task queue")
    print("  2. Memory pool allocates GPU buffers")
    print("  3. Transfer engine handles H2D/D2H")
    print("  4. Dispatcher launches kernels")
    print("  5. Tiled matmul for compute")

    # Reset components
    pool.clear()
    scheduler.clear()

    # Simulate batch processing
    batch_size = 32
    hidden_dim = 1024
    num_layers = 4

    print(f"\nBatch={batch_size}, Hidden={hidden_dim}, Layers={num_layers}")

    total_time = 0
    total_flops = 0

    for layer in range(num_layers):
        # Submit scheduler task
        task = rust.TaskMeta(
            id=f"layer_{layer}",
            name=f"Layer {layer}",
            memory_estimate=batch_size * hidden_dim * 4 * 2,  # input + output
            priority=0
        )
        task_id = scheduler.submit(task)

        # Allocate memory
        input_block_id = pool.allocate(batch_size * hidden_dim * 4)
        weight_block_id = pool.allocate(hidden_dim * hidden_dim * 4)
        output_block_id = pool.allocate(batch_size * hidden_dim * 4)

        # Queue transfer
        h2d_id = transfer_engine.enqueue_h2d(
            host_ptr=0x1000,
            device_ptr=input_block_id,
            size=batch_size * hidden_dim * 4
        )

        # Queue kernel
        config = rust.LaunchConfig.linear(batch_size * hidden_dim, 256)
        kernel_id = dispatcher.queue(
            kernel_handle=0xFFFF0000 + layer,
            config=config,
            task_id=task_id,
            priority=0
        )

        # Actual compute
        A = np.random.randn(batch_size, hidden_dim).astype(np.float32)
        W = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)

        A_gpu = native.from_numpy(A)
        W_gpu = native.from_numpy(W)

        start = time.perf_counter()
        out_gpu = native.matmul(A_gpu, W_gpu)
        layer_time = time.perf_counter() - start

        total_time += layer_time
        total_flops += 2 * batch_size * hidden_dim * hidden_dim

        # Complete task
        scheduler.complete_task(task_id)

        # Free memory (except output for next layer)
        pool.free(input_block_id)
        pool.free(weight_block_id)

    throughput = total_flops / total_time / 1e9

    print(f"\nPipeline completed:")
    print(f"  Total time: {total_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} GFLOPS")
    print(f"  Tasks completed: {scheduler.stats().completed_count}")
    print(f"  Memory allocations: {pool.stats().allocation_count}")
    print(f"  Transfers queued: {transfer_engine.stats().total_queued}")
    print(f"  Kernels dispatched: {dispatcher.stats().total_queued}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Summary - PyGPUkit v0.2 Features")

    print("""
    1. Tiled Matmul Kernel
       - Shared memory tiling (64x64 tiles)
       - Double-buffered prefetch
       - Up to 6+ TFLOPS on RTX 3090 Ti

    2. Rust Memory Pool
       - LRU eviction
       - Size-class free lists
       - Thread-safe (parking_lot::RwLock)

    3. Rust Scheduler
       - Priority-based task queue
       - Memory reservation
       - Bandwidth pacing

    4. Rust Async Transfer Engine
       - Separate H2D/D2H streams
       - Priority ordering
       - Concurrent transfer limits

    5. Rust Kernel Dispatch Controller
       - Per-stream in-flight limits
       - Scheduler task integration
       - Launch lifecycle tracking

    6. Driver-Only Mode (Infrastructure)
       - CUDA Driver API wrappers
       - Context management
       - No cudart dependency (when enabled)
    """)

    print("=" * 70)
    print(" Demo Complete!")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
