#!/usr/bin/env python3
"""Benchmark: Tiled matmul vs NumPy.

Demonstrates the performance improvement from shared memory tiling.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

import pygpukit as gp
from pygpukit.core.backend import get_backend


def benchmark_matmul(size: int, iterations: int = 10) -> dict:
    """Benchmark matmul for a given matrix size."""
    np.random.seed(42)

    # Create test data
    a_np = np.random.rand(size, size).astype(np.float32)
    b_np = np.random.rand(size, size).astype(np.float32)

    # NumPy benchmark
    numpy_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = np.matmul(a_np, b_np)
        numpy_times.append(time.perf_counter() - start)
    numpy_avg = np.mean(numpy_times) * 1000  # ms

    # PyGPUkit benchmark
    a_gpu = gp.from_numpy(a_np)
    b_gpu = gp.from_numpy(b_np)

    # Warm-up
    _ = gp.matmul(a_gpu, b_gpu)

    gpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = gp.matmul(a_gpu, b_gpu)
        gpu_times.append(time.perf_counter() - start)
    gpu_avg = np.mean(gpu_times) * 1000  # ms

    # Calculate GFLOPS (2 * N^3 FLOPs for matmul)
    flops = 2 * size * size * size
    gpu_gflops = flops / (gpu_avg / 1000) / 1e9
    numpy_gflops = flops / (numpy_avg / 1000) / 1e9

    return {
        "size": size,
        "numpy_ms": numpy_avg,
        "gpu_ms": gpu_avg,
        "speedup": numpy_avg / gpu_avg,
        "numpy_gflops": numpy_gflops,
        "gpu_gflops": gpu_gflops,
    }


def main():
    print("=" * 70)
    print("  PyGPUkit Tiled Matmul Benchmark")
    print("=" * 70)
    print()

    # Get backend info
    backend = get_backend()
    props = backend.get_device_properties()
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"SMs: {props.multiprocessor_count}")
    print()

    # Benchmark various sizes
    sizes = [128, 256, 512, 1024, 2048]

    print("Running benchmarks (10 iterations each)...")
    print()

    results = []
    for size in sizes:
        print(f"  Testing {size}x{size}...", end=" ", flush=True)
        result = benchmark_matmul(size)
        results.append(result)
        print(f"done ({result['gpu_ms']:.2f} ms)")

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(
        f"{'Size':>8} | {'NumPy (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>8} | {'GPU GFLOPS':>12}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['size']:>8} | {r['numpy_ms']:>12.3f} | {r['gpu_ms']:>12.3f} | {r['speedup']:>7.1f}x | {r['gpu_gflops']:>12.1f}"
        )

    print()
    print("=" * 70)
    print()

    # Peak performance
    best = max(results, key=lambda x: x["gpu_gflops"])
    print(f"Peak GPU Performance: {best['gpu_gflops']:.1f} GFLOPS at {best['size']}x{best['size']}")
    print(f"Best Speedup vs NumPy: {max(r['speedup'] for r in results):.1f}x")
    print()

    # Verify correctness
    print("Verifying correctness...")
    a_np = np.random.rand(256, 256).astype(np.float32)
    b_np = np.random.rand(256, 256).astype(np.float32)

    expected = np.matmul(a_np, b_np)
    result = gp.matmul(gp.from_numpy(a_np), gp.from_numpy(b_np)).to_numpy()

    max_diff = np.max(np.abs(expected - result))
    print(f"Max difference from NumPy: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("[OK] Results match NumPy (within tolerance)")
    else:
        print("[FAIL] Results differ from NumPy!")

    print()


if __name__ == "__main__":
    main()
