#!/usr/bin/env python3
"""Benchmark large matrices."""

import sys

sys.path.insert(0, "src")
import time

import numpy as np

import pygpukit as gp

sizes = [4096]
for size in sizes:
    np.random.seed(42)
    a_np = np.random.rand(size, size).astype(np.float32)
    b_np = np.random.rand(size, size).astype(np.float32)

    # NumPy
    start = time.perf_counter()
    _ = np.matmul(a_np, b_np)
    numpy_ms = (time.perf_counter() - start) * 1000

    # GPU
    a_gpu = gp.from_numpy(a_np)
    b_gpu = gp.from_numpy(b_np)
    _ = gp.matmul(a_gpu, b_gpu)  # warmup

    start = time.perf_counter()
    _ = gp.matmul(a_gpu, b_gpu)
    gpu_ms = (time.perf_counter() - start) * 1000

    flops = 2 * size * size * size
    gflops = flops / (gpu_ms / 1000) / 1e9

    print(
        f"{size}x{size}: NumPy={numpy_ms:.1f}ms, GPU={gpu_ms:.1f}ms, Speedup={numpy_ms/gpu_ms:.1f}x, {gflops:.0f} GFLOPS"
    )
