"""Base benchmark class and utilities."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from .results import BenchmarkResult, GPUInfo


def get_gpu_info() -> GPUInfo:
    """Get GPU information."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    props = native.get_device_properties(0)

    return GPUInfo(
        name=props.name,
        sm_major=props.compute_capability_major,
        sm_minor=props.compute_capability_minor,
        memory_gb=props.total_memory / (1024**3),
    )


def measure_kernel(
    fn: Callable[[], Any],
    warmup: int = 10,
    iterations: int = 50,
    sync_fn: Callable[[], None] | None = None,
) -> tuple[float, float, float, float]:
    """Measure kernel execution time.

    Args:
        fn: Function to benchmark
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        sync_fn: Optional sync function (e.g., device_synchronize)

    Returns:
        (median_us, min_us, max_us, std_us)
    """
    if sync_fn is None:
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        sync_fn = native.device_synchronize

    # Warmup
    for _ in range(warmup):
        fn()
    sync_fn()

    # Benchmark
    times = []
    for _ in range(iterations):
        sync_fn()
        start = time.perf_counter()
        fn()
        sync_fn()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    times_arr = np.array(times)
    return (
        float(np.median(times_arr)),
        float(np.min(times_arr)),
        float(np.max(times_arr)),
        float(np.std(times_arr)),
    )


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    category: str = "unknown"
    warmup: int = 10
    iterations: int = 50

    def __init__(self, warmup: int | None = None, iterations: int | None = None):
        if warmup is not None:
            self.warmup = warmup
        if iterations is not None:
            self.iterations = iterations

    @abstractmethod
    def run(self) -> list[BenchmarkResult]:
        """Run the benchmark and return results."""
        pass

    def _measure(
        self,
        name: str,
        fn: Callable[[], Any],
        params: dict[str, Any],
        flops: float | None = None,
        bytes_moved: float | None = None,
        check_fn: Callable[[], tuple[bool, float]] | None = None,
    ) -> BenchmarkResult:
        """Measure a single benchmark case."""
        median_us, min_us, max_us, std_us = measure_kernel(
            fn, warmup=self.warmup, iterations=self.iterations
        )

        # Calculate TFLOPS if flops provided
        tflops = None
        if flops is not None and median_us > 0:
            tflops = flops / median_us / 1e6  # TFLOPS = flops / us / 1e6

        # Calculate bandwidth if bytes provided
        bandwidth = None
        if bytes_moved is not None and median_us > 0:
            bandwidth = bytes_moved / median_us / 1e3  # GB/s = bytes / us / 1e3

        # Check correctness
        correct = True
        rel_error = 0.0
        if check_fn is not None:
            correct, rel_error = check_fn()

        return BenchmarkResult(
            name=name,
            category=self.category,
            params=params,
            median_us=median_us,
            min_us=min_us,
            max_us=max_us,
            std_us=std_us,
            tflops=tflops,
            bandwidth_gbps=bandwidth,
            correct=correct,
            rel_error=rel_error,
            iterations=self.iterations,
        )
