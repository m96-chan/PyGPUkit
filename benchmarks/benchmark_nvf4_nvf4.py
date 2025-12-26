#!/usr/bin/env python3
"""
Pure NVF4 GEMM Benchmark for SM120 (Blackwell GeForce)

Benchmarks NVF4 GEMM without quantization overhead to measure
pure tensor core performance.
"""

import time

import numpy as np


def benchmark_nvf4_nvf4(sizes: list[int], warmup: int = 5, iterations: int = 20):
    """Benchmark pure NVF4 GEMM at various sizes."""
    from pygpukit.core.backend import get_native_module
    from pygpukit.core.factory import zeros

    native = get_native_module()

    if not native.nvf4_nvf4_sm120_available():
        print("NVF4-NVF4 SM120 not available")
        return

    print("=" * 70)
    print("Pure NVF4 GEMM Benchmark (SM120 Blackwell GeForce)")
    print("=" * 70)

    # Get GPU info
    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SM: {props.compute_capability_major}.{props.compute_capability_minor}")
    print()
    print("Pre-quantized NVF4 data (no quantization overhead)")
    print()

    results = []

    for size in sizes:
        M, N, K = size, size, size
        flops = 2.0 * M * N * K  # FLOPs for GEMM

        # Allocate output buffer (BF16)
        D_gpu = zeros((M, N), dtype="bfloat16")
        D_native = D_gpu._get_native()  # Get native GPUArray

        # Warmup
        for _ in range(warmup):
            native.benchmark_gemm_nvf4_sm120(D_native, M, N, K)
        native.device_synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            native.device_synchronize()
            start = time.perf_counter()
            native.benchmark_gemm_nvf4_sm120(D_native, M, N, K)
            native.device_synchronize()
            end = time.perf_counter()
            times.append(end - start)

        median_time = np.median(times)
        min_time = np.min(times)
        tflops_median = flops / median_time / 1e12
        tflops_max = flops / min_time / 1e12

        results.append(
            {
                "size": size,
                "tflops_median": tflops_median,
                "tflops_max": tflops_max,
                "time_ms": median_time * 1000,
            }
        )

        print(
            f"{M}x{N}x{K}: {tflops_median:.2f} TFLOPS (median), "
            f"{tflops_max:.2f} TFLOPS (max), "
            f"time={median_time * 1000:.2f}ms"
        )

    print()
    print("=" * 70)
    print("Summary Table")
    print("=" * 70)
    print("| Size | TFLOPS (median) | TFLOPS (max) | Time (ms) |")
    print("|------|-----------------|--------------|-----------|")
    for r in results:
        print(
            f"| {r['size']}x{r['size']} | {r['tflops_median']:.2f} | "
            f"{r['tflops_max']:.2f} | {r['time_ms']:.2f} |"
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pure NVF4 GEMM Benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192, 12288, 16384],
        help="Matrix sizes to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")

    args = parser.parse_args()

    benchmark_nvf4_nvf4(args.sizes, args.warmup, args.iterations)
