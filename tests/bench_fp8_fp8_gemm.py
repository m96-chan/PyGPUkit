#!/usr/bin/env python3
"""Benchmark FP8xFP8 GEMM - Comparing v2 (uncached) vs v5 (cached scale buffers)."""

import time

import numpy as np

from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def bench_fp8_fp8_gemm():
    """Benchmark FP8xFP8 GEMM variants."""
    native = get_native_module()

    print("=" * 70)
    print("FP8xFP8 GEMM Benchmark - v2 (uncached) vs v5 (cached)")
    print("=" * 70)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print()

    # Test configurations
    configs = [
        (1024, 4096, 14336),
        (2048, 4096, 14336),
        (4096, 4096, 14336),
        (8192, 4096, 14336),
    ]

    # Kernel variants to test (only the working ones)
    variants = [
        ("v2 (uncached 128x256x64)", "gemm_fp8_fp8_sm120_v2"),
        ("v5 (cached 128x128x128)", "gemm_fp8_fp8_sm120_v5"),
    ]

    warmup = 5
    iterations = 20

    print(f"{'Config':<25} {'v2 (uncached)':<18} {'v5 (cached)':<18} {'Speedup':<10}")
    print("-" * 70)

    for M, K, N in configs:
        # Create FP8 tensors
        A_fp8 = from_numpy(np.random.randint(0, 256, (M, K), dtype=np.uint8))
        B_fp8 = from_numpy(np.random.randint(0, 256, (K, N), dtype=np.uint8))
        C_fp8 = from_numpy(np.zeros((M, N), dtype=np.uint8))

        flops = 2 * M * N * K
        results = {}

        for name, func_name in variants:
            func = getattr(native, func_name, None)
            if func is None:
                results[name] = None
                continue

            try:
                # Warmup
                for _ in range(warmup):
                    func(A_fp8._get_native(), B_fp8._get_native(), C_fp8._get_native())
                native.device_synchronize()

                # Benchmark
                times = []
                for _ in range(iterations):
                    native.device_synchronize()
                    start = time.perf_counter()
                    func(A_fp8._get_native(), B_fp8._get_native(), C_fp8._get_native())
                    native.device_synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1e6)

                median_us = np.median(times)
                tflops = flops / median_us / 1e6
                results[name] = tflops

            except Exception as e:
                results[name] = None

        # Print results
        config_str = f"M={M}, K={K}, N={N}"
        v2_tflops = results.get("v2 (uncached 128x256x64)")
        v5_tflops = results.get("v5 (cached 128x128x128)")

        v2_str = f"{v2_tflops:.1f} TFLOPS" if v2_tflops else "N/A"
        v5_str = f"{v5_tflops:.1f} TFLOPS" if v5_tflops else "N/A"

        if v2_tflops and v5_tflops:
            speedup = v5_tflops / v2_tflops
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{config_str:<25} {v2_str:<18} {v5_str:<18} {speedup_str:<10}")

    print()
    print("v5 uses cached scale factor buffers to avoid per-call allocation overhead.")


if __name__ == "__main__":
    bench_fp8_fp8_gemm()
