#!/usr/bin/env python3
"""Quick benchmark for CUTLASS FP8×FP8 GEMM."""

import time

import numpy as np

from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def bench_fp8_fp8_gemm():
    """Benchmark FP8×FP8 GEMM."""
    native = get_native_module()

    print("=" * 60)
    print("FP8×FP8 GEMM Benchmark (CUTLASS SM120)")
    print("=" * 60)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print()

    # Test configurations
    configs = [
        (128, 4096, 14336),
        (256, 4096, 14336),
        (512, 4096, 14336),
        (1024, 4096, 14336),
        (2048, 4096, 14336),
        (4096, 4096, 14336),
        (8192, 4096, 14336),
    ]

    warmup = 5
    iterations = 20

    for M, K, N in configs:
        print(f"\nM={M}, K={K}, N={N}")

        # Create FP8 tensors (random uint8 as FP8)
        # A: [M, K] row-major
        # B: [K, N] row-major
        # C: [M, N] output
        A_fp8 = from_numpy(np.random.randint(0, 256, (M, K), dtype=np.uint8))
        B_fp8 = from_numpy(np.random.randint(0, 256, (K, N), dtype=np.uint8))
        C_fp8 = from_numpy(np.zeros((M, N), dtype=np.uint8))

        # FLOPS calculation
        flops = 2 * M * N * K

        try:
            # Warmup
            for _ in range(warmup):
                native.gemm_fp8_fp8_sm120(
                    A_fp8._get_native(), B_fp8._get_native(), C_fp8._get_native()
                )
            native.device_synchronize()

            # Benchmark
            times = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.gemm_fp8_fp8_sm120(
                    A_fp8._get_native(), B_fp8._get_native(), C_fp8._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)

            median_us = np.median(times)
            tflops = flops / median_us / 1e6

            print(f"  Time: {median_us:.1f} us")
            print(f"  Performance: {tflops:.1f} TFLOPS")

        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    bench_fp8_fp8_gemm()
