#!/usr/bin/env python3
"""Benchmark Int8 GEMM via FP8 approximation (SM120)."""

import time

import numpy as np

from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def bench_int8_gemm():
    """Benchmark Int8 GEMM performance."""
    native = get_native_module()

    print("=" * 70)
    print("Int8 GEMM Benchmark (SM120 via FP8 TensorCore)")
    print("=" * 70)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")

    # Check availability
    if not native.int8_gemm_available():
        print("Int8 GEMM not available on this GPU (requires SM120)")
        return

    print("Int8 GEMM: Available (via FP8 approximation)")
    print()

    # Test configurations (M, K, N) - typical LLM shapes
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

    print(f"{'M':>6} {'K':>6} {'N':>6} | {'Int8->Int32':>14} | {'Int8->Int8':>14} | {'Correct':>8}")
    print("-" * 70)

    for M, K, N in configs:
        # A: [M, K] Int8 (RowMajor)
        A_np = np.random.randint(-64, 64, (M, K), dtype=np.int8)
        A = from_numpy(A_np)

        # B: [N, K] Int8 (transposed for ColumnMajor)
        B_np = np.random.randint(-64, 64, (N, K), dtype=np.int8)
        B = from_numpy(B_np)

        # Output buffers (use from_numpy for int8/int32)
        D_int32_np = np.zeros((M, N), dtype=np.int32)
        D_int32 = from_numpy(D_int32_np)
        D_int8_np = np.zeros((M, N), dtype=np.int8)
        D_int8 = from_numpy(D_int8_np)

        # Theoretical OPs (2 * M * N * K)
        flops = 2 * M * N * K

        # Benchmark Int8 -> Int32
        try:
            # Warmup
            for _ in range(warmup):
                native.int8_gemm_int32_sm120(
                    A._get_native(), B._get_native(), D_int32._get_native()
                )
            native.device_synchronize()

            # Benchmark
            times_int32 = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.int8_gemm_int32_sm120(
                    A._get_native(), B._get_native(), D_int32._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times_int32.append((end - start) * 1e6)

            median_int32_us = np.median(times_int32)
            tflops_int32 = flops / median_int32_us / 1e6
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | Int32 ERROR: {e}")
            continue

        # Benchmark Int8 -> Int8
        try:
            # Warmup
            for _ in range(warmup):
                native.int8_gemm_int8_sm120(A._get_native(), B._get_native(), D_int8._get_native())
            native.device_synchronize()

            # Benchmark
            times_int8 = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.int8_gemm_int8_sm120(A._get_native(), B._get_native(), D_int8._get_native())
                native.device_synchronize()
                end = time.perf_counter()
                times_int8.append((end - start) * 1e6)

            median_int8_us = np.median(times_int8)
            tflops_int8 = flops / median_int8_us / 1e6
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | {tflops_int32:>10.1f} T  | Int8 ERROR: {e}")
            continue

        # Correctness check (compare with numpy)
        # Note: FP8 approximation won't be exact, so we check relative error
        D_int32_np = np.asarray(D_int32.to_numpy())
        ref_np = A_np.astype(np.int32) @ B_np.astype(np.int32).T

        # Calculate relative error
        max_val = np.abs(ref_np).max() + 1e-8
        max_diff = np.abs(D_int32_np - ref_np).max()
        rel_error = max_diff / max_val

        # FP8 approximation: allow larger error (FP8 E4M3 precision is ~1-2%)
        is_correct = rel_error < 0.15  # 15% tolerance for FP8 approximation

        status = "PASS" if is_correct else f"FAIL({rel_error:.1%})"
        print(
            f"{M:>6} {K:>6} {N:>6} | {tflops_int32:>10.1f} T  | {tflops_int8:>10.1f} T  | {status:>8}"
        )

    print()
    print("T = TFLOPS (effective Int8 ops)")
    print("Note: Uses FP8 TensorCore internally (~3.5% precision loss)")


if __name__ == "__main__":
    bench_int8_gemm()
