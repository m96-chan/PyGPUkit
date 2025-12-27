#!/usr/bin/env python3
"""Benchmark W8A16 GEMM: Hand-written vs CUTLASS."""

import time

import numpy as np

import pygpukit as gk
from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module
from pygpukit.ops.matmul import fp8_init_lut


def f32_to_bf16_numpy(f32: np.ndarray) -> np.ndarray:
    """Convert float32 to bfloat16 (stored as uint16)."""
    uint32_view = f32.view(np.uint32)
    bf16_data = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)
    return bf16_data


def bench_w8a16_gemm():
    """Benchmark W8A16 GEMM variants."""
    native = get_native_module()
    fp8_init_lut()

    print("=" * 70)
    print("W8A16 GEMM Benchmark: Hand-written vs CUTLASS (SM120)")
    print("=" * 70)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
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

    print(f"{'M':>6} {'K':>6} {'N':>6} | {'Hand-written':>14} | {'CUTLASS':>14} | {'Speedup':>8}")
    print("-" * 70)

    for M, K, N in configs:
        # A: [M, K] BF16 activation
        A_f32 = np.random.randn(M, K).astype(np.float32)
        A_bf16_np = f32_to_bf16_numpy(A_f32)
        A_bf16 = from_numpy(A_bf16_np)
        A_bf16._dtype = gk.core.dtypes.bfloat16

        # B_fp8: [K, N] FP8 weights (as uint8) - RowMajor for hand-written kernel
        B_fp8_row = from_numpy(np.random.randint(0, 256, (K, N), dtype=np.uint8))
        # B_fp8_col: [N, K] for CUTLASS ColumnMajor (transposed storage)
        B_fp8_col = from_numpy(
            np.ascontiguousarray(np.random.randint(0, 256, (K, N), dtype=np.uint8).T)
        )

        # B_scale: [K/128, N/128] BF16 scale factors for hand-written kernel
        scale_k = (K + 127) // 128
        scale_n = (N + 127) // 128
        scale_f32 = np.ones((scale_k, scale_n), dtype=np.float32)
        scale_bf16_np = f32_to_bf16_numpy(scale_f32)
        B_scale = from_numpy(scale_bf16_np)
        B_scale._dtype = gk.core.dtypes.bfloat16

        # Output buffers
        C_hand = gk.empty((M, N), dtype="bfloat16")
        C_cutlass = gk.empty((M, N), dtype="bfloat16")

        flops = 2 * M * N * K

        # Benchmark hand-written kernel
        try:
            # Warmup
            for _ in range(warmup):
                native.w8a16_gemm_sm120(
                    A_bf16._get_native(),
                    B_fp8_row._get_native(),
                    B_scale._get_native(),
                    C_hand._get_native(),
                )
            native.device_synchronize()

            # Benchmark
            times_hand = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.w8a16_gemm_sm120(
                    A_bf16._get_native(),
                    B_fp8_row._get_native(),
                    B_scale._get_native(),
                    C_hand._get_native(),
                )
                native.device_synchronize()
                end = time.perf_counter()
                times_hand.append((end - start) * 1e6)

            median_hand_us = np.median(times_hand)
            tflops_hand = flops / median_hand_us / 1e6
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | Hand-written ERROR: {e}")
            continue

        # Benchmark CUTLASS kernel
        try:
            # Warmup
            for _ in range(warmup):
                native.w8a16_cutlass_sm120(
                    A_bf16._get_native(), B_fp8_col._get_native(), C_cutlass._get_native()
                )
            native.device_synchronize()

            # Benchmark
            times_cutlass = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.w8a16_cutlass_sm120(
                    A_bf16._get_native(), B_fp8_col._get_native(), C_cutlass._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times_cutlass.append((end - start) * 1e6)

            median_cutlass_us = np.median(times_cutlass)
            tflops_cutlass = flops / median_cutlass_us / 1e6
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | {tflops_hand:>10.1f} T  | CUTLASS ERROR: {e}")
            continue

        speedup = tflops_cutlass / tflops_hand if tflops_hand > 0 else 0

        print(
            f"{M:>6} {K:>6} {N:>6} | {tflops_hand:>10.1f} T  | {tflops_cutlass:>10.1f} T  | {speedup:>6.2f}x"
        )

    print()
    print("T = TFLOPS")


if __name__ == "__main__":
    bench_w8a16_gemm()
