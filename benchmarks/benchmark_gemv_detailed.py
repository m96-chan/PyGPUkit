#!/usr/bin/env python3
"""
Detailed GEMV Benchmark with individual timing per iteration.

Compares: BF16, FP8, NVFP4 GEMV kernels.
"""

import time

import numpy as np

import pygpukit as gk
from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def benchmark_gemv_detailed():
    """Detailed GEMV benchmark with per-iteration timing."""
    from pygpukit.ops.matmul import (
        fp8_init_lut,
        gemv_bf16,
        gemv_fp8_bf16,
        gemv_nvf4_available,
        gemv_nvf4_bf16,
    )

    native = get_native_module()
    fp8_init_lut()

    print("=" * 80)
    print("Detailed GEMV Benchmark")
    print("=" * 80)

    # Get GPU info
    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print("Memory Bandwidth: ~1792 GB/s (theoretical)")
    print()

    configs = [
        (4096, 4096),
        (14336, 4096),
        (4096, 14336),
    ]

    warmup = 10
    iterations = 50

    for N, K in configs:
        print(f"\n{'=' * 60}")
        print(f"N={N}, K={K}")
        print(f"{'=' * 60}")

        # Calculate theoretical bandwidth
        # BF16: B is K*N*2 bytes, A is K*2 bytes
        bf16_bytes = K * N * 2 + K * 2
        # FP8: B is N*K bytes, A is K*2 bytes, scale is (N/128)*(K/128)*2 bytes
        fp8_bytes = N * K + K * 2 + ((N + 127) // 128) * ((K + 127) // 128) * 2
        # NVF4: B is N*K/2 bytes, A is K*2 bytes, scale is (K/32)*N bytes
        nvf4_bytes = N * (K // 2) + K * 2 + ((K + 31) // 32) * N

        print(
            f"Data sizes: BF16={bf16_bytes / 1e6:.1f}MB, FP8={fp8_bytes / 1e6:.1f}MB, NVF4={nvf4_bytes / 1e6:.1f}MB"
        )
        print(
            f"Theoretical time @1000GB/s: BF16={bf16_bytes / 1e9 * 1e6:.1f}us, FP8={fp8_bytes / 1e9 * 1e6:.1f}us"
        )
        print()

        # ===== BF16 GEMV =====
        A_bf16 = gk.empty((K,), dtype="bfloat16")
        B_bf16 = gk.empty((K, N), dtype="bfloat16")
        C_bf16 = gk.empty((N,), dtype="bfloat16")

        # Warmup
        for _ in range(warmup):
            gemv_bf16(A_bf16, B_bf16, out=C_bf16)
        native.device_synchronize()

        # Benchmark with individual timing
        times_bf16 = []
        for _ in range(iterations):
            native.device_synchronize()
            start = time.perf_counter()
            gemv_bf16(A_bf16, B_bf16, out=C_bf16)
            native.device_synchronize()
            end = time.perf_counter()
            times_bf16.append((end - start) * 1e6)

        median_bf16 = np.median(times_bf16)
        min_bf16 = np.min(times_bf16)
        print(
            f"BF16:  median={median_bf16:.1f}us, min={min_bf16:.1f}us, "
            f"BW={bf16_bytes / median_bf16 / 1e3:.0f}GB/s"
        )

        # ===== FP8 GEMV (optimized, B[N,K] layout) =====
        A_fp8 = gk.empty((K,), dtype="bfloat16")
        B_fp8_nk = from_numpy(np.zeros((N, K), dtype=np.uint8))  # [N, K] layout
        n_blocks = (N + 127) // 128
        k_blocks = (K + 127) // 128
        B_scale_fp8 = from_numpy(np.ones((n_blocks, k_blocks), dtype=np.float16).view(np.uint16))
        C_fp8 = gk.empty((N,), dtype="bfloat16")

        for _ in range(warmup):
            gemv_fp8_bf16(A_fp8, B_fp8_nk, B_scale_fp8, out=C_fp8)
        native.device_synchronize()

        times_fp8 = []
        for _ in range(iterations):
            native.device_synchronize()
            start = time.perf_counter()
            gemv_fp8_bf16(A_fp8, B_fp8_nk, B_scale_fp8, out=C_fp8)
            native.device_synchronize()
            end = time.perf_counter()
            times_fp8.append((end - start) * 1e6)

        median_fp8 = np.median(times_fp8)
        min_fp8 = np.min(times_fp8)
        print(
            f"FP8:   median={median_fp8:.1f}us, min={min_fp8:.1f}us, "
            f"BW={fp8_bytes / median_fp8 / 1e3:.0f}GB/s"
        )

        # ===== NVFP4 GEMV =====
        if gemv_nvf4_available():
            A_nvf4 = gk.empty((K,), dtype="bfloat16")
            B_nvf4 = from_numpy(np.zeros((K // 2, N), dtype=np.uint8))
            k_scale_blocks = (K + 31) // 32
            B_scale_nvf4 = from_numpy(np.ones((k_scale_blocks, N), dtype=np.uint8))
            C_nvf4 = gk.empty((N,), dtype="bfloat16")

            for _ in range(warmup):
                gemv_nvf4_bf16(A_nvf4, B_nvf4, B_scale_nvf4, out=C_nvf4)
            native.device_synchronize()

            times_nvf4 = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                gemv_nvf4_bf16(A_nvf4, B_nvf4, B_scale_nvf4, out=C_nvf4)
                native.device_synchronize()
                end = time.perf_counter()
                times_nvf4.append((end - start) * 1e6)

            median_nvf4 = np.median(times_nvf4)
            min_nvf4 = np.min(times_nvf4)
            print(
                f"NVFP4: median={median_nvf4:.1f}us, min={min_nvf4:.1f}us, "
                f"BW={nvf4_bytes / median_nvf4 / 1e3:.0f}GB/s"
            )
        else:
            median_nvf4 = float("inf")
            print("NVFP4: N/A")

        # Summary
        print()
        print("Speedup vs BF16:")
        print(f"  FP8:   {median_bf16 / median_fp8:.2f}x")
        if gemv_nvf4_available():
            print(f"  NVFP4: {median_bf16 / median_nvf4:.2f}x")


if __name__ == "__main__":
    benchmark_gemv_detailed()
