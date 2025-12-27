#!/usr/bin/env python3
"""
Comprehensive GEMV Benchmark for README.md

All GEMV kernels with LLM-relevant sizes, reporting in microseconds.
"""

import time

import numpy as np

import pygpukit as gk
from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def benchmark_gemv_all():
    """Comprehensive GEMV benchmark for all formats."""
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
    print("Comprehensive GEMV Benchmark (RTX 5090)")
    print("=" * 80)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print()

    # LLM-relevant configurations
    # (K, N) - K is hidden dim, N is output dim
    configs = [
        # Qwen-7B style
        (4096, 4096, "Qwen-7B hidden"),
        (4096, 14336, "Qwen-7B MLP up"),
        (14336, 4096, "Qwen-7B MLP down"),
        # Qwen-72B style
        (8192, 8192, "Qwen-72B hidden"),
        (8192, 29568, "Qwen-72B MLP up"),
        (29568, 8192, "Qwen-72B MLP down"),
    ]

    warmup = 10
    iterations = 50

    # Results table
    results = []

    for K, N, label in configs:
        print(f"\n{label}: K={K}, N={N}")

        # ===== BF16 GEMV =====
        A_bf16 = gk.empty((K,), dtype="bfloat16")
        B_bf16 = gk.empty((K, N), dtype="bfloat16")
        C_bf16 = gk.empty((N,), dtype="bfloat16")

        for _ in range(warmup):
            gemv_bf16(A_bf16, B_bf16, out=C_bf16)
        native.device_synchronize()

        times_bf16 = []
        for _ in range(iterations):
            native.device_synchronize()
            start = time.perf_counter()
            gemv_bf16(A_bf16, B_bf16, out=C_bf16)
            native.device_synchronize()
            end = time.perf_counter()
            times_bf16.append((end - start) * 1e6)

        median_bf16 = np.median(times_bf16)

        # ===== FP8 GEMV =====
        try:
            A_fp8 = gk.empty((K,), dtype="bfloat16")
            B_fp8_nk = from_numpy(np.zeros((N, K), dtype=np.uint8))
            n_blocks = (N + 127) // 128
            k_blocks = (K + 127) // 128
            B_scale_fp8 = from_numpy(
                np.ones((n_blocks, k_blocks), dtype=np.float16).view(np.uint16)
            )
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
        except Exception:
            median_fp8 = float("inf")

        # ===== NVF4 GEMV =====
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
        else:
            median_nvf4 = float("inf")

        # ===== Int4 GEMV =====
        try:
            if native.int4_gemv_available():

                def pack_int4(values: np.ndarray) -> np.ndarray:
                    flat = values.reshape(-1)
                    low = flat[0::2].astype(np.int32) & 0x0F
                    high = flat[1::2].astype(np.int32) & 0x0F
                    packed = (high << 4) | low
                    new_shape = list(values.shape)
                    new_shape[-1] //= 2
                    return packed.astype(np.uint8).reshape(new_shape)

                A_int4_raw = np.random.randint(-8, 8, K, dtype=np.int8)
                B_int4_raw = np.random.randint(-8, 8, (N, K), dtype=np.int8)
                A_int4 = from_numpy(pack_int4(A_int4_raw.reshape(1, -1)).reshape(-1))
                B_int4 = from_numpy(pack_int4(B_int4_raw))
                C_int4 = from_numpy(np.zeros(N, dtype=np.int32))

                for _ in range(warmup):
                    native.int4_gemv_int32_sm120(
                        A_int4._get_native(), B_int4._get_native(), C_int4._get_native()
                    )
                native.device_synchronize()

                times_int4 = []
                for _ in range(iterations):
                    native.device_synchronize()
                    start = time.perf_counter()
                    native.int4_gemv_int32_sm120(
                        A_int4._get_native(), B_int4._get_native(), C_int4._get_native()
                    )
                    native.device_synchronize()
                    end = time.perf_counter()
                    times_int4.append((end - start) * 1e6)

                median_int4 = np.median(times_int4)
            else:
                median_int4 = float("inf")
        except Exception:
            median_int4 = float("inf")

        results.append(
            {
                "label": label,
                "K": K,
                "N": N,
                "bf16": median_bf16,
                "fp8": median_fp8,
                "nvf4": median_nvf4,
                "int4": median_int4,
            }
        )

        print(f"  BF16: {median_bf16:.1f} us")
        print(f"  FP8:  {median_fp8:.1f} us")
        if median_nvf4 != float("inf"):
            print(f"  NVF4: {median_nvf4:.1f} us")
        if median_int4 != float("inf"):
            print(f"  Int4: {median_int4:.1f} us")

    # Print README table
    print("\n" + "=" * 80)
    print("README.md Table (GEMV Performance)")
    print("=" * 80)
    print()
    print("| Layer | K | N | BF16 | FP8 | NVF4 | Int4 |")
    print("|-------|------|-------|------|-----|------|------|")

    for r in results:
        bf16_str = f"{r['bf16']:.0f} us"
        fp8_str = f"{r['fp8']:.0f} us"
        nvf4_str = f"{r['nvf4']:.0f} us" if r["nvf4"] != float("inf") else "—"
        int4_str = f"{r['int4']:.0f} us" if r["int4"] != float("inf") else "—"
        print(
            f"| {r['label']} | {r['K']} | {r['N']} | {bf16_str} | {fp8_str} | {nvf4_str} | {int4_str} |"
        )


if __name__ == "__main__":
    benchmark_gemv_all()
