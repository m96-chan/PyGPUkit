#!/usr/bin/env python3
"""Benchmark Native Int8 GEMM using dp4a CUDA cores (SM120)"""

import time

import numpy as np

from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def test_int8_native_gemm():
    """Test Native Int8 GEMM performance and correctness"""
    native = get_native_module()

    print("=" * 70)
    print("Native Int8 GEMM via dp4a Benchmark (SM120)")
    print("=" * 70)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")

    # Check availability
    if not native.int8_native_gemm_available():
        print("Native Int8 GEMM not available (requires SM61+ for dp4a)")
        return

    print("Native Int8 GEMM: available")
    print()

    # Correctness test with small values
    print("=== Correctness Test (small values -5 to 5) ===")
    M, N, K = 128, 128, 256

    np.random.seed(42)
    A_int8 = np.random.randint(-5, 6, (M, K), dtype=np.int8)
    B_int8 = np.random.randint(-5, 6, (N, K), dtype=np.int8)  # [N, K] for transposed B

    # Reference: C = A @ B.T
    C_ref = A_int8.astype(np.int32) @ B_int8.T.astype(np.int32)

    # GPU computation
    A_gpu = from_numpy(A_int8)
    B_gpu = from_numpy(B_int8)
    D_gpu = from_numpy(np.zeros((M, N), dtype=np.int32))

    native.int8_native_gemm_sm120(A_gpu._get_native(), B_gpu._get_native(), D_gpu._get_native())
    native.device_synchronize()

    D_result = D_gpu.to_numpy()

    # Check correctness
    diff = np.abs(D_result.astype(np.float64) - C_ref.astype(np.float64))
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_error = diff.sum() / (np.abs(C_ref).sum() + 1e-10)

    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"Max absolute diff: {max_diff}")
    print(f"Mean absolute diff: {mean_diff:.4f}")
    print(f"Relative error: {rel_error * 100:.4f}%")
    print(f"Sample expected: {C_ref[0, :5]}")
    print(f"Sample got:      {D_result[0, :5]}")
    print()

    # Test with full Int8 range (-128 to 127)
    print("=== Correctness Test (full Int8 range -128 to 127) ===")
    A_int8_full = np.random.randint(-128, 128, (M, K), dtype=np.int8)
    B_int8_full = np.random.randint(-128, 128, (N, K), dtype=np.int8)

    C_ref_full = A_int8_full.astype(np.int32) @ B_int8_full.T.astype(np.int32)

    A_gpu_full = from_numpy(A_int8_full)
    B_gpu_full = from_numpy(B_int8_full)
    D_gpu_full = from_numpy(np.zeros((M, N), dtype=np.int32))

    native.int8_native_gemm_sm120(
        A_gpu_full._get_native(), B_gpu_full._get_native(), D_gpu_full._get_native()
    )
    native.device_synchronize()

    D_result_full = D_gpu_full.to_numpy()

    diff_full = np.abs(D_result_full.astype(np.float64) - C_ref_full.astype(np.float64))
    max_diff_full = diff_full.max()
    mean_diff_full = diff_full.mean()
    rel_error_full = diff_full.sum() / (np.abs(C_ref_full).sum() + 1e-10)

    print(f"Max absolute diff: {max_diff_full}")
    print(f"Mean absolute diff: {mean_diff_full:.4f}")
    print(f"Relative error: {rel_error_full * 100:.4f}%")
    print(f"Sample expected: {C_ref_full[0, :5]}")
    print(f"Sample got:      {D_result_full[0, :5]}")
    print()

    # Performance benchmark
    print("=== Performance Benchmark ===")
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'TFLOPS':>10} | {'us':>10}")
    print("-" * 50)

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
        # Generate random Int8 values
        A_int8 = np.random.randint(-128, 128, (M, K), dtype=np.int8)
        B_int8 = np.random.randint(-128, 128, (N, K), dtype=np.int8)

        A_gpu = from_numpy(A_int8)
        B_gpu = from_numpy(B_int8)
        D_gpu = from_numpy(np.zeros((M, N), dtype=np.int32))

        flops = 2 * M * N * K

        try:
            for _ in range(warmup):
                native.int8_native_gemm_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), D_gpu._get_native()
                )
            native.device_synchronize()

            times = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.int8_native_gemm_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), D_gpu._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)

            median_us = np.median(times)
            tflops = flops / median_us / 1e6
            print(f"{M:>6} {K:>6} {N:>6} | {tflops:>10.2f} | {median_us:>10.1f}")
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | ERROR: {e}")

    print()
    print("Note: Native Int8 GEMM uses dp4a CUDA cores (not TensorCore)")
    print("      Expect lower TFLOPS than FP8 TensorCore (~1.2 TOPS on RTX 5090)")
    print("      This kernel provides EXACT Int8 computation with Int32 accumulation")


if __name__ == "__main__":
    test_int8_native_gemm()
