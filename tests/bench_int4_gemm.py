#!/usr/bin/env python3
"""Benchmark Int4 GEMM via Int8/FP8 approximation (SM120)"""

import time

import numpy as np

from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module


def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack signed 4-bit values into uint8 (2 values per byte, low nibble first)"""
    assert values.dtype == np.int8
    assert values.shape[-1] % 2 == 0

    flat = values.reshape(-1)
    # Convert to unsigned 4-bit (0-15 for signed -8 to 7)
    low = flat[0::2].astype(np.int32) & 0x0F
    high = flat[1::2].astype(np.int32) & 0x0F
    packed = (high << 4) | low

    new_shape = list(values.shape)
    new_shape[-1] //= 2
    return packed.astype(np.uint8).reshape(new_shape)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8 to signed 4-bit values"""
    flat = packed.reshape(-1)
    low = (flat & 0x0F).astype(np.int8)
    high = ((flat >> 4) & 0x0F).astype(np.int8)

    # Sign extend
    low = np.where(low > 7, low - 16, low).astype(np.int8)
    high = np.where(high > 7, high - 16, high).astype(np.int8)

    result = np.empty(len(flat) * 2, dtype=np.int8)
    result[0::2] = low
    result[1::2] = high

    new_shape = list(packed.shape)
    new_shape[-1] *= 2
    return result.reshape(new_shape)


def test_int4_gemm():
    """Test Int4 GEMM performance and correctness"""
    native = get_native_module()

    print("=" * 70)
    print("Int4 GEMM via Int8/FP8 Benchmark (SM120)")
    print("=" * 70)

    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")

    # Check availability
    if not native.int4_gemm_available():
        print("Int4 GEMM not available (requires SM120)")
        return

    print("Int4 GEMM: available")
    print()

    # Test with small values first (correctness)
    print("=== Correctness Test (small values -2 to 2) ===")
    M, N, K = 128, 128, 256

    # Generate small random Int4 values (-2 to 2 range)
    np.random.seed(42)
    A_int8 = np.random.randint(-2, 3, (M, K), dtype=np.int8)
    B_int8 = np.random.randint(-2, 3, (N, K), dtype=np.int8)  # [N, K] for transposed B

    # Pack to Int4
    A_packed = pack_int4(A_int8)
    B_packed = pack_int4(B_int8)

    print(f"A shape: {A_int8.shape} -> packed: {A_packed.shape}")
    print(f"B shape: {B_int8.shape} -> packed: {B_packed.shape}")

    # Reference: C = A @ B.T
    C_ref = A_int8.astype(np.int32) @ B_int8.T.astype(np.int32)

    # GPU computation
    A_gpu = from_numpy(A_packed)
    B_gpu = from_numpy(B_packed)
    D_gpu = from_numpy(np.zeros((M, N), dtype=np.int32))

    native.int4_gemm_int32_sm120(A_gpu._get_native(), B_gpu._get_native(), D_gpu._get_native())
    native.device_synchronize()

    D_result = D_gpu.to_numpy()

    # Check correctness
    diff = np.abs(D_result.astype(np.float64) - C_ref.astype(np.float64))
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_error = diff.sum() / (np.abs(C_ref).sum() + 1e-10)

    print(f"Max absolute diff: {max_diff}")
    print(f"Mean absolute diff: {mean_diff:.4f}")
    print(f"Relative error: {rel_error * 100:.4f}%")
    print(f"Sample expected: {C_ref[0, :5]}")
    print(f"Sample got:      {D_result[0, :5]}")
    print()

    # Test with full Int4 range (-8 to 7)
    print("=== Correctness Test (full Int4 range -8 to 7) ===")
    A_int8_full = np.random.randint(-8, 8, (M, K), dtype=np.int8)
    B_int8_full = np.random.randint(-8, 8, (N, K), dtype=np.int8)

    A_packed_full = pack_int4(A_int8_full)
    B_packed_full = pack_int4(B_int8_full)

    C_ref_full = A_int8_full.astype(np.int32) @ B_int8_full.T.astype(np.int32)

    A_gpu_full = from_numpy(A_packed_full)
    B_gpu_full = from_numpy(B_packed_full)
    D_gpu_full = from_numpy(np.zeros((M, N), dtype=np.int32))

    native.int4_gemm_int32_sm120(
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
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'Int4->Int32':>14} | {'Int4->Int8':>14}")
    print("-" * 56)

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
        # Generate random Int4 values
        A_int8 = np.random.randint(-8, 8, (M, K), dtype=np.int8)
        B_int8 = np.random.randint(-8, 8, (N, K), dtype=np.int8)

        A_packed = pack_int4(A_int8)
        B_packed = pack_int4(B_int8)

        A_gpu = from_numpy(A_packed)
        B_gpu = from_numpy(B_packed)
        D_int32 = from_numpy(np.zeros((M, N), dtype=np.int32))
        D_int8 = from_numpy(np.zeros((M, N), dtype=np.int8))

        flops = 2 * M * N * K

        # Benchmark Int4 -> Int32
        try:
            for _ in range(warmup):
                native.int4_gemm_int32_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), D_int32._get_native()
                )
            native.device_synchronize()

            times_int32 = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.int4_gemm_int32_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), D_int32._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times_int32.append((end - start) * 1e6)

            median_int32_us = np.median(times_int32)
            tflops_int32 = flops / median_int32_us / 1e6
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | ERROR: {e}")
            continue

        # Benchmark Int4 -> Int8
        try:
            for _ in range(warmup):
                native.int4_gemm_int8_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), D_int8._get_native()
                )
            native.device_synchronize()

            times_int8 = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.int4_gemm_int8_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), D_int8._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times_int8.append((end - start) * 1e6)

            median_int8_us = np.median(times_int8)
            tflops_int8 = flops / median_int8_us / 1e6
        except Exception as e:
            print(f"{M:>6} {K:>6} {N:>6} | {tflops_int32:>10.1f} T  | ERROR: {e}")
            continue

        print(f"{M:>6} {K:>6} {N:>6} | {tflops_int32:>10.1f} T  | {tflops_int8:>10.1f} T")

    print()
    print("T = TFLOPS (effective Int4 ops)")
    print("Note: Uses Int8->FP8 TensorCore internally")
    print("      Unpacking Int4->Int8 adds overhead vs native Int4")


def test_int4_gemv():
    """Test Int4 GEMV for M=1 decode path"""
    native = get_native_module()

    print()
    print("=" * 70)
    print("Int4 GEMV (M=1 decode) Benchmark (SM120)")
    print("=" * 70)

    # Check availability
    if not native.int4_gemv_available():
        print("Int4 GEMV not available (requires SM120)")
        return

    print("Int4 GEMV: available")
    print()

    # Correctness test
    print("=== Correctness Test ===")
    K, N = 4096, 14336

    np.random.seed(42)
    A_int8 = np.random.randint(-8, 8, K, dtype=np.int8)
    B_int8 = np.random.randint(-8, 8, (N, K), dtype=np.int8)

    # Pack to Int4
    A_packed = pack_int4(A_int8.reshape(1, -1)).reshape(-1)
    B_packed = pack_int4(B_int8)

    # Reference: C = A @ B.T (dot product per row of B)
    C_ref = (A_int8.astype(np.int32).reshape(1, -1) @ B_int8.T.astype(np.int32)).reshape(-1)

    # GPU computation
    A_gpu = from_numpy(A_packed)
    B_gpu = from_numpy(B_packed)
    C_gpu = from_numpy(np.zeros(N, dtype=np.int32))

    native.int4_gemv_int32_sm120(A_gpu._get_native(), B_gpu._get_native(), C_gpu._get_native())
    native.device_synchronize()

    C_result = C_gpu.to_numpy()

    diff = np.abs(C_result.astype(np.float64) - C_ref.astype(np.float64))
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_error = diff.sum() / (np.abs(C_ref).sum() + 1e-10)

    print(f"K={K}, N={N}")
    print(f"Max absolute diff: {max_diff}")
    print(f"Mean absolute diff: {mean_diff:.4f}")
    print(f"Relative error: {rel_error * 100:.4f}%")
    print(f"Sample expected: {C_ref[:5]}")
    print(f"Sample got:      {C_result[:5]}")
    print()

    # Performance benchmark (M=1 GEMV typical for LLM decode)
    print("=== Performance Benchmark (M=1 GEMV) ===")
    print(f"{'K':>6} {'N':>6} | {'TFLOPS':>10} | {'us':>10}")
    print("-" * 42)

    configs = [
        (4096, 4096),
        (4096, 14336),
        (4096, 18944),
        (8192, 8192),
        (8192, 28672),
    ]

    warmup = 10
    iterations = 50

    for K, N in configs:
        A_int8 = np.random.randint(-8, 8, K, dtype=np.int8)
        B_int8 = np.random.randint(-8, 8, (N, K), dtype=np.int8)

        A_packed = pack_int4(A_int8.reshape(1, -1)).reshape(-1)
        B_packed = pack_int4(B_int8)

        A_gpu = from_numpy(A_packed)
        B_gpu = from_numpy(B_packed)
        C_gpu = from_numpy(np.zeros(N, dtype=np.int32))

        flops = 2 * K * N  # M=1 GEMV

        try:
            for _ in range(warmup):
                native.int4_gemv_int32_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), C_gpu._get_native()
                )
            native.device_synchronize()

            times = []
            for _ in range(iterations):
                native.device_synchronize()
                start = time.perf_counter()
                native.int4_gemv_int32_sm120(
                    A_gpu._get_native(), B_gpu._get_native(), C_gpu._get_native()
                )
                native.device_synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)

            median_us = np.median(times)
            tflops = flops / median_us / 1e6
            print(f"{K:>6} {N:>6} | {tflops:>10.2f} | {median_us:>10.1f}")
        except Exception as e:
            print(f"{K:>6} {N:>6} | ERROR: {e}")

    print()
    print("Note: GEMV is memory-bandwidth bound, TFLOPS is lower than GEMM")


if __name__ == "__main__":
    test_int4_gemm()
    test_int4_gemv()
