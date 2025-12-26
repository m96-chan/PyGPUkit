#!/usr/bin/env python3
"""
NVF4-BF16 GEMM Benchmark for SM120 (Blackwell GeForce)

Benchmarks NVF4 (4-bit) GEMM with BF16 I/O.
NVF4 provides 2x memory bandwidth compared to FP8.
"""

import time

import numpy as np


def bf16_to_f32(bf16_uint16: np.ndarray) -> np.ndarray:
    """Convert BFloat16 (stored as uint16) to float32."""
    bf16_uint16 = bf16_uint16.astype(np.uint16)
    f32_bits = bf16_uint16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    """Convert float32 to BFloat16 (stored as uint16)."""
    f32 = f32.astype(np.float32)
    f32_bits = f32.view(np.uint32)
    bf16_bits = (f32_bits >> 16).astype(np.uint16)
    return bf16_bits


def benchmark_nvf4_bf16(sizes: list[int], warmup: int = 5, iterations: int = 20):
    """Benchmark NVF4-BF16 GEMM at various sizes."""
    from pygpukit.core.backend import get_native_module
    from pygpukit.core.factory import from_numpy
    from pygpukit.ops import matmul_nvf4_bf16_sm120, nvf4_bf16_sm120_available

    native = get_native_module()

    if not nvf4_bf16_sm120_available():
        print("NVF4-BF16 SM120 not available")
        return

    print("=" * 70)
    print("NVF4-BF16 GEMM Benchmark (SM120 Blackwell GeForce)")
    print("=" * 70)

    # Get GPU info
    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SM: {props.compute_capability_major}.{props.compute_capability_minor}")
    print()
    print("GPU-side quantization: BF16 -> NVF4 (no H2D copies)")
    print()

    results = []

    for size in sizes:
        M, N, K = size, size, size
        flops = 2.0 * M * N * K  # FLOPs for GEMM

        # Create NVF4-appropriate data (values in representable range)
        nvf4_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=np.float32)
        A = np.random.choice(nvf4_values, size=(M, K)).astype(np.float32)
        B = np.random.choice(nvf4_values, size=(K, N)).astype(np.float32)

        A_bf16 = f32_to_bf16(A)
        B_bf16 = f32_to_bf16(B)

        A_gpu = from_numpy(A_bf16)
        B_gpu = from_numpy(B_bf16)

        # Warmup
        for _ in range(warmup):
            C_gpu = matmul_nvf4_bf16_sm120(A_gpu, B_gpu)
        native.device_synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            native.device_synchronize()
            start = time.perf_counter()
            C_gpu = matmul_nvf4_bf16_sm120(A_gpu, B_gpu)
            native.device_synchronize()
            end = time.perf_counter()
            times.append(end - start)

        # Get result and verify
        C_uint16 = C_gpu.to_numpy()
        C_f32 = bf16_to_f32(C_uint16)
        C_ref = bf16_to_f32(A_bf16) @ bf16_to_f32(B_bf16)

        rel_error = np.linalg.norm(C_f32 - C_ref) / np.linalg.norm(C_ref)

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
                "rel_error": rel_error,
            }
        )

        status = "PASS" if rel_error < 0.05 else "FAIL"
        print(
            f"{M}x{N}x{K}: {tflops_median:.2f} TFLOPS (median), "
            f"{tflops_max:.2f} TFLOPS (max), "
            f"rel_error={rel_error:.2e} [{status}]"
        )

    print()
    print("=" * 70)
    print("Summary Table (for README)")
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

    parser = argparse.ArgumentParser(description="NVF4-BF16 GEMM Benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="Matrix sizes to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")

    args = parser.parse_args()

    benchmark_nvf4_bf16(args.sizes, args.warmup, args.iterations)
