#!/usr/bin/env python3
"""
PyGPUkit Comprehensive Benchmark

Benchmarks all supported dtypes and runtime modes:
- FP32, TF32, FP16, BF16
- Driver-Only mode vs Full (JIT) mode

Usage:
    python benchmark_all.py [--sizes SIZES] [--quick]

Output format matches README.md tables for easy updates.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# =============================================================================
# Setup CUDA DLL path (Windows)
# =============================================================================
cuda_path = os.environ.get(
    "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
)
cuda_bin = os.path.join(cuda_path, "bin")
if os.path.isdir(cuda_bin):
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_bin)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class BenchmarkResult:
    dtype: str
    size: int
    tflops_median: float
    tflops_max: float
    time_ms: float
    correct: bool
    rel_error: float


@dataclass
class GPUInfo:
    name: str
    sm_major: int
    sm_minor: int
    nvrtc_available: bool


# =============================================================================
# Native Module Import Helper
# =============================================================================
_native_module = None

def get_native_module():
    """Get native module with fallback."""
    global _native_module
    if _native_module is not None:
        return _native_module
    try:
        import _pygpukit_native as native
        _native_module = native
    except ImportError:
        from pygpukit import _pygpukit_native as native
        _native_module = native
    return _native_module


# =============================================================================
# Benchmark Functions
# =============================================================================
def get_gpu_info() -> GPUInfo:
    """Get GPU information."""
    native = get_native_module()
    props = native.get_device_properties(0)

    # Check NVRTC availability
    try:
        import pygpukit as gpk
        nvrtc = gpk.is_nvrtc_available()
    except:
        nvrtc = False

    return GPUInfo(
        name=props.name,
        sm_major=props.compute_capability_major,
        sm_minor=props.compute_capability_minor,
        nvrtc_available=nvrtc,
    )


def benchmark_fp32(size: int, warmup: int = 5, iterations: int = 10) -> BenchmarkResult:
    """Benchmark FP32 matmul."""
    native = get_native_module()

    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)

    # Correctness
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_result = C_gpu.to_numpy()
    C_expected = A @ B
    rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
    correct = rel_error < 1e-3  # FP32 matmul has some numerical error due to order of operations

    # Warmup
    for _ in range(warmup):
        _ = native.matmul(A_gpu, B_gpu)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = native.matmul(A_gpu, B_gpu)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    min_time = np.min(times)
    flops = 2.0 * size * size * size

    return BenchmarkResult(
        dtype="FP32",
        size=size,
        tflops_median=flops / median_time / 1e12,
        tflops_max=flops / min_time / 1e12,
        time_ms=median_time * 1000,
        correct=correct,
        rel_error=rel_error,
    )


def benchmark_tf32(size: int, warmup: int = 5, iterations: int = 10) -> BenchmarkResult:
    """Benchmark TF32 TensorCore matmul."""
    native = get_native_module()

    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)

    # Correctness (TF32 tolerance is higher)
    C_gpu = native.matmul_tf32(A_gpu, B_gpu, True)
    C_result = C_gpu.to_numpy()
    C_expected = A @ B
    rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
    correct = rel_error < 1e-2  # TF32 has ~0.1% per-op error

    # Warmup
    for _ in range(warmup):
        _ = native.matmul_tf32(A_gpu, B_gpu, True)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = native.matmul_tf32(A_gpu, B_gpu, True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    min_time = np.min(times)
    flops = 2.0 * size * size * size

    return BenchmarkResult(
        dtype="TF32",
        size=size,
        tflops_median=flops / median_time / 1e12,
        tflops_max=flops / min_time / 1e12,
        time_ms=median_time * 1000,
        correct=correct,
        rel_error=rel_error,
    )


def benchmark_fp16(size: int, warmup: int = 5, iterations: int = 10) -> BenchmarkResult:
    """Benchmark FP16 matmul."""
    native = get_native_module()

    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)

    # Correctness
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_result = C_gpu.to_numpy()
    C_expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    rel_error = np.max(np.abs(C_result.astype(np.float32) - C_expected.astype(np.float32))) / (np.max(np.abs(C_expected.astype(np.float32))) + 1e-7)
    correct = rel_error < 0.05

    # Warmup
    for _ in range(warmup):
        _ = native.matmul(A_gpu, B_gpu)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = native.matmul(A_gpu, B_gpu)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    min_time = np.min(times)
    flops = 2.0 * size * size * size

    return BenchmarkResult(
        dtype="FP16",
        size=size,
        tflops_median=flops / median_time / 1e12,
        tflops_max=flops / min_time / 1e12,
        time_ms=median_time * 1000,
        correct=correct,
        rel_error=rel_error,
    )


def benchmark_bf16(size: int, warmup: int = 5, iterations: int = 10) -> BenchmarkResult:
    """Benchmark BF16 matmul."""
    native = get_native_module()
    import pygpukit as gpk

    A_fp32 = np.random.randn(size, size).astype(np.float32)
    B_fp32 = np.random.randn(size, size).astype(np.float32)

    # Convert to BF16 via GPUArray
    A_gpu = gpk.from_numpy(A_fp32).astype(gpk.bfloat16)._get_native()
    B_gpu = gpk.from_numpy(B_fp32).astype(gpk.bfloat16)._get_native()

    # Correctness
    C_gpu = native.matmul(A_gpu, B_gpu)
    # Convert result back to FP32 for comparison
    C_gpk = gpk.GPUArray._wrap_native(C_gpu).astype(gpk.float32)
    C_result = C_gpk.to_numpy()
    C_expected = A_fp32 @ B_fp32
    rel_error = np.max(np.abs(C_result - C_expected)) / (np.max(np.abs(C_expected)) + 1e-7)
    correct = rel_error < 0.05

    # Re-create arrays for benchmark (previous ones consumed)
    A_gpu = gpk.from_numpy(A_fp32).astype(gpk.bfloat16)._get_native()
    B_gpu = gpk.from_numpy(B_fp32).astype(gpk.bfloat16)._get_native()

    # Warmup
    for _ in range(warmup):
        _ = native.matmul(A_gpu, B_gpu)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = native.matmul(A_gpu, B_gpu)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    min_time = np.min(times)
    flops = 2.0 * size * size * size

    return BenchmarkResult(
        dtype="BF16",
        size=size,
        tflops_median=flops / median_time / 1e12,
        tflops_max=flops / min_time / 1e12,
        time_ms=median_time * 1000,
        correct=correct,
        rel_error=rel_error,
    )


# =============================================================================
# Output Functions
# =============================================================================
def print_header(gpu_info: GPUInfo):
    """Print benchmark header."""
    print("=" * 70)
    print(" PyGPUkit Comprehensive Benchmark")
    print("=" * 70)
    print()
    print(f"GPU: {gpu_info.name}")
    print(f"SM: {gpu_info.sm_major}.{gpu_info.sm_minor}")
    print(f"NVRTC (JIT): {'Available' if gpu_info.nvrtc_available else 'Not Available'}")
    print(f"Mode: {'Full (Driver + JIT)' if gpu_info.nvrtc_available else 'Driver-Only'}")
    print()


def print_correctness_results(results: list):
    """Print correctness verification results."""
    print("=" * 70)
    print(" Correctness Verification")
    print("=" * 70)
    print()
    print(f"{'Dtype':<8} {'Size':<12} {'Rel Error':<12} {'Status':<8}")
    print("-" * 44)

    for r in results:
        status = "PASS" if r.correct else "FAIL"
        print(f"{r.dtype:<8} {r.size}x{r.size:<6} {r.rel_error:<12.2e} {status:<8}")
    print()


def print_benchmark_results(results: list, sizes: list):
    """Print benchmark results in README-compatible table format."""
    print("=" * 70)
    print(" Performance Results (TFLOPS)")
    print("=" * 70)
    print()

    # Group by size
    by_size = {}
    for r in results:
        if r.size not in by_size:
            by_size[r.size] = {}
        by_size[r.size][r.dtype] = r

    # Print table
    print(f"{'Size':<14} {'FP32':<10} {'TF32':<10} {'FP16':<10} {'BF16':<10}")
    print("-" * 54)

    for size in sizes:
        if size not in by_size:
            continue
        row = by_size[size]
        fp32 = row.get("FP32")
        tf32 = row.get("TF32")
        fp16 = row.get("FP16")
        bf16 = row.get("BF16")

        fp32_str = f"{fp32.tflops_median:.1f}" if fp32 else "-"
        tf32_str = f"{tf32.tflops_median:.1f}" if tf32 else "-"
        fp16_str = f"{fp16.tflops_median:.1f}" if fp16 else "-"
        bf16_str = f"{bf16.tflops_median:.1f}" if bf16 else "-"

        print(f"{size}x{size:<8} {fp32_str:<10} {tf32_str:<10} {fp16_str:<10} {bf16_str:<10}")

    print()


def print_readme_table(results: list, sizes: list, mode: str):
    """Print README.md compatible markdown table."""
    print("=" * 70)
    print(f" README.md Table ({mode})")
    print("=" * 70)
    print()

    # Group by size
    by_size = {}
    for r in results:
        if r.size not in by_size:
            by_size[r.size] = {}
        by_size[r.size][r.dtype] = r

    print("| Matrix Size | FP32 | TF32 | FP16 | BF16 |")
    print("|-------------|------|------|------|------|")

    for size in sizes:
        if size not in by_size:
            continue
        row = by_size[size]
        fp32 = row.get("FP32")
        tf32 = row.get("TF32")
        fp16 = row.get("FP16")
        bf16 = row.get("BF16")

        fp32_str = f"{fp32.tflops_median:.1f} TFLOPS" if fp32 else "-"
        tf32_str = f"{tf32.tflops_median:.1f} TFLOPS" if tf32 else "-"
        fp16_str = f"{fp16.tflops_median:.1f} TFLOPS" if fp16 else "-"
        bf16_str = f"{bf16.tflops_median:.1f} TFLOPS" if bf16 else "-"

        print(f"| {size}x{size} | {fp32_str} | {tf32_str} | {fp16_str} | {bf16_str} |")

    print()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PyGPUkit Comprehensive Benchmark")
    parser.add_argument("--sizes", type=str, default="2048,4096,8192",
                        help="Comma-separated matrix sizes (default: 2048,4096,8192)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer iterations")
    parser.add_argument("--dtypes", type=str, default="fp32,tf32,fp16,bf16",
                        help="Comma-separated dtypes to benchmark")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    dtypes = [d.strip().lower() for d in args.dtypes.split(",")]

    warmup = 3 if args.quick else 5
    iterations = 5 if args.quick else 10

    # Setup environment for TF32
    os.environ["PYGPUKIT_ALLOW_TF32"] = "1"
    os.environ["PYGPUKIT_TF32_V2"] = "1"

    # Get GPU info
    gpu_info = get_gpu_info()
    print_header(gpu_info)

    mode = "Full (Driver + JIT)" if gpu_info.nvrtc_available else "Driver-Only"

    # Run benchmarks
    results = []

    print("Running benchmarks...")
    print()

    for size in sizes:
        iters = iterations // 2 if size >= 8192 else iterations

        if "fp32" in dtypes:
            print(f"  FP32 {size}x{size}...", end=" ", flush=True)
            r = benchmark_fp32(size, warmup, iters)
            results.append(r)
            print(f"{r.tflops_median:.1f} TFLOPS")

        if "tf32" in dtypes:
            print(f"  TF32 {size}x{size}...", end=" ", flush=True)
            r = benchmark_tf32(size, warmup, iters)
            results.append(r)
            print(f"{r.tflops_median:.1f} TFLOPS")

        if "fp16" in dtypes:
            print(f"  FP16 {size}x{size}...", end=" ", flush=True)
            r = benchmark_fp16(size, warmup, iters)
            results.append(r)
            print(f"{r.tflops_median:.1f} TFLOPS")

        if "bf16" in dtypes:
            print(f"  BF16 {size}x{size}...", end=" ", flush=True)
            r = benchmark_bf16(size, warmup, iters)
            results.append(r)
            print(f"{r.tflops_median:.1f} TFLOPS")

    print()

    # Print results
    print_correctness_results(results)
    print_benchmark_results(results, sizes)
    print_readme_table(results, sizes, mode)

    # Summary
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print()
    print(f"Mode: {mode}")
    print(f"GPU: {gpu_info.name}")

    # Find peak performance
    if results:
        peak = max(results, key=lambda r: r.tflops_median)
        print(f"Peak: {peak.tflops_median:.1f} TFLOPS ({peak.dtype}, {peak.size}x{peak.size})")

    print()
    print("RTX 3090 Ti Theoretical:")
    print("  FP32: ~40 TFLOPS")
    print("  TF32 TensorCore: ~80 TFLOPS (Sparse: ~156 TFLOPS)")
    print("  FP16 TensorCore: ~160 TFLOPS")
    print()


if __name__ == "__main__":
    main()
