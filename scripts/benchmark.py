#!/usr/bin/env python3
"""
PyGPUkit Comprehensive Benchmark

Benchmarks all supported dtypes:
- FP32 (Ampere optimized kernel)
- TF32 v1 (WMMA TensorCore)
- TF32 v2 (PTX mma.sync TensorCore, optimized)
- FP16 (simple kernel, TensorCore planned)
- BF16 (simple kernel, TensorCore planned)

Runtime Modes:
- Driver-Only: Uses pre-compiled kernels, no CUDA Toolkit needed
- Full (JIT): Same kernels + JIT compilation for custom ops

Note: Built-in matmul kernels are pre-compiled, so Driver-Only and Full
modes have identical performance for matmul operations.

Usage:
    python benchmark_all.py [--sizes SIZES] [--quick] [--tf32-version v1|v2]

Output format matches README.md tables for easy updates.
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

# =============================================================================
# Setup CUDA DLL path (Windows)
# =============================================================================
cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
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

    try:
        import pygpukit as gpk

        nvrtc = gpk.is_nvrtc_available()
    except Exception:
        nvrtc = False

    return GPUInfo(
        name=props.name,
        sm_major=props.compute_capability_major,
        sm_minor=props.compute_capability_minor,
        nvrtc_available=nvrtc,
    )


def benchmark_fp32(size: int, warmup: int = 5, iterations: int = 10) -> BenchmarkResult:
    """Benchmark FP32 matmul (Ampere optimized kernel)."""
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
    correct = rel_error < 1e-3

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


def benchmark_tf32(
    size: int, warmup: int = 5, iterations: int = 10, use_v2: bool = True
) -> BenchmarkResult:
    """Benchmark TF32 TensorCore matmul.

    Uses environment variables to control kernel selection:
    - PYGPUKIT_ALLOW_TF32=1: Enable TF32 kernels
    - PYGPUKIT_TF32_V2=1: Use optimized v2 kernel (PTX mma.sync)
    """
    native = get_native_module()

    # Set environment for TF32
    os.environ["PYGPUKIT_ALLOW_TF32"] = "1"
    if use_v2:
        os.environ["PYGPUKIT_TF32_V2"] = "1"
    else:
        os.environ.pop("PYGPUKIT_TF32_V2", None)

    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)

    # Correctness - use native.matmul which respects env vars
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_result = C_gpu.to_numpy()
    C_expected = A @ B
    rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
    correct = rel_error < 1e-2  # TF32 has ~0.1% per-op error

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

    version = "v2" if use_v2 else "v1"
    return BenchmarkResult(
        dtype=f"TF32 {version}",
        size=size,
        tflops_median=flops / median_time / 1e12,
        tflops_max=flops / min_time / 1e12,
        time_ms=median_time * 1000,
        correct=correct,
        rel_error=rel_error,
    )


def benchmark_fp16(size: int, warmup: int = 5, iterations: int = 10) -> BenchmarkResult:
    """Benchmark FP16 matmul (simple kernel, no TensorCore yet)."""
    native = get_native_module()

    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)

    # Correctness
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_result = C_gpu.to_numpy()
    C_expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    rel_error = np.max(np.abs(C_result.astype(np.float32) - C_expected.astype(np.float32))) / (
        np.max(np.abs(C_expected.astype(np.float32))) + 1e-7
    )
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
    """Benchmark BF16 matmul (simple kernel, no TensorCore yet)."""
    native = get_native_module()
    import pygpukit as gpk

    A_fp32 = np.random.randn(size, size).astype(np.float32)
    B_fp32 = np.random.randn(size, size).astype(np.float32)

    # Convert to BF16 via GPUArray
    A_gpu = gpk.from_numpy(A_fp32).astype(gpk.bfloat16)._get_native()
    B_gpu = gpk.from_numpy(B_fp32).astype(gpk.bfloat16)._get_native()

    # Correctness
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_gpk = gpk.GPUArray._wrap_native(C_gpu).astype(gpk.float32)
    C_result = C_gpk.to_numpy()
    C_expected = A_fp32 @ B_fp32
    rel_error = np.max(np.abs(C_result - C_expected)) / (np.max(np.abs(C_expected)) + 1e-7)
    correct = rel_error < 0.05

    # Re-create arrays for benchmark
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
def print_header(gpu_info: GPUInfo, tf32_version: str):
    """Print benchmark header."""
    print("=" * 70)
    print(" PyGPUkit Comprehensive Benchmark")
    print("=" * 70)
    print()
    print(f"GPU: {gpu_info.name}")
    print(f"SM: {gpu_info.sm_major}.{gpu_info.sm_minor}")
    print(f"NVRTC (JIT): {'Available' if gpu_info.nvrtc_available else 'Not Available'}")
    print(f"TF32 Kernel: {tf32_version}")
    print()
    print("Note: Built-in matmul kernels are pre-compiled.")
    print("      Driver-Only and Full modes have identical matmul performance.")
    print()


def print_correctness_results(results: list):
    """Print correctness verification results."""
    print("=" * 70)
    print(" Correctness Verification")
    print("=" * 70)
    print()
    print(f"{'Dtype':<12} {'Size':<12} {'Rel Error':<12} {'Status':<8}")
    print("-" * 48)

    for r in results:
        status = "PASS" if r.correct else "FAIL"
        print(f"{r.dtype:<12} {r.size}x{r.size:<6} {r.rel_error:<12.2e} {status:<8}")
    print()


def print_benchmark_results(results: list, sizes: list):
    """Print benchmark results."""
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

    # Get all dtypes
    all_dtypes = []
    for r in results:
        if r.dtype not in all_dtypes:
            all_dtypes.append(r.dtype)

    # Print header
    header = f"{'Size':<14}"
    for dt in all_dtypes:
        header += f"{dt:<12}"
    print(header)
    print("-" * (14 + 12 * len(all_dtypes)))

    # Print rows
    for size in sizes:
        if size not in by_size:
            continue
        row = by_size[size]
        line = f"{size}x{size:<8}"
        for dt in all_dtypes:
            r = row.get(dt)
            if r:
                line += f"{r.tflops_median:<12.1f}"
            else:
                line += f"{'-':<12}"
        print(line)

    print()


def print_readme_table(results: list, sizes: list):
    """Print README.md compatible markdown table."""
    print("=" * 70)
    print(" README.md Table")
    print("=" * 70)
    print()

    # Group by size
    by_size = {}
    for r in results:
        if r.size not in by_size:
            by_size[r.size] = {}
        by_size[r.size][r.dtype] = r

    # Get dtypes
    all_dtypes = []
    for r in results:
        if r.dtype not in all_dtypes:
            all_dtypes.append(r.dtype)

    # Print markdown table
    header = "| Matrix Size |"
    separator = "|-------------|"
    for dt in all_dtypes:
        header += f" {dt} |"
        separator += "------|"
    print(header)
    print(separator)

    for size in sizes:
        if size not in by_size:
            continue
        row = by_size[size]
        line = f"| {size}x{size} |"
        for dt in all_dtypes:
            r = row.get(dt)
            if r:
                line += f" {r.tflops_median:.1f} TFLOPS |"
            else:
                line += " - |"
        print(line)

    print()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PyGPUkit Comprehensive Benchmark")
    parser.add_argument(
        "--sizes",
        type=str,
        default="2048,4096,8192",
        help="Comma-separated matrix sizes (default: 2048,4096,8192)",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer iterations")
    parser.add_argument(
        "--dtypes",
        type=str,
        default="fp32,tf32,fp16,bf16",
        help="Comma-separated dtypes to benchmark",
    )
    parser.add_argument(
        "--tf32-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="TF32 kernel version: v1 (WMMA) or v2 (PTX mma.sync, default)",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    dtypes = [d.strip().lower() for d in args.dtypes.split(",")]
    use_tf32_v2 = args.tf32_version == "v2"

    warmup = 3 if args.quick else 5
    iterations = 5 if args.quick else 10

    # Get GPU info
    gpu_info = get_gpu_info()
    print_header(gpu_info, args.tf32_version.upper())

    # Run benchmarks
    results = []

    print("Running benchmarks...")
    print()

    for size in sizes:
        iters = max(2, iterations // 2) if size >= 8192 else iterations

        if "fp32" in dtypes:
            # Disable TF32 for FP32 benchmark
            os.environ.pop("PYGPUKIT_ALLOW_TF32", None)
            os.environ.pop("PYGPUKIT_TF32_V2", None)
            print(f"  FP32 {size}x{size}...", end=" ", flush=True)
            r = benchmark_fp32(size, warmup, iters)
            results.append(r)
            print(f"{r.tflops_median:.1f} TFLOPS")

        if "tf32" in dtypes:
            print(f"  TF32 {args.tf32_version} {size}x{size}...", end=" ", flush=True)
            r = benchmark_tf32(size, warmup, iters, use_v2=use_tf32_v2)
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
    print_readme_table(results, sizes)

    # Summary
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print()
    print(f"GPU: {gpu_info.name}")
    print(f"TF32 Kernel: {args.tf32_version.upper()}")

    if results:
        peak = max(results, key=lambda r: r.tflops_median)
        print(f"Peak: {peak.tflops_median:.1f} TFLOPS ({peak.dtype}, {peak.size}x{peak.size})")

    print()
    print("RTX 3090 Ti Theoretical:")
    print("  FP32: ~40 TFLOPS")
    print("  TF32 TensorCore: ~80 TFLOPS (Sparse: ~156 TFLOPS)")
    print("  FP16 TensorCore: ~160 TFLOPS (not yet optimized)")
    print()
    print("Note: FP16/BF16 use simple kernels. TensorCore optimization in Issue #60.")
    print()


if __name__ == "__main__":
    main()
