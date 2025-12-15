#!/usr/bin/env python3
"""
PyGPUkit v0.2.5 Full Feature Demo

Demonstrates all features available in v0.2.5:
- Data types: FP32, FP16, BF16
- Elementwise operations: add, mul, sub, div
- Matrix multiplication: FP32, TF32 (TensorCore), FP16, BF16
- Reduction operations: sum, mean, max
- Type conversion: astype()
"""

import os
import time

import numpy as np

# Set TF32 environment before import
os.environ["PYGPUKIT_ALLOW_TF32"] = "1"

import pygpukit as gpk


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def benchmark_matmul(a, b, name: str, warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark matmul and return TFLOPS."""
    M, K = a.shape
    _, N = b.shape

    # Warmup
    for _ in range(warmup):
        c = a @ b
        _ = c.to_numpy()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = a @ b
        _ = c.to_numpy()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    flops = 2.0 * M * N * K
    tflops = flops / avg_time / 1e12

    print(f"  {name}: {avg_time * 1000:.2f} ms, {tflops:.2f} TFLOPS")
    return tflops


def demo_dtypes():
    """Demonstrate supported data types."""
    section("Data Types")

    print("Supported dtypes:")
    print(f"  - gpk.float32: {gpk.float32}")
    print(f"  - gpk.float64: {gpk.float64}")
    print(f"  - gpk.float16: {gpk.float16}")
    print(f"  - gpk.bfloat16: {gpk.bfloat16}")
    print(f"  - gpk.int32: {gpk.int32}")
    print(f"  - gpk.int64: {gpk.int64}")

    # Create arrays with different dtypes
    print()
    print("Creating arrays:")

    a_fp32 = gpk.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    print(f"  FP32: {a_fp32}")

    a_fp16 = gpk.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    print(f"  FP16: {a_fp16}")

    a_bf16 = gpk.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32)).astype(gpk.bfloat16)
    print(f"  BF16: {a_bf16}")


def demo_elementwise():
    """Demonstrate elementwise operations."""
    section("Elementwise Operations")

    for dtype_name, np_dtype, gpk_dtype in [
        ("FP32", np.float32, None),
        ("FP16", np.float16, None),
        ("BF16", np.float32, gpk.bfloat16),
    ]:
        print(f"\n{dtype_name}:")

        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)
        b_np = np.array([0.5, 1.5, 2.5, 3.5], dtype=np_dtype)

        if gpk_dtype == gpk.bfloat16:
            a = gpk.from_numpy(a_np).astype(gpk.bfloat16)
            b = gpk.from_numpy(b_np).astype(gpk.bfloat16)
        else:
            a = gpk.from_numpy(a_np)
            b = gpk.from_numpy(b_np)

        # Operations
        add_result = a + b
        mul_result = a * b
        sub_result = a - b
        div_result = a / b

        # Convert back for display
        if gpk_dtype == gpk.bfloat16:
            add_np = add_result.astype(gpk.float32).to_numpy()
            mul_np = mul_result.astype(gpk.float32).to_numpy()
            sub_np = sub_result.astype(gpk.float32).to_numpy()
            div_np = div_result.astype(gpk.float32).to_numpy()
        else:
            add_np = add_result.to_numpy()
            mul_np = mul_result.to_numpy()
            sub_np = sub_result.to_numpy()
            div_np = div_result.to_numpy()

        print(f"  a = {a_np}")
        print(f"  b = {b_np}")
        print(f"  a + b = {add_np}")
        print(f"  a * b = {mul_np}")
        print(f"  a - b = {sub_np}")
        print(f"  a / b = {np.round(div_np, 3)}")


def demo_matmul():
    """Demonstrate matrix multiplication."""
    section("Matrix Multiplication")

    size = 1024
    print(f"Matrix size: {size}x{size}")
    print()

    # FP32
    print("FP32 Matmul:")
    a_fp32 = gpk.from_numpy(np.random.randn(size, size).astype(np.float32))
    b_fp32 = gpk.from_numpy(np.random.randn(size, size).astype(np.float32))
    c = a_fp32 @ b_fp32
    print(f"  Result shape: {c.shape}, dtype: {c.dtype}")
    benchmark_matmul(a_fp32, b_fp32, "Performance")

    # TF32 (TensorCore)
    print("\nTF32 Matmul (TensorCore):")
    c_tf32 = gpk.matmul(a_fp32, b_fp32, use_tf32=True)
    print(f"  Result shape: {c_tf32.shape}, dtype: {c_tf32.dtype}")

    # Accuracy check
    c_np = c.to_numpy()
    c_tf32_np = c_tf32.to_numpy()
    rel_err = np.max(np.abs(c_np - c_tf32_np)) / np.max(np.abs(c_np))
    print(f"  TF32 vs FP32 rel error: {rel_err:.6f}")

    # FP16
    print("\nFP16 Matmul:")
    a_fp16 = gpk.from_numpy(np.random.randn(size, size).astype(np.float16))
    b_fp16 = gpk.from_numpy(np.random.randn(size, size).astype(np.float16))
    c_fp16 = a_fp16 @ b_fp16
    print(f"  Result shape: {c_fp16.shape}, dtype: {c_fp16.dtype}")
    benchmark_matmul(a_fp16, b_fp16, "Performance")

    # BF16
    print("\nBF16 Matmul:")
    a_bf16 = gpk.from_numpy(np.random.randn(size, size).astype(np.float32)).astype(gpk.bfloat16)
    b_bf16 = gpk.from_numpy(np.random.randn(size, size).astype(np.float32)).astype(gpk.bfloat16)
    c_bf16 = a_bf16 @ b_bf16
    print(f"  Result shape: {c_bf16.shape}, dtype: {c_bf16.dtype}")
    benchmark_matmul(a_bf16, b_bf16, "Performance")


def demo_reductions():
    """Demonstrate reduction operations."""
    section("Reduction Operations")

    a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    a = gpk.from_numpy(a_np)

    print(f"Input: {a_np}")
    print()

    # Sum
    s = gpk.sum(a)
    print(f"sum(a) = {s.to_numpy()[0]:.4f} (expected: {np.sum(a_np):.4f})")

    # Mean
    m = gpk.mean(a)
    print(f"mean(a) = {m.to_numpy()[0]:.4f} (expected: {np.mean(a_np):.4f})")

    # Max
    mx = gpk.max(a)
    print(f"max(a) = {mx.to_numpy()[0]:.4f} (expected: {np.max(a_np):.4f})")


def demo_astype():
    """Demonstrate type conversion."""
    section("Type Conversion (astype)")

    # FP32 -> FP16
    a_fp32 = gpk.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    a_fp16 = a_fp32.astype(gpk.float16)
    print(f"FP32 -> FP16: {a_fp32} -> {a_fp16}")

    # FP32 -> BF16
    a_bf16 = a_fp32.astype(gpk.bfloat16)
    print(f"FP32 -> BF16: {a_fp32} -> {a_bf16}")

    # BF16 -> FP32
    a_back = a_bf16.astype(gpk.float32)
    print(f"BF16 -> FP32: {a_bf16} -> {a_back}")
    print(f"  Values: {a_back.to_numpy()}")


def demo_benchmark_full():
    """Full benchmark across all dtypes and sizes."""
    section("Full Benchmark")

    sizes = [1024, 2048, 4096]

    print("Matmul Performance (TFLOPS):")
    print()
    print(f"{'Size':<12} {'FP32':<10} {'TF32':<10} {'FP16':<10} {'BF16':<10}")
    print("-" * 52)

    for size in sizes:
        results = {}

        # FP32
        a = gpk.from_numpy(np.random.randn(size, size).astype(np.float32))
        b = gpk.from_numpy(np.random.randn(size, size).astype(np.float32))

        # Warmup & benchmark FP32
        for _ in range(3):
            _ = (a @ b).to_numpy()

        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = (a @ b).to_numpy()
            times.append(time.perf_counter() - start)
        flops = 2.0 * size**3
        results["FP32"] = flops / np.mean(times) / 1e12

        # TF32
        for _ in range(3):
            _ = gpk.matmul(a, b, use_tf32=True).to_numpy()

        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = gpk.matmul(a, b, use_tf32=True).to_numpy()
            times.append(time.perf_counter() - start)
        results["TF32"] = flops / np.mean(times) / 1e12

        # FP16
        a16 = gpk.from_numpy(np.random.randn(size, size).astype(np.float16))
        b16 = gpk.from_numpy(np.random.randn(size, size).astype(np.float16))

        for _ in range(3):
            _ = (a16 @ b16).to_numpy()

        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = (a16 @ b16).to_numpy()
            times.append(time.perf_counter() - start)
        results["FP16"] = flops / np.mean(times) / 1e12

        # BF16
        abf = gpk.from_numpy(np.random.randn(size, size).astype(np.float32)).astype(gpk.bfloat16)
        bbf = gpk.from_numpy(np.random.randn(size, size).astype(np.float32)).astype(gpk.bfloat16)

        for _ in range(3):
            _ = (abf @ bbf).to_numpy()

        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = (abf @ bbf).to_numpy()
            times.append(time.perf_counter() - start)
        results["BF16"] = flops / np.mean(times) / 1e12

        print(
            f"{size}x{size:<7} {results['FP32']:<10.2f} {results['TF32']:<10.2f} {results['FP16']:<10.2f} {results['BF16']:<10.2f}"
        )


def main():
    print("=" * 60)
    print(" PyGPUkit v0.2.5 - Full Feature Demo")
    print("=" * 60)

    # Show version and backend info
    print("\nBackend: Native C++/CUDA")
    print(f"TF32 enabled: {os.environ.get('PYGPUKIT_ALLOW_TF32', '0') == '1'}")

    demo_dtypes()
    demo_elementwise()
    demo_matmul()
    demo_reductions()
    demo_astype()
    demo_benchmark_full()

    section("Demo Complete")
    print("All v0.2.5 features demonstrated successfully!")


if __name__ == "__main__":
    main()
