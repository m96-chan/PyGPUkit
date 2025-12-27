#!/usr/bin/env python3
"""
W8A16 GEMM Benchmark for SM120.

Tests FP8 weight x BF16 activation -> BF16 output.
"""

import time

import numpy as np

import pygpukit as gk
from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module
from pygpukit.ops.matmul import w8a16_gemm_sm120


def benchmark_w8a16_gemm():
    """Benchmark W8A16 GEMM kernel."""
    native = get_native_module()

    print("=" * 80)
    print("W8A16 GEMM Benchmark (SM120)")
    print("=" * 80)

    # Get GPU info
    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print()

    # Test configurations (typical LLM layer sizes)
    # Qwen3-30B-A3B MoE: hidden=2048, intermediate varies by expert
    configs = [
        # (M, K, N) - prefill batch sizes
        (1, 2048, 8192),  # Single token, small MLP
        (16, 2048, 8192),  # Small batch
        (64, 2048, 8192),  # Medium batch
        (128, 4096, 14336),  # Large batch, Qwen-7B MLP
        (256, 4096, 14336),  # Larger batch
        (512, 4096, 14336),  # Prefill size
        (1024, 4096, 14336),  # Long prefill
    ]

    warmup = 10
    iterations = 50

    for M, K, N in configs:
        print(f"\n{'=' * 60}")
        print(f"M={M}, K={K}, N={N}")
        print(f"{'=' * 60}")

        # Calculate data sizes
        A_bytes = M * K * 2  # BF16
        B_bytes = K * N * 1  # FP8
        C_bytes = M * N * 2  # BF16
        scale_k = (K + 127) // 128
        scale_n = (N + 127) // 128
        scale_bytes = scale_k * scale_n * 2  # BF16 scale
        total_bytes = A_bytes + B_bytes + C_bytes + scale_bytes

        print(f"Data: A={A_bytes / 1e6:.2f}MB, B={B_bytes / 1e6:.2f}MB, C={C_bytes / 1e6:.2f}MB")
        print(f"Total I/O: {total_bytes / 1e6:.2f}MB")

        # Calculate FLOPS (2*M*N*K for matmul)
        flops = 2 * M * N * K

        # Create tensors
        A_bf16 = gk.empty((M, K), dtype="bfloat16")
        B_fp8 = from_numpy(np.random.randint(0, 256, (K, N), dtype=np.uint8))
        B_scale = gk.empty((scale_k, scale_n), dtype="bfloat16")
        C_out = gk.empty((M, N), dtype="bfloat16")

        # Warmup
        for _ in range(warmup):
            w8a16_gemm_sm120(A_bf16, B_fp8, B_scale, out=C_out)
        native.device_synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            native.device_synchronize()
            start = time.perf_counter()
            w8a16_gemm_sm120(A_bf16, B_fp8, B_scale, out=C_out)
            native.device_synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1e6)  # microseconds

        median_us = np.median(times)
        min_us = np.min(times)
        max_us = np.max(times)

        # Calculate performance
        tflops = flops / median_us / 1e6  # TFLOPS
        bw = total_bytes / median_us / 1e3  # GB/s

        print(f"Time: median={median_us:.1f}us, min={min_us:.1f}us, max={max_us:.1f}us")
        print(f"Performance: {tflops:.2f} TFLOPS, BW={bw:.0f} GB/s")


if __name__ == "__main__":
    benchmark_w8a16_gemm()
