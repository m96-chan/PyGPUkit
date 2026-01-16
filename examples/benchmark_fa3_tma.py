#!/usr/bin/env python3
"""
Flash Attention 3 TMA Benchmark

Compares TMA-enabled FA3 vs baseline FA3.
"""

import os
import time
import numpy as np

# Disable all advanced attention initially
os.environ["PYGPUKIT_FA3"] = "0"
os.environ["PYGPUKIT_FA3_TMA"] = "0"
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"

import pygpukit as gk
from pygpukit.core.dtypes import DataType
from pygpukit.ops.nn import sdpa_causal
from pygpukit.core.backend import get_native_module

native = get_native_module()


def run_benchmark(Q_gpu, K_gpu, V_gpu, mode, n_warmup=5, n_iters=20):
    """Run attention benchmark with specified mode."""
    # Configure mode
    if mode == "baseline":
        os.environ["PYGPUKIT_FA3_TMA"] = "0"
        os.environ["PYGPUKIT_FA3"] = "1"
        os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"
    elif mode == "tma":
        os.environ["PYGPUKIT_FA3_TMA"] = "1"
        os.environ["PYGPUKIT_FA3"] = "0"
        os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"
    elif mode == "fa2":
        os.environ["PYGPUKIT_FA3_TMA"] = "0"
        os.environ["PYGPUKIT_FA3"] = "0"
        os.environ["PYGPUKIT_FLASH_ATTENTION"] = "1"

    # Warmup
    for _ in range(n_warmup):
        out = sdpa_causal(Q_gpu, K_gpu, V_gpu)
    native.device_synchronize()

    # Measure
    native.device_synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        out = sdpa_causal(Q_gpu, K_gpu, V_gpu)
    native.device_synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / n_iters) * 1e6, out


def main():
    print("=" * 70)
    print("Flash Attention 3: TMA vs Baseline Benchmark")
    print("=" * 70)
    print()

    # Benchmark configurations
    configs = [
        (32, 512, 128),
        (32, 1024, 128),
        (32, 2048, 128),
        (32, 4096, 128),
    ]

    print(f"{'Config':<25} {'Baseline (us)':<15} {'TMA (us)':<15} {'Speedup':<10}")
    print("-" * 70)

    for heads, seq_len, head_dim in configs:
        np.random.seed(42)
        Q_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32)
        K_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32)
        V_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32)

        bf16 = DataType.from_string("bfloat16")
        Q_gpu = gk.from_numpy(Q_np).astype(bf16)
        K_gpu = gk.from_numpy(K_np).astype(bf16)
        V_gpu = gk.from_numpy(V_np).astype(bf16)

        # Benchmark baseline
        baseline_time, out_baseline = run_benchmark(Q_gpu, K_gpu, V_gpu, "baseline")

        # Benchmark TMA
        tma_time, out_tma = run_benchmark(Q_gpu, K_gpu, V_gpu, "tma")

        # Compute speedup
        speedup = baseline_time / tma_time if tma_time > 0 else 0

        config_str = f"[{heads}, {seq_len}, {head_dim}]"
        print(f"{config_str:<25} {baseline_time:>12.1f}   {tma_time:>12.1f}   {speedup:>8.2f}x")

        # Verify correctness
        fp32 = DataType.from_string("float32")
        out_baseline_fp32 = out_baseline.astype(fp32).to_numpy()
        out_tma_fp32 = out_tma.astype(fp32).to_numpy()
        rel_error = np.abs(out_baseline_fp32 - out_tma_fp32).mean() / (
            np.abs(out_baseline_fp32).mean() + 1e-6
        )
        if rel_error > 0.05:
            print(f"  WARNING: High relative error: {rel_error:.4f}")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
