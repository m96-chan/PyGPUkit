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


def calc_tflops(heads, seq_len, head_dim, time_us):
    """Calculate TFLOPS for attention."""
    # FLOPs: 4 * heads * seq^2 * head_dim (Q@K^T + softmax approx + P@V)
    flops = 4 * heads * seq_len * seq_len * head_dim
    return (flops / (time_us / 1e6)) / 1e12


def main():
    print("=" * 90)
    print("Flash Attention 3: TMA vs Baseline vs FA2 Benchmark")
    print("=" * 90)
    print()

    # Benchmark configurations
    configs = [
        (32, 512, 128),
        (32, 1024, 128),
        (32, 2048, 128),
        (32, 4096, 128),
    ]

    print(f"{'Config':<20} {'FA2 (us)':<12} {'FA3 (us)':<12} {'TMA (us)':<12} {'FA3 TFLOPS':<12} {'TMA TFLOPS':<12}")
    print("-" * 90)

    for heads, seq_len, head_dim in configs:
        np.random.seed(42)
        Q_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32)
        K_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32)
        V_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32)

        bf16 = DataType.from_string("bfloat16")
        Q_gpu = gk.from_numpy(Q_np).astype(bf16)
        K_gpu = gk.from_numpy(K_np).astype(bf16)
        V_gpu = gk.from_numpy(V_np).astype(bf16)

        # Benchmark FA2
        try:
            fa2_time, out_fa2 = run_benchmark(Q_gpu, K_gpu, V_gpu, "fa2")
        except Exception as e:
            fa2_time = float('nan')
            out_fa2 = None

        # Benchmark FA3 baseline
        try:
            baseline_time, out_baseline = run_benchmark(Q_gpu, K_gpu, V_gpu, "baseline")
        except Exception as e:
            baseline_time = float('nan')
            out_baseline = None

        # Benchmark FA3 TMA
        try:
            tma_time, out_tma = run_benchmark(Q_gpu, K_gpu, V_gpu, "tma")
        except Exception as e:
            tma_time = float('nan')
            out_tma = None

        # Calculate TFLOPS
        fa3_tflops = calc_tflops(heads, seq_len, head_dim, baseline_time) if baseline_time else 0
        tma_tflops = calc_tflops(heads, seq_len, head_dim, tma_time) if tma_time else 0

        config_str = f"[{heads}, {seq_len}, {head_dim}]"
        fa2_str = f"{fa2_time:>10.1f}" if not np.isnan(fa2_time) else "N/A".rjust(10)
        fa3_str = f"{baseline_time:>10.1f}" if not np.isnan(baseline_time) else "N/A".rjust(10)
        tma_str = f"{tma_time:>10.1f}" if not np.isnan(tma_time) else "N/A".rjust(10)
        print(f"{config_str:<20} {fa2_str}   {fa3_str}   {tma_str}   {fa3_tflops:>10.2f}   {tma_tflops:>10.2f}")

        # Verify correctness (TMA vs FA3)
        if out_baseline is not None and out_tma is not None:
            fp32 = DataType.from_string("float32")
            out_baseline_fp32 = out_baseline.astype(fp32).to_numpy()
            out_tma_fp32 = out_tma.astype(fp32).to_numpy()
            rel_error = np.abs(out_baseline_fp32 - out_tma_fp32).mean() / (
                np.abs(out_baseline_fp32).mean() + 1e-6
            )
            if rel_error > 0.05:
                print(f"  WARNING: TMA vs FA3 relative error: {rel_error:.4f}")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
