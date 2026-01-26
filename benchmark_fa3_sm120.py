"""
FA3 SM120 Configuration Benchmark

Uses sdpa_causal_timed to measure attention kernel performance.
Environment variables control which FA3 variant is used:
- PYGPUKIT_FA3=1: Force FA3 on
- PYGPUKIT_FA3_TMA=1: Force TMA variant

Current: FA3 TMA at 51.97 TFLOPS (baseline)
Target: 60+ TFLOPS with SM120 tuning
"""

import numpy as np
import time
import os
import sys

import pygpukit as gpk
from pygpukit.core.backend import get_native_module
from pygpukit.core.dtypes import DataType

native = get_native_module()


def compute_attention_flops(batch: int, heads: int, seq_q: int, seq_kv: int, head_dim: int) -> int:
    """Compute total FLOPs for attention forward pass."""
    # Q@K^T: 2 * batch * heads * seq_q * seq_kv * head_dim
    qk_flops = 2 * batch * heads * seq_q * seq_kv * head_dim
    # P@V: 2 * batch * heads * seq_q * head_dim * seq_kv
    pv_flops = 2 * batch * heads * seq_q * head_dim * seq_kv
    return qk_flops + pv_flops


def benchmark_sdpa_timed(heads: int, seq_len: int, head_dim: int, num_iters: int = 50):
    """Benchmark SDPA using kernel-only timing (sdpa_causal_timed)."""
    bf16 = DataType.from_string("bfloat16")

    # Allocate tensors [n_heads, seq_len, head_dim]
    Q_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32) * 0.1
    K_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32) * 0.1
    V_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32) * 0.1

    Q = gpk.from_numpy(Q_np).astype(bf16)
    K = gpk.from_numpy(K_np).astype(bf16)
    V = gpk.from_numpy(V_np).astype(bf16)
    O = gpk.zeros((heads, seq_len, head_dim), dtype=bf16)

    scale = 1.0 / np.sqrt(head_dim)

    # Warmup
    for _ in range(3):
        native.sdpa_causal_(Q._native, K._native, V._native, O._native, scale)

    # Benchmark using kernel timing
    native.device_synchronize()
    total_time_us = 0.0
    for _ in range(num_iters):
        kernel_us = native.sdpa_causal_timed(Q._native, K._native, V._native, O._native, scale)
        total_time_us += kernel_us

    avg_time_us = total_time_us / num_iters

    # Compute TFLOPS (batch=1 for single head group)
    flops = compute_attention_flops(1, heads, seq_len, seq_len, head_dim)
    tflops = flops / (avg_time_us * 1e-6) / 1e12

    return avg_time_us, tflops


def benchmark_sdpa_python_timing(heads: int, seq_len: int, head_dim: int, num_iters: int = 50):
    """Benchmark SDPA using Python-side timing (includes overhead)."""
    bf16 = DataType.from_string("bfloat16")

    # Allocate tensors
    Q_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32) * 0.1
    K_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32) * 0.1
    V_np = np.random.randn(heads, seq_len, head_dim).astype(np.float32) * 0.1

    Q = gpk.from_numpy(Q_np).astype(bf16)
    K = gpk.from_numpy(K_np).astype(bf16)
    V = gpk.from_numpy(V_np).astype(bf16)
    O = gpk.zeros((heads, seq_len, head_dim), dtype=bf16)

    scale = 1.0 / np.sqrt(head_dim)

    # Warmup
    for _ in range(3):
        native.sdpa_causal_(Q._native, K._native, V._native, O._native, scale)

    # Benchmark
    native.device_synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        native.sdpa_causal_(Q._native, K._native, V._native, O._native, scale)
    native.device_synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    # Compute TFLOPS
    flops = compute_attention_flops(1, heads, seq_len, seq_len, head_dim)
    tflops = flops / elapsed / 1e12

    return elapsed * 1e6, tflops


def main():
    print("=" * 70)
    print("FA3 SM120 Attention Benchmark")
    print("=" * 70)

    # Print environment
    fa3_env = os.environ.get("PYGPUKIT_FA3", "auto")
    fa3_tma_env = os.environ.get("PYGPUKIT_FA3_TMA", "auto")
    print(f"PYGPUKIT_FA3={fa3_env}")
    print(f"PYGPUKIT_FA3_TMA={fa3_tma_env}")

    # Get device info
    print(f"\nDevice: SM{native.get_sm_version()}")

    # Test configurations
    configs = [
        # (heads, seq_len, head_dim)
        (32, 512, 128),
        (32, 1024, 128),
        (32, 2048, 128),
        (32, 4096, 128),
    ]

    num_iters = 50

    print(f"\n{'Config':<25} {'Kernel (us)':<12} {'TFLOPS':<10} {'Python (us)':<12} {'TFLOPS':<10}")
    print("-" * 80)

    for heads, seq_len, head_dim in configs:
        config_str = f"h={heads}, s={seq_len}, d={head_dim}"

        try:
            # Kernel-only timing
            kernel_us, kernel_tflops = benchmark_sdpa_timed(heads, seq_len, head_dim, num_iters)

            # Python-side timing (for comparison)
            python_us, python_tflops = benchmark_sdpa_python_timing(heads, seq_len, head_dim, num_iters)

            print(f"{config_str:<25} {kernel_us:<12.1f} {kernel_tflops:<10.2f} {python_us:<12.1f} {python_tflops:<10.2f}")

        except Exception as e:
            print(f"{config_str:<25} ERROR: {e}")

    # Print TMA cache stats
    print("\n" + "=" * 70)
    print("TMA Descriptor Cache Stats:")
    native.print_tma_cache_stats()

    print("\n" + "=" * 70)
    print("Notes:")
    print("- Kernel timing uses CUDA Events (excludes Python/host overhead)")
    print("- Python timing includes launch overhead")
    print("- TFLOPS calculated from kernel timing")
    print("=" * 70)


if __name__ == "__main__":
    main()
