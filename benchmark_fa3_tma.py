"""
Benchmark FA3 TMA Attention Kernel

Reports:
1. Kernel-only time (via cudaEvent, excludes host overhead)
2. E2E time (includes Python + allocation overhead)
3. TMA descriptor cache statistics

Usage:
    python benchmark_fa3_tma.py [seq_len] [num_iterations]
    python benchmark_fa3_tma.py 1024 100
"""
import os
import sys
import time

# Force TMA path
os.environ["PYGPUKIT_FA3_TMA"] = "1"
os.environ["PYGPUKIT_FA3"] = "0"
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"

import numpy as np
import pygpukit as gpk
from pygpukit.ops.nn import sdpa_causal
from pygpukit.core.backend import get_native_module
from pygpukit.core.dtypes import DataType

native = get_native_module()


def compute_tflops(seq_len: int, num_heads: int, head_dim: int, time_us: float) -> float:
    """Compute TFLOPS for SDPA operation."""
    # SDPA FLOPs: 4 * seq * seq * head_dim * num_heads (Q@K + softmax + P@V)
    flops = 4 * seq_len * seq_len * head_dim * num_heads
    return flops / (time_us * 1e-6) / 1e12


def benchmark_kernel_only(Q, K, V, out, num_iters: int = 100) -> tuple[float, float]:
    """Benchmark using cudaEvent timing (kernel-only)."""
    # Get native arrays
    Q_n, K_n, V_n, out_n = Q._native, K._native, V._native, out._native

    # Warmup
    for _ in range(3):
        native.sdpa_causal_timed(Q_n, K_n, V_n, out_n, 0.0)

    times_us = []
    for _ in range(num_iters):
        kernel_time_us = native.sdpa_causal_timed(Q_n, K_n, V_n, out_n, 0.0)
        times_us.append(kernel_time_us)

    avg_us = np.mean(times_us)
    std_us = np.std(times_us)
    return avg_us, std_us


def benchmark_e2e(Q, K, V, num_iters: int = 100) -> tuple[float, float]:
    """Benchmark end-to-end (includes Python overhead)."""
    # Warmup
    for _ in range(3):
        out = sdpa_causal(Q, K, V)
    native.device_synchronize()

    times_us = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = sdpa_causal(Q, K, V)
        native.device_synchronize()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    avg_us = np.mean(times_us)
    std_us = np.std(times_us)
    return avg_us, std_us


def benchmark_e2e_cached(Q, K, V, out, num_iters: int = 100) -> tuple[float, float]:
    """Benchmark E2E with pre-allocated output (realistic usage)."""
    # Get native arrays
    Q_n, K_n, V_n, out_n = Q._native, K._native, V._native, out._native

    # Warmup
    for _ in range(3):
        native.sdpa_causal_(Q_n, K_n, V_n, out_n, 0.0)
    native.device_synchronize()

    times_us = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        native.sdpa_causal_(Q_n, K_n, V_n, out_n, 0.0)
        native.device_synchronize()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    avg_us = np.mean(times_us)
    std_us = np.std(times_us)
    return avg_us, std_us


def main():
    # Parse args
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    num_heads = 32
    head_dim = 128

    print("=" * 60)
    print("FA3 TMA Attention Benchmark")
    print("=" * 60)
    print(f"  seq_len    = {seq_len}")
    print(f"  num_heads  = {num_heads}")
    print(f"  head_dim   = {head_dim}")
    print(f"  iterations = {num_iters}")
    print()

    # Create inputs
    np.random.seed(42)
    Q_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    K_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    V_np = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)

    bf16 = DataType.from_string("bfloat16")
    Q = gpk.from_numpy(Q_np).astype(bf16)
    K = gpk.from_numpy(K_np).astype(bf16)
    V = gpk.from_numpy(V_np).astype(bf16)

    # Pre-allocate output for cached benchmarks
    out = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)

    # Clear cache for fresh start
    native.clear_tma_cache()

    # First call - cold cache (creates descriptors)
    print("Cold cache (first call)...")
    cold_time_us = native.sdpa_causal_timed(Q._native, K._native, V._native, out._native, 0.0)
    print(f"  Cold time: {cold_time_us:.1f} us")
    print()
    native.print_tma_cache_stats()
    print()

    # Kernel-only benchmark (cudaEvent)
    print("Kernel-only benchmark (cudaEvent timing)...")
    kernel_avg_us, kernel_std_us = benchmark_kernel_only(Q, K, V, out, num_iters)
    kernel_tflops = compute_tflops(seq_len, num_heads, head_dim, kernel_avg_us)
    print(f"  Avg time: {kernel_avg_us:.1f} +/- {kernel_std_us:.1f} us")
    print(f"  TFLOPS:   {kernel_tflops:.2f}")
    print()

    # E2E with pre-allocated output (realistic reuse)
    print("E2E benchmark (pre-allocated output, realistic reuse)...")
    e2e_cached_avg_us, e2e_cached_std_us = benchmark_e2e_cached(Q, K, V, out, num_iters)
    e2e_cached_tflops = compute_tflops(seq_len, num_heads, head_dim, e2e_cached_avg_us)
    print(f"  Avg time: {e2e_cached_avg_us:.1f} +/- {e2e_cached_std_us:.1f} us")
    print(f"  TFLOPS:   {e2e_cached_tflops:.2f}")
    print()

    # E2E with allocation (worst case)
    print("E2E benchmark (with allocation, worst case)...")
    e2e_avg_us, e2e_std_us = benchmark_e2e(Q, K, V, num_iters)
    e2e_tflops = compute_tflops(seq_len, num_heads, head_dim, e2e_avg_us)
    print(f"  Avg time: {e2e_avg_us:.1f} +/- {e2e_std_us:.1f} us")
    print(f"  TFLOPS:   {e2e_tflops:.2f}")
    print()

    # Final cache stats
    print("Final TMA cache statistics:")
    native.print_tma_cache_stats()
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Kernel-only:  {kernel_avg_us:8.1f} us  ({kernel_tflops:.2f} TFLOPS)")
    print(f"  E2E cached:   {e2e_cached_avg_us:8.1f} us  ({e2e_cached_tflops:.2f} TFLOPS)")
    print(f"  E2E allocate: {e2e_avg_us:8.1f} us  ({e2e_tflops:.2f} TFLOPS)")
    print()
    overhead_us = e2e_cached_avg_us - kernel_avg_us
    print(f"  Host overhead (cached): {overhead_us:.1f} us ({100*overhead_us/e2e_cached_avg_us:.1f}%)")

    # Verify correctness
    print()
    print("Verifying correctness...")

    # Reset output and run fresh timed call
    out_test = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)
    native.sdpa_causal_timed(Q._native, K._native, V._native, out_test._native, 0.0)

    # Get reference using standard path
    out_ref = sdpa_causal(Q, K, V)

    # Convert to FP32 for comparison (BF16 to_numpy returns raw uint16)
    fp32 = DataType.from_string("float32")
    out_test_fp32 = out_test.astype(fp32).to_numpy()
    out_ref_fp32 = out_ref.astype(fp32).to_numpy()

    # Debug: check for NaNs/Infs
    if np.any(np.isnan(out_test_fp32)):
        print("  WARNING: Output contains NaN values")
    if np.any(np.isinf(out_test_fp32)):
        print("  WARNING: Output contains Inf values")

    max_diff = np.max(np.abs(out_test_fp32 - out_ref_fp32))
    rel_diff = max_diff / (np.max(np.abs(out_ref_fp32)) + 1e-8)
    print(f"  Max abs difference: {max_diff:.6e}")
    print(f"  Relative difference: {rel_diff:.6e}")
    print(f"  Output range: [{out_test_fp32.min():.4f}, {out_test_fp32.max():.4f}]")
    print(f"  Reference range: [{out_ref_fp32.min():.4f}, {out_ref_fp32.max():.4f}]")

    if max_diff < 1e-1 or rel_diff < 1e-2:
        print("  Correctness: PASS")
    else:
        print("  Correctness: FAIL")


if __name__ == "__main__":
    main()
