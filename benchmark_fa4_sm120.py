"""
Benchmark FA4 SM120 Attention Kernel

Phases:
- Phase 1: BF16 Baseline (same as FA3)
- Phase 2: NVFP4 Q@K^T
- Phase 3: Full NVFP4

Usage:
    python benchmark_fa4_sm120.py [seq_len] [num_iterations]
"""
import os
import sys
import time

import numpy as np

# Check for FA4 availability
try:
    import pygpukit as gpk
    from pygpukit.core.backend import get_native_module
    from pygpukit.core.dtypes import DataType

    native = get_native_module()
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def compute_tflops(seq_len: int, num_heads: int, head_dim: int, time_us: float) -> float:
    """Compute TFLOPS for SDPA operation."""
    # SDPA FLOPs: 4 * seq * seq * head_dim * num_heads
    flops = 4 * seq_len * seq_len * head_dim * num_heads
    return flops / (time_us * 1e-6) / 1e12


def run_fa3_reference(Q, K, V, out, num_iters: int = 10):
    """Run FA3 TMA as reference."""
    Q_n, K_n, V_n, out_n = Q._native, K._native, V._native, out._native

    # Warmup
    for _ in range(3):
        native.sdpa_causal_timed(Q_n, K_n, V_n, out_n, 0.0)

    times_us = []
    for _ in range(num_iters):
        kernel_time_us = native.sdpa_causal_timed(Q_n, K_n, V_n, out_n, 0.0)
        times_us.append(kernel_time_us)

    return np.mean(times_us), np.std(times_us)


def run_fa4_phase1(Q, K, V, out, num_iters: int = 10):
    """Run FA4 Phase 1 (BF16 Baseline)."""
    Q_n, K_n, V_n, out_n = Q._native, K._native, V._native, out._native

    # Check if FA4 is available
    if not hasattr(native, 'fa4_phase1_timed'):
        return None, None

    # Warmup
    for _ in range(3):
        native.fa4_phase1_timed(Q_n, K_n, V_n, out_n, 0.0)

    times_us = []
    for _ in range(num_iters):
        kernel_time_us = native.fa4_phase1_timed(Q_n, K_n, V_n, out_n, 0.0)
        times_us.append(kernel_time_us)

    return np.mean(times_us), np.std(times_us)


def verify_correctness(out_test, out_ref, name: str, rtol: float = 1e-2, atol: float = 1e-2):
    """Verify output against reference."""
    fp32 = DataType.from_string("float32")
    test_np = out_test.astype(fp32).to_numpy()
    ref_np = out_ref.astype(fp32).to_numpy()

    max_diff = np.max(np.abs(test_np - ref_np))
    rel_diff = max_diff / (np.max(np.abs(ref_np)) + 1e-8)

    passed = max_diff < atol or rel_diff < rtol

    print(f"  {name}:")
    print(f"    Max abs diff: {max_diff:.6e}")
    print(f"    Rel diff: {rel_diff:.6e}")
    print(f"    Status: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    num_heads = 32
    head_dim = 128

    print("=" * 70)
    print("FA4 SM120 Benchmark")
    print("=" * 70)
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

    # Pre-allocate outputs
    out_fa3 = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)
    out_fa4 = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)

    # Clear TMA cache
    native.clear_tma_cache()

    # =========================================================================
    # FA3 Reference (TMA)
    # =========================================================================
    print("FA3 TMA Reference:")
    fa3_avg, fa3_std = run_fa3_reference(Q, K, V, out_fa3, num_iters)
    fa3_tflops = compute_tflops(seq_len, num_heads, head_dim, fa3_avg)
    print(f"  Time: {fa3_avg:.1f} +/- {fa3_std:.1f} us")
    print(f"  TFLOPS: {fa3_tflops:.2f}")
    print()

    # =========================================================================
    # FA4 Phase 1 (BF16 Baseline)
    # =========================================================================
    print("FA4 Phase 1 (BF16 Baseline):")
    fa4_avg, fa4_std = run_fa4_phase1(Q, K, V, out_fa4, num_iters)

    if fa4_avg is not None:
        fa4_tflops = compute_tflops(seq_len, num_heads, head_dim, fa4_avg)
        print(f"  Time: {fa4_avg:.1f} +/- {fa4_std:.1f} us")
        print(f"  TFLOPS: {fa4_tflops:.2f}")

        # Compare with FA3
        speedup = fa3_avg / fa4_avg if fa4_avg > 0 else 0
        print(f"  vs FA3: {speedup:.2f}x")
    else:
        print("  FA4 not available (native binding missing)")
        fa4_tflops = 0
    print()

    # =========================================================================
    # Correctness Verification
    # =========================================================================
    print("Correctness Verification:")

    # Re-run single iteration for clean comparison
    native.clear_tma_cache()
    out_fa3_verify = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)
    out_fa4_verify = gpk.zeros((num_heads, seq_len, head_dim), dtype=bf16)

    native.sdpa_causal_timed(Q._native, K._native, V._native, out_fa3_verify._native, 0.0)

    if hasattr(native, 'fa4_phase1_timed'):
        native.fa4_phase1_timed(Q._native, K._native, V._native, out_fa4_verify._native, 0.0)
        verify_correctness(out_fa4_verify, out_fa3_verify, "FA4 Phase 1 vs FA3")
    else:
        print("  FA4 not available for verification")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  FA3 TMA:      {fa3_avg:8.1f} us  ({fa3_tflops:.2f} TFLOPS)")
    if fa4_avg is not None:
        print(f"  FA4 Phase 1:  {fa4_avg:8.1f} us  ({fa4_tflops:.2f} TFLOPS)")
    print()


if __name__ == "__main__":
    main()
