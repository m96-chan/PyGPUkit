#!/usr/bin/env python3
"""
Flash Attention 3 Benchmark & Correctness Test

Compares FA3 (SM120+) vs FA2 vs standard SDPA:
- Correctness: relative error vs reference
- Performance: latency and throughput
"""

import os
import time
import numpy as np

# Set environment before import
os.environ["PYGPUKIT_FA3"] = "0"  # Start with FA3 off
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"  # Start with FA2 off

import pygpukit as gk
from pygpukit.core.backend import get_native_module
from pygpukit.core.dtypes import DataType
from pygpukit.ops.nn import sdpa_causal


def reference_attention_cpu(Q, K, V, scale):
    """CPU reference implementation for correctness check."""
    # Q: [n_heads, q_len, head_dim]
    # K: [n_heads, kv_len, head_dim]
    # V: [n_heads, kv_len, head_dim]
    n_heads, q_len, head_dim = Q.shape
    kv_len = K.shape[1]

    output = np.zeros_like(Q)

    for h in range(n_heads):
        # Compute attention scores
        scores = np.matmul(Q[h], K[h].T) * scale  # [q_len, kv_len]

        # Apply causal mask
        for i in range(q_len):
            causal_offset = kv_len - q_len
            max_attend = causal_offset + i + 1
            scores[i, max_attend:] = -np.inf

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
        weights = scores_exp / scores_sum

        # Output
        output[h] = np.matmul(weights, V[h])

    return output


def run_sdpa_with_mode(Q_np, K_np, V_np, scale, mode, native):
    """Run SDPA with specific mode."""
    if mode == "sdpa":
        os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"
        os.environ["PYGPUKIT_FA3"] = "0"
    elif mode == "fa2":
        os.environ["PYGPUKIT_FLASH_ATTENTION"] = "1"
        os.environ["PYGPUKIT_FA3"] = "0"
    elif mode == "fa3":
        os.environ["PYGPUKIT_FLASH_ATTENTION"] = "0"
        os.environ["PYGPUKIT_FA3"] = "1"

    # Create GPU arrays (bfloat16)
    bf16 = DataType.from_string("bfloat16")
    Q_gpu = gk.from_numpy(Q_np.astype(np.float32)).astype(bf16)
    K_gpu = gk.from_numpy(K_np.astype(np.float32)).astype(bf16)
    V_gpu = gk.from_numpy(V_np.astype(np.float32)).astype(bf16)

    # Warmup
    for _ in range(3):
        out = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    native.device_synchronize()

    # Benchmark
    n_iters = 20
    native.device_synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        out = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    native.device_synchronize()
    elapsed = time.perf_counter() - start

    avg_time_us = (elapsed / n_iters) * 1e6

    # Convert bfloat16 output to float32 for comparison
    fp32 = DataType.from_string("float32")
    out_fp32 = out.astype(fp32)
    return out_fp32.to_numpy(), avg_time_us


def compute_error(output, reference):
    """Compute relative error with proper handling of near-zero values."""
    diff = np.abs(output - reference)

    # Use a combination of absolute and relative error
    # For values near zero, absolute error is more meaningful
    abs_tol = 1e-4  # Absolute tolerance
    ref_abs = np.abs(reference)

    # Relative error where reference is large enough
    # Absolute error otherwise
    mask = ref_abs > abs_tol
    rel_error = np.zeros_like(diff)
    rel_error[mask] = diff[mask] / ref_abs[mask]
    rel_error[~mask] = diff[~mask]  # Use absolute error for small values

    return np.max(rel_error), np.mean(rel_error[mask]) if np.any(mask) else np.mean(diff)


def benchmark_config(n_heads, seq_len, head_dim, native, sm_version):
    """Benchmark a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Config: n_heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"{'='*60}")

    # Generate random data
    np.random.seed(42)
    Q = np.random.randn(n_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(n_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(n_heads, seq_len, head_dim).astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)

    # CPU reference
    print("\nComputing CPU reference...")
    ref_output = reference_attention_cpu(Q, K, V, scale)

    results = {}

    # Determine which modes to test
    modes = ["sdpa", "fa2"]
    if sm_version >= 120:
        modes.append("fa3")

    # Test each mode
    for mode in modes:
        print(f"\nTesting {mode.upper()}...")
        try:
            output, time_us = run_sdpa_with_mode(Q, K, V, scale, mode, native)
            max_err, mean_err = compute_error(output, ref_output)
            # BF16 precision is ~7 bits mantissa vs FP32 ~23 bits
            # Mean error < 5% is good for BF16
            status = "PASS" if mean_err < 0.05 else "FAIL"
            results[mode] = {
                "time_us": time_us,
                "max_rel_error": max_err,
                "mean_rel_error": mean_err,
                "status": status
            }
            print(f"  Time: {time_us:.1f} us")
            print(f"  Max rel error: {max_err:.2e}")
            print(f"  Mean rel error: {mean_err:.2e}")
            print(f"  Status: {results[mode]['status']}")
        except Exception as e:
            results[mode] = {
                "time_us": float('inf'),
                "max_rel_error": float('inf'),
                "mean_rel_error": float('inf'),
                "status": f"ERROR: {e}"
            }
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Mode':<10} {'Time (us)':<15} {'Max Err':<15} {'Status':<10}")
    print(f"{'-'*50}")
    for mode, r in results.items():
        print(f"{mode.upper():<10} {r['time_us']:<15.1f} {r['max_rel_error']:<15.2e} {r['status']:<10}")

    # Speedup calculations
    if "fa3" in results and results.get("sdpa", {}).get("time_us", float('inf')) < float('inf'):
        if results["fa3"]["time_us"] < float('inf'):
            speedup_vs_sdpa = results["sdpa"]["time_us"] / results["fa3"]["time_us"]
            print(f"\nFA3 vs SDPA speedup: {speedup_vs_sdpa:.2f}x")

    if "fa3" in results and results.get("fa2", {}).get("time_us", float('inf')) < float('inf'):
        if results["fa3"]["time_us"] < float('inf'):
            speedup_vs_fa2 = results["fa2"]["time_us"] / results["fa3"]["time_us"]
            print(f"FA3 vs FA2 speedup: {speedup_vs_fa2:.2f}x")

    return results


def main():
    print("=" * 60)
    print("Flash Attention 3 Benchmark & Correctness Test")
    print("=" * 60)

    # Get native module
    native = get_native_module()

    # Check GPU
    props = native.get_device_properties(0)
    sm_version = props.compute_capability_major * 10 + props.compute_capability_minor
    print(f"\nGPU: {props.name}")
    print(f"SM Version: {sm_version}")
    print(f"FA3 Available: {'Yes' if sm_version >= 120 else 'No (requires SM120+)'}")

    if sm_version < 120:
        print("\nWARNING: FA3 requires SM120+. Running FA2/SDPA comparison only.")

    # Test configurations (n_heads, seq_len, head_dim)
    configs = [
        # Small config for quick correctness check
        (8, 128, 128),
        # Medium config
        (32, 512, 128),
        # Large config (typical LLM)
        (32, 1024, 128),
        (32, 2048, 128),
    ]

    all_results = {}
    for config in configs:
        try:
            results = benchmark_config(*config, native, sm_version)
            all_results[config] = results
        except Exception as e:
            print(f"Config {config} failed: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    header = f"{'Config':<25} {'SDPA':<12} {'FA2':<12}"
    if sm_version >= 120:
        header += f" {'FA3':<12} {'FA3/SDPA':<10}"
    print(header)
    print("-" * 70)

    for config, results in all_results.items():
        config_str = f"{config[0]}h x {config[1]}seq x {config[2]}d"
        sdpa_time = results.get("sdpa", {}).get("time_us", float('inf'))
        fa2_time = results.get("fa2", {}).get("time_us", float('inf'))

        row = f"{config_str:<25} {sdpa_time:<12.0f} {fa2_time:<12.0f}"

        if sm_version >= 120:
            fa3_time = results.get("fa3", {}).get("time_us", float('inf'))
            if sdpa_time < float('inf') and fa3_time < float('inf'):
                speedup = f"{sdpa_time/fa3_time:.2f}x"
            else:
                speedup = "N/A"
            row += f" {fa3_time:<12.0f} {speedup:<10}"

        print(row)


if __name__ == "__main__":
    main()
