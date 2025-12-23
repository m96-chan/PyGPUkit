#!/usr/bin/env python3
"""Test Flash-Decoding correctness against standard SDPA.

This test must be run twice:
1. PYGPUKIT_FLASH_DECODING=0 python test_flash_decoding.py --save-ref
2. PYGPUKIT_FLASH_DECODING=1 python test_flash_decoding.py --compare-ref
"""

import os
import sys
import time

import numpy as np

from pygpukit.core import default_stream, from_numpy
from pygpukit.ops.basic import sdpa_causal_fixed_cache

print("=" * 60)
print("Flash-Decoding Correctness Test")
print(f"PYGPUKIT_FLASH_DECODING = {os.environ.get('PYGPUKIT_FLASH_DECODING', 'not set')}")
print("=" * 60)

# Qwen3-8B dimensions
n_heads = 32
head_dim = 128
max_seq_len = 512
context_len = 256

np.random.seed(42)

# Create random Q, K, V in SDPA format: [n_heads, seq_len, head_dim]
q_np = np.random.randn(n_heads, 1, head_dim).astype(np.float16) * 0.1
k_np = np.random.randn(n_heads, max_seq_len, head_dim).astype(np.float16) * 0.1
v_np = np.random.randn(n_heads, max_seq_len, head_dim).astype(np.float16) * 0.1

q = from_numpy(q_np)
k = from_numpy(k_np)
v = from_numpy(v_np)
out = from_numpy(np.zeros((n_heads, 1, head_dim), dtype=np.float16))

# Warm up
for _ in range(3):
    sdpa_causal_fixed_cache(q, k, v, out, context_len)
default_stream().synchronize()

# Benchmark
n_iters = 100
default_stream().synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    sdpa_causal_fixed_cache(q, k, v, out, context_len)
default_stream().synchronize()
elapsed = (time.perf_counter() - start) / n_iters * 1000

result = out.to_numpy()

print(f"\nContext length: {context_len}")
print(f"Time per call: {elapsed:.3f} ms")
print(f"Output shape: {result.shape}")
print(f"Output sample: {result[0, 0, :5]}")

# Save/compare reference
ref_file = "flash_decoding_ref.npy"
if "--save-ref" in sys.argv:
    np.save(ref_file, result)
    print(f"\nReference saved to {ref_file}")
elif "--compare-ref" in sys.argv:
    if os.path.exists(ref_file):
        ref = np.load(ref_file)
        diff = np.abs(result.astype(np.float32) - ref.astype(np.float32))
        max_diff = diff.max()
        mean_diff = diff.mean()
        print("\n=== Comparison with reference ===")
        print(f"Max abs diff: {max_diff:.6f}")
        print(f"Mean abs diff: {mean_diff:.6f}")
        print(f"Status: {'PASS' if max_diff < 0.01 else 'FAIL'}")
    else:
        print(f"\nReference file {ref_file} not found. Run with --save-ref first.")
else:
    # Single test mode - just run both configurations
    print("\n=== Running both configurations ===")

    # Test with different context lengths
    test_contexts = [64, 128, 256, 512]

    for ctx in test_contexts:
        q = from_numpy(q_np)
        k = from_numpy(k_np)
        v = from_numpy(v_np)
        out = from_numpy(np.zeros((n_heads, 1, head_dim), dtype=np.float16))

        # Time the SDPA call
        default_stream().synchronize()
        start = time.perf_counter()
        for _ in range(100):
            sdpa_causal_fixed_cache(q, k, v, out, ctx)
        default_stream().synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1000

        print(f"  context_len={ctx:3d}: {elapsed:.3f} ms/call")

print("\n" + "=" * 60)
print("Done")
print("=" * 60)
