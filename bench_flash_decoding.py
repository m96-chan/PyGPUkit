#!/usr/bin/env python3
"""Benchmark Flash-Decoding vs Standard SDPA.

Compares performance across different context lengths.
"""

import subprocess
import sys

# Test configurations
test_contexts = [64, 128, 256, 512, 1024, 2048]

results = {"standard": {}, "flash": {}}

print("=" * 70)
print("Flash-Decoding vs Standard SDPA Benchmark")
print("=" * 70)

# Run benchmark for each configuration
script = """
import os
import numpy as np
import time
from pygpukit.core import from_numpy, default_stream
from pygpukit.ops.basic import sdpa_causal_fixed_cache

n_heads = 32
head_dim = 128
max_seq_len = {max_seq_len}
context_len = {context_len}

np.random.seed(42)
q_np = np.random.randn(n_heads, 1, head_dim).astype(np.float16) * 0.1
k_np = np.random.randn(n_heads, max_seq_len, head_dim).astype(np.float16) * 0.1
v_np = np.random.randn(n_heads, max_seq_len, head_dim).astype(np.float16) * 0.1

q = from_numpy(q_np)
k = from_numpy(k_np)
v = from_numpy(v_np)
out = from_numpy(np.zeros((n_heads, 1, head_dim), dtype=np.float16))

# Warm up
for _ in range(10):
    sdpa_causal_fixed_cache(q, k, v, out, context_len)
default_stream().synchronize()

# Benchmark
n_iters = 200
default_stream().synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    sdpa_causal_fixed_cache(q, k, v, out, context_len)
default_stream().synchronize()
elapsed = (time.perf_counter() - start) / n_iters * 1000

print(f"{{elapsed:.4f}}")
"""

print(f"\n{'Context':<10} {'Standard':<12} {'Flash-Dec':<12} {'Speedup':<10}")
print("-" * 44)

for ctx in test_contexts:
    max_seq = max(ctx, 512)

    # Standard SDPA
    code = script.format(max_seq_len=max_seq, context_len=ctx)
    env = {"PYGPUKIT_FLASH_DECODING": "0"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )
    std_time = float(result.stdout.strip()) if result.returncode == 0 else -1

    # Flash-Decoding
    env = {"PYGPUKIT_FLASH_DECODING": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )
    flash_time = float(result.stdout.strip()) if result.returncode == 0 else -1

    speedup = std_time / flash_time if flash_time > 0 else 0
    print(f"{ctx:<10} {std_time:>8.3f} ms  {flash_time:>8.3f} ms  {speedup:>6.2f}x")

print("\n" + "=" * 70)
print("Notes:")
print("- Flash-Decoding CHUNK_SIZE = 256")
print("- Speedup < 1.0x means Flash-Decoding is slower")
print("- Expected benefit when context_len > 256 (multiple chunks)")
print("=" * 70)
