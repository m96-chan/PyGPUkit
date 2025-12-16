#!/usr/bin/env python3
"""
PyGPUkit CUDA Graph Demo

Demonstrates CUDA Graph capture and replay for optimized inference:
1. Basic CUDA Graph capture/replay with matmul
2. Fixed-length KV cache for decode optimization
3. Performance comparison with/without CUDA Graph

Usage:
    python demo_cuda_graph.py

Requirements:
    - PyGPUkit v0.2.10+
    - CUDA capable GPU (SM >= 80)
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer(name: str):
    """Simple timer context manager."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {name}: {elapsed:.2f} ms")


def demo_basic_cuda_graph():
    """Demo 1: Basic CUDA Graph capture and replay."""
    print("\n" + "=" * 70)
    print(" Demo 1: Basic CUDA Graph Capture/Replay")
    print("=" * 70)

    import pygpukit as pk

    native = pk._pygpukit_native

    # Create test tensors
    print("\nCreating test tensors [4096, 4096]...")
    A_np = np.random.randn(4096, 4096).astype(np.float16)
    B_np = np.random.randn(4096, 4096).astype(np.float16)

    A = native.from_numpy(A_np)
    B = native.from_numpy(B_np)
    C = native.from_numpy(np.zeros((4096, 4096), dtype=np.float16))

    # Create CUDA Graph
    print("\nCreating CUDA Graph...")
    graph = native.CudaGraph()

    # Capture
    print("  Capturing matmul operation...")
    graph.begin_capture()
    native.matmul_(A, B, C)  # In-place matmul into C
    graph.end_capture()

    print(f"  Graph ready: {graph.is_ready()}")
    print(f"  Graph nodes: {graph.num_nodes}")

    # Benchmark: Without CUDA Graph
    print("\nBenchmark: Without CUDA Graph")

    # Warmup
    for _ in range(3):
        native.matmul_(A, B, C)

    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        native.matmul_(A, B, C)
    elapsed_no_graph = (time.perf_counter() - start) * 1000 / iterations
    print(f"  Average per iteration: {elapsed_no_graph:.2f} ms")

    # Benchmark: With CUDA Graph
    print("\nBenchmark: With CUDA Graph")

    # Warmup replays
    for _ in range(3):
        graph.replay()

    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    elapsed_with_graph = (time.perf_counter() - start) * 1000 / iterations
    print(f"  Average per iteration: {elapsed_with_graph:.2f} ms")

    # Speedup
    speedup = elapsed_no_graph / elapsed_with_graph
    print(f"\n  Speedup: {speedup:.2f}x")

    return True


def demo_fixed_kv_cache():
    """Demo 2: Fixed-length KV cache operations."""
    print("\n" + "=" * 70)
    print(" Demo 2: Fixed-Length KV Cache Operations")
    print("=" * 70)

    import pygpukit as pk

    native = pk._pygpukit_native

    # Model config (Qwen3-8B like)
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 512
    prefill_len = 10

    print("\nKV Cache Config:")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  max_seq_len: {max_seq_len}")

    # Allocate fixed-length KV cache (using native API directly)
    print("\nAllocating fixed-length KV cache...")
    k_cache_np = np.zeros((max_seq_len, num_kv_heads, head_dim), dtype=np.float16)
    v_cache_np = np.zeros((max_seq_len, num_kv_heads, head_dim), dtype=np.float16)

    k_cache = native.from_numpy(k_cache_np)
    v_cache = native.from_numpy(v_cache_np)

    cache_size_mb = (k_cache_np.nbytes + v_cache_np.nbytes) / 1024 / 1024
    print(f"  Cache size per layer: {cache_size_mb:.2f} MB")

    # Test prefill (using native API directly)
    print("\nTesting prefill...")
    prefill_k = np.random.randn(prefill_len, num_kv_heads, head_dim).astype(np.float16)
    prefill_v = np.random.randn(prefill_len, num_kv_heads, head_dim).astype(np.float16)

    prefill_k_gpu = native.from_numpy(prefill_k)
    prefill_v_gpu = native.from_numpy(prefill_v)

    native.kv_cache_prefill(prefill_k_gpu, k_cache, 0)
    native.kv_cache_prefill(prefill_v_gpu, v_cache, 0)

    # Verify prefill
    k_cache_result = k_cache.to_numpy()
    prefill_match = np.allclose(k_cache_result[:prefill_len], prefill_k, rtol=1e-3)
    print(f"  Prefill correctness: {'PASS' if prefill_match else 'FAIL'}")

    # Test decode update (using native API directly)
    print("\nTesting decode updates...")
    for pos in range(prefill_len, prefill_len + 5):
        new_k = np.random.randn(1, num_kv_heads, head_dim).astype(np.float16)
        new_v = np.random.randn(1, num_kv_heads, head_dim).astype(np.float16)

        new_k_gpu = native.from_numpy(new_k)
        new_v_gpu = native.from_numpy(new_v)

        native.kv_cache_update(new_k_gpu, k_cache, pos)
        native.kv_cache_update(new_v_gpu, v_cache, pos)

        # Verify
        k_cache_result = k_cache.to_numpy()
        update_match = np.allclose(k_cache_result[pos], new_k[0], rtol=1e-3)
        print(f"  Position {pos} update: {'PASS' if update_match else 'FAIL'}")

    return True


def demo_sdpa_fixed_cache():
    """Demo 3: SDPA with fixed-length KV cache."""
    print("\n" + "=" * 70)
    print(" Demo 3: SDPA with Fixed-Length KV Cache")
    print("=" * 70)

    import pygpukit as pk

    native = pk._pygpukit_native

    # Config
    n_heads = 8
    max_seq_len = 256
    head_dim = 64
    context_len = 50  # Actual valid tokens
    q_len = 1  # Single query (decode)

    print("\nSDPA Config:")
    print(f"  n_heads: {n_heads}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  context_len: {context_len}")
    print(f"  head_dim: {head_dim}")

    # Create tensors (using native API directly)
    print("\nCreating tensors...")

    # Q: [n_heads, q_len, head_dim]
    Q = native.from_numpy(np.random.randn(n_heads, q_len, head_dim).astype(np.float16))

    # K, V: [n_heads, max_seq_len, head_dim] - fixed cache size
    K = native.from_numpy(np.random.randn(n_heads, max_seq_len, head_dim).astype(np.float16))
    V = native.from_numpy(np.random.randn(n_heads, max_seq_len, head_dim).astype(np.float16))

    # Output: [n_heads, q_len, head_dim]
    out = native.from_numpy(np.zeros((n_heads, q_len, head_dim), dtype=np.float16))

    # Call SDPA with fixed cache (using native API directly)
    print("\nRunning SDPA with fixed cache...")
    native.sdpa_causal_fixed_cache(Q, K, V, out, context_len, 0.0)

    result = out.to_numpy()
    print(f"  Output shape: {result.shape}")
    print(f"  Output mean: {result.mean():.6f}")
    print(f"  Output std: {result.std():.6f}")

    # Verify output is not all zeros (computation happened)
    if np.abs(result.mean()) > 1e-6 or result.std() > 1e-6:
        print("  [PASS] SDPA with fixed cache working")
        return True
    else:
        print("  [FAIL] Output appears to be zeros")
        return False


def demo_cuda_graph_with_kv_cache():
    """Demo 4: CUDA Graph with KV cache update."""
    print("\n" + "=" * 70)
    print(" Demo 4: CUDA Graph with KV Cache Update")
    print("=" * 70)

    import pygpukit as pk

    native = pk._pygpukit_native

    # Config
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 512

    print("\nCapturing KV cache update into CUDA Graph...")

    # Allocate buffers
    k_cache = native.from_numpy(np.zeros((max_seq_len, num_kv_heads, head_dim), dtype=np.float16))
    new_k = native.from_numpy(np.random.randn(1, num_kv_heads, head_dim).astype(np.float16))

    # Create and capture graph
    graph = native.CudaGraph()

    graph.begin_capture()
    native.kv_cache_update(new_k, k_cache, 0)  # Position is fixed at capture time
    graph.end_capture()

    print(f"  Graph ready: {graph.is_ready()}")
    print(f"  Graph nodes: {graph.num_nodes}")

    # Benchmark
    iterations = 100

    # Without graph
    start = time.perf_counter()
    for i in range(iterations):
        native.kv_cache_update(new_k, k_cache, i % max_seq_len)
    elapsed_no_graph = (time.perf_counter() - start) * 1000 / iterations

    # With graph (note: position is fixed, just for kernel launch overhead comparison)
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    elapsed_with_graph = (time.perf_counter() - start) * 1000 / iterations

    print(f"\n  Without graph: {elapsed_no_graph * 1000:.2f} us/iter")
    print(f"  With graph: {elapsed_with_graph * 1000:.2f} us/iter")
    print(f"  Speedup: {elapsed_no_graph / elapsed_with_graph:.2f}x")

    return True


def main():
    print("\n" + "=" * 70)
    print(" PyGPUkit CUDA Graph Demo")
    print("=" * 70)

    results = []

    # Demo 1: Basic CUDA Graph
    try:
        results.append(("Basic CUDA Graph", demo_basic_cuda_graph()))
    except Exception as e:
        print(f"  [FAIL] Basic CUDA Graph: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Basic CUDA Graph", False))

    # Demo 2: Fixed KV Cache
    try:
        results.append(("Fixed KV Cache", demo_fixed_kv_cache()))
    except Exception as e:
        print(f"  [FAIL] Fixed KV Cache: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Fixed KV Cache", False))

    # Demo 3: SDPA with fixed cache
    try:
        results.append(("SDPA Fixed Cache", demo_sdpa_fixed_cache()))
    except Exception as e:
        print(f"  [FAIL] SDPA Fixed Cache: {e}")
        import traceback

        traceback.print_exc()
        results.append(("SDPA Fixed Cache", False))

    # Demo 4: CUDA Graph with KV cache
    try:
        results.append(("CUDA Graph + KV Cache", demo_cuda_graph_with_kv_cache()))
    except Exception as e:
        print(f"  [FAIL] CUDA Graph + KV Cache: {e}")
        import traceback

        traceback.print_exc()
        results.append(("CUDA Graph + KV Cache", False))

    # Summary
    print("\n" + "=" * 70)
    print(" Demo Summary")
    print("=" * 70)

    print("\nResults:")
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("All demos completed successfully!")
    else:
        print("Some demos failed. Check the output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
