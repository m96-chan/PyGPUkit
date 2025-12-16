#!/usr/bin/env python3
"""Test Flash Attention kernel correctness."""

import os

import numpy as np

# Enable Flash Attention
os.environ["PYGPUKIT_FLASH_ATTENTION"] = "1"

from pygpukit.core.factory import from_numpy
from pygpukit.ops import sdpa_causal


def test_flash_attention_correctness():
    """Compare Flash Attention output with NumPy reference."""
    print("Testing Flash Attention correctness...")

    # Test parameters
    n_heads = 4
    q_len = 16
    kv_len = 16
    head_dim = 64
    scale = 1.0 / np.sqrt(head_dim)

    # Generate random inputs
    np.random.seed(42)
    Q_np = np.random.randn(n_heads, q_len, head_dim).astype(np.float32)
    K_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float32)
    V_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float32)

    # NumPy reference (standard attention with causal mask)
    scores = np.matmul(Q_np, K_np.transpose(0, 2, 1)) * scale

    # Apply causal mask
    causal_offset = kv_len - q_len
    for i in range(q_len):
        max_attend = causal_offset + i + 1
        if max_attend < kv_len:
            scores[:, i, max_attend:] = -np.inf

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Output
    ref_output = np.matmul(weights, V_np)

    # GPU computation with Flash Attention
    Q_gpu = from_numpy(Q_np)
    K_gpu = from_numpy(K_np)
    V_gpu = from_numpy(V_np)

    result_gpu = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    result_np = result_gpu.to_numpy()

    # Compare
    max_diff = np.abs(result_np - ref_output).max()
    mean_diff = np.abs(result_np - ref_output).mean()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("  PASS: Flash Attention matches reference")
        return True
    else:
        print("  FAIL: Flash Attention differs from reference")
        return False


def test_flash_attention_kv_cache():
    """Test Flash Attention with KV cache (kv_len > q_len)."""
    print("\nTesting Flash Attention with KV cache...")

    n_heads = 4
    q_len = 1  # Single token decode
    kv_len = 32  # Cached KV
    head_dim = 64
    scale = 1.0 / np.sqrt(head_dim)

    np.random.seed(123)
    Q_np = np.random.randn(n_heads, q_len, head_dim).astype(np.float32)
    K_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float32)
    V_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float32)

    # NumPy reference
    scores = np.matmul(Q_np, K_np.transpose(0, 2, 1)) * scale

    # Causal mask for decode (can attend to all kv_len positions)
    # No masking needed since we're decoding the last position

    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    ref_output = np.matmul(weights, V_np)

    # GPU computation
    Q_gpu = from_numpy(Q_np)
    K_gpu = from_numpy(K_np)
    V_gpu = from_numpy(V_np)

    result_gpu = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    result_np = result_gpu.to_numpy()

    max_diff = np.abs(result_np - ref_output).max()
    mean_diff = np.abs(result_np - ref_output).mean()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("  PASS: KV cache test matches reference")
        return True
    else:
        print("  FAIL: KV cache test differs from reference")
        return False


def test_flash_attention_fp16():
    """Test Flash Attention with FP16."""
    print("\nTesting Flash Attention with FP16...")

    n_heads = 4
    q_len = 16
    kv_len = 16
    head_dim = 64
    scale = 1.0 / np.sqrt(head_dim)

    np.random.seed(456)
    Q_np = np.random.randn(n_heads, q_len, head_dim).astype(np.float16)
    K_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float16)
    V_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float16)

    # NumPy reference (in float32 for accuracy)
    Q_f32 = Q_np.astype(np.float32)
    K_f32 = K_np.astype(np.float32)
    V_f32 = V_np.astype(np.float32)

    scores = np.matmul(Q_f32, K_f32.transpose(0, 2, 1)) * scale

    causal_offset = kv_len - q_len
    for i in range(q_len):
        max_attend = causal_offset + i + 1
        if max_attend < kv_len:
            scores[:, i, max_attend:] = -np.inf

    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    ref_output = np.matmul(weights, V_f32).astype(np.float16)

    # GPU computation
    Q_gpu = from_numpy(Q_np)
    K_gpu = from_numpy(K_np)
    V_gpu = from_numpy(V_np)

    result_gpu = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    result_np = result_gpu.to_numpy()

    max_diff = np.abs(result_np.astype(np.float32) - ref_output.astype(np.float32)).max()
    mean_diff = np.abs(result_np.astype(np.float32) - ref_output.astype(np.float32)).mean()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    # FP16 has lower precision
    if max_diff < 5e-2:
        print("  PASS: FP16 Flash Attention matches reference")
        return True
    else:
        print("  FAIL: FP16 Flash Attention differs from reference")
        return False


def test_flash_attention_long_sequence():
    """Test Flash Attention with long sequences."""
    print("\nTesting Flash Attention with long sequences...")

    # Qwen3-8B-like dimensions
    n_heads = 32
    q_len = 1  # Single token decode
    kv_len = 128  # Long KV cache
    head_dim = 128
    scale = 1.0 / np.sqrt(head_dim)

    np.random.seed(789)
    Q_np = np.random.randn(n_heads, q_len, head_dim).astype(np.float16)
    K_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float16)
    V_np = np.random.randn(n_heads, kv_len, head_dim).astype(np.float16)

    # NumPy reference
    Q_f32 = Q_np.astype(np.float32)
    K_f32 = K_np.astype(np.float32)
    V_f32 = V_np.astype(np.float32)

    scores = np.matmul(Q_f32, K_f32.transpose(0, 2, 1)) * scale
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    ref_output = np.matmul(weights, V_f32).astype(np.float16)

    # GPU computation
    Q_gpu = from_numpy(Q_np)
    K_gpu = from_numpy(K_np)
    V_gpu = from_numpy(V_np)

    result_gpu = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    result_np = result_gpu.to_numpy()

    # Check for NaN
    if np.any(np.isnan(result_np)):
        print("  FAIL: Output contains NaN!")
        nan_count = np.sum(np.isnan(result_np))
        print(f"  NaN count: {nan_count} / {result_np.size}")
        return False

    max_diff = np.abs(result_np.astype(np.float32) - ref_output.astype(np.float32)).max()
    mean_diff = np.abs(result_np.astype(np.float32) - ref_output.astype(np.float32)).mean()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 5e-2:
        print("  PASS: Long sequence test matches reference")
        return True
    else:
        print("  FAIL: Long sequence test differs from reference")
        return False


def test_flash_attention_prefill():
    """Test Flash Attention during prefill (q_len = kv_len)."""
    print("\nTesting Flash Attention during prefill...")

    n_heads = 32
    seq_len = 64
    head_dim = 128
    scale = 1.0 / np.sqrt(head_dim)

    np.random.seed(321)
    Q_np = np.random.randn(n_heads, seq_len, head_dim).astype(np.float16)
    K_np = np.random.randn(n_heads, seq_len, head_dim).astype(np.float16)
    V_np = np.random.randn(n_heads, seq_len, head_dim).astype(np.float16)

    # NumPy reference with causal mask
    Q_f32 = Q_np.astype(np.float32)
    K_f32 = K_np.astype(np.float32)
    V_f32 = V_np.astype(np.float32)

    scores = np.matmul(Q_f32, K_f32.transpose(0, 2, 1)) * scale

    # Apply causal mask
    for i in range(seq_len):
        scores[:, i, i + 1 :] = -np.inf

    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    ref_output = np.matmul(weights, V_f32).astype(np.float16)

    # GPU computation
    Q_gpu = from_numpy(Q_np)
    K_gpu = from_numpy(K_np)
    V_gpu = from_numpy(V_np)

    result_gpu = sdpa_causal(Q_gpu, K_gpu, V_gpu, scale)
    result_np = result_gpu.to_numpy()

    # Check for NaN
    if np.any(np.isnan(result_np)):
        print("  FAIL: Output contains NaN!")
        nan_count = np.sum(np.isnan(result_np))
        print(f"  NaN count: {nan_count} / {result_np.size}")
        return False

    max_diff = np.abs(result_np.astype(np.float32) - ref_output.astype(np.float32)).max()
    mean_diff = np.abs(result_np.astype(np.float32) - ref_output.astype(np.float32)).mean()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 5e-2:
        print("  PASS: Prefill test matches reference")
        return True
    else:
        print("  FAIL: Prefill test differs from reference")
        return False


def main():
    print("=" * 60)
    print(" Flash Attention 2 Test Suite")
    print("=" * 60)

    print(f"\nPYGPUKIT_FLASH_ATTENTION = {os.environ.get('PYGPUKIT_FLASH_ATTENTION', 'not set')}")

    passed = 0
    failed = 0

    if test_flash_attention_correctness():
        passed += 1
    else:
        failed += 1

    if test_flash_attention_kv_cache():
        passed += 1
    else:
        failed += 1

    if test_flash_attention_fp16():
        passed += 1
    else:
        failed += 1

    if test_flash_attention_long_sequence():
        passed += 1
    else:
        failed += 1

    if test_flash_attention_prefill():
        passed += 1
    else:
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
