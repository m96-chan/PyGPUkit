#!/usr/bin/env python3
"""Test batch decode zero-allocation path."""

import numpy as np

MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"

from pygpukit.core import default_stream
from pygpukit.llm import detect_model_spec, load_model_from_safetensors, load_safetensors
from pygpukit.llm.model import DecodeBuffers
from pygpukit.ops.basic import kv_cache_prefill_gqa

MAX_SEQ_LEN = 64
MAX_BATCH_SIZE = 8


def main():
    print("=" * 70)
    print("TEST: Batch Decode Zero-Allocation Path")
    print("=" * 70)

    # Load model
    st = load_safetensors(MODEL_PATH)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(MODEL_PATH, dtype="float16", spec=spec)
    dtype = "float16"
    lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
    vocab_size = lm_head.shape[0]

    print("\nModel: Qwen3-8B")
    print(f"  Layers: {model.config.num_layers}")

    # Initialize KV cache
    print("\nInitializing KV cache...")
    for block in model.blocks:
        block.attn.init_fixed_cache(MAX_SEQ_LEN, dtype=dtype)

    # Prefill with some tokens
    input_ids = list(range(100, 110))  # 10 tokens
    print(f"Prefill with {len(input_ids)} tokens...")
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)
    default_stream().synchronize()

    # Backup KV cache
    kv_backup = model.snapshot_kv_cache()

    # Allocate batch decode buffers
    print(f"\nAllocating batch buffers (max_batch_size={MAX_BATCH_SIZE})...")
    use_qk_norm = spec is not None and spec.use_qk_norm
    batch_buffers = DecodeBuffers.allocate(
        model.config,
        dtype=dtype,
        use_qk_norm=use_qk_norm,
        vocab_size=vocab_size,
        max_batch_size=MAX_BATCH_SIZE,
    )
    print(f"  max_batch_size: {batch_buffers.max_batch_size}")
    print(
        f"  hidden_batch shape: {batch_buffers.hidden_batch.shape if batch_buffers.hidden_batch else None}"
    )

    # Test with different batch sizes
    test_batch_sizes = [2, 4, 8]
    test_tokens = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012]

    for batch_size in test_batch_sizes:
        print(f"\n--- Testing batch_size={batch_size} ---")

        # Restore KV cache
        model.restore_kv_cache(kv_backup)
        default_stream().synchronize()

        tokens = test_tokens[:batch_size]
        start_pos = len(input_ids)
        ctx_len = start_pos + batch_size

        # Baseline: existing batch path (with allocations)
        hidden_baseline = model._decode_step_fixed_cache_batch(tokens, start_pos, ctx_len)
        hidden_baseline_np = hidden_baseline.to_numpy()

        # Restore KV cache again
        model.restore_kv_cache(kv_backup)
        default_stream().synchronize()

        # Test: zero-alloc path
        hidden_zero_alloc = model._decode_step_fixed_cache_batch_zero_alloc(
            tokens, start_pos, ctx_len, batch_buffers
        )
        hidden_zero_alloc_np = hidden_zero_alloc.to_numpy()

        # Compare
        max_diff = np.max(np.abs(hidden_baseline_np - hidden_zero_alloc_np))
        rel_diff = max_diff / (np.max(np.abs(hidden_baseline_np)) + 1e-10)
        match = np.allclose(hidden_baseline_np, hidden_zero_alloc_np, rtol=1e-3, atol=1e-4)

        print(f"  Baseline shape: {hidden_baseline_np.shape}")
        print(f"  Zero-alloc shape: {hidden_zero_alloc_np.shape}")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Rel diff: {rel_diff:.6e}")
        print(f"  Match: {'PASS' if match else 'FAIL'}")

        if not match:
            print(f"  Baseline[:, :5]: {hidden_baseline_np[0, :5]}")
            print(f"  Zero-alloc[:, :5]: {hidden_zero_alloc_np[0, :5]}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
