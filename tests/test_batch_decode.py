#!/usr/bin/env python3
"""Test batch decode correctness by comparing with sequential single-token decode."""

import numpy as np

model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

from tokenizers import Tokenizer

from pygpukit.core import from_numpy
from pygpukit.llm import (
    ChatMessage,
    detect_model_spec,
    format_chat_messages,
    load_model_from_safetensors,
    load_safetensors,
)
from pygpukit.llm.model import precompute_freqs_cis, sample_token
from pygpukit.ops.basic import kv_cache_prefill_gqa

MAX_SEQ_LEN = 512
BATCH_SIZE = 4  # Number of tokens to decode at once


def main():
    print("=" * 70)
    print("BATCH DECODE CORRECTNESS TEST")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
    ]
    prompt = format_chat_messages(messages, model_type="qwen3")
    input_ids = tokenizer.encode(prompt).ids
    prefill_len = len(input_ids)

    print(f"\nLoading model... (prefill_len={prefill_len})")
    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)
    dtype = str(model.embed_tokens.dtype)

    print("Initializing KV cache...")
    for block in model.blocks:
        block.attn.init_fixed_cache(MAX_SEQ_LEN, dtype=dtype)

    if model.config.use_rope:
        cos_np, sin_np = precompute_freqs_cis(
            model.config.head_dim, MAX_SEQ_LEN, model.config.rope_theta
        )
        np_dtype = np.float16 if dtype == "float16" else np.float32
        model._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
        model._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))

    # =========================================================================
    # Run prefill and get first token
    # =========================================================================
    print("\nRunning prefill...")
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    logits = model.get_logits(hidden)
    last_logits = logits.to_numpy()[-1]
    first_token = sample_token(last_logits, 0.7, 50, 0.9)

    # =========================================================================
    # Test 1: Sequential single-token decode (baseline)
    # =========================================================================
    print(f"\n--- Test 1: Sequential single-token decode ({BATCH_SIZE} tokens) ---")

    # Store KV cache state after prefill for later reset
    kv_cache_backup = []
    for block in model.blocks:
        k_backup = block.attn._k_cache.to_numpy().copy()
        v_backup = block.attn._v_cache.to_numpy().copy()
        kv_cache_backup.append((k_backup, v_backup))

    # Generate tokens sequentially
    sequential_hiddens = []
    sequential_tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    for i in range(BATCH_SIZE):
        token_id = sequential_tokens[-1] if i == 0 else sequential_tokens[-1]
        hidden = model._decode_step_fixed_cache(token_id, position, context_len)
        sequential_hiddens.append(hidden.to_numpy().copy())

        logits = model.get_logits(hidden)
        next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
        sequential_tokens.append(next_token)

        position += 1
        context_len += 1

    print(f"Sequential tokens: {sequential_tokens[: BATCH_SIZE + 1]}")
    print(f"Sequential hidden shapes: {[h.shape for h in sequential_hiddens]}")

    # =========================================================================
    # Test 2: Batch decode
    # =========================================================================
    print(f"\n--- Test 2: Batch decode ({BATCH_SIZE} tokens at once) ---")

    # Restore KV cache to post-prefill state
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = kv_cache_backup[i]
        # Restore by copying back (need to re-upload to GPU)
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

    # Use the same token IDs from sequential decode
    batch_tokens = sequential_tokens[:BATCH_SIZE]
    start_position = prefill_len
    context_len_batch = prefill_len + BATCH_SIZE

    print(f"Batch tokens: {batch_tokens}")
    print(f"Start position: {start_position}, Context len: {context_len_batch}")

    batch_hidden = model._decode_step_fixed_cache_batch(
        batch_tokens, start_position, context_len_batch
    )
    batch_hidden_np = batch_hidden.to_numpy()
    print(f"Batch hidden shape: {batch_hidden_np.shape}")

    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n--- Comparison ---")

    all_pass = True
    for i in range(BATCH_SIZE):
        seq_h = sequential_hiddens[i]
        batch_h = batch_hidden_np[i : i + 1]  # [1, hidden_size]

        # Compare
        diff = np.abs(seq_h - batch_h)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = np.max(diff / (np.abs(seq_h) + 1e-8))

        status = "PASS" if max_diff < 0.1 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  Token {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_error={rel_error:.6f} [{status}]"
        )

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL TESTS PASSED!")
    else:
        print("RESULT: SOME TESTS FAILED!")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
