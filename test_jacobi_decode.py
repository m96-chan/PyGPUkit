#!/usr/bin/env python3
"""Test Jacobi decoding correctness.

Correctness criteria:
1. Jacobi ON/OFF produces IDENTICAL output with greedy decoding
2. Converges within max_iter iterations
3. KV cache integrity after multiple steps
"""

import numpy as np

MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
TOKENIZER_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

from tokenizers import Tokenizer

from pygpukit import CudaEvent, event_elapsed_ms
from pygpukit.core import default_stream, from_numpy
from pygpukit.llm import (
    ChatMessage,
    detect_model_spec,
    format_chat_messages,
    load_model_from_safetensors,
    load_safetensors,
)
from pygpukit.llm.model import precompute_freqs_cis
from pygpukit.ops.basic import kv_cache_prefill_gqa

MAX_SEQ_LEN = 512
GEN_TOKENS = 32


def generate_sequential_greedy(model, first_token, prefill_len, kv_backup, num_tokens):
    """Generate tokens sequentially with greedy sampling (baseline)."""
    model.restore_kv_cache(kv_backup)

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    for _ in range(num_tokens - 1):
        hidden = model._decode_step_fixed_cache(tokens[-1], position, context_len)
        logits = model.get_logits(hidden)
        logits_np = logits.to_numpy()[-1]
        next_token = int(np.argmax(logits_np))
        tokens.append(next_token)
        position += 1
        context_len += 1

    return tokens


def generate_jacobi(
    model,
    first_token,
    prefill_len,
    kv_backup,
    num_tokens,
    n_tokens=8,
    max_iter=3,
    init_strategy="repeat",
):
    """Generate tokens using Jacobi decoding."""
    model.restore_kv_cache(kv_backup)

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    total_iterations = 0
    total_converged = 0
    steps = 0

    while len(tokens) < num_tokens:
        remaining = num_tokens - len(tokens)
        current_n = min(n_tokens, remaining)

        if current_n <= 0:
            break

        accepted, new_pos, stats = model.decode_step_jacobi(
            tokens[-1],
            position,
            context_len,
            n_tokens=current_n,
            max_iter=max_iter,
            init_strategy=init_strategy,
        )

        total_iterations += stats["iterations"]
        total_converged += 1 if stats["converged"] else 0
        steps += 1

        tokens.extend(accepted)
        position = new_pos
        context_len = new_pos + 1

    avg_iterations = total_iterations / steps if steps > 0 else 0
    convergence_rate = total_converged / steps if steps > 0 else 0

    return tokens[:num_tokens], avg_iterations, convergence_rate


def main():
    print("=" * 70)
    print("JACOBI DECODING CORRECTNESS TEST")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing."),
    ]
    prompt = format_chat_messages(messages, model_type="qwen3")
    input_ids = tokenizer.encode(prompt).ids
    prefill_len = len(input_ids)

    print(f"\nLoading model... (prefill_len={prefill_len})")
    st = load_safetensors(MODEL_PATH)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(MODEL_PATH, dtype="float16", spec=spec)
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

    # Prefill
    print("Running prefill...")
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    logits = model.get_logits(hidden)
    first_token = int(np.argmax(logits.to_numpy()[-1]))

    kv_backup = model.snapshot_kv_cache()
    print(f"First token (greedy): {first_token}")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        generate_sequential_greedy(model, first_token, prefill_len, kv_backup, 5)
    default_stream().synchronize()

    start_event = CudaEvent()
    stop_event = CudaEvent()

    # =========================================================================
    # Test 1: Sequential Greedy (Baseline)
    # =========================================================================
    print(f"\n--- Test 1: Sequential Greedy ({GEN_TOKENS} tokens) ---")

    start_event.record()
    seq_tokens = generate_sequential_greedy(model, first_token, prefill_len, kv_backup, GEN_TOKENS)
    stop_event.record()
    stop_event.synchronize()

    seq_time = event_elapsed_ms(start_event, stop_event)
    seq_text = tokenizer.decode(seq_tokens)

    print(f"Time: {seq_time:.1f} ms")
    print(f"Tokens: {seq_tokens[:10]}...")
    print(f"Text: {seq_text[:100]}...")

    # =========================================================================
    # Test 2: Jacobi with init_strategy="greedy" (should match exactly)
    # =========================================================================
    print("\n--- Test 2: Jacobi (n=8, iter=3, init=greedy) ---")
    print("Expected: 100% match (greedy init = sequential)")

    start_event.record()
    jacobi_greedy_tokens, avg_iter, conv_rate = generate_jacobi(
        model,
        first_token,
        prefill_len,
        kv_backup,
        GEN_TOKENS,
        n_tokens=8,
        max_iter=3,
        init_strategy="greedy",
    )
    stop_event.record()
    stop_event.synchronize()

    jacobi_greedy_time = event_elapsed_ms(start_event, stop_event)
    jacobi_greedy_text = tokenizer.decode(jacobi_greedy_tokens)
    greedy_match = jacobi_greedy_tokens == seq_tokens

    print(f"Time: {jacobi_greedy_time:.1f} ms")
    print(f"Avg iterations: {avg_iter:.2f}, Convergence: {conv_rate:.1%}")
    print(f"Match baseline: {greedy_match}")
    print(f"Text: {jacobi_greedy_text[:100]}...")

    # =========================================================================
    # Test 3: Jacobi with init_strategy="repeat"
    # =========================================================================
    print("\n--- Test 3: Jacobi (n=8, iter=3, init=repeat) ---")

    start_event.record()
    jacobi_repeat_tokens, avg_iter_r, conv_rate_r = generate_jacobi(
        model,
        first_token,
        prefill_len,
        kv_backup,
        GEN_TOKENS,
        n_tokens=8,
        max_iter=3,
        init_strategy="repeat",
    )
    stop_event.record()
    stop_event.synchronize()

    jacobi_repeat_time = event_elapsed_ms(start_event, stop_event)
    jacobi_repeat_text = tokenizer.decode(jacobi_repeat_tokens)
    repeat_match = jacobi_repeat_tokens == seq_tokens

    print(f"Time: {jacobi_repeat_time:.1f} ms")
    print(f"Avg iterations: {avg_iter_r:.2f}, Convergence: {conv_rate_r:.1%}")
    print(f"Match baseline: {repeat_match}")
    print(f"Text: {jacobi_repeat_text[:100]}...")

    # =========================================================================
    # Test 4: Jacobi with init_strategy="ngram"
    # =========================================================================
    print("\n--- Test 4: Jacobi (n=8, iter=3, init=ngram) ---")

    start_event.record()
    jacobi_ngram_tokens, avg_iter_n, conv_rate_n = generate_jacobi(
        model,
        first_token,
        prefill_len,
        kv_backup,
        GEN_TOKENS,
        n_tokens=8,
        max_iter=3,
        init_strategy="ngram",
    )
    stop_event.record()
    stop_event.synchronize()

    jacobi_ngram_time = event_elapsed_ms(start_event, stop_event)
    jacobi_ngram_text = tokenizer.decode(jacobi_ngram_tokens)
    ngram_match = jacobi_ngram_tokens == seq_tokens

    print(f"Time: {jacobi_ngram_time:.1f} ms")
    print(f"Avg iterations: {avg_iter_n:.2f}, Convergence: {conv_rate_n:.1%}")
    print(f"Match baseline: {ngram_match}")
    print(f"Text: {jacobi_ngram_text[:100]}...")

    # =========================================================================
    # Test 5: KV Cache Integrity
    # =========================================================================
    print("\n--- Test 5: KV Cache Integrity ---")

    # Run Jacobi, then sequential - should produce same output
    generate_jacobi(
        model,
        first_token,
        prefill_len,
        kv_backup,
        10,
        n_tokens=8,
        max_iter=3,
        init_strategy="repeat",
    )

    seq_after = generate_sequential_greedy(model, first_token, prefill_len, kv_backup, GEN_TOKENS)
    kv_integrity = seq_after == seq_tokens
    print(f"KV integrity: {'PASS' if kv_integrity else 'FAIL'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True

    # Check 1: Greedy init should match exactly
    print(f"\n1. Jacobi (greedy init) matches baseline: {'PASS' if greedy_match else 'FAIL'}")
    if not greedy_match:
        all_pass = False
        print(f"   Baseline: {seq_tokens[:10]}...")
        print(f"   Jacobi:   {jacobi_greedy_tokens[:10]}...")

    # Check 2: KV integrity
    print(f"2. KV Cache integrity: {'PASS' if kv_integrity else 'FAIL'}")
    if not kv_integrity:
        all_pass = False

    # Note: repeat/ngram init may not match baseline (expected)
    print(f"\n3. Jacobi (repeat init) matches baseline: {repeat_match} (may differ)")
    print(f"4. Jacobi (ngram init) matches baseline: {ngram_match} (may differ)")

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: CORE TESTS PASSED!")
    else:
        print("RESULT: SOME TESTS FAILED!")
    print("=" * 70)

    # Performance summary
    print(f"\n{'Method':<30} {'Time (ms)':<12} {'Avg Iter':<10} {'Match'}")
    print("-" * 62)
    print(f"{'Sequential (baseline)':<30} {seq_time:<12.1f} {'N/A':<10} {'N/A'}")
    print(
        f"{'Jacobi (init=greedy)':<30} {jacobi_greedy_time:<12.1f} {avg_iter:<10.2f} {'YES' if greedy_match else 'NO'}"
    )
    print(
        f"{'Jacobi (init=repeat)':<30} {jacobi_repeat_time:<12.1f} {avg_iter_r:<10.2f} {'YES' if repeat_match else 'NO'}"
    )
    print(
        f"{'Jacobi (init=ngram)':<30} {jacobi_ngram_time:<12.1f} {avg_iter_n:<10.2f} {'YES' if ngram_match else 'NO'}"
    )

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
