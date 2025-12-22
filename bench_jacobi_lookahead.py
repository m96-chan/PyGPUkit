#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark Jacobi decoding: Original (CPU copies) vs Lookahead (GPU-side).

Compares:
1. Sequential baseline (no Jacobi)
2. Jacobi with CPU KV snapshot/restore
3. Jacobi Lookahead (GPU-side, no CPU copies)
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


def generate_jacobi_original(
    model,
    first_token,
    prefill_len,
    kv_backup,
    num_tokens,
    n_tokens=8,
    max_iter=3,
    init_strategy="repeat",
):
    """Generate tokens using Jacobi decoding (original, with CPU copies)."""
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


def generate_jacobi_lookahead(
    model, first_token, prefill_len, num_tokens, n_tokens=8, max_iter=3, init_strategy="repeat"
):
    """Generate tokens using Jacobi decoding with lookahead KV (GPU-side)."""
    # Set confirmed position after prefill
    model.set_lookahead_confirmed_pos(prefill_len)

    tokens = [first_token]

    total_iterations = 0
    total_converged = 0
    steps = 0

    while len(tokens) < num_tokens:
        remaining = num_tokens - len(tokens)
        current_n = min(n_tokens, remaining)

        if current_n <= 0:
            break

        accepted, stats = model.decode_step_jacobi_lookahead(
            tokens[-1],
            n_tokens=current_n,
            max_iter=max_iter,
            init_strategy=init_strategy,
        )

        total_iterations += stats["iterations"]
        total_converged += 1 if stats["converged"] else 0
        steps += 1

        tokens.extend(accepted)

    avg_iterations = total_iterations / steps if steps > 0 else 0
    convergence_rate = total_converged / steps if steps > 0 else 0

    return tokens[:num_tokens], avg_iterations, convergence_rate


def main():
    print("=" * 70)
    print("JACOBI LOOKAHEAD KV BENCHMARK")
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
    # Test 1: Sequential Baseline
    # =========================================================================
    print(f"\n--- Sequential Baseline ({GEN_TOKENS} tokens) ---")

    start_event.record()
    seq_tokens = generate_sequential_greedy(model, first_token, prefill_len, kv_backup, GEN_TOKENS)
    stop_event.record()
    stop_event.synchronize()

    seq_time = event_elapsed_ms(start_event, stop_event)
    seq_tps = (GEN_TOKENS - 1) * 1000 / seq_time
    seq_text = tokenizer.decode(seq_tokens)

    print(f"Time: {seq_time:.1f} ms, {seq_tps:.2f} tok/s")
    print(f"Text: {seq_text[:80]}...")

    # =========================================================================
    # Test 2: Jacobi Original (CPU copies)
    # =========================================================================
    print("\n--- Jacobi Original (n=8, iter=3, init=repeat) ---")

    start_event.record()
    jacobi_orig_tokens, avg_iter_o, conv_rate_o = generate_jacobi_original(
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

    jacobi_orig_time = event_elapsed_ms(start_event, stop_event)
    jacobi_orig_tps = (GEN_TOKENS - 1) * 1000 / jacobi_orig_time
    match_orig = jacobi_orig_tokens == seq_tokens

    print(f"Time: {jacobi_orig_time:.1f} ms, {jacobi_orig_tps:.2f} tok/s")
    print(f"Avg iterations: {avg_iter_o:.2f}, Convergence: {conv_rate_o:.1%}")
    print(f"Match baseline: {match_orig}")

    # =========================================================================
    # Test 3: Jacobi Lookahead (GPU-side)
    # =========================================================================
    print("\n--- Jacobi Lookahead (n=8, iter=3, init=repeat) ---")

    # Restore KV from backup for fresh start
    model.restore_kv_cache(kv_backup)

    start_event.record()
    jacobi_look_tokens, avg_iter_l, conv_rate_l = generate_jacobi_lookahead(
        model, first_token, prefill_len, GEN_TOKENS, n_tokens=8, max_iter=3, init_strategy="repeat"
    )
    stop_event.record()
    stop_event.synchronize()

    jacobi_look_time = event_elapsed_ms(start_event, stop_event)
    jacobi_look_tps = (GEN_TOKENS - 1) * 1000 / jacobi_look_time
    match_look = jacobi_look_tokens == seq_tokens

    print(f"Time: {jacobi_look_time:.1f} ms, {jacobi_look_tps:.2f} tok/s")
    print(f"Avg iterations: {avg_iter_l:.2f}, Convergence: {conv_rate_l:.1%}")
    print(f"Match baseline: {match_look}")

    # =========================================================================
    # Test 4: Jacobi Lookahead with greedy init
    # =========================================================================
    print("\n--- Jacobi Lookahead (n=8, iter=3, init=greedy) ---")

    # Restore KV from backup for fresh start
    model.restore_kv_cache(kv_backup)

    start_event.record()
    jacobi_greedy_tokens, avg_iter_g, conv_rate_g = generate_jacobi_lookahead(
        model, first_token, prefill_len, GEN_TOKENS, n_tokens=8, max_iter=3, init_strategy="greedy"
    )
    stop_event.record()
    stop_event.synchronize()

    jacobi_greedy_time = event_elapsed_ms(start_event, stop_event)
    jacobi_greedy_tps = (GEN_TOKENS - 1) * 1000 / jacobi_greedy_time
    match_greedy = jacobi_greedy_tokens == seq_tokens

    print(f"Time: {jacobi_greedy_time:.1f} ms, {jacobi_greedy_tps:.2f} tok/s")
    print(f"Avg iterations: {avg_iter_g:.2f}, Convergence: {conv_rate_g:.1%}")
    print(f"Match baseline: {match_greedy}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    speedup_orig = seq_time / jacobi_orig_time if jacobi_orig_time > 0 else 0
    speedup_look = seq_time / jacobi_look_time if jacobi_look_time > 0 else 0
    speedup_look_vs_orig = jacobi_orig_time / jacobi_look_time if jacobi_look_time > 0 else 0

    print(f"\n{'Method':<35} {'Time (ms)':<12} {'tok/s':<10} {'Speedup':<10} {'Match'}")
    print("-" * 77)
    print(f"{'Sequential (baseline)':<35} {seq_time:<12.1f} {seq_tps:<10.2f} {'1.00x':<10} {'N/A'}")
    print(
        f"{'Jacobi Original (CPU copies)':<35} {jacobi_orig_time:<12.1f} {jacobi_orig_tps:<10.2f} {speedup_orig:.2f}x{'':<5} {'YES' if match_orig else 'NO'}"
    )
    print(
        f"{'Jacobi Lookahead (GPU-side)':<35} {jacobi_look_time:<12.1f} {jacobi_look_tps:<10.2f} {speedup_look:.2f}x{'':<5} {'YES' if match_look else 'NO'}"
    )
    print(
        f"{'Jacobi Lookahead (greedy init)':<35} {jacobi_greedy_time:<12.1f} {jacobi_greedy_tps:<10.2f} {(seq_time / jacobi_greedy_time):.2f}x{'':<5} {'YES' if match_greedy else 'NO'}"
    )

    print(f"\nLookahead vs Original speedup: {speedup_look_vs_orig:.2f}x")

    # Correctness check
    all_pass = match_orig and match_look and match_greedy
    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL CORRECTNESS TESTS PASSED!")
    else:
        print("RESULT: SOME TESTS FAILED!")
        if not match_orig:
            print(f"  Jacobi Original mismatch: {jacobi_orig_tokens[:10]}...")
        if not match_look:
            print(f"  Jacobi Lookahead mismatch: {jacobi_look_tokens[:10]}...")
        if not match_greedy:
            print(f"  Jacobi Greedy mismatch: {jacobi_greedy_tokens[:10]}...")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
