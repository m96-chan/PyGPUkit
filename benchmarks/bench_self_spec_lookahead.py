#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark Self-Speculative: Original (CPU copies) vs Lookahead (GPU-side).

Compares:
1. Sequential baseline
2. Self-Speculative with CPU KV snapshot/restore
3. Self-Speculative Lookahead (GPU-side, no CPU copies)
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


def generate_self_spec_original(
    model, first_token, prefill_len, kv_backup, num_tokens, max_draft_tokens=4, draft_layers=8
):
    """Generate using self-speculative decoding (original, with CPU copies)."""
    model.restore_kv_cache(kv_backup)

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    total_draft = 0
    total_accepted = 0

    while len(tokens) < num_tokens:
        remaining = num_tokens - len(tokens)
        current_draft = min(max_draft_tokens, remaining)

        if current_draft <= 0:
            break

        accepted, new_pos, stats = model.decode_step_self_speculative(
            tokens[-1],
            position,
            context_len,
            max_draft_tokens=current_draft,
            draft_layers=draft_layers,
        )

        total_draft += stats["draft_count"]
        total_accepted += stats["accepted_count"]

        tokens.extend(accepted)
        position = new_pos
        context_len = new_pos + 1

    acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
    return tokens[:num_tokens], acceptance_rate


def generate_self_spec_lookahead(
    model, first_token, prefill_len, num_tokens, max_draft_tokens=4, draft_layers=8
):
    """Generate using self-speculative decoding with lookahead KV (GPU-side)."""
    # Set confirmed position after prefill
    model.set_lookahead_confirmed_pos(prefill_len)

    tokens = [first_token]

    total_draft = 0
    total_accepted = 0

    while len(tokens) < num_tokens:
        remaining = num_tokens - len(tokens)
        current_draft = min(max_draft_tokens, remaining)

        if current_draft <= 0:
            break

        accepted, stats = model.decode_step_self_speculative_lookahead(
            tokens[-1],
            max_draft_tokens=current_draft,
            draft_layers=draft_layers,
        )

        total_draft += stats["draft_count"]
        total_accepted += stats["accepted_count"]

        tokens.extend(accepted)

    acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
    return tokens[:num_tokens], acceptance_rate


def main():
    print("=" * 70)
    print("SELF-SPECULATIVE LOOKAHEAD KV BENCHMARK")
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
    num_layers = len(model.blocks)

    print(f"Model: {num_layers} layers")

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

    # Test different draft layer counts
    draft_layer_configs = [32, 34, 35, 36]  # High layer counts for better acceptance

    results = []

    # =========================================================================
    # Sequential Baseline
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

    for draft_layers in draft_layer_configs:
        # =====================================================================
        # Self-Speculative Original (CPU copies)
        # =====================================================================
        print(f"\n--- Self-Spec Original (draft_layers={draft_layers}) ---")

        start_event.record()
        orig_tokens, orig_accept = generate_self_spec_original(
            model,
            first_token,
            prefill_len,
            kv_backup,
            GEN_TOKENS,
            max_draft_tokens=4,
            draft_layers=draft_layers,
        )
        stop_event.record()
        stop_event.synchronize()

        orig_time = event_elapsed_ms(start_event, stop_event)
        orig_tps = (GEN_TOKENS - 1) * 1000 / orig_time
        match_orig = orig_tokens == seq_tokens

        print(f"Time: {orig_time:.1f} ms, {orig_tps:.2f} tok/s")
        print(f"Acceptance: {orig_accept:.1%}, Match: {match_orig}")

        # =====================================================================
        # Self-Speculative Lookahead (GPU-side)
        # =====================================================================
        print(f"--- Self-Spec Lookahead (draft_layers={draft_layers}) ---")

        # Restore KV from backup
        model.restore_kv_cache(kv_backup)

        start_event.record()
        look_tokens, look_accept = generate_self_spec_lookahead(
            model,
            first_token,
            prefill_len,
            GEN_TOKENS,
            max_draft_tokens=4,
            draft_layers=draft_layers,
        )
        stop_event.record()
        stop_event.synchronize()

        look_time = event_elapsed_ms(start_event, stop_event)
        look_tps = (GEN_TOKENS - 1) * 1000 / look_time
        match_look = look_tokens == seq_tokens

        print(f"Time: {look_time:.1f} ms, {look_tps:.2f} tok/s")
        print(f"Acceptance: {look_accept:.1%}, Match: {match_look}")

        speedup = orig_time / look_time if look_time > 0 else 0

        results.append(
            {
                "layers": draft_layers,
                "orig_time": orig_time,
                "look_time": look_time,
                "orig_accept": orig_accept,
                "look_accept": look_accept,
                "match_orig": match_orig,
                "match_look": match_look,
                "speedup": speedup,
            }
        )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Draft Layers':<15} {'Original (ms)':<15} {'Lookahead (ms)':<15} {'Speedup':<10} {'Match'}"
    )
    print("-" * 65)
    print(f"{'Sequential':<15} {seq_time:<15.1f} {'-':<15} {'-':<10} {'N/A'}")

    all_pass = True
    for r in results:
        match_str = "YES" if (r["match_orig"] and r["match_look"]) else "NO"
        if not (r["match_orig"] and r["match_look"]):
            all_pass = False
        print(
            f"{r['layers']:<15} {r['orig_time']:<15.1f} {r['look_time']:<15.1f} {r['speedup']:.2f}x{'':<5} {match_str}"
        )

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL CORRECTNESS TESTS PASSED!")
    else:
        print("RESULT: SOME TESTS FAILED!")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
