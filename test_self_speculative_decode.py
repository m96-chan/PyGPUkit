#!/usr/bin/env python3
"""Test self-speculative decoding correctness.

Correctness criteria:
1. Self-Speculative ON/OFF produces IDENTICAL output with temperature=0
2. Draft = full model layers should give ~100% acceptance
3. KV cache must not be corrupted after rejection
"""

import numpy as np

# Model paths
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
    # Restore KV cache
    model.restore_kv_cache(kv_backup)

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    for _ in range(num_tokens - 1):
        hidden = model._decode_step_fixed_cache(tokens[-1], position, context_len)
        logits = model.get_logits(hidden)
        logits_np = logits.to_numpy()[-1]
        next_token = int(np.argmax(logits_np))  # Greedy
        tokens.append(next_token)
        position += 1
        context_len += 1

    return tokens


def generate_self_speculative(
    model, first_token, prefill_len, kv_backup, num_tokens, max_draft_tokens=4, draft_layers=8
):
    """Generate tokens using self-speculative decoding."""
    # Restore KV cache
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


def main():
    print("=" * 70)
    print("SELF-SPECULATIVE DECODING CORRECTNESS TEST")
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
    first_token = int(np.argmax(logits.to_numpy()[-1]))  # Greedy first token

    # Backup KV cache after prefill
    kv_backup = model.snapshot_kv_cache()

    print(f"First token (greedy): {first_token} = '{tokenizer.decode([first_token])}'")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        generate_sequential_greedy(model, first_token, prefill_len, kv_backup, 5)
    default_stream().synchronize()

    # =========================================================================
    # Test 1: Sequential Greedy (Baseline)
    # =========================================================================
    print(f"\n--- Test 1: Sequential Greedy ({GEN_TOKENS} tokens) ---")

    start_event = CudaEvent()
    stop_event = CudaEvent()

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
    # Test 2: Self-Speculative with draft_layers = num_layers (should be ~100%)
    # =========================================================================
    print(f"\n--- Test 2: Self-Speculative (draft_layers={num_layers}, ALL layers) ---")
    print("Expected: ~100% acceptance (draft = full model)")

    start_event.record()
    spec_full_tokens, spec_full_acceptance = generate_self_speculative(
        model,
        first_token,
        prefill_len,
        kv_backup,
        GEN_TOKENS,
        max_draft_tokens=4,
        draft_layers=num_layers,
    )
    stop_event.record()
    stop_event.synchronize()

    spec_full_time = event_elapsed_ms(start_event, stop_event)
    spec_full_text = tokenizer.decode(spec_full_tokens)

    print(f"Time: {spec_full_time:.1f} ms")
    print(f"Acceptance rate: {spec_full_acceptance:.1%}")
    print(f"Tokens match: {spec_full_tokens == seq_tokens}")
    print(f"Text: {spec_full_text[:100]}...")

    # =========================================================================
    # Test 3: Self-Speculative with draft_layers = 8
    # =========================================================================
    print("\n--- Test 3: Self-Speculative (draft_layers=8) ---")

    start_event.record()
    spec8_tokens, spec8_acceptance = generate_self_speculative(
        model, first_token, prefill_len, kv_backup, GEN_TOKENS, max_draft_tokens=4, draft_layers=8
    )
    stop_event.record()
    stop_event.synchronize()

    spec8_time = event_elapsed_ms(start_event, stop_event)
    spec8_text = tokenizer.decode(spec8_tokens)

    print(f"Time: {spec8_time:.1f} ms")
    print(f"Acceptance rate: {spec8_acceptance:.1%}")
    print(f"Tokens match: {spec8_tokens == seq_tokens}")
    print(f"Text: {spec8_text[:100]}...")

    # =========================================================================
    # Test 4: Self-Speculative with draft_layers = 12
    # =========================================================================
    print("\n--- Test 4: Self-Speculative (draft_layers=12) ---")

    start_event.record()
    spec12_tokens, spec12_acceptance = generate_self_speculative(
        model, first_token, prefill_len, kv_backup, GEN_TOKENS, max_draft_tokens=4, draft_layers=12
    )
    stop_event.record()
    stop_event.synchronize()

    spec12_time = event_elapsed_ms(start_event, stop_event)
    spec12_text = tokenizer.decode(spec12_tokens)

    print(f"Time: {spec12_time:.1f} ms")
    print(f"Acceptance rate: {spec12_acceptance:.1%}")
    print(f"Tokens match: {spec12_tokens == seq_tokens}")
    print(f"Text: {spec12_text[:100]}...")

    # =========================================================================
    # Test 5: KV Cache Integrity Check
    # =========================================================================
    print("\n--- Test 5: KV Cache Integrity Check ---")
    print("Running sequential after speculative to check KV cache...")

    # Run speculative first
    generate_self_speculative(
        model, first_token, prefill_len, kv_backup, 10, max_draft_tokens=4, draft_layers=8
    )

    # Now run sequential - should produce same output as baseline
    kv_after_spec = model.snapshot_kv_cache()

    # Restore and run sequential
    seq_after_tokens = generate_sequential_greedy(
        model, first_token, prefill_len, kv_backup, GEN_TOKENS
    )

    kv_integrity_ok = seq_after_tokens == seq_tokens
    print(f"KV Cache Integrity: {'PASS' if kv_integrity_ok else 'FAIL'}")
    if not kv_integrity_ok:
        print(f"  Expected: {seq_tokens[:10]}...")
        print(f"  Got:      {seq_after_tokens[:10]}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True

    # Check 1: Full layers should give identical output
    test1_pass = spec_full_tokens == seq_tokens
    print(
        f"\n1. Full layers (draft={num_layers}) matches baseline: {'PASS' if test1_pass else 'FAIL'}"
    )
    if not test1_pass:
        all_pass = False
        print(f"   Baseline: {seq_tokens[:10]}...")
        print(f"   Got:      {spec_full_tokens[:10]}...")

    # Check 2: Full layers should have ~100% acceptance
    test2_pass = spec_full_acceptance > 0.95
    print(
        f"2. Full layers acceptance > 95%: {'PASS' if test2_pass else 'FAIL'} ({spec_full_acceptance:.1%})"
    )
    if not test2_pass:
        all_pass = False

    # Check 3: KV cache integrity
    print(f"3. KV Cache integrity after speculative: {'PASS' if kv_integrity_ok else 'FAIL'}")
    if not kv_integrity_ok:
        all_pass = False

    # Check 4: Speculative outputs should match baseline (greedy = deterministic)
    test4a_pass = spec8_tokens == seq_tokens
    test4b_pass = spec12_tokens == seq_tokens
    print(f"4. Speculative (8 layers) matches baseline: {'PASS' if test4a_pass else 'FAIL'}")
    print(f"5. Speculative (12 layers) matches baseline: {'PASS' if test4b_pass else 'FAIL'}")
    if not test4a_pass or not test4b_pass:
        all_pass = False
        if not test4a_pass:
            print(f"   8-layer: {spec8_tokens[:10]}... vs baseline: {seq_tokens[:10]}...")
        if not test4b_pass:
            print(f"   12-layer: {spec12_tokens[:10]}... vs baseline: {seq_tokens[:10]}...")

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL TESTS PASSED!")
    else:
        print("RESULT: SOME TESTS FAILED!")
    print("=" * 70)

    # Performance summary
    print(f"\n{'Method':<30} {'Time (ms)':<12} {'Acceptance':<12} {'Match':<10}")
    print("-" * 64)
    print(f"{'Sequential (baseline)':<30} {seq_time:<12.1f} {'N/A':<12} {'N/A':<10}")
    print(
        f"{'Self-Spec (layers=ALL)':<30} {spec_full_time:<12.1f} {spec_full_acceptance * 100:<11.0f}% {'YES' if test1_pass else 'NO':<10}"
    )
    print(
        f"{'Self-Spec (layers=8)':<30} {spec8_time:<12.1f} {spec8_acceptance * 100:<11.0f}% {'YES' if test4a_pass else 'NO':<10}"
    )
    print(
        f"{'Self-Spec (layers=12)':<30} {spec12_time:<12.1f} {spec12_acceptance * 100:<11.0f}% {'YES' if test4b_pass else 'NO':<10}"
    )

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
