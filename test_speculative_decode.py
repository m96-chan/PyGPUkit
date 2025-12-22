#!/usr/bin/env python3
"""Test speculative decoding with Qwen3-0.6B (draft) and Qwen3-8B (target)."""

import numpy as np

# Model paths
DRAFT_MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
TARGET_MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
TOKENIZER_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"

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
from pygpukit.llm.model import precompute_freqs_cis, sample_token
from pygpukit.ops.basic import kv_cache_prefill_gqa

MAX_SEQ_LEN = 512
DRAFT_TOKENS = 4  # Number of draft tokens to generate per step
GEN_TOKENS = 32  # Total tokens to generate


def load_draft_model():
    """Load the smaller draft model (Qwen3-0.6B)."""
    print("Loading draft model (Qwen3-0.6B)...")
    st = load_safetensors(DRAFT_MODEL_PATH)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(DRAFT_MODEL_PATH, dtype="float16", spec=spec)
    return model


def load_target_model():
    """Load the larger target model (Qwen3-8B)."""
    print("Loading target model (Qwen3-8B)...")
    st = load_safetensors(TARGET_MODEL_PATH)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(TARGET_MODEL_PATH, dtype="float16", spec=spec)
    return model


def init_model_cache(model, max_seq_len):
    """Initialize KV cache and RoPE for a model."""
    dtype = str(model.embed_tokens.dtype)
    for block in model.blocks:
        block.attn.init_fixed_cache(max_seq_len, dtype=dtype)

    if model.config.use_rope:
        cos_np, sin_np = precompute_freqs_cis(
            model.config.head_dim, max_seq_len, model.config.rope_theta
        )
        np_dtype = np.float16 if dtype == "float16" else np.float32
        model._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
        model._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))


def run_prefill(model, input_ids):
    """Run prefill and return first token."""
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    logits = model.get_logits(hidden)
    last_logits = logits.to_numpy()[-1]
    first_token = sample_token(last_logits, 0.7, 50, 0.9)
    return first_token


def backup_kv_cache(model):
    """Backup KV cache state."""
    backup = []
    for block in model.blocks:
        k_backup = block.attn._k_cache.to_numpy().copy()
        v_backup = block.attn._v_cache.to_numpy().copy()
        backup.append((k_backup, v_backup))
    return backup


def restore_kv_cache(model, backup):
    """Restore KV cache from backup."""
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = backup[i]
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))


def generate_sequential(model, first_token, prefill_len, kv_backup, num_tokens):
    """Generate tokens sequentially (baseline)."""
    restore_kv_cache(model, kv_backup)

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    for _ in range(num_tokens - 1):
        hidden = model._decode_step_fixed_cache(tokens[-1], position, context_len)
        logits = model.get_logits(hidden)
        next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
        tokens.append(next_token)
        position += 1
        context_len += 1

    return tokens


def generate_speculative(
    draft_model,
    target_model,
    first_token,
    prefill_len,
    draft_kv_backup,
    target_kv_backup,
    num_tokens,
    num_draft_tokens=4,
):
    """Generate tokens using speculative decoding.

    Algorithm:
    1. Draft model generates K tokens sequentially
    2. Target model verifies all K tokens in one batch forward pass
    3. Accept tokens until first disagreement, then sample from target
    4. Update both KV caches with accepted tokens only
    """
    # Restore initial KV cache state
    restore_kv_cache(draft_model, draft_kv_backup)
    restore_kv_cache(target_model, target_kv_backup)

    tokens = [first_token]
    draft_pos = prefill_len
    target_pos = prefill_len
    draft_ctx = prefill_len + 1
    target_ctx = prefill_len + 1

    total_draft = 0
    total_accepted = 0

    while len(tokens) < num_tokens:
        remaining = num_tokens - len(tokens)
        current_draft = min(num_draft_tokens, remaining)

        if current_draft <= 0:
            break

        # Save KV cache state before speculation
        draft_kv_before = backup_kv_cache(draft_model)
        target_kv_before = backup_kv_cache(target_model)

        # === Step 1: Draft model generates K tokens sequentially ===
        draft_tokens = []
        draft_pos_temp = draft_pos
        draft_ctx_temp = draft_ctx
        current_token = tokens[-1]

        for _ in range(current_draft):
            hidden = draft_model._decode_step_fixed_cache(
                current_token, draft_pos_temp, draft_ctx_temp
            )
            logits = draft_model.get_logits(hidden)
            next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
            draft_tokens.append(next_token)
            current_token = next_token
            draft_pos_temp += 1
            draft_ctx_temp += 1

        total_draft += len(draft_tokens)

        # === Step 2: Target model verifies in batch ===
        # Restore target cache to before-speculation state
        restore_kv_cache(target_model, target_kv_before)

        # Verify: input is [last_accepted, d0, d1, ..., d(K-2)] to get logits for [d0, d1, ..., d(K-1)]
        verify_input = [tokens[-1]] + draft_tokens[:-1]  # K tokens
        target_ctx_batch = target_ctx + len(verify_input)

        hidden = target_model._decode_step_fixed_cache_batch(
            verify_input, target_pos, target_ctx_batch
        )
        target_logits = target_model.get_logits(hidden)
        target_logits_np = target_logits.to_numpy()  # [K, vocab_size]

        # === Step 3: Accept/Reject tokens ===
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            # Sample from target distribution
            target_token = sample_token(target_logits_np[i], 0.7, 50, 0.9)

            if target_token == draft_token:
                # Draft matches target - accept
                accepted.append(draft_token)
            else:
                # Disagreement - use target's token and stop
                accepted.append(target_token)
                break

        total_accepted += len(
            [t for i, t in enumerate(accepted) if i < len(draft_tokens) and t == draft_tokens[i]]
        )

        # === Step 4: Update KV caches with only accepted tokens ===
        # Restore to before-speculation state
        restore_kv_cache(draft_model, draft_kv_before)
        restore_kv_cache(target_model, target_kv_before)

        # Re-run forward pass with accepted tokens only
        for acc_token in accepted:
            # Draft model - single token decode
            draft_model._decode_step_fixed_cache(tokens[-1], draft_pos, draft_ctx)
            draft_pos += 1
            draft_ctx += 1

            # Target model - single token decode
            target_model._decode_step_fixed_cache(tokens[-1], target_pos, target_ctx)
            target_pos += 1
            target_ctx += 1

            tokens.append(acc_token)

            if len(tokens) >= num_tokens:
                break

    acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
    return tokens[:num_tokens], acceptance_rate


def main():
    print("=" * 70)
    print("SPECULATIVE DECODING TEST")
    print("Draft: Qwen3-0.6B, Target: Qwen3-8B")
    print(f"Draft tokens per step: {DRAFT_TOKENS}")
    print("=" * 70)

    # Load tokenizer (shared between both models)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Prepare input
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing."),
    ]
    prompt = format_chat_messages(messages, model_type="qwen3")
    input_ids = tokenizer.encode(prompt).ids
    prefill_len = len(input_ids)
    print(f"\nPrefill length: {prefill_len}")

    # Load models
    draft_model = load_draft_model()
    target_model = load_target_model()

    # Initialize caches
    print("\nInitializing KV caches...")
    init_model_cache(draft_model, MAX_SEQ_LEN)
    init_model_cache(target_model, MAX_SEQ_LEN)

    # Run prefill on both models
    print("Running prefill on both models...")
    draft_first = run_prefill(draft_model, input_ids)
    target_first = run_prefill(target_model, input_ids)

    print(f"Draft first token: {draft_first} = '{tokenizer.decode([draft_first])}'")
    print(f"Target first token: {target_first} = '{tokenizer.decode([target_first])}'")

    # Use target's first token for generation
    first_token = target_first

    # Backup KV caches
    draft_kv_backup = backup_kv_cache(draft_model)
    target_kv_backup = backup_kv_cache(target_model)

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        generate_sequential(target_model, first_token, prefill_len, target_kv_backup, 5)
    default_stream().synchronize()

    # Benchmark: Sequential with target model only
    print(f"\n--- Sequential Decode (target only, {GEN_TOKENS} tokens) ---")
    start_event = CudaEvent()
    stop_event = CudaEvent()

    restore_kv_cache(target_model, target_kv_backup)
    start_event.record()
    seq_tokens = generate_sequential(
        target_model, first_token, prefill_len, target_kv_backup, GEN_TOKENS
    )
    stop_event.record()
    stop_event.synchronize()

    seq_time = event_elapsed_ms(start_event, stop_event)
    seq_tps = (GEN_TOKENS - 1) * 1000 / seq_time

    seq_text = tokenizer.decode(seq_tokens)
    print(f"Time: {seq_time:.1f} ms")
    print(f"Throughput: {seq_tps:.2f} tok/s")
    print(f"Text: {seq_text[:100]}...")

    # Benchmark: Speculative decoding
    print(f"\n--- Speculative Decode (draft={DRAFT_TOKENS} tokens) ---")
    restore_kv_cache(draft_model, draft_kv_backup)
    restore_kv_cache(target_model, target_kv_backup)

    start_event.record()
    spec_tokens, acceptance_rate = generate_speculative(
        draft_model,
        target_model,
        first_token,
        prefill_len,
        draft_kv_backup,
        target_kv_backup,
        GEN_TOKENS,
        DRAFT_TOKENS,
    )
    stop_event.record()
    stop_event.synchronize()

    spec_time = event_elapsed_ms(start_event, stop_event)
    spec_tps = (GEN_TOKENS - 1) * 1000 / spec_time

    spec_text = tokenizer.decode(spec_tokens)
    print(f"Time: {spec_time:.1f} ms")
    print(f"Throughput: {spec_tps:.2f} tok/s")
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    print(f"Speedup: {spec_tps / seq_tps:.2f}x")
    print(f"Text: {spec_text[:100]}...")

    # Verify output quality
    print("\n--- Output Comparison ---")
    print(f"Sequential: {seq_text[:150]}...")
    print(f"Speculative: {spec_text[:150]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Time (ms)':<12} {'tok/s':<10} {'Speedup':<10}")
    print("-" * 57)
    print(f"{'Sequential (8B only)':<25} {seq_time:<12.1f} {seq_tps:<10.2f} {'1.00x':<10}")
    print(
        f"{'Speculative (0.6B+8B)':<25} {spec_time:<12.1f} {spec_tps:<10.2f} {spec_tps / seq_tps:.2f}x"
    )
    print(f"\nAcceptance rate: {acceptance_rate:.1%}")
    print("\nNote: Current implementation re-runs forward pass for accepted tokens.")
    print("Optimization: Use KV cache rollback instead of re-computation.")


if __name__ == "__main__":
    main()
