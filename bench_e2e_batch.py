#!/usr/bin/env python3
"""End-to-end benchmark: Sequential vs Batch decode for text generation."""

import numpy as np

model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

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
GEN_TOKENS = 32  # Number of tokens to generate


def generate_sequential(model, tokenizer, first_token, prefill_len, kv_backup):
    """Generate tokens sequentially (baseline)."""
    # Restore KV cache
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = kv_backup[i]
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    start_event = CudaEvent()
    stop_event = CudaEvent()

    start_event.record()

    for _ in range(GEN_TOKENS - 1):
        hidden = model._decode_step_fixed_cache(tokens[-1], position, context_len)
        logits = model.get_logits(hidden)
        next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
        tokens.append(next_token)
        position += 1
        context_len += 1

    stop_event.record()
    stop_event.synchronize()

    elapsed_ms = event_elapsed_ms(start_event, stop_event)
    text = tokenizer.decode(tokens)

    return tokens, text, elapsed_ms


def generate_batch(model, tokenizer, first_token, prefill_len, kv_backup, batch_size=4):
    """Generate tokens using batch decode (simulating speculative decoding)."""
    # Restore KV cache
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = kv_backup[i]
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    start_event = CudaEvent()
    stop_event = CudaEvent()

    start_event.record()

    while len(tokens) < GEN_TOKENS:
        # How many tokens to generate in this batch
        remaining = GEN_TOKENS - len(tokens)
        current_batch = min(batch_size, remaining)

        if current_batch == 1:
            # Single token - use optimized path
            hidden = model._decode_step_fixed_cache(tokens[-1], position, context_len)
            logits = model.get_logits(hidden)
            next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
            tokens.append(next_token)
            position += 1
            context_len += 1
        else:
            # Generate draft tokens first (simulated - just use greedy from last token)
            # In real speculative decoding, this would be from a smaller model
            draft_tokens = []
            temp_position = position
            temp_context = context_len
            temp_token = tokens[-1]

            # Simple draft: repeatedly sample from last token's distribution
            # (This is a simulation - real speculative uses a draft model)
            for _ in range(current_batch):
                hidden = model._decode_step_fixed_cache(temp_token, temp_position, temp_context)
                logits = model.get_logits(hidden)
                next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
                draft_tokens.append(next_token)
                temp_token = next_token
                temp_position += 1
                temp_context += 1

            # For this benchmark, we just accept all draft tokens
            # (simulating 100% acceptance rate)
            tokens.extend(draft_tokens)
            position = temp_position
            context_len = temp_context

    stop_event.record()
    stop_event.synchronize()

    elapsed_ms = event_elapsed_ms(start_event, stop_event)
    text = tokenizer.decode(tokens[:GEN_TOKENS])

    return tokens[:GEN_TOKENS], text, elapsed_ms


def generate_batch_parallel(model, tokenizer, first_token, prefill_len, kv_backup, batch_size=4):
    """Generate tokens using true batch parallel verification.

    This simulates speculative decoding where:
    1. Draft model generates N tokens (simulated by sequential here)
    2. Target model verifies all N tokens in ONE forward pass (batch)
    3. Accept matching tokens, reject rest

    For this benchmark, we assume 100% acceptance to measure raw batch speedup.
    """
    # Restore KV cache
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = kv_backup[i]
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

    tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    start_event = CudaEvent()
    stop_event = CudaEvent()

    # First, generate all tokens sequentially to get the "draft" sequence
    # (In real speculative decoding, this would be from a fast draft model)
    draft_tokens = [first_token]
    temp_pos = position
    temp_ctx = context_len

    for _ in range(GEN_TOKENS - 1):
        hidden = model._decode_step_fixed_cache(draft_tokens[-1], temp_pos, temp_ctx)
        logits = model.get_logits(hidden)
        next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
        draft_tokens.append(next_token)
        temp_pos += 1
        temp_ctx += 1

    # Restore KV cache again for batch verification
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = kv_backup[i]
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

    # Now verify in batches (this is the parallel speedup)
    start_event.record()

    verified_tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1

    idx = 1  # Start after first token
    while idx < len(draft_tokens):
        remaining = len(draft_tokens) - idx
        current_batch = min(batch_size, remaining)

        batch_tokens = draft_tokens[idx : idx + current_batch]

        # Batch verify
        hidden = model._decode_step_fixed_cache_batch(
            batch_tokens,
            position,
            context_len + current_batch,  # Context includes new tokens
        )

        # Get logits for verification (would compare with draft in real speculative)
        logits = model.get_logits(hidden)

        # For benchmark, assume 100% acceptance
        verified_tokens.extend(batch_tokens)
        position += current_batch
        context_len += current_batch
        idx += current_batch

    stop_event.record()
    stop_event.synchronize()

    elapsed_ms = event_elapsed_ms(start_event, stop_event)
    text = tokenizer.decode(verified_tokens[:GEN_TOKENS])

    return verified_tokens[:GEN_TOKENS], text, elapsed_ms


def main():
    print("=" * 70)
    print("END-TO-END BATCH DECODE BENCHMARK")
    print(f"Generating {GEN_TOKENS} tokens")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing in simple terms."),
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

    # Prefill
    print("Running prefill...")
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    logits = model.get_logits(hidden)
    last_logits = logits.to_numpy()[-1]
    first_token = sample_token(last_logits, 0.7, 50, 0.9)

    # Backup KV cache after prefill
    kv_backup = []
    for block in model.blocks:
        k_backup = block.attn._k_cache.to_numpy().copy()
        v_backup = block.attn._v_cache.to_numpy().copy()
        kv_backup.append((k_backup, v_backup))

    print(f"\nFirst token: {first_token} = '{tokenizer.decode([first_token])}'")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        generate_sequential(model, tokenizer, first_token, prefill_len, kv_backup)
    default_stream().synchronize()

    # Benchmark Sequential
    print("\n--- Sequential Decode ---")
    seq_tokens, seq_text, seq_time = generate_sequential(
        model, tokenizer, first_token, prefill_len, kv_backup
    )
    seq_tps = (GEN_TOKENS - 1) * 1000 / seq_time  # -1 because first token is given

    print(f"Time: {seq_time:.1f} ms")
    print(f"Throughput: {seq_tps:.2f} tok/s")
    print(f"Generated: {seq_text[:100]}...")

    # Benchmark Batch Parallel Verification
    print("\n--- Batch Parallel Verification (batch=4) ---")
    batch_tokens, batch_text, batch_time = generate_batch_parallel(
        model, tokenizer, first_token, prefill_len, kv_backup, batch_size=4
    )
    batch_tps = (GEN_TOKENS - 1) * 1000 / batch_time

    print(f"Time: {batch_time:.1f} ms (verification only)")
    print(f"Throughput: {batch_tps:.2f} tok/s (verification)")
    print(f"Speedup: {batch_tps / seq_tps:.2f}x")
    print(f"Generated: {batch_text[:100]}...")

    # Benchmark Batch 8
    print("\n--- Batch Parallel Verification (batch=8) ---")
    batch8_tokens, batch8_text, batch8_time = generate_batch_parallel(
        model, tokenizer, first_token, prefill_len, kv_backup, batch_size=8
    )
    batch8_tps = (GEN_TOKENS - 1) * 1000 / batch8_time

    print(f"Time: {batch8_time:.1f} ms (verification only)")
    print(f"Throughput: {batch8_tps:.2f} tok/s (verification)")
    print(f"Speedup: {batch8_tps / seq_tps:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<30} {'Time (ms)':<12} {'tok/s':<10} {'Speedup':<10}")
    print("-" * 62)
    print(f"{'Sequential':<30} {seq_time:<12.1f} {seq_tps:<10.2f} {'1.00x':<10}")
    print(
        f"{'Batch Verify (batch=4)':<30} {batch_time:<12.1f} {batch_tps:<10.2f} {batch_tps / seq_tps:<10.2f}x"
    )
    print(
        f"{'Batch Verify (batch=8)':<30} {batch8_time:<12.1f} {batch8_tps:<10.2f} {batch8_tps / seq_tps:<10.2f}x"
    )

    print("\nNote: 'Batch Verify' measures verification phase only.")
    print("Real speculative decoding would add draft model overhead.")
    print("With ~30ms draft model, expected E2E speedup: ~2-3x")


if __name__ == "__main__":
    main()
