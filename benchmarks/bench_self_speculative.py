#!/usr/bin/env python3
"""Benchmark self-speculative decoding with various draft layer counts."""

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
    """Generate tokens sequentially with greedy sampling."""
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


def generate_self_speculative(
    model, first_token, prefill_len, kv_backup, num_tokens, max_draft_tokens=4, draft_layers=8
):
    """Generate tokens using self-speculative decoding."""
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
    print("SELF-SPECULATIVE DECODING BENCHMARK")
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

    # Backup KV cache
    kv_backup = model.snapshot_kv_cache()

    print(f"First token (greedy): {first_token}")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        generate_sequential_greedy(model, first_token, prefill_len, kv_backup, 5)
    default_stream().synchronize()

    start_event = CudaEvent()
    stop_event = CudaEvent()

    # Baseline
    print(f"\n--- Sequential Baseline ({GEN_TOKENS} tokens) ---")
    start_event.record()
    seq_tokens = generate_sequential_greedy(model, first_token, prefill_len, kv_backup, GEN_TOKENS)
    stop_event.record()
    stop_event.synchronize()
    seq_time = event_elapsed_ms(start_event, stop_event)
    seq_tps = (GEN_TOKENS - 1) * 1000 / seq_time
    print(f"Time: {seq_time:.1f} ms, {seq_tps:.2f} tok/s")

    # Test different draft layer counts
    results = []
    draft_layer_counts = [18, 24, 28, 32, 34, 35, 36]

    for draft_layers in draft_layer_counts:
        print(f"\n--- Self-Speculative (draft_layers={draft_layers}/{num_layers}) ---")

        start_event.record()
        spec_tokens, acceptance_rate = generate_self_speculative(
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

        spec_time = event_elapsed_ms(start_event, stop_event)
        spec_tps = (GEN_TOKENS - 1) * 1000 / spec_time
        matches = spec_tokens == seq_tokens
        speedup = seq_time / spec_time if spec_time > 0 else 0

        print(f"Time: {spec_time:.1f} ms, {spec_tps:.2f} tok/s")
        print(f"Acceptance: {acceptance_rate:.1%}, Match: {matches}, Speedup: {speedup:.2f}x")

        results.append(
            {
                "layers": draft_layers,
                "time": spec_time,
                "tps": spec_tps,
                "acceptance": acceptance_rate,
                "matches": matches,
                "speedup": speedup,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Layers':<10} {'Time (ms)':<12} {'tok/s':<10} {'Accept':<10} {'Speedup':<10} {'Match'}"
    )
    print("-" * 62)
    print(f"{'Baseline':<10} {seq_time:<12.1f} {seq_tps:<10.2f} {'N/A':<10} {'1.00x':<10} {'N/A'}")
    for r in results:
        print(
            f"{r['layers']:<10} {r['time']:<12.1f} {r['tps']:<10.2f} {r['acceptance'] * 100:<9.0f}% {r['speedup']:.2f}x{'':<5} {'YES' if r['matches'] else 'NO'}"
        )

    print("\nNote: Current implementation has high overhead from KV cache CPU-GPU copies.")
    print("Performance will improve with GPU-side KV cache management.")


if __name__ == "__main__":
    main()
