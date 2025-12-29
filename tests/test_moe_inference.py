#!/usr/bin/env python3
"""Test MoE inference with various prompt lengths."""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("PYGPUKIT_CUBLASLT_DEBUG", "0")

import numpy as np
from tokenizers import Tokenizer

MODEL_PATH = "F:/LLM/Qwen3-30B-A3B-Instruct-2507-FP8"


def logits_to_f32(logits_gpu) -> np.ndarray:
    """Convert logits GPU array to numpy float32."""
    logits_np = logits_gpu.to_numpy()
    if logits_np.dtype == np.uint16:
        return (logits_np.astype(np.uint32) << 16).view(np.float32)
    return logits_np.astype(np.float32)


def sample_top_k(logits: np.ndarray, k: int = 50, temperature: float = 0.7) -> int:
    """Sample from logits with top-k and temperature."""
    logits = logits / temperature
    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]
    top_k_probs = np.exp(top_k_logits - top_k_logits.max())
    top_k_probs /= top_k_probs.sum()
    return int(top_k_idx[np.random.choice(len(top_k_idx), p=top_k_probs)])


def test_prompt_lengths():
    """Test inference with various prompt lengths."""
    from pygpukit.llm import load_safetensors, detect_model_spec, MIXTRAL_SPEC
    from pygpukit.llm.loader import load_model_from_safetensors

    print(f"Loading model from {MODEL_PATH}...")

    # Find the index file
    index_file = f"{MODEL_PATH}/model.safetensors.index.json"

    st = load_safetensors(index_file)
    spec = detect_model_spec(st.tensor_names)
    if spec is None:
        spec = MIXTRAL_SPEC

    model = load_model_from_safetensors(index_file, dtype="bfloat16", spec=spec)
    tokenizer = Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")

    # Initialize KV cache
    MAX_SEQ_LEN = 512
    for block in model.blocks:
        block.attn.init_fixed_cache(MAX_SEQ_LEN, dtype="bfloat16")

    # Test cases with different token counts - focus on M >= 16 threshold
    test_cases = [
        ("Hi", "short (9)"),
        ("What is 2+2?", "medium (15)"),
        ("What is two plus two? Please answer briefly.", "longer (18)"),
        ("The quick brown fox jumps over the lazy dog. This is a test of the emergency broadcast system.", "long (28)"),
        ("Please write a haiku about programming in Python. Make sure to include references to debugging, testing, and code review.", "very long (35)"),
    ]

    for prompt, label in test_cases:
        print(f"\n=== Testing {label} prompt ===")

        # Reset KV cache for each test
        for block in model.blocks:
            block.attn.init_fixed_cache(MAX_SEQ_LEN, dtype="bfloat16")

        messages = [{"role": "user", "content": prompt}]
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        input_ids = tokenizer.encode(full_prompt).ids
        print(f"Prompt: {prompt!r}")
        print(f"Token count: {len(input_ids)}")

        # Prefill - get hidden states and past_key_values
        hidden, past_key_values = model(input_ids, use_cache=True)

        # Store KV cache
        from pygpukit.ops.basic import kv_cache_prefill_gqa
        for i, block in enumerate(model.blocks):
            past_k, past_v = past_key_values[i]
            kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
            kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

        # Get logits from hidden states
        logits = model.get_logits(hidden)
        logits_np = logits_to_f32(logits)

        # Check logits - shape is [seq_len, vocab_size]
        if logits_np.ndim == 3:
            last_logits = logits_np[0, -1, :]
        else:
            last_logits = logits_np[-1, :]
        print(f"Logits stats: min={last_logits.min():.2f}, max={last_logits.max():.2f}, mean={last_logits.mean():.4f}")

        # Get top tokens
        top_indices = np.argsort(last_logits)[-5:][::-1]
        print("Top 5 tokens:")
        for idx in top_indices:
            token = tokenizer.decode([int(idx)])
            print(f"  {idx}: {last_logits[idx]:.2f} -> {token!r}")

        # Generate a few tokens using decode step
        from pygpukit.core import default_stream
        generated = []
        current_token = sample_top_k(last_logits)
        generated.append(current_token)

        position = len(input_ids)
        context_len = position + 1

        for _ in range(9):
            hidden = model._decode_step_fixed_cache(current_token, position, context_len)
            logits = model.get_logits(hidden)
            logits_np = logits_to_f32(logits)
            last_logits = logits_np[-1, :]
            current_token = sample_top_k(last_logits)
            generated.append(current_token)
            position += 1
            context_len += 1

        default_stream().synchronize()
        output_text = tokenizer.decode(generated)
        print(f"Generated (10 tokens): {output_text!r}")

        # Check for garbage
        is_garbage = any([
            output_text.count(output_text[0]) > 8 if output_text else False,  # Repetitive single char
            "{{{{" in output_text,
            "}}}}}" in output_text,
            all(c in "0123456789" for c in output_text.strip()),
        ])

        if is_garbage:
            print("WARNING: Output looks like garbage!")
        else:
            print("Output looks reasonable.")


if __name__ == "__main__":
    test_prompt_lengths()
