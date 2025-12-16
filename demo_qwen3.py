#!/usr/bin/env python3
"""
Qwen3-8B FP16 Demo for PyGPUkit v0.2.10

Demonstrates text generation with:
- Weight repacking for optimal GPU memory placement
- FP16 inference via CUTLASS
- KV-cache enabled autoregressive generation
"""

import time

from transformers import AutoTokenizer

from pygpukit.llm import detect_model_spec, load_model_from_safetensors, load_safetensors

# Model path (cached from HuggingFace Hub)
MODEL_ID = "Aratako/Qwen3-8B-ERP-v0.1"
MODEL_PATH = None


def find_model_path():
    """Find the cached model path."""
    import os
    from pathlib import Path

    # Check HF cache
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_dirs = list(cache_dir.glob(f"models--{MODEL_ID.replace('/', '--')}"))

    if model_dirs:
        snapshots = list(model_dirs[0].glob("snapshots/*"))
        if snapshots:
            # Find the index file
            for snapshot in snapshots:
                index_file = snapshot / "model.safetensors.index.json"
                if index_file.exists():
                    return str(index_file)

    return None


def main():
    print("=" * 70)
    print(" PyGPUkit v0.2.10 - Qwen3-8B FP16 Demo")
    print("=" * 70)

    # Find model
    model_path = find_model_path()
    if not model_path:
        print(f"\nError: Model not found in cache: {MODEL_ID}")
        print("Please run: huggingface-cli download Aratako/Qwen3-8B-ERP-v0.1")
        return 1

    print(f"\nModel path: {model_path}")

    # Load tokenizer from HuggingFace
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # Detect model spec
    print("\n[2/3] Detecting model type...")
    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    print(f"  Model type: {spec.name}")
    print(f"  Norm type: {spec.norm_type}")
    print(f"  Activation: {spec.activation}")

    # Load model with weight repacking
    print("\n[3/3] Loading model (FP16 with weight repacking)...")
    start = time.perf_counter()
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)
    load_time = time.perf_counter() - start

    config = model.config
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num KV heads: {config.num_kv_heads}")
    print(f"  Load time: {load_time:.1f}s")

    # Warmup
    print("\nWarming up...")
    test_ids = tokenizer.encode("Hello", add_special_tokens=False)
    _ = model.generate(test_ids, max_new_tokens=2, temperature=0.0, use_cache=True)

    # Text generation with streaming
    print("\n" + "=" * 70)
    print(" Text Generation (Streaming)")
    print("=" * 70)

    prompt = "The future of artificial intelligence is"
    print(f'\nPrompt: "{prompt}"')

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Input tokens: {len(input_ids)}")

    # Generate with streaming
    max_new_tokens = 50
    print(f"\nGenerating {max_new_tokens} tokens (streaming)...\n")

    print("-" * 70)
    print(prompt, end="", flush=True)

    start = time.perf_counter()
    generated_ids = []
    for token_id in model.generate_stream(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    ):
        generated_ids.append(token_id)
        # Decode and print token immediately
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        print(token_str, end="", flush=True)

    total_time = time.perf_counter() - start
    print("\n" + "-" * 70)

    new_tokens = len(generated_ids)
    print(f"\nGenerated {new_tokens} tokens in {total_time:.2f}s")
    print(f"Throughput: {new_tokens / total_time:.1f} tok/s")

    # Benchmark decode speed
    print("\n" + "=" * 70)
    print(" Decode Performance")
    print("=" * 70)

    # Prefill
    hidden, past_kv = model(input_ids, use_cache=True)
    logits = model.get_logits(hidden)
    logits_np = logits.to_numpy()
    next_token = int(logits_np[-1].argmax())

    # Time decode
    decode_times = []
    for _ in range(10):
        start = time.perf_counter()
        hidden, past_kv = model([next_token], past_key_values=past_kv, use_cache=True)
        _ = model.get_logits(hidden)
        elapsed = (time.perf_counter() - start) * 1000
        decode_times.append(elapsed)

    avg_decode = sum(decode_times) / len(decode_times)
    print(f"\nSingle token decode: {avg_decode:.1f} ms")
    print(f"Decode throughput: {1000 / avg_decode:.1f} tok/s")
    print(f"Per-layer time: {avg_decode / config.num_layers:.2f} ms")

    print("\n" + "=" * 70)
    print(" Demo Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
