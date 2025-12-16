#!/usr/bin/env python3
"""
PyGPUkit v0.2.9 - End-to-End LLM Inference Demo

Demonstrates the unified Transformer implementation with:
- Model loading from safetensors (GPT-2 or LLaMA)
- Tokenization
- KV-cache enabled autoregressive generation
- Hybrid Attention: CPU for decode (seq_len=1), GPU for prefill
- Performance benchmarking (prefill + decode)

Usage:
    python demo_llm_e2e.py --model /path/to/model.safetensors --tokenizer /path/to/tokenizer.json

Supported models:
    - LLaMA 2/3 (any size in safetensors format)
    - GPT-2 (safetensors format)

Note on Tokenizer:
    The built-in pygpukit.llm.Tokenizer is EXPERIMENTAL and intended for demos only.
    It may not work with all tokenizer.json formats (e.g., Qwen3).
    For production use, we recommend HuggingFace tokenizers:
        pip install tokenizers
        from tokenizers import Tokenizer
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms * 1000:.2f} us"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


def main():
    parser = argparse.ArgumentParser(description="PyGPUkit E2E LLM Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.safetensors",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "llama", "gpt2"],
        default="auto",
        help="Model type (auto-detect by default)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run benchmark without text generation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Weight dtype (float32 or float16)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" PyGPUkit v0.2.9 - End-to-End LLM Inference Demo")
    print("=" * 70)

    # Check paths
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)

    if not model_path.exists():
        print(f"\nError: Model not found: {model_path}")
        return 1

    if not tokenizer_path.exists():
        print(f"\nError: Tokenizer not found: {tokenizer_path}")
        return 1

    # Import PyGPUkit
    try:
        import pygpukit as gpk
        from pygpukit.llm import (
            Tokenizer,
            detect_model_spec,
            load_model_from_safetensors,
            load_safetensors,
        )

        print("\nPyGPUkit loaded successfully")
        print(f"  CUDA available: {gpk.is_cuda_available()}")
    except ImportError as e:
        print(f"\nError importing PyGPUkit: {e}")
        return 1

    # =========================================================================
    # Load Tokenizer
    # =========================================================================
    section("Loading Tokenizer")

    start = time.perf_counter()
    tokenizer = Tokenizer(str(tokenizer_path))
    tokenizer_time = (time.perf_counter() - start) * 1000

    print(f"  Path: {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  BOS token: {tokenizer.bos_token_id}")
    print(f"  EOS token: {tokenizer.eos_token_id}")
    print(f"  Load time: {format_time(tokenizer_time)}")

    # =========================================================================
    # Detect Model Type
    # =========================================================================
    section("Detecting Model Type")

    st = load_safetensors(str(model_path))
    tensor_names = st.tensor_names

    # Use ModelSpec-based detection
    spec = detect_model_spec(tensor_names)
    model_type = spec.name

    print(f"  Model type: {model_type.upper()}")
    print(f"  ModelSpec: {spec.name}")
    print(f"  Total tensors: {len(tensor_names)}")
    print(f"  Norm type: {spec.norm_type}")
    print(f"  Activation: {spec.activation}")
    print(f"  Use RoPE: {spec.use_rope}")
    print(f"  Use QK Norm: {spec.use_qk_norm}")

    # =========================================================================
    # Load Model
    # =========================================================================
    section("Loading Model")

    start = time.perf_counter()
    model = load_model_from_safetensors(str(model_path), dtype=args.dtype, spec=spec)
    load_time = (time.perf_counter() - start) * 1000

    config = model.config
    print(f"  Architecture: {model_type.upper()}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num KV heads: {config.num_kv_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Norm type: {config.norm_type}")
    print(f"  Activation: {config.activation}")
    print(f"  Use RoPE: {config.use_rope}")
    print(f"  Load time: {format_time(load_time)}")
    print(f"  model.spec: {model.spec.name if model.spec else 'None'}")
    print(f"  dtype: {args.dtype}")

    # Estimate model size
    params = (
        config.vocab_size * config.hidden_size  # embed_tokens
        + config.num_layers
        * (
            4 * config.hidden_size * config.hidden_size  # Q, K, V, O projections
            + 3 * config.hidden_size * config.intermediate_size  # gate, up, down
            + 2 * config.hidden_size  # norms
        )
        + config.hidden_size  # final norm
    )
    print(f"  Estimated params: {params / 1e9:.2f}B")

    # =========================================================================
    # Tokenize Prompt
    # =========================================================================
    section("Tokenizing Prompt")

    prompt = args.prompt
    input_ids = tokenizer.encode(prompt)

    print(f'  Prompt: "{prompt}"')
    print(f"  Token IDs: {input_ids}")
    print(f"  Num tokens: {len(input_ids)}")

    # Show token breakdown
    print("  Tokens: ", end="")
    for tid in input_ids[:10]:
        token_str = tokenizer.id_to_token(tid)
        # Use ASCII-safe representation to avoid encoding issues
        safe_repr = repr(token_str).encode("ascii", "replace").decode("ascii")
        print(f"[{tid}:{safe_repr}] ", end="")
    if len(input_ids) > 10:
        print("...")
    else:
        print()

    # =========================================================================
    # Benchmark: Prefill + Decode
    # =========================================================================
    section("Benchmark: Prefill + Decode")

    # Warmup
    print("  Warming up...")
    _ = model.generate(input_ids[:4], max_new_tokens=2, temperature=0.0, use_cache=True)

    # Prefill benchmark (various sequence lengths)
    print("\n  Prefill Performance (GPU SDPA):")
    for seq_len in [8, 16, 32, 64, 128]:
        # Create test sequence
        test_ids = (input_ids * ((seq_len // len(input_ids)) + 1))[:seq_len]

        # Time prefill only
        times = []
        for _ in range(3):
            start = time.perf_counter()
            hidden, past_kv = model(test_ids, use_cache=True)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        tokens_per_sec = seq_len / (avg_time / 1000)
        print(f"    seq_len={seq_len:3d}: {avg_time:6.2f} ms ({tokens_per_sec:,.0f} tok/s)")

    # Decode benchmark (single token)
    print("\n  Decode Performance (Hybrid CPU/GPU):")

    # First do a prefill to get KV cache
    hidden, past_kv = model(input_ids, use_cache=True)
    logits = model.get_logits(hidden)
    logits_np = logits.to_numpy()
    next_token = int(logits_np[-1].argmax())

    # Time single-token decode
    decode_times = []
    for _ in range(10):
        start = time.perf_counter()
        hidden, past_kv = model([next_token], past_key_values=past_kv, use_cache=True)
        _ = model.get_logits(hidden)
        elapsed = (time.perf_counter() - start) * 1000
        decode_times.append(elapsed)

    avg_decode = sum(decode_times) / len(decode_times)
    min_decode = min(decode_times)
    max_decode = max(decode_times)
    tokens_per_sec = 1000 / avg_decode

    print(
        f"    Single token decode: {avg_decode:.2f} ms (min={min_decode:.2f}, max={max_decode:.2f})"
    )
    print(f"    Decode throughput: {tokens_per_sec:.1f} tok/s")

    # Per-layer estimate
    per_layer = avg_decode / config.num_layers
    print(f"    Per-layer time: {per_layer:.2f} ms")

    if args.benchmark_only:
        section("Benchmark Complete")
        return 0

    # =========================================================================
    # Text Generation
    # =========================================================================
    section("Text Generation")

    print(f'  Prompt: "{prompt}"')
    print(f"  Max new tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print()

    # Generate with timing
    start = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    total_time = (time.perf_counter() - start) * 1000

    # Decode output
    output_text = tokenizer.decode(output_ids)
    new_tokens = len(output_ids) - len(input_ids)

    print("  Generated text:")
    print(f"  {'-' * 60}")
    print(f"  {output_text}")
    print(f"  {'-' * 60}")
    print()
    print(f"  Input tokens: {len(input_ids)}")
    print(f"  Output tokens: {len(output_ids)}")
    print(f"  New tokens: {new_tokens}")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Throughput: {new_tokens / (total_time / 1000):.1f} tok/s")

    # =========================================================================
    # Summary
    # =========================================================================
    section("Summary")

    print(f"  Model: {model_type.upper()} ({config.num_layers}L, {config.hidden_size}H)")
    print("  Prefill: GPU SDPA with causal mask")
    print("  Decode: Hybrid (CPU for seq_len=1, GPU otherwise)")
    print("  KV Cache: numpy (CPU)")
    print()
    print("  Performance:")
    print(f"    Prefill (64 tokens): ~{sum(times) / len(times):.1f} ms")
    print(f"    Decode (per token): ~{avg_decode:.1f} ms")
    print(f"    Generation: {new_tokens / (total_time / 1000):.1f} tok/s")

    print()
    print("  PyGPUkit v0.2.9 Features Used:")
    print("    - Unified TransformerConfig")
    print("    - CausalTransformerModel with generate()")
    print("    - Hybrid Attention (CPU decode / GPU prefill)")
    print("    - GPU: RMSNorm, SDPA, SiLU, RoPE, matmul")
    print("    - SafeTensors loading with BFloat16 support")
    print("    - Rust-based tokenizer")

    return 0


if __name__ == "__main__":
    exit(main())
