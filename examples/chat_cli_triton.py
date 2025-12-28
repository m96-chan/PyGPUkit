#!/usr/bin/env python3
"""
PyGPUkit - Hybrid Chat CLI (Triton + Native CUDA)

Demonstrates mixing Triton kernels with native CUDA kernels:
- Triton: RMSNorm (rapid prototyping, easy to modify)
- Native CUDA: MatMul (cuBLASLt), Attention (SDPA), KV cache

This shows how to use Triton for quick kernel iteration while
keeping performance-critical paths on optimized CUDA kernels.

Usage:
    python examples/chat_cli_triton.py --model /path/to/model --tokenizer /path/to/tokenizer.json

Requirements:
    pip install triton  # or: pip install pygpukit[triton]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("PYGPUKIT_CUBLASLT_DEBUG", "0")

import numpy as np


def logits_to_f32(logits_gpu) -> np.ndarray:
    """Convert logits GPU array to numpy float32."""
    logits_np = logits_gpu.to_numpy()
    if logits_np.dtype == np.uint16:
        return (logits_np.astype(np.uint32) << 16).view(np.float32)
    return logits_np.astype(np.float32)


def _build_byte_decoder() -> dict[str, int]:
    """Build unicode-to-byte mapping for GPT-2/Qwen tokenizers."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def _token_str_to_bytes(token_str: str) -> bytes:
    """Convert GPT-2/Qwen token string to bytes."""
    result = []
    for char in token_str:
        if char in _BYTE_DECODER:
            result.append(_BYTE_DECODER[char])
        else:
            result.extend(char.encode("utf-8"))
    return bytes(result)


class StreamingDecoder:
    """UTF-8 streaming decoder for token output."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending_bytes = b""
        self._cache: dict[int, bytes] = {}

    def _get_token_bytes(self, token_id: int) -> bytes:
        cached = self._cache.get(token_id)
        if cached is not None:
            return cached
        token_str = self.tokenizer.id_to_token(token_id)
        if token_str is None:
            result = b""
        else:
            result = _token_str_to_bytes(token_str)
        self._cache[token_id] = result
        return result

    def add_token(self, token_id: int) -> str:
        new_bytes = self._get_token_bytes(token_id)
        if not new_bytes:
            return ""

        all_bytes = self.pending_bytes + new_bytes
        valid_end = 0
        i = 0
        while i < len(all_bytes):
            byte = all_bytes[i]
            if byte < 0x80:
                valid_end = i + 1
                i += 1
            elif byte < 0xC0:
                i += 1
            elif byte < 0xE0:
                if i + 1 < len(all_bytes) and 0x80 <= all_bytes[i + 1] < 0xC0:
                    valid_end = i + 2
                    i += 2
                else:
                    break
            elif byte < 0xF0:
                if (
                    i + 2 < len(all_bytes)
                    and 0x80 <= all_bytes[i + 1] < 0xC0
                    and 0x80 <= all_bytes[i + 2] < 0xC0
                ):
                    valid_end = i + 3
                    i += 3
                else:
                    break
            elif byte < 0xF8:
                if (
                    i + 3 < len(all_bytes)
                    and 0x80 <= all_bytes[i + 1] < 0xC0
                    and 0x80 <= all_bytes[i + 2] < 0xC0
                    and 0x80 <= all_bytes[i + 3] < 0xC0
                ):
                    valid_end = i + 4
                    i += 4
                else:
                    break
            else:
                i += 1

        complete_bytes = all_bytes[:valid_end]
        self.pending_bytes = all_bytes[valid_end:]
        if complete_bytes:
            return complete_bytes.decode("utf-8", errors="replace")
        return ""

    def flush(self) -> str:
        if self.pending_bytes:
            text = self.pending_bytes.decode("utf-8", errors="replace")
            self.pending_bytes = b""
            return text
        return ""


# =============================================================================
# Triton-based Norm Layer (replaces native CUDA RMSNorm)
# =============================================================================


class TritonNorm:
    """Norm layer using Triton RMSNorm kernel.

    This demonstrates using Triton for rapid kernel prototyping.
    The Triton kernel can be easily modified in kernels/rmsnorm.py
    without recompiling C++ code.
    """

    def __init__(self, original_norm):
        """Wrap an existing Norm layer with Triton implementation."""
        self.weight = original_norm.weight
        self.bias = original_norm.bias
        self.norm_type = original_norm.norm_type
        self.eps = original_norm.eps

        # Import Triton components
        from pygpukit.triton import from_gpuarray, kernels

        self._from_gpuarray = from_gpuarray
        self._triton_rmsnorm = kernels.rmsnorm
        self._triton_layernorm = kernels.layernorm

        # Pre-wrap weight for Triton
        self._weight_triton = from_gpuarray(self.weight)

    def __call__(self, x):
        """Forward pass using Triton kernel."""
        from pygpukit.core.factory import zeros

        # Create output buffer with same shape/dtype as input
        out = zeros(list(x.shape), dtype=x.dtype)

        # Wrap for Triton
        x_triton = self._from_gpuarray(x)
        out_triton = self._from_gpuarray(out)

        # Call Triton kernel
        if self.norm_type == "rmsnorm":
            self._triton_rmsnorm(x_triton, self._weight_triton, out_triton, self.eps)
        else:
            if self.bias is None:
                raise ValueError("LayerNorm requires bias")
            bias_triton = self._from_gpuarray(self.bias)
            self._triton_layernorm(x_triton, self._weight_triton, bias_triton, out_triton, self.eps)

        return out


def patch_model_with_triton(model, verbose: bool = True) -> int:
    """Replace all Norm layers in model with TritonNorm.

    Returns:
        Number of layers patched
    """
    from pygpukit.llm.layers import Norm

    patched = 0

    # Patch block norms
    for i, block in enumerate(model.blocks):
        # Attention norm
        if isinstance(block.attn_norm, Norm):
            block.attn_norm = TritonNorm(block.attn_norm)
            patched += 1

        # MLP norm
        if isinstance(block.mlp_norm, Norm):
            block.mlp_norm = TritonNorm(block.mlp_norm)
            patched += 1

        # QK norms in attention (Qwen3 style)
        if hasattr(block.attn, "q_norm") and block.attn.q_norm is not None:
            if isinstance(block.attn.q_norm, Norm):
                block.attn.q_norm = TritonNorm(block.attn.q_norm)
                patched += 1
        if hasattr(block.attn, "k_norm") and block.attn.k_norm is not None:
            if isinstance(block.attn.k_norm, Norm):
                block.attn.k_norm = TritonNorm(block.attn.k_norm)
                patched += 1

    # Final norm
    if hasattr(model, "_norm") and isinstance(model._norm, Norm):
        model._norm = TritonNorm(model._norm)
        patched += 1

    if verbose:
        print(f"  Patched {patched} Norm layers with Triton RMSNorm")

    return patched


def main():
    parser = argparse.ArgumentParser(
        description="PyGPUkit Hybrid Chat (Triton + Native CUDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--no-triton", action="store_true", help="Disable Triton (use native only)")
    args = parser.parse_args()

    # Check Triton availability
    print("Checking Triton availability...")
    try:
        from pygpukit.triton import triton_available, triton_version

        if triton_available():
            print(f"  Triton {triton_version()} available")
            use_triton = not args.no_triton
        else:
            print("  Triton not available, using native CUDA only")
            use_triton = False
    except ImportError:
        print("  Triton not installed, using native CUDA only")
        use_triton = False

    # Load model
    print(f"\nLoading model from: {args.model}")
    print(f"  dtype: {args.dtype}")
    t0 = time.perf_counter()

    from tokenizers import Tokenizer

    from pygpukit.core import default_stream, from_numpy
    from pygpukit.llm import (
        ChatMessage,
        DecodeM1,
        detect_model_spec,
        format_chat_messages,
        load_model_from_safetensors,
        load_safetensors,
    )
    from pygpukit.llm.buffers import DecodeBuffers
    from pygpukit.llm.layers import precompute_freqs_cis
    from pygpukit.llm.sampling import sample_token
    from pygpukit.ops.basic import kv_cache_prefill_gqa

    tokenizer = Tokenizer.from_file(args.tokenizer)
    st = load_safetensors(args.model)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(args.model, dtype=args.dtype, spec=spec)

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    config = model.config
    print(f"  Architecture: {spec.name if spec else 'unknown'}")
    print(f"  Layers: {config.num_layers}, Hidden: {config.hidden_size}")

    # ==========================================================================
    # HYBRID SETUP: Patch Norm layers with Triton
    # ==========================================================================
    if use_triton:
        print("\nApplying Triton backend...")
        patch_model_with_triton(model)
        print("  Kernel routing:")
        print("    - RMSNorm: Triton (kernels/rmsnorm.py)")
        print("    - MatMul:  Native CUDA (cuBLASLt)")
        print("    - SDPA:    Native CUDA (optimized)")
        print("    - KV Cache: Native CUDA")
    else:
        print("\nUsing native CUDA for all operations")

    # Initialize KV cache
    print(f"\nInitializing KV cache (max_seq_len={args.max_seq_len})...")
    for block in model.blocks:
        block.attn.init_fixed_cache(args.max_seq_len, dtype=args.dtype)

    # Initialize decode strategy
    use_qk_norm = model.spec is not None and model.spec.use_qk_norm
    lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
    vocab_size = lm_head.shape[0]

    decode_buffers = DecodeBuffers.allocate(
        config, dtype=args.dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
    )

    m1 = DecodeM1()
    m1.bind(model)

    # Precompute RoPE
    if config.use_rope:
        cos_np, sin_np = precompute_freqs_cis(config.head_dim, args.max_seq_len, config.rope_theta)
        if args.dtype == "float16":
            model._rope_cos_gpu = from_numpy(cos_np.astype(np.float16))
            model._rope_sin_gpu = from_numpy(sin_np.astype(np.float16))
        elif args.dtype == "bfloat16":
            cos_u32 = cos_np.view(np.uint32)
            sin_u32 = sin_np.view(np.uint32)
            cos_bf16 = ((cos_u32 + 0x7FFF + ((cos_u32 >> 16) & 1)) >> 16).astype(np.uint16)
            sin_bf16 = ((sin_u32 + 0x7FFF + ((sin_u32 >> 16) & 1)) >> 16).astype(np.uint16)
            model._rope_cos_gpu = from_numpy(cos_bf16)
            model._rope_sin_gpu = from_numpy(sin_bf16)

    default_stream().synchronize()
    print("Ready!")

    # Chat state
    conversation: list[ChatMessage] = []
    system_msg = ChatMessage(role="system", content=args.system)

    model_type = "llama"
    if spec and "qwen" in spec.name.lower():
        model_type = "qwen3"

    # Get special tokens
    eos_token_id = None
    for tok in ["<|endoftext|>", "</s>", "<|im_end|>"]:
        tid = tokenizer.token_to_id(tok)
        if tid is not None:
            eos_token_id = tid
            break

    qwen_end_tokens = set()
    if model_type == "qwen3":
        for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = tokenizer.token_to_id(tok)
            if tid is not None:
                qwen_end_tokens.add(tid)

    skip_tokens: set[int] = set()
    if model_type == "qwen3":
        tid = tokenizer.token_to_id("<|im_start|>")
        if tid is not None:
            skip_tokens.add(tid)
        for tok in ["assistant", "think", "user", "system", "\n"]:
            tid = tokenizer.token_to_id(tok)
            if tid is not None:
                skip_tokens.add(tid)
    skip_tokens -= qwen_end_tokens
    if eos_token_id is not None:
        skip_tokens.discard(eos_token_id)

    def is_end_token(token_id: int) -> bool:
        return token_id == eos_token_id or token_id in qwen_end_tokens

    def should_skip_token(token_id: int, at_start: bool, skip_count: int) -> bool:
        if not at_start or skip_count >= 10:
            return False
        return token_id in skip_tokens

    def apply_rep_penalty(logits: np.ndarray, ids: list[int], penalty: float) -> np.ndarray:
        if penalty == 1.0 or not ids:
            return logits
        logits = logits.copy()
        for tid in set(ids):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty
        return logits

    rep_penalty = 1.1

    def generate(messages: list[ChatMessage]) -> tuple[str, float, float]:
        prompt = format_chat_messages(messages, model_type=model_type)
        input_ids = tokenizer.encode(prompt).ids

        if len(input_ids) >= args.max_seq_len - 10:
            return "[Error: Conversation too long. Use /clear to reset.]", 0, 0

        # Prefill
        t_prefill_start = time.perf_counter()
        hidden, past_kv = model(input_ids, use_cache=True)
        for i, block in enumerate(model.blocks):
            past_k, past_v = past_kv[i]
            kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
            kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)
        default_stream().synchronize()
        prefill_time = time.perf_counter() - t_prefill_start

        # Decode
        t_decode_start = time.perf_counter()
        logits = model.get_logits(hidden)
        last_logits = logits_to_f32(logits)[-1]
        next_token = sample_token(last_logits, args.temperature, args.top_k, args.top_p)

        generated_ids: list[int] = []
        position = len(input_ids)
        context_len = position + 1
        at_start = True
        skip_count = 0

        # Skip special tokens
        while should_skip_token(next_token, at_start, skip_count):
            if context_len >= args.max_seq_len:
                break
            logits = m1.step(next_token, position, context_len, decode_buffers)
            logits_np = logits_to_f32(logits)[-1]
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
            position += 1
            context_len += 1
            skip_count += 1

        if is_end_token(next_token):
            default_stream().synchronize()
            return "", prefill_time, time.perf_counter() - t_decode_start

        stream_decoder = StreamingDecoder(tokenizer)
        text_chunk = stream_decoder.add_token(next_token)
        if text_chunk:
            print(text_chunk, end="", flush=True)
        generated_ids.append(next_token)
        at_start = False

        while len(generated_ids) < args.max_new_tokens and context_len < args.max_seq_len:
            logits = m1.step(next_token, position, context_len, decode_buffers)
            logits_raw = logits_to_f32(logits)[-1]
            logits_np = apply_rep_penalty(logits_raw, generated_ids, rep_penalty)
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

            if is_end_token(next_token):
                break

            generated_ids.append(next_token)
            position += 1
            context_len += 1

            text_chunk = stream_decoder.add_token(next_token)
            if text_chunk:
                print(text_chunk, end="", flush=True)

        remaining = stream_decoder.flush()
        if remaining:
            print(remaining, end="", flush=True)

        default_stream().synchronize()
        decode_time = time.perf_counter() - t_decode_start
        print()
        return tokenizer.decode(generated_ids), prefill_time, decode_time

    # Chat loop
    print("\n" + "=" * 60)
    print(" PyGPUkit Hybrid Chat (Triton + Native CUDA)")
    backend_str = "Triton RMSNorm + Native CUDA" if use_triton else "Native CUDA only"
    print(f" Backend: {backend_str}")
    print(" Commands: /clear (reset), /quit (exit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            conversation.clear()
            print("[Conversation cleared]")
            continue

        conversation.append(ChatMessage(role="user", content=user_input))
        messages = [system_msg] + conversation

        print("\nAssistant: ", end="", flush=True)
        response, prefill_time, decode_time = generate(messages)

        conversation.append(ChatMessage(role="assistant", content=response))

        tokens_generated = len(tokenizer.encode(response).ids) if response else 0
        decode_tps = tokens_generated / decode_time if decode_time > 0 else 0
        print(
            f"  [prefill: {prefill_time:.1f}s, decode: {tokens_generated} tok / {decode_time:.1f}s = {decode_tps:.1f} tok/s]"
        )

    print("\nUnloading model...")
    del model
    print("Done.")


if __name__ == "__main__":
    main()
