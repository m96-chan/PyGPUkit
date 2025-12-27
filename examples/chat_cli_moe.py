#!/usr/bin/env python3
"""
PyGPUkit - MoE (Mixture of Experts) Chat CLI

A minimal chat interface for MoE models (Mixtral, Qwen3-MoE, etc.).
Supports multiple chat templates with auto-detection.

Usage:
    python examples/chat_cli_moe.py --model /path/to/model.safetensors.index.json --tokenizer /path/to/tokenizer.json

Example (Qwen3-30B-A3B MoE):
    python examples/chat_cli_moe.py \
        --model /path/to/Qwen3-30B-A3B/model.safetensors.index.json \
        --tokenizer /path/to/Qwen3-30B-A3B/tokenizer.json

Example (Mixtral-8x7B):
    python examples/chat_cli_moe.py \
        --model /path/to/Mixtral-8x7B/model.safetensors.index.json \
        --tokenizer /path/to/Mixtral-8x7B/tokenizer.json

Example with explicit chat template:
    python examples/chat_cli_moe.py \
        --model /path/to/model --chat-template qwen

Example with CUDA Graph (faster decode):
    python examples/chat_cli_moe.py \
        --model /path/to/model --cuda-graph

Supported chat templates:
    qwen     - Qwen2/Qwen3 (<|im_start|>...<|im_end|>)
    mistral  - Mistral/Mixtral ([INST]...[/INST])
    llama2   - LLaMA 2 (<<SYS>>...<</SYS>>)
    llama3   - LLaMA 3 (<|start_header_id|>...<|eot_id|>)
    chatml   - Generic ChatML

Commands:
    /clear  - Clear conversation history
    /quit   - Exit chat
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress cuBLASLt debug output
os.environ.setdefault("PYGPUKIT_CUBLASLT_DEBUG", "0")

import numpy as np


def logits_to_f32(logits_gpu) -> np.ndarray:
    """Convert logits GPU array to numpy float32."""
    logits_np = logits_gpu.to_numpy()
    if logits_np.dtype == np.uint16:
        # bf16 stored as uint16 - convert to fp32
        return (logits_np.astype(np.uint32) << 16).view(np.float32)
    return logits_np.astype(np.float32)


def _build_byte_decoder() -> dict[str, int]:
    """Build the unicode-to-byte mapping used by GPT-2/Mistral style tokenizers."""
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
    """Convert a token string to raw bytes."""
    result = []
    for char in token_str:
        if char in _BYTE_DECODER:
            result.append(_BYTE_DECODER[char])
        else:
            result.extend(char.encode("utf-8"))
    return bytes(result)


class StreamingDecoder:
    """Streaming decoder for UTF-8 safe output."""

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

    def reset(self):
        self.pending_bytes = b""


def detect_chat_template(spec_name: str) -> str:
    """Detect chat template from model spec name."""
    name = spec_name.lower()
    if "qwen" in name:
        return "qwen"
    elif "mixtral" in name or "mistral" in name:
        return "mistral"
    elif "llama3" in name or "llama-3" in name:
        return "llama3"
    elif "llama" in name:
        return "llama2"
    return "chatml"


def main():
    parser = argparse.ArgumentParser(
        description="PyGPUkit MoE Chat CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.safetensors or model.safetensors.index.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1, 1.0 = disabled)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA Graph for faster decode (reduces kernel launch overhead)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        choices=["qwen", "mistral", "llama2", "llama3", "chatml"],
        help="Chat template (auto-detected from model if not specified)",
    )
    args = parser.parse_args()

    # Lazy imports for faster --help
    print("Loading PyGPUkit...")
    from tokenizers import Tokenizer

    from pygpukit.core import default_stream, from_numpy
    from pygpukit.llm import (
        MIXTRAL_SPEC,
        DecodeM1Graph,
        detect_model_spec,
        load_model_from_safetensors,
        load_safetensors,
    )
    from pygpukit.llm.buffers import DecodeBuffers
    from pygpukit.llm.chat import format_chat_messages
    from pygpukit.llm.layers import precompute_freqs_cis
    from pygpukit.llm.sampling import sample_token
    from pygpukit.ops.basic import kv_cache_prefill_gqa

    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"\nLoading MoE model from: {args.model}")
    print(f"  dtype: {args.dtype}")
    t0 = time.perf_counter()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    st = load_safetensors(args.model)
    spec = detect_model_spec(st.tensor_names)

    # Verify it's a MoE model
    if spec is None:
        print("Warning: Could not auto-detect model spec, using MIXTRAL_SPEC")
        spec = MIXTRAL_SPEC
    elif not spec.is_moe:
        print(f"Warning: Detected {spec.name} which is not a MoE model")
        print("This example is optimized for MoE models like Mixtral")

    model = load_model_from_safetensors(args.model, dtype=args.dtype, spec=spec)

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Model info
    config = model.config
    print(f"  Architecture: {spec.name if spec else 'unknown'}")
    print(f"  Layers: {config.num_layers}, Hidden: {config.hidden_size}")
    print(f"  Vocab size: {model.embed_tokens.shape[0]}")
    if config.num_experts:
        print(f"  MoE: {config.num_experts} experts, top-{config.num_experts_per_tok}")

    # Determine chat template
    chat_template = args.chat_template
    if chat_template is None:
        chat_template = detect_chat_template(spec.name if spec else "")
    print(f"  Chat template: {chat_template}")

    # =========================================================================
    # Initialize KV Cache
    # =========================================================================
    print(f"\nInitializing KV cache (max_seq_len={args.max_seq_len})...")

    for block in model.blocks:
        block.attn.init_fixed_cache(args.max_seq_len, dtype=args.dtype)

    # =========================================================================
    # Initialize Decode Buffers
    # =========================================================================
    use_qk_norm = model.spec is not None and model.spec.use_qk_norm
    lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
    vocab_size = lm_head.shape[0]

    decode_buffers = DecodeBuffers.allocate(
        config, dtype=args.dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
    )

    # Precompute RoPE frequencies
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
        else:
            model._rope_cos_gpu = from_numpy(cos_np.astype(np.float32))
            model._rope_sin_gpu = from_numpy(sin_np.astype(np.float32))

    default_stream().synchronize()

    # =========================================================================
    # Initialize CUDA Graph (optional)
    # =========================================================================
    use_cuda_graph = args.cuda_graph
    m1_graph = None

    if use_cuda_graph:
        print("\nInitializing CUDA Graph...")
        m1_graph = DecodeM1Graph()
        m1_graph.bind(model)
        m1_graph.init_graph(max_seq_len=args.max_seq_len)
        print(f"  CUDA Graph ready (max_seq_len={args.max_seq_len})")

    print("Ready!")

    # =========================================================================
    # Chat State
    # =========================================================================
    conversation: list[dict] = []
    system_msg = {"role": "system", "content": args.system}

    # Get EOS tokens (model-specific)
    eos_token_ids: set[int] = set()
    for eos_str in ["</s>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>"]:
        tid = tokenizer.token_to_id(eos_str)
        if tid is not None:
            eos_token_ids.add(tid)

    def is_end_token(token_id: int) -> bool:
        return token_id in eos_token_ids

    def apply_repetition_penalty(
        logits: np.ndarray, generated_ids: list[int], penalty: float
    ) -> np.ndarray:
        if penalty == 1.0 or not generated_ids:
            return logits
        logits = logits.copy()
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits

    # =========================================================================
    # Decode Helper (CUDA Graph or Non-Graph)
    # =========================================================================
    def decode_one_token(token_id: int, position: int, context_len: int) -> np.ndarray:
        """Decode one token and return logits as numpy array.

        Uses CUDA Graph if enabled, otherwise falls back to standard decode.
        """
        if use_cuda_graph and m1_graph is not None:
            logits = m1_graph.step_graph(token_id, position, context_len)
            return logits_to_f32(logits)[-1]
        else:
            hidden = model._decode_step_fixed_cache(token_id, position, context_len)
            logits = model.get_logits(hidden)
            return logits_to_f32(logits)[-1]

    # =========================================================================
    # Generation Function
    # =========================================================================
    def generate(messages: list[dict]) -> tuple[str, float, float, int]:
        """Generate response using M=1 decode."""
        prompt = format_chat_messages(messages, model_type=chat_template)
        input_ids = tokenizer.encode(prompt).ids

        if len(input_ids) >= args.max_seq_len - 10:
            return "[Error: Conversation too long. Use /clear to reset.]", 0, 0, 0

        # Prefill
        t_prefill_start = time.perf_counter()
        hidden, past_key_values = model(input_ids, use_cache=True)
        for i, block in enumerate(model.blocks):
            past_k, past_v = past_key_values[i]
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

        # Check if first token is end token
        if is_end_token(next_token):
            default_stream().synchronize()
            decode_time = time.perf_counter() - t_decode_start
            return "", prefill_time, decode_time, 0

        # Use streaming decoder for UTF-8 safe output
        stream_decoder = StreamingDecoder(tokenizer)

        # Output first token
        text_chunk = stream_decoder.add_token(next_token)
        if text_chunk:
            print(text_chunk, end="", flush=True)
        generated_ids.append(next_token)

        while len(generated_ids) < args.max_new_tokens:
            if context_len >= args.max_seq_len:
                break

            # Decode one token (CUDA Graph or standard)
            logits_np = decode_one_token(next_token, position, context_len)
            logits_np = apply_repetition_penalty(logits_np, generated_ids, args.repetition_penalty)
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

            if is_end_token(next_token):
                break

            generated_ids.append(next_token)
            position += 1
            context_len += 1

            text_chunk = stream_decoder.add_token(next_token)
            if text_chunk:
                print(text_chunk, end="", flush=True)

        # Flush any remaining buffered text
        remaining = stream_decoder.flush()
        if remaining:
            print(remaining, end="", flush=True)

        default_stream().synchronize()
        decode_time = time.perf_counter() - t_decode_start

        print()
        return tokenizer.decode(generated_ids), prefill_time, decode_time, len(generated_ids)

    # =========================================================================
    # Chat Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print(" PyGPUkit MoE Chat")
    if config.num_experts:
        print(
            f" Model: {spec.name} ({config.num_experts} experts, top-{config.num_experts_per_tok})"
        )
    else:
        print(f" Model: {spec.name}")
    print(f" CUDA Graph: {'ON' if use_cuda_graph else 'OFF'}")
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

        # Commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            conversation.clear()
            print("[Conversation cleared]")
            continue

        # Add user message
        conversation.append({"role": "user", "content": user_input})

        # Build full message list with system prompt
        messages = [system_msg] + conversation

        # Generate response
        print("\nAssistant: ", end="", flush=True)

        response, prefill_time, decode_time, tokens_generated = generate(messages)

        # Add assistant response to history
        conversation.append({"role": "assistant", "content": response})

        # Stats
        decode_tps = tokens_generated / decode_time if decode_time > 0 else 0
        print(
            f"  [prefill: {prefill_time:.1f}s, "
            f"decode: {tokens_generated} tok / {decode_time:.1f}s = {decode_tps:.1f} tok/s]"
        )

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\nUnloading model...")
    del model
    print("Done.")


if __name__ == "__main__":
    main()
