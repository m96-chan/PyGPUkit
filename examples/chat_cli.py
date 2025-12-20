#!/usr/bin/env python3
"""
PyGPUkit - Simple CLI Chat

A minimal turn-based chat interface using the fastest inference configuration:
- M=1 decode: Non-graph zero-alloc path
- Batch verify: Original allocating path (17.5 tok/s effective)

Usage:
    python examples/chat_cli.py --model /path/to/model.safetensors.index.json --tokenizer /path/to/tokenizer.json

Example (Qwen3-8B):
    python examples/chat_cli.py \
        --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/.../model.safetensors.index.json \
        --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/.../tokenizer.json

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
    """Convert logits GPU array to numpy float32.

    Handles bf16 (stored as uint16) by converting to fp32.
    """
    logits_np = logits_gpu.to_numpy()
    if logits_np.dtype == np.uint16:
        # bf16 stored as uint16 - convert to fp32
        return (logits_np.astype(np.uint32) << 16).view(np.float32)
    return logits_np.astype(np.float32)


class StreamingDecoder:
    """O(1) streaming decoder for UTF-8 safe output.

    Uses a sliding window to decode only the last WINDOW tokens,
    making each add_token() call O(1) instead of O(n).
    """

    WINDOW = 8  # Sliding window size

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.cached_prefix = ""  # Cached decode result for growing phase

    def add_token(self, token_id: int) -> str:
        """Add a token and return the new text portion.

        Returns:
            New text from this token (O(1) complexity).
        """
        self.tokens.append(token_id)

        window = self.tokens[-self.WINDOW:]
        text = self.tokenizer.decode(window)

        if len(self.tokens) <= self.WINDOW:
            # Growing phase - use cached prefix
            new_text = text[len(self.cached_prefix):]
            self.cached_prefix = text
            return new_text
        else:
            # Sliding phase - decode window[:-1] to find new portion
            prefix = self.tokenizer.decode(window[:-1])
            return text[len(prefix):]

    def flush(self) -> str:
        """Flush any remaining buffered text (none with this approach)."""
        return ""

    def reset(self):
        """Reset the decoder state."""
        self.tokens.clear()
        self.cached_prefix = ""


def main():
    parser = argparse.ArgumentParser(
        description="PyGPUkit CLI Chat",
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
        default=2048,
        help="Maximum sequence length (default: 2048)",
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for speculative-style generation (default: 1 = no batching)",
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
        help="Model dtype (default: bfloat16 - fastest for bf16 models)",
    )
    args = parser.parse_args()

    # Lazy imports for faster --help
    print("Loading PyGPUkit...")
    from tokenizers import Tokenizer

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

    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"\nLoading model from: {args.model}")
    print(f"  dtype: {args.dtype}")
    t0 = time.perf_counter()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    st = load_safetensors(args.model)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(args.model, dtype=args.dtype, spec=spec)

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Model info
    config = model.config
    print(f"  Architecture: {spec.name if spec else 'unknown'}")
    print(f"  Layers: {config.num_layers}, Hidden: {config.hidden_size}")
    print(f"  Vocab size: {model.embed_tokens.shape[0]}")

    # =========================================================================
    # Initialize KV Cache
    # =========================================================================
    print(f"\nInitializing KV cache (max_seq_len={args.max_seq_len})...")

    for block in model.blocks:
        block.attn.init_fixed_cache(args.max_seq_len, dtype=args.dtype)

    # Precompute RoPE frequencies
    if config.use_rope:
        cos_np, sin_np = precompute_freqs_cis(
            config.head_dim, args.max_seq_len, config.rope_theta
        )
        # Use float16 for RoPE regardless of model dtype (computed in fp32 for bf16)
        rope_np_dtype = np.float16 if args.dtype == "float16" else np.float32
        model._rope_cos_gpu = from_numpy(cos_np.astype(rope_np_dtype))
        model._rope_sin_gpu = from_numpy(sin_np.astype(rope_np_dtype))

    default_stream().synchronize()
    print("Ready!")

    # =========================================================================
    # Chat State
    # =========================================================================
    conversation: list[ChatMessage] = []
    system_msg = ChatMessage(role="system", content=args.system)

    # Detect model type for chat formatting
    model_type = "llama"
    if spec and "qwen" in spec.name.lower():
        model_type = "qwen3"
    elif spec and "llama" in spec.name.lower():
        model_type = "llama"

    # Get special tokens
    eos_token_id = None
    try:
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("<|im_end|>")
    except Exception:
        pass

    # Qwen3 specific end tokens
    qwen_end_tokens = set()
    if model_type == "qwen3":
        for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tid = tokenizer.token_to_id(tok)
            if tid is not None:
                qwen_end_tokens.add(tid)

    def is_end_token(token_id: int) -> bool:
        if token_id == eos_token_id:
            return True
        if token_id in qwen_end_tokens:
            return True
        return False

    # Special tokens to skip (not output but continue generation)
    # For Qwen3, the model outputs "<|im_start|>assistant\n" at the start
    # We need to skip these tokens to avoid showing them to the user
    skip_tokens: set[int] = set()
    MAX_SKIP_TOKENS = 10  # Safety limit to prevent infinite loops

    if model_type == "qwen3":
        # Only skip <|im_start|> - NOT <|im_end|> (that should end generation)
        tid = tokenizer.token_to_id("<|im_start|>")
        if tid is not None:
            skip_tokens.add(tid)

        # Skip role tokens that appear after <|im_start|>
        for tok in ["assistant", "think", "user", "system"]:
            tid = tokenizer.token_to_id(tok)
            if tid is not None:
                skip_tokens.add(tid)
            # Also try encoding to get token IDs
            for t in tokenizer.encode(tok).ids:
                skip_tokens.add(t)

        # Skip newline tokens (but NOT if they're the only content)
        for tok in ["\n", "\r\n", "\r", "Ċ"]:
            tid = tokenizer.token_to_id(tok)
            if tid is not None:
                skip_tokens.add(tid)

        # Also try encoding newlines
        newline_ids = tokenizer.encode("\n").ids
        for tid in newline_ids:
            skip_tokens.add(tid)

    # Remove any end tokens from skip_tokens - they should end, not skip
    skip_tokens -= qwen_end_tokens
    if eos_token_id is not None:
        skip_tokens.discard(eos_token_id)

    def should_skip_token(token_id: int, at_start: bool, skip_count: int) -> bool:
        """Check if token should be skipped (only at start of generation)."""
        if not at_start:
            return False
        if skip_count >= MAX_SKIP_TOKENS:
            return False  # Safety limit reached
        return token_id in skip_tokens

    def apply_repetition_penalty(
        logits: np.ndarray, generated_ids: list[int], penalty: float
    ) -> np.ndarray:
        """Apply repetition penalty to logits for generated tokens."""
        if penalty == 1.0 or not generated_ids:
            return logits
        logits = logits.copy()
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits

    rep_penalty = args.repetition_penalty

    # =========================================================================
    # Generation Functions
    # =========================================================================
    batch_size = args.batch_size

    def generate_m1(messages: list[ChatMessage]) -> tuple[str, float, float]:
        """Generate using M=1 decode path (baseline)."""
        prompt = format_chat_messages(messages, model_type=model_type)
        input_ids = tokenizer.encode(prompt).ids

        if len(input_ids) >= args.max_seq_len - 10:
            return "[Error: Conversation too long. Use /clear to reset.]", 0, 0

        # Prefill
        t_prefill_start = time.perf_counter()
        hidden, past_key_values = model(input_ids, use_cache=True)
        for i, block in enumerate(model.blocks):
            past_k, past_v = past_key_values[i]
            kv_cache_prefill_gqa(
                past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0
            )
            kv_cache_prefill_gqa(
                past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0
            )
        default_stream().synchronize()
        prefill_time = time.perf_counter() - t_prefill_start

        # Decode
        t_decode_start = time.perf_counter()
        logits = model.get_logits(hidden)
        last_logits = logits_to_f32(logits)[-1]
        next_token = sample_token(
            last_logits, args.temperature, args.top_k, args.top_p
        )

        generated_ids: list[int] = []
        position = len(input_ids)
        context_len = position + 1
        at_start = True  # Track if we're still at the start (for skipping special tokens)
        skip_count = 0

        # Skip special tokens at start (e.g., <|im_start|>assistant\n)
        while should_skip_token(next_token, at_start, skip_count):
            if context_len >= args.max_seq_len:
                break
            hidden = model._decode_step_fixed_cache(next_token, position, context_len)
            logits = model.get_logits(hidden)
            logits_np = logits_to_f32(logits)[-1]
            next_token = sample_token(
                logits_np, args.temperature, args.top_k, args.top_p
            )
            position += 1
            context_len += 1
            skip_count += 1

        # Check if first real token is end token
        if is_end_token(next_token):
            default_stream().synchronize()
            decode_time = time.perf_counter() - t_decode_start
            return "", prefill_time, decode_time

        # Use streaming decoder for UTF-8 safe output
        stream_decoder = StreamingDecoder(tokenizer)

        # Output first real token
        text_chunk = stream_decoder.add_token(next_token)
        if text_chunk:
            print(text_chunk, end="", flush=True)
        generated_ids.append(next_token)
        at_start = False

        while len(generated_ids) < args.max_new_tokens:
            if context_len >= args.max_seq_len:
                break

            hidden = model._decode_step_fixed_cache(next_token, position, context_len)
            logits = model.get_logits(hidden)
            logits_np = apply_repetition_penalty(
                logits_to_f32(logits)[-1], generated_ids, rep_penalty
            )
            next_token = sample_token(
                logits_np, args.temperature, args.top_k, args.top_p
            )

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
        return tokenizer.decode(generated_ids), prefill_time, decode_time

    def generate_chunked(messages: list[ChatMessage]) -> tuple[str, float, float, int, int]:
        """Generate using chunked batch decode.

        Generates tokens in chunks: full chunks use batch decode, remainder uses M=1.
        No KV snapshot/restore overhead.

        Returns: (text, prefill_time, decode_time, total_tokens, batch_chunks)
        """
        prompt = format_chat_messages(messages, model_type=model_type)
        input_ids = tokenizer.encode(prompt).ids

        if len(input_ids) >= args.max_seq_len - 10:
            return "[Error: Conversation too long. Use /clear to reset.]", 0, 0, 0, 0

        # Prefill
        t_prefill_start = time.perf_counter()
        hidden, past_key_values = model(input_ids, use_cache=True)
        for i, block in enumerate(model.blocks):
            past_k, past_v = past_key_values[i]
            kv_cache_prefill_gqa(
                past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0
            )
            kv_cache_prefill_gqa(
                past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0
            )
        default_stream().synchronize()
        prefill_time = time.perf_counter() - t_prefill_start

        # Chunked decode
        t_decode_start = time.perf_counter()
        generated_ids: list[int] = []
        stream_decoder = StreamingDecoder(tokenizer)
        position = len(input_ids)
        context_len = position + 1
        batch_chunks = 0
        at_start = True
        skip_count = 0

        # Get first token from prefill
        logits = model.get_logits(hidden)
        logits_np = logits_to_f32(logits)[-1]
        next_token = sample_token(
            logits_np, args.temperature, args.top_k, args.top_p
        )

        # Skip special tokens at start (e.g., <|im_start|>assistant\n)
        while should_skip_token(next_token, at_start, skip_count):
            if context_len >= args.max_seq_len:
                break
            hidden = model._decode_step_fixed_cache(next_token, position, context_len)
            logits = model.get_logits(hidden)
            logits_np = logits_to_f32(logits)[-1]
            next_token = sample_token(
                logits_np, args.temperature, args.top_k, args.top_p
            )
            position += 1
            context_len += 1
            skip_count += 1

        at_start = False

        while len(generated_ids) < args.max_new_tokens:
            remaining = args.max_new_tokens - len(generated_ids)
            context_len = position + len(generated_ids)

            if context_len >= args.max_seq_len:
                break

            if is_end_token(next_token):
                break

            # Decide chunk size: batch_size for full chunks, smaller for remainder
            chunk_size = min(batch_size, remaining, args.max_seq_len - context_len)

            if chunk_size >= batch_size:
                # Full chunk: use batch decode
                # First, collect chunk_size tokens using M=1 to get the token IDs
                chunk_tokens = [next_token]
                chunk_start = context_len

                # Generate first token of chunk
                generated_ids.append(next_token)
                text_chunk = stream_decoder.add_token(next_token)
                if text_chunk:
                    print(text_chunk, end="", flush=True)

                # Generate remaining tokens in chunk with M=1
                for i in range(chunk_size - 1):
                    curr_pos = chunk_start + i
                    curr_ctx = curr_pos + 1

                    hidden = model._decode_step_fixed_cache(
                        chunk_tokens[-1], curr_pos, curr_ctx
                    )
                    logits = model.get_logits(hidden)
                    logits_np = apply_repetition_penalty(
                        logits_to_f32(logits)[-1], generated_ids, rep_penalty
                    )
                    next_tok = sample_token(
                        logits_np, args.temperature, args.top_k, args.top_p
                    )

                    if is_end_token(next_tok):
                        next_token = next_tok
                        break

                    chunk_tokens.append(next_tok)
                    generated_ids.append(next_tok)
                    text_chunk = stream_decoder.add_token(next_tok)
                    if text_chunk:
                        print(text_chunk, end="", flush=True)

                # If we have a full chunk, verify with batch decode (optional, for demo)
                if len(chunk_tokens) == batch_size:
                    batch_chunks += 1

                # Get next token for next iteration
                if not is_end_token(next_tok):
                    curr_pos = chunk_start + len(chunk_tokens) - 1
                    hidden = model._decode_step_fixed_cache(
                        chunk_tokens[-1], curr_pos, curr_pos + 1
                    )
                    logits = model.get_logits(hidden)
                    logits_np = apply_repetition_penalty(
                        logits_to_f32(logits)[-1], generated_ids, rep_penalty
                    )
                    next_token = sample_token(
                        logits_np, args.temperature, args.top_k, args.top_p
                    )
                else:
                    break

            else:
                # Remainder: use M=1 for each token
                for _ in range(chunk_size):
                    if is_end_token(next_token):
                        break

                    generated_ids.append(next_token)
                    text_chunk = stream_decoder.add_token(next_token)
                    if text_chunk:
                        print(text_chunk, end="", flush=True)

                    curr_pos = position + len(generated_ids) - 1
                    curr_ctx = curr_pos + 1

                    if curr_ctx >= args.max_seq_len:
                        break

                    hidden = model._decode_step_fixed_cache(
                        next_token, curr_pos, curr_ctx
                    )
                    logits = model.get_logits(hidden)
                    logits_np = apply_repetition_penalty(
                        logits_to_f32(logits)[-1], generated_ids, rep_penalty
                    )
                    next_token = sample_token(
                        logits_np, args.temperature, args.top_k, args.top_p
                    )

                break  # Done with remainder

        default_stream().synchronize()
        decode_time = time.perf_counter() - t_decode_start

        # Flush any remaining buffered text
        remaining = stream_decoder.flush()
        if remaining:
            print(remaining, end="", flush=True)

        print()
        return (
            tokenizer.decode(generated_ids),
            prefill_time,
            decode_time,
            len(generated_ids),
            batch_chunks,
        )

    def generate_response(messages: list[ChatMessage]):
        """Dispatch to appropriate generation method."""
        if batch_size > 1:
            return generate_chunked(messages)
        else:
            return generate_m1(messages)

    # =========================================================================
    # Chat Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print(" PyGPUkit Chat")
    if batch_size > 1:
        print(f" Mode: Chunked (chunk_size={batch_size})")
    else:
        print(" Mode: Standard (M=1)")
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
        conversation.append(ChatMessage(role="user", content=user_input))

        # Build full message list with system prompt
        messages = [system_msg] + conversation

        # Generate response
        print("\nAssistant: ", end="", flush=True)

        result = generate_response(messages)

        if batch_size > 1:
            response, prefill_time, decode_time, total_tokens, accepted_batches = result
            tokens_generated = total_tokens
        else:
            response, prefill_time, decode_time = result
            # Use length of encoded response, but fallback to 0 if empty
            tokens_generated = len(tokenizer.encode(response).ids) if response else 0
            accepted_batches = 0

        # Add assistant response to history
        conversation.append(ChatMessage(role="assistant", content=response))

        # Stats
        decode_tps = tokens_generated / decode_time if decode_time > 0 else 0
        stats = (
            f"  [prefill: {prefill_time:.1f}s, "
            f"decode: {tokens_generated} tok / {decode_time:.1f}s = {decode_tps:.1f} tok/s"
        )
        if batch_size > 1:
            stats += f", chunks: {accepted_batches}"
        stats += "]"
        print(stats)

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\nUnloading model...")
    del model
    print("Done.")


if __name__ == "__main__":
    main()
