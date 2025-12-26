#!/usr/bin/env python3
"""
PyGPUkit - Simple CLI Chat

A minimal turn-based chat interface using the Strategy pattern:
- DecodeM1: Single token decode (baseline)
- DecodeBatch: Batch decode for higher throughput

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


def _build_byte_decoder() -> dict[str, int]:
    """Build the unicode-to-byte mapping used by GPT-2/Qwen style tokenizers.

    These tokenizers encode raw bytes as unicode characters to avoid control chars.
    This function builds the reverse mapping to convert token strings back to bytes.
    """
    # Characters that map directly to their byte values
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # Other bytes are mapped to higher unicode code points
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


# Global byte decoder for GPT-2/Qwen style tokenizers
_BYTE_DECODER = _build_byte_decoder()


def _token_str_to_bytes(token_str: str) -> bytes:
    """Convert a GPT-2/Qwen style token string to raw bytes."""
    result = []
    for char in token_str:
        if char in _BYTE_DECODER:
            result.append(_BYTE_DECODER[char])
        else:
            # Fallback: encode as UTF-8
            result.extend(char.encode("utf-8"))
    return bytes(result)


class StreamingDecoder:
    """Streaming decoder for UTF-8 safe output.

    Bypasses tokenizer.decode() and manually converts token strings to bytes,
    then buffers incomplete UTF-8 sequences until they are complete.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending_bytes = b""  # Incomplete UTF-8 bytes waiting for more
        self._cache: dict[int, bytes] = {}  # Cache: token_id -> bytes

    def _get_token_bytes(self, token_id: int) -> bytes:
        """Get bytes for a token ID, with caching."""
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
        """Add a token and return the new text portion.

        Returns:
            New complete UTF-8 text from this token.
        """
        new_bytes = self._get_token_bytes(token_id)
        if not new_bytes:
            return ""

        all_bytes = self.pending_bytes + new_bytes

        # Find the longest valid UTF-8 prefix
        valid_end = 0
        i = 0
        while i < len(all_bytes):
            byte = all_bytes[i]
            if byte < 0x80:
                # ASCII
                valid_end = i + 1
                i += 1
            elif byte < 0xC0:
                # Orphan continuation byte - skip it
                i += 1
            elif byte < 0xE0:
                # 2-byte sequence
                if i + 1 < len(all_bytes) and 0x80 <= all_bytes[i + 1] < 0xC0:
                    valid_end = i + 2
                    i += 2
                else:
                    break  # Incomplete - wait for more bytes
            elif byte < 0xF0:
                # 3-byte sequence
                if (
                    i + 2 < len(all_bytes)
                    and 0x80 <= all_bytes[i + 1] < 0xC0
                    and 0x80 <= all_bytes[i + 2] < 0xC0
                ):
                    valid_end = i + 3
                    i += 3
                else:
                    break  # Incomplete - wait for more bytes
            elif byte < 0xF8:
                # 4-byte sequence
                if (
                    i + 3 < len(all_bytes)
                    and 0x80 <= all_bytes[i + 1] < 0xC0
                    and 0x80 <= all_bytes[i + 2] < 0xC0
                    and 0x80 <= all_bytes[i + 3] < 0xC0
                ):
                    valid_end = i + 4
                    i += 4
                else:
                    break  # Incomplete - wait for more bytes
            else:
                # Invalid start byte - skip it
                i += 1

        # Output complete bytes, keep incomplete ones pending
        complete_bytes = all_bytes[:valid_end]
        self.pending_bytes = all_bytes[valid_end:]

        if complete_bytes:
            return complete_bytes.decode("utf-8", errors="replace")
        return ""

    def flush(self) -> str:
        """Flush any remaining buffered bytes."""
        if self.pending_bytes:
            text = self.pending_bytes.decode("utf-8", errors="replace")
            self.pending_bytes = b""
            return text
        return ""

    def reset(self):
        """Reset the decoder state."""
        self.pending_bytes = b""


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
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA Graph for faster decode (reduces kernel launch overhead)",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="[EXPERIMENTAL] Enable self-speculative decoding (uses argmax, may cause repetition)",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=4,
        help="Number of draft tokens per speculation round (default: 4)",
    )
    parser.add_argument(
        "--draft-layers",
        type=int,
        default=8,
        help="Number of early layers to use as draft model (default: 8)",
    )
    args = parser.parse_args()

    # Lazy imports for faster --help
    print("Loading PyGPUkit...")
    from tokenizers import Tokenizer

    from pygpukit.core import default_stream, from_numpy
    from pygpukit.llm import (
        ChatMessage,
        DecodeM1,
        DecodeM1Graph,
        DecodeSpeculative,
        detect_model_spec,
        format_chat_messages,
        load_model_from_safetensors,
        load_safetensors,
    )
    from pygpukit.llm.buffers import DecodeBuffers
    from pygpukit.llm.layers import precompute_freqs_cis
    from pygpukit.llm.sampling import sample_token
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

    # =========================================================================
    # Initialize Decode Strategy
    # =========================================================================
    use_qk_norm = model.spec is not None and model.spec.use_qk_norm
    lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
    vocab_size = lm_head.shape[0]

    decode_buffers = DecodeBuffers.allocate(
        config, dtype=args.dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
    )

    # Initialize decode strategy
    use_cuda_graph = args.cuda_graph
    use_speculative = args.speculative
    m1_graph = None
    speculative_strategy = None

    if use_speculative:
        # Use DecodeSpeculative for self-speculative decoding
        print("\nInitializing Self-Speculative Decode...")
        print(f"  draft_tokens={args.draft_tokens}, draft_layers={args.draft_layers}")
        print("  WARNING: Uses argmax (greedy) decoding - may produce repetitive output")
        print("  For production use, prefer --cuda-graph instead")
        speculative_strategy = DecodeSpeculative(
            max_draft_tokens=args.draft_tokens,
            draft_layers=args.draft_layers,
        )
        speculative_strategy.bind(model)
        m1 = None  # Not used in speculative mode
    elif use_cuda_graph:
        # Use DecodeM1Graph for CUDA Graph mode
        print("\nInitializing CUDA Graph...")
        m1_graph = DecodeM1Graph()
        m1_graph.bind(model)
        m1_graph.init_graph(max_seq_len=args.max_seq_len)
        print(f"  CUDA Graph ready (max_seq_len={args.max_seq_len})")
        m1 = None  # Not used in graph mode
    else:
        # Use DecodeM1 for non-graph mode
        m1 = DecodeM1()
        m1.bind(model)

    if not use_cuda_graph and config.use_rope:
        # Precompute RoPE frequencies for non-CUDA-Graph path
        cos_np, sin_np = precompute_freqs_cis(config.head_dim, args.max_seq_len, config.rope_theta)
        if args.dtype == "float16":
            model._rope_cos_gpu = from_numpy(cos_np.astype(np.float16))
            model._rope_sin_gpu = from_numpy(sin_np.astype(np.float16))
        elif args.dtype == "bfloat16":
            # Convert float32 -> bfloat16 via bit manipulation
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

    def decode_one_token(token_id: int, position: int, context_len: int):
        """Decode one token, using CUDA Graph if available.

        Returns:
            Logits array [1, vocab_size] or [vocab_size].
        """
        if use_cuda_graph and m1_graph is not None:
            return m1_graph.step_graph(token_id, position, context_len)
        else:
            # m1.step() now returns logits directly [1, vocab_size]
            return m1.step(token_id, position, context_len, decode_buffers)

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
        at_start = True  # Track if we're still at the start (for skipping special tokens)
        skip_count = 0

        # Skip special tokens at start (e.g., <|im_start|>assistant\n)
        while should_skip_token(next_token, at_start, skip_count):
            if context_len >= args.max_seq_len:
                break
            logits = decode_one_token(next_token, position, context_len)
            logits_np = logits_to_f32(logits)[-1]
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
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

            logits = decode_one_token(next_token, position, context_len)
            logits_np = apply_repetition_penalty(
                logits_to_f32(logits)[-1], generated_ids, rep_penalty
            )
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
            kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
            kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)
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
        next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

        # Skip special tokens at start (e.g., <|im_start|>assistant\n)
        while should_skip_token(next_token, at_start, skip_count):
            if context_len >= args.max_seq_len:
                break
            logits = decode_one_token(next_token, position, context_len)
            logits_np = logits_to_f32(logits)[-1]
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
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

                    logits = decode_one_token(chunk_tokens[-1], curr_pos, curr_ctx)
                    logits_np = apply_repetition_penalty(
                        logits_to_f32(logits)[-1], generated_ids, rep_penalty
                    )
                    next_tok = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

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
                    logits = decode_one_token(chunk_tokens[-1], curr_pos, curr_pos + 1)
                    logits_np = apply_repetition_penalty(
                        logits_to_f32(logits)[-1], generated_ids, rep_penalty
                    )
                    next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
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

                    logits = decode_one_token(next_token, curr_pos, curr_ctx)
                    logits_np = apply_repetition_penalty(
                        logits_to_f32(logits)[-1], generated_ids, rep_penalty
                    )
                    next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

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

    def generate_speculative(
        messages: list[ChatMessage],
    ) -> tuple[str, float, float, int, int, float]:
        """Generate using self-speculative decoding.

        Uses early layers as draft model, verifies with full model in batch.
        Uses KV snapshot/restore for correctness.

        Returns: (text, prefill_time, decode_time, total_tokens, total_drafts, accept_rate)
        """
        prompt = format_chat_messages(messages, model_type=model_type)
        input_ids = tokenizer.encode(prompt).ids

        if len(input_ids) >= args.max_seq_len - 10:
            return "[Error: Conversation too long. Use /clear to reset.]", 0, 0, 0, 0, 0.0

        # Prefill
        t_prefill_start = time.perf_counter()
        hidden, past_key_values = model(input_ids, use_cache=True)
        for i, block in enumerate(model.blocks):
            past_k, past_v = past_key_values[i]
            kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
            kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)
        default_stream().synchronize()
        prefill_time = time.perf_counter() - t_prefill_start

        # Self-speculative decode
        t_decode_start = time.perf_counter()
        generated_ids: list[int] = []
        stream_decoder = StreamingDecoder(tokenizer)
        position = len(input_ids)
        context_len = position + 1
        at_start = True
        skip_count = 0

        # Stats
        total_drafts = 0
        total_accepted = 0

        # Get first token from prefill
        logits = model.get_logits(hidden)
        logits_np = logits_to_f32(logits)[-1]
        next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

        # Skip special tokens at start (e.g., <|im_start|>assistant\n)
        while should_skip_token(next_token, at_start, skip_count):
            if context_len >= args.max_seq_len:
                break
            # Use fixed cache decode for skipping
            hidden = model._decode_step_fixed_cache(next_token, position, context_len)
            logits = model.get_logits(hidden)
            logits_np = logits_to_f32(logits)[-1]
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
            position += 1
            context_len += 1
            skip_count += 1

        at_start = False

        # Check if first real token is end token
        if is_end_token(next_token):
            default_stream().synchronize()
            decode_time = time.perf_counter() - t_decode_start
            return "", prefill_time, decode_time, 0, 0, 0.0

        # Output first real token (step_speculative takes this as input and returns NEXT tokens)
        text_chunk = stream_decoder.add_token(next_token)
        if text_chunk:
            print(text_chunk, end="", flush=True)
        generated_ids.append(next_token)

        # Main speculative decode loop
        while len(generated_ids) < args.max_new_tokens:
            if context_len >= args.max_seq_len:
                break

            if is_end_token(next_token):
                break

            # Run speculative decode step (uses KV snapshot/restore)
            accepted_tokens, new_position, stats = speculative_strategy.step_speculative(
                next_token, position, context_len
            )

            # Track stats
            total_drafts += stats["draft_count"]
            total_accepted += stats["accepted_count"]

            # Stream out accepted tokens
            for tok in accepted_tokens:
                if is_end_token(tok):
                    break
                generated_ids.append(tok)
                text_chunk = stream_decoder.add_token(tok)
                if text_chunk:
                    print(text_chunk, end="", flush=True)

            # Check if we hit end token
            if any(is_end_token(tok) for tok in accepted_tokens):
                break

            # Update position for next iteration
            position = new_position
            context_len = position + 1

            # Get next token for next speculation round
            if accepted_tokens:
                next_token = accepted_tokens[-1]
            else:
                break

        # Flush any remaining buffered text
        remaining = stream_decoder.flush()
        if remaining:
            print(remaining, end="", flush=True)

        default_stream().synchronize()
        decode_time = time.perf_counter() - t_decode_start

        # Calculate acceptance rate
        accept_rate = total_accepted / total_drafts if total_drafts > 0 else 0.0

        print()
        return (
            tokenizer.decode(generated_ids),
            prefill_time,
            decode_time,
            len(generated_ids),
            total_drafts,
            accept_rate,
        )

    def generate_response(messages: list[ChatMessage]):
        """Dispatch to appropriate generation method."""
        if use_speculative:
            return generate_speculative(messages)
        elif batch_size > 1:
            return generate_chunked(messages)
        else:
            return generate_m1(messages)

    # =========================================================================
    # Chat Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print(" PyGPUkit Chat")
    if use_speculative:
        mode_str = (
            f"Self-Speculative (draft_tokens={args.draft_tokens}, draft_layers={args.draft_layers})"
        )
    elif batch_size > 1:
        mode_str = f"Chunked (chunk_size={batch_size})"
    elif use_cuda_graph:
        mode_str = "M=1 + CUDA Graph"
    else:
        mode_str = "M=1 (standard)"
    print(f" Mode: {mode_str}")
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

        if use_speculative:
            response, prefill_time, decode_time, total_tokens, total_drafts, accept_rate = result
            tokens_generated = total_tokens
        elif batch_size > 1:
            response, prefill_time, decode_time, total_tokens, accepted_batches = result
            tokens_generated = total_tokens
        else:
            response, prefill_time, decode_time = result
            # Use length of encoded response, but fallback to 0 if empty
            tokens_generated = len(tokenizer.encode(response).ids) if response else 0

        # Add assistant response to history
        conversation.append(ChatMessage(role="assistant", content=response))

        # Stats
        decode_tps = tokens_generated / decode_time if decode_time > 0 else 0
        stats = (
            f"  [prefill: {prefill_time:.1f}s, "
            f"decode: {tokens_generated} tok / {decode_time:.1f}s = {decode_tps:.1f} tok/s"
        )
        if use_speculative:
            stats += f", drafts: {total_drafts}, accept: {accept_rate:.1%}"
        elif batch_size > 1:
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
