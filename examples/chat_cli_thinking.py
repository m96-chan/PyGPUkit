#!/usr/bin/env python3
"""
PyGPUkit - Thinking Model Chat CLI

A chat interface for Qwen3 Thinking models that display reasoning process.

Usage:
    python examples/chat_cli_thinking.py --model /path/to/model --tokenizer /path/to/tokenizer.json

Example (Qwen3-4B-Thinking):
    python examples/chat_cli_thinking.py \
        --model F:/LLM/Qwen3-4B-Thinking-2507 \
        --tokenizer F:/LLM/Qwen3-4B-Thinking-2507/tokenizer.json

Example with CUDA Graph (faster decode):
    python examples/chat_cli_thinking.py \
        --model F:/LLM/Qwen3-4B-Thinking-2507 \
        --cuda-graph

Commands:
    /clear  - Clear conversation history
    /think  - Toggle thinking display (default: on)
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
        return (logits_np.astype(np.uint32) << 16).view(np.float32)
    return logits_np.astype(np.float32)


def _build_byte_decoder() -> dict[str, int]:
    """Build the unicode-to-byte mapping used by tokenizers."""
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


class ThinkingParser:
    """Parser for <think>...</think> blocks in streaming output."""

    def __init__(self):
        self.in_thinking = False
        self.thinking_content = ""
        self.response_content = ""
        self.buffer = ""

    def add_text(self, text: str) -> tuple[str | None, str | None]:
        """Process text and return (thinking_chunk, response_chunk).

        Returns chunks to display for thinking and response sections.
        """
        self.buffer += text
        thinking_out = None
        response_out = None

        while True:
            if not self.in_thinking:
                # Look for <think> start
                think_start = self.buffer.find("<think>")
                if think_start != -1:
                    # Output anything before <think>
                    if think_start > 0:
                        response_out = (response_out or "") + self.buffer[:think_start]
                        self.response_content += self.buffer[:think_start]
                    self.buffer = self.buffer[think_start + 7 :]  # Skip "<think>"
                    self.in_thinking = True
                else:
                    # Check if we might have partial "<think" at end
                    for i in range(1, min(7, len(self.buffer) + 1)):
                        if self.buffer.endswith("<think>"[:i]):
                            # Keep potential partial tag
                            safe_text = self.buffer[:-i]
                            if safe_text:
                                response_out = (response_out or "") + safe_text
                                self.response_content += safe_text
                            self.buffer = self.buffer[-i:]
                            break
                    else:
                        # No partial match, output all
                        if self.buffer:
                            response_out = (response_out or "") + self.buffer
                            self.response_content += self.buffer
                        self.buffer = ""
                    break
            else:
                # Look for </think> end
                think_end = self.buffer.find("</think>")
                if think_end != -1:
                    # Output thinking content
                    if think_end > 0:
                        thinking_out = (thinking_out or "") + self.buffer[:think_end]
                        self.thinking_content += self.buffer[:think_end]
                    self.buffer = self.buffer[think_end + 8 :]  # Skip "</think>"
                    self.in_thinking = False
                else:
                    # Check for partial "</think" at end
                    for i in range(1, min(8, len(self.buffer) + 1)):
                        if self.buffer.endswith("</think>"[:i]):
                            safe_text = self.buffer[:-i]
                            if safe_text:
                                thinking_out = (thinking_out or "") + safe_text
                                self.thinking_content += safe_text
                            self.buffer = self.buffer[-i:]
                            break
                    else:
                        if self.buffer:
                            thinking_out = (thinking_out or "") + self.buffer
                            self.thinking_content += self.buffer
                        self.buffer = ""
                    break

        return thinking_out, response_out

    def flush(self) -> tuple[str | None, str | None]:
        """Flush remaining buffer."""
        if self.buffer:
            if self.in_thinking:
                self.thinking_content += self.buffer
                result = (self.buffer, None)
            else:
                self.response_content += self.buffer
                result = (None, self.buffer)
            self.buffer = ""
            return result
        return None, None

    def reset(self):
        self.in_thinking = False
        self.thinking_content = ""
        self.response_content = ""
        self.buffer = ""


def format_qwen3_thinking_chat(messages: list[dict]) -> str:
    """Format messages for Qwen3 Thinking model.

    Qwen3 Thinking uses ChatML format with thinking enabled.
    """
    result = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    result += "<|im_start|>assistant\n"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="PyGPUkit Thinking Model Chat CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory or model.safetensors.index.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (default: auto-detect in model dir)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens per response (default: 4096, thinking needs more)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6, recommended for thinking)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20, recommended for thinking)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant. Think step by step.",
        help="System prompt",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0 = disabled)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--hide-thinking",
        action="store_true",
        help="Hide thinking process (only show final answer)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA Graph for faster decode (reduces kernel launch overhead)",
    )
    args = parser.parse_args()

    # Auto-detect tokenizer path
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        from pathlib import Path

        model_path = Path(args.model)
        if model_path.is_dir():
            tokenizer_path = str(model_path / "tokenizer.json")
        else:
            tokenizer_path = str(model_path.parent / "tokenizer.json")

    # Auto-detect model file
    model_path = args.model
    from pathlib import Path

    mp = Path(model_path)
    if mp.is_dir():
        # Look for index.json or single safetensors
        index_file = mp / "model.safetensors.index.json"
        if index_file.exists():
            model_path = str(index_file)
        else:
            st_files = list(mp.glob("*.safetensors"))
            if st_files:
                model_path = str(st_files[0])

    # Lazy imports for faster --help
    print("Loading PyGPUkit...")
    from tokenizers import Tokenizer

    from pygpukit.core import default_stream, from_numpy
    from pygpukit.llm import (
        DecodeM1Graph,
        detect_model_spec,
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
    print(f"\nLoading Thinking model from: {model_path}")
    print(f"  dtype: {args.dtype}")
    t0 = time.perf_counter()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(model_path, dtype=args.dtype, spec=spec)

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
    # Initialize Decode Buffers
    # =========================================================================
    use_qk_norm = model.spec is not None and model.spec.use_qk_norm
    lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
    vocab_size = lm_head.shape[0]

    decode_buffers = DecodeBuffers.allocate(
        config, dtype=args.dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
    )

    # Precompute RoPE frequencies (needed for non-graph path)
    if config.use_rope and not args.cuda_graph:
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

    default_stream().synchronize()
    print("Ready!")

    # =========================================================================
    # Chat State
    # =========================================================================
    conversation: list[dict] = []
    system_msg = {"role": "system", "content": args.system}
    show_thinking = not args.hide_thinking

    # Get special tokens
    eos_token_id = tokenizer.token_to_id("<|im_end|>")
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")

    # Tokens to skip at start
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    assistant_ids = set(tokenizer.encode("assistant").ids)

    def is_end_token(token_id: int) -> bool:
        return token_id == eos_token_id

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
    def generate(messages: list[dict]) -> tuple[str, str, float, float, int]:
        """Generate response with thinking.

        Returns: (thinking, response, prefill_time, decode_time, tokens)
        """
        prompt = format_qwen3_thinking_chat(messages)
        input_ids = tokenizer.encode(prompt).ids

        if len(input_ids) >= args.max_seq_len - 10:
            return "", "[Error: Conversation too long. Use /clear to reset.]", 0, 0, 0

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

        # Skip <|im_start|>assistant\n at start
        skip_count = 0
        max_skip = 5
        while skip_count < max_skip:
            if next_token == im_start_id or next_token in assistant_ids:
                skip_count += 1
                logits_np = decode_one_token(next_token, position, context_len)
                next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
                position += 1
                context_len += 1
            else:
                # Check for newline after assistant
                token_str = tokenizer.id_to_token(next_token)
                if token_str and token_str.strip() == "":
                    skip_count += 1
                    logits_np = decode_one_token(next_token, position, context_len)
                    next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)
                    position += 1
                    context_len += 1
                else:
                    break

        if is_end_token(next_token):
            default_stream().synchronize()
            decode_time = time.perf_counter() - t_decode_start
            return "", "", prefill_time, decode_time, 0

        # Streaming decode with thinking parser
        stream_decoder = StreamingDecoder(tokenizer)
        thinking_parser = ThinkingParser()

        # Display mode
        in_thinking_display = False

        while len(generated_ids) < args.max_new_tokens:
            if context_len >= args.max_seq_len:
                break

            if is_end_token(next_token):
                break

            generated_ids.append(next_token)
            position += 1
            context_len += 1

            # Decode token to text
            text_chunk = stream_decoder.add_token(next_token)
            if text_chunk:
                thinking_chunk, response_chunk = thinking_parser.add_text(text_chunk)

                # Display thinking
                if thinking_chunk and show_thinking:
                    if not in_thinking_display:
                        print("\n[Thinking]", flush=True)
                        in_thinking_display = True
                    print(f"\033[90m{thinking_chunk}\033[0m", end="", flush=True)

                # Display response
                if response_chunk:
                    if in_thinking_display:
                        print("\n[Answer]", flush=True)
                        in_thinking_display = False
                    print(response_chunk, end="", flush=True)

            # Get next token
            logits_np = decode_one_token(next_token, position - 1, context_len - 1)
            logits_np = apply_repetition_penalty(logits_np, generated_ids, args.repetition_penalty)
            next_token = sample_token(logits_np, args.temperature, args.top_k, args.top_p)

        # Flush remaining
        remaining = stream_decoder.flush()
        if remaining:
            thinking_chunk, response_chunk = thinking_parser.add_text(remaining)
            if thinking_chunk and show_thinking:
                print(f"\033[90m{thinking_chunk}\033[0m", end="", flush=True)
            if response_chunk:
                print(response_chunk, end="", flush=True)

        thinking_chunk, response_chunk = thinking_parser.flush()
        if thinking_chunk and show_thinking:
            print(f"\033[90m{thinking_chunk}\033[0m", end="", flush=True)
        if response_chunk:
            print(response_chunk, end="", flush=True)

        default_stream().synchronize()
        decode_time = time.perf_counter() - t_decode_start

        print()
        return (
            thinking_parser.thinking_content,
            thinking_parser.response_content,
            prefill_time,
            decode_time,
            len(generated_ids),
        )

    # =========================================================================
    # Chat Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print(" PyGPUkit Thinking Chat")
    print(f" Model: {spec.name if spec else 'unknown'}")
    print(f" Thinking display: {'ON' if show_thinking else 'OFF'}")
    print(f" CUDA Graph: {'ON' if use_cuda_graph else 'OFF'}")
    print(" Commands: /clear (reset), /think (toggle), /quit (exit)")
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
        elif user_input.lower() == "/think":
            show_thinking = not show_thinking
            print(f"[Thinking display: {'ON' if show_thinking else 'OFF'}]")
            continue

        # Add user message
        conversation.append({"role": "user", "content": user_input})

        # Build full message list with system prompt
        messages = [system_msg] + conversation

        # Generate response
        print("\nAssistant: ", end="", flush=True)

        thinking, response, prefill_time, decode_time, tokens_generated = generate(messages)

        # Add response to history (without thinking)
        conversation.append({"role": "assistant", "content": response})

        # Stats
        decode_tps = tokens_generated / decode_time if decode_time > 0 else 0
        thinking_tokens = len(tokenizer.encode(thinking).ids) if thinking else 0
        response_tokens = len(tokenizer.encode(response).ids) if response else 0
        print(
            f"  [prefill: {prefill_time:.1f}s, "
            f"decode: {tokens_generated} tok / {decode_time:.1f}s = {decode_tps:.1f} tok/s, "
            f"think: {thinking_tokens}, answer: {response_tokens}]"
        )

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\nUnloading model...")
    del model
    print("Done.")


if __name__ == "__main__":
    main()
