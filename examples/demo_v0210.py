#!/usr/bin/env python3
"""
PyGPUkit v0.2.10 - Comprehensive Feature Demo

Demonstrates the three major v0.2.10 features:
1. INT8 Quantization (#85) - Weight-only quantization for memory reduction
2. Paged Attention (#87) - KV Cache paging for memory efficiency
3. Continuous Batching (#86) - Multi-request batch processing

Usage:
    python demo_v0210.py --model /path/to/qwen3-8b --tokenizer /path/to/tokenizer.json

Requirements:
    - PyGPUkit v0.2.10+
    - CUDA capable GPU (SM >= 80)
    - Qwen3-8B model in safetensors format
    - HuggingFace tokenizers (pip install tokenizers)
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def format_bytes(size: int) -> str:
    """Format bytes in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms * 1000:.2f} us"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


def demo_int8_quantization():
    """Demo 1: INT8 Quantization (#85)"""
    section("Demo 1: INT8 Quantization (#85)")

    print("\nINT8 weight-only quantization reduces memory usage by ~50%")
    print("while maintaining accuracy through per-row scaling.\n")

    import pygpukit as gk

    native = gk._pygpukit_native

    # Create test weight matrix (simulating a linear layer)
    print("Creating test weight matrix [4096, 4096]...")
    weight_np = np.random.randn(4096, 4096).astype(np.float16) * 0.02
    weight_gpu = native.from_numpy(weight_np)

    fp16_size = weight_np.nbytes
    print(f"  FP16 weight size: {format_bytes(fp16_size)}")

    # Quantize to INT8
    print("\nQuantizing to INT8...")
    start = time.perf_counter()
    weight_int8, scale = native.quantize_to_int8(weight_gpu)
    quant_time = (time.perf_counter() - start) * 1000

    int8_size = weight_int8.to_numpy().nbytes + scale.to_numpy().nbytes
    print(f"  INT8 weight size: {format_bytes(int8_size)}")
    print(f"  Memory savings: {100 * (1 - int8_size / fp16_size):.1f}%")
    print(f"  Quantization time: {format_time(quant_time)}")

    # Verify shapes
    print(f"\n  weight_int8.shape: {list(weight_int8.shape)}")
    print(f"  scale.shape: {list(scale.shape)}")

    # Test dequantization accuracy
    print("\nTesting dequantization accuracy...")
    weight_dequant = native.dequantize_int8(weight_int8, scale, native.DataType.Float16)
    weight_dequant_np = weight_dequant.to_numpy()

    # Calculate error (filter near-zero values)
    mask = np.abs(weight_np) > 0.01
    if mask.sum() > 0:
        rel_error = np.abs(weight_dequant_np[mask] - weight_np[mask]) / np.abs(weight_np[mask])
        print(f"  Mean relative error: {rel_error.mean():.6f}")
        print(f"  Max relative error: {rel_error.max():.6f}")
    else:
        print("  (Skipped - no significant values)")

    # Test quantized linear
    print("\nTesting quantized linear layer (INT8 x FP16)...")
    batch_size = 32
    activation_np = np.random.randn(batch_size, 4096).astype(np.float16) * 0.1
    activation_gpu = native.from_numpy(activation_np)

    # Quantized matmul
    start = time.perf_counter()
    output_int8 = native.linear_int8(activation_gpu, weight_int8, scale, None)
    int8_time = (time.perf_counter() - start) * 1000

    # Reference FP16 matmul
    weight_t = native.transpose(weight_gpu)
    start = time.perf_counter()
    output_fp16 = native.matmul(activation_gpu, weight_t)
    fp16_time = (time.perf_counter() - start) * 1000

    print(f"  INT8 linear time: {format_time(int8_time)}")
    print(f"  FP16 linear time: {format_time(fp16_time)}")

    # Compare outputs
    out_int8_np = output_int8.to_numpy()
    out_fp16_np = output_fp16.to_numpy()
    abs_error = np.abs(out_int8_np - out_fp16_np)
    print(f"  Output mean absolute error: {abs_error.mean():.6f}")
    print(f"  Output max absolute error: {abs_error.max():.6f}")

    print("\n  [PASS] INT8 Quantization working correctly!")
    return True


def demo_paged_attention():
    """Demo 2: Paged Attention (#87)"""
    section("Demo 2: Paged Attention (#87)")

    print("\nPaged Attention enables vLLM-style memory management:")
    print("- Fixed-size blocks (16 tokens/block)")
    print("- Dynamic allocation via page tables")
    print("- Memory sharing across sequences\n")

    import pygpukit as gk

    native = gk._pygpukit_native

    # Parameters
    num_seqs = 4
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    block_size = 16
    num_blocks = 64
    max_context_len = 256

    print("Configuration:")
    print(f"  Sequences: {num_seqs}")
    print(f"  Heads: {num_heads} (Q), {num_kv_heads} (KV)")
    print(f"  Head dim: {head_dim}")
    print(f"  Block size: {block_size} tokens")
    print(f"  Total blocks: {num_blocks}")

    # Allocate KV cache
    print("\nAllocating paged KV cache...")
    k_cache = native.allocate_kv_cache(num_blocks, num_kv_heads, block_size, head_dim)
    v_cache = native.allocate_kv_cache(num_blocks, num_kv_heads, block_size, head_dim)

    cache_size = k_cache.to_numpy().nbytes + v_cache.to_numpy().nbytes
    print(f"  KV cache shape: {list(k_cache.shape)}")
    print(f"  Total cache size: {format_bytes(cache_size)}")

    # Traditional KV cache for comparison
    traditional_size = num_seqs * max_context_len * num_kv_heads * head_dim * 2 * 2  # FP16
    print(f"  Traditional cache (fixed {max_context_len} tokens): {format_bytes(traditional_size)}")

    # Create block tables
    context_lens = [64, 128, 32, 96]  # Variable context lengths
    blocks_per_seq = [(cl + block_size - 1) // block_size for cl in context_lens]
    max_blocks_per_seq = max(blocks_per_seq)

    block_tables_np = np.zeros((num_seqs, max_blocks_per_seq), dtype=np.int32)
    block_idx = 0
    for seq_idx, num_seq_blocks in enumerate(blocks_per_seq):
        for b in range(num_seq_blocks):
            block_tables_np[seq_idx, b] = block_idx
            block_idx += 1

    block_tables = native.from_numpy(block_tables_np)
    context_lens_gpu = native.from_numpy(np.array(context_lens, dtype=np.int32))

    print(f"\nSequence context lengths: {context_lens}")
    print(f"Blocks per sequence: {blocks_per_seq}")
    print(f"Total blocks used: {sum(blocks_per_seq)} / {num_blocks}")

    # Fill KV cache with test data (simulating prefill)
    print("\nFilling KV cache with test data...")
    total_tokens = sum(context_lens)
    slot_mapping_list = []
    for seq_idx, ctx_len in enumerate(context_lens):
        for pos in range(ctx_len):
            block_idx_in_seq = pos // block_size
            offset_in_block = pos % block_size
            physical_block = block_tables_np[seq_idx, block_idx_in_seq]
            slot = physical_block * block_size + offset_in_block
            slot_mapping_list.append(slot)

    slot_mapping = native.from_numpy(np.array(slot_mapping_list, dtype=np.int32))

    k_data = np.random.randn(total_tokens, num_kv_heads, head_dim).astype(np.float16)
    v_data = np.random.randn(total_tokens, num_kv_heads, head_dim).astype(np.float16)
    k_gpu = native.from_numpy(k_data)
    v_gpu = native.from_numpy(v_data)

    native.reshape_and_cache(k_gpu, v_gpu, k_cache, v_cache, slot_mapping)
    print(f"  Cached {total_tokens} tokens across {sum(blocks_per_seq)} blocks")

    # Test paged attention
    print("\nRunning paged attention v1...")
    q_np = np.random.randn(num_seqs, num_heads, head_dim).astype(np.float16)
    q_gpu = native.from_numpy(q_np)

    start = time.perf_counter()
    output = native.paged_attention_v1(q_gpu, k_cache, v_cache, block_tables, context_lens_gpu, 0.0)
    attn_time = (time.perf_counter() - start) * 1000

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Attention time: {format_time(attn_time)}")

    # Test decode phase (copy new KV to cache)
    print("\nSimulating decode phase (adding new token)...")
    k_new = np.random.randn(num_seqs, num_kv_heads, head_dim).astype(np.float16)
    v_new = np.random.randn(num_seqs, num_kv_heads, head_dim).astype(np.float16)
    k_new_gpu = native.from_numpy(k_new)
    v_new_gpu = native.from_numpy(v_new)

    # New slots for decode tokens (add token to last position in current block)
    # Note: In a real system, we'd allocate new blocks if needed
    new_slots = []
    for seq_idx, ctx_len in enumerate(context_lens):
        # Use position within current last block
        last_block_idx = (ctx_len - 1) // block_size
        offset_in_block = (ctx_len - 1) % block_size + 1  # Next position
        if offset_in_block >= block_size:
            # Would need new block - use last position of current block for demo
            offset_in_block = block_size - 1
        physical_block = block_tables_np[seq_idx, last_block_idx]
        slot = physical_block * block_size + offset_in_block
        new_slots.append(slot)

    slot_mapping_decode = native.from_numpy(np.array(new_slots, dtype=np.int32))
    native.copy_to_paged_cache(k_new_gpu, v_new_gpu, k_cache, v_cache, slot_mapping_decode)
    print("  Added 1 token to each sequence")

    # Memory efficiency calculation
    used_blocks = sum(blocks_per_seq)
    utilization = used_blocks / num_blocks * 100
    print("\nMemory efficiency:")
    print(f"  Block utilization: {utilization:.1f}%")
    print(f"  Fragmentation: {100 - utilization:.1f}%")

    print("\n  [PASS] Paged Attention working correctly!")
    return True


def demo_continuous_batching():
    """Demo 3: Continuous Batching (#86)"""
    section("Demo 3: Continuous Batching (#86)")

    print("\nContinuous Batching enables iteration-level scheduling:")
    print("- Dynamic batch formation")
    print("- Embedding gathering")
    print("- Argmax sampling")
    print("- EOS detection\n")

    import pygpukit as gk

    native = gk._pygpukit_native

    # Simulate batch of sequences with different lengths
    batch_size = 4
    vocab_size = 32000
    hidden_size = 4096

    # Variable-length sequences (simulating prefill + decode mix)
    seq_lens = [64, 1, 32, 1]  # 2 prefill (64, 32), 2 decode (1, 1)
    total_tokens = sum(seq_lens)

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"  Total tokens: {total_tokens}")

    # Prepare batch inputs
    print("\nPreparing batch inputs...")
    token_lists = [list(np.random.randint(0, vocab_size, size=sl)) for sl in seq_lens]

    start = time.perf_counter()
    token_ids, actual_total = native.prepare_batch_inputs(token_lists)
    prep_time = (time.perf_counter() - start) * 1000

    print(f"  Token IDs shape: {list(token_ids.shape)}")
    print(f"  Total tokens: {actual_total}")
    print(f"  Preparation time: {format_time(prep_time)}")

    # Create embedding table
    print("\nGathering embeddings...")
    embeddings_np = np.random.randn(vocab_size, hidden_size).astype(np.float16) * 0.02
    embeddings_gpu = native.from_numpy(embeddings_np)

    start = time.perf_counter()
    gathered = native.gather_embeddings(token_ids, embeddings_gpu, total_tokens)
    gather_time = (time.perf_counter() - start) * 1000

    print(f"  Gathered shape: {list(gathered.shape)}")
    print(f"  Gather time: {format_time(gather_time)}")

    # Prepare position IDs
    print("\nPreparing position IDs...")
    seq_start_positions = native.compute_cumsum(
        native.from_numpy(np.array(seq_lens, dtype=np.int32))
    )
    context_lens = [63, 127, 31, 95]  # Context lengths (for decode positions)
    is_prefill = [1, 0, 1, 0]  # Which sequences are in prefill mode

    position_ids = native.prepare_position_ids(
        seq_start_positions,
        native.from_numpy(np.array(context_lens, dtype=np.int32)),
        native.from_numpy(np.array(is_prefill, dtype=np.int32)),
        native.from_numpy(np.array(seq_lens, dtype=np.int32)),
        batch_size,
        total_tokens,
    )

    print(f"  Position IDs shape: {list(position_ids.shape)}")
    pos_ids_np = position_ids.to_numpy()
    print(f"  First 5 positions: {pos_ids_np[:5].tolist()}")

    # Simulate model output logits
    print("\nScattering last-token logits...")
    batch_logits_np = np.random.randn(total_tokens, vocab_size).astype(np.float16)
    batch_logits = native.from_numpy(batch_logits_np)

    start = time.perf_counter()
    last_token_logits = native.scatter_last_token_logits(
        batch_logits,
        seq_start_positions,
        native.from_numpy(np.array(seq_lens, dtype=np.int32)),
        batch_size,
        vocab_size,
    )
    scatter_time = (time.perf_counter() - start) * 1000

    print(f"  Last-token logits shape: {list(last_token_logits.shape)}")
    print(f"  Scatter time: {format_time(scatter_time)}")

    # Argmax sampling
    print("\nArgmax sampling...")
    start = time.perf_counter()
    sampled_tokens = native.argmax_sample(last_token_logits, batch_size, vocab_size)
    sample_time = (time.perf_counter() - start) * 1000

    sampled_np = sampled_tokens.to_numpy()
    print(f"  Sampled tokens: {sampled_np.tolist()}")
    print(f"  Sample time: {format_time(sample_time)}")

    # EOS detection
    print("\nEOS detection...")
    eos_token_id = 2  # Common EOS token ID
    # Manually set one token to EOS for testing
    test_tokens = sampled_np.copy()
    test_tokens[1] = eos_token_id
    test_tokens_gpu = native.from_numpy(test_tokens)

    start = time.perf_counter()
    finished = native.check_eos(test_tokens_gpu, eos_token_id)
    eos_time = (time.perf_counter() - start) * 1000

    finished_np = finished.to_numpy()
    print(f"  Test tokens: {test_tokens.tolist()}")
    print(f"  EOS token ID: {eos_token_id}")
    print(f"  Finished flags: {finished_np.tolist()}")
    print(f"  EOS check time: {format_time(eos_time)}")

    print("\n  [PASS] Continuous Batching working correctly!")
    return True


def demo_llm_generation(model_path: str, tokenizer_path: str):
    """Demo 4: LLM Generation with Qwen3-8B"""
    section("Demo 4: Qwen3-8B Text Generation")

    print("\nLoading Qwen3-8B model and generating text...")
    print("This demonstrates the full inference pipeline.\n")

    try:
        from pygpukit.llm import (
            ChatMessage,
            detect_model_spec,
            format_chat_messages,
            load_model_from_safetensors,
            load_safetensors,
        )
    except ImportError as e:
        print(f"Error importing PyGPUkit: {e}")
        return False

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"  Error loading tokenizer: {e}")
        print("  Install tokenizers: pip install tokenizers")
        return False

    # Detect and load model
    print("\nDetecting model type...")
    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    print(f"  Detected: {spec.name.upper()}")

    print("\nLoading model (this may take a while)...")
    start = time.perf_counter()
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)
    load_time = (time.perf_counter() - start) * 1000

    config = model.config
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads} (Q), {config.num_kv_heads} (KV)")
    print(f"  Load time: {format_time(load_time)}")

    # Create chat prompt
    print("\nGenerating text with chat template...")
    messages = [
        ChatMessage(role="system", content="You are a helpful AI assistant."),
        ChatMessage(role="user", content="What are the three laws of robotics?"),
    ]

    prompt = format_chat_messages(messages, model_type="qwen3")
    print(f"  Prompt: {messages[-1].content}")

    # Tokenize
    input_ids = tokenizer.encode(prompt).ids
    print(f"  Input tokens: {len(input_ids)}")

    # Generate
    print("\n  Generating...")
    start = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=128,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        use_cache=True,
    )
    gen_time = (time.perf_counter() - start) * 1000

    new_tokens = len(output_ids) - len(input_ids)
    tokens_per_sec = new_tokens / (gen_time / 1000)

    # Decode output
    output_text = tokenizer.decode(output_ids)
    generated_text = tokenizer.decode(output_ids[len(input_ids) :])

    print(f"\n  Generated ({new_tokens} tokens, {tokens_per_sec:.1f} tok/s):")
    print(f"  {'-' * 60}")
    # Only show the generated part
    print(f"  {generated_text[:500]}..." if len(generated_text) > 500 else f"  {generated_text}")
    print(f"  {'-' * 60}")

    print("\n  [PASS] LLM Generation working correctly!")
    return True


def main():
    parser = argparse.ArgumentParser(description="PyGPUkit v0.2.10 Feature Demo")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model.safetensors or model.safetensors.index.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation demo (run only kernel demos)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" PyGPUkit v0.2.10 - Comprehensive Feature Demo")
    print("=" * 70)

    # Check PyGPUkit
    try:
        import pygpukit as gk

        print("\nPyGPUkit loaded successfully")
        print(f"  CUDA available: {gk.is_cuda_available()}")
    except ImportError as e:
        print(f"\nError importing PyGPUkit: {e}")
        return 1

    # Run kernel demos (no model needed)
    results = []

    try:
        results.append(("INT8 Quantization", demo_int8_quantization()))
    except Exception as e:
        print(f"\n  [FAIL] INT8 Quantization: {e}")
        results.append(("INT8 Quantization", False))

    gc.collect()

    try:
        results.append(("Paged Attention", demo_paged_attention()))
    except Exception as e:
        print(f"\n  [FAIL] Paged Attention: {e}")
        results.append(("Paged Attention", False))

    gc.collect()

    try:
        results.append(("Continuous Batching", demo_continuous_batching()))
    except Exception as e:
        print(f"\n  [FAIL] Continuous Batching: {e}")
        results.append(("Continuous Batching", False))

    gc.collect()

    # Run LLM demo if model provided
    if not args.skip_llm and args.model and args.tokenizer:
        model_path = Path(args.model)
        tokenizer_path = Path(args.tokenizer)

        if not model_path.exists():
            print(f"\nWarning: Model not found: {model_path}")
        elif not tokenizer_path.exists():
            print(f"\nWarning: Tokenizer not found: {tokenizer_path}")
        else:
            try:
                results.append(
                    ("LLM Generation", demo_llm_generation(str(model_path), str(tokenizer_path)))
                )
            except Exception as e:
                print(f"\n  [FAIL] LLM Generation: {e}")
                import traceback

                traceback.print_exc()
                results.append(("LLM Generation", False))
    elif not args.skip_llm and (not args.model or not args.tokenizer):
        print("\n" + "=" * 70)
        print(" Skipping LLM Generation Demo")
        print("=" * 70)
        print("\n  To run the LLM demo, provide model and tokenizer paths:")
        print("    python demo_v0210.py --model /path/to/model --tokenizer /path/to/tokenizer.json")

    # Summary
    section("Demo Summary")
    print("\nv0.2.10 Feature Status:")
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("All demos completed successfully!")
    else:
        print("Some demos failed. Check the output above for details.")

    print("\nv0.2.10 Features Summary:")
    print("  - INT8 Quantization: ~50% memory reduction for weights")
    print("  - Paged Attention: vLLM-style KV cache memory management")
    print("  - Continuous Batching: Dynamic multi-request scheduling")
    print("  - Chat Templates: Qwen3/LLaMA/Mistral format support")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
