"""Profile individual block operations with proper CUDA synchronization."""

import time

import numpy as np

from pygpukit.core import GPUArray, default_stream, from_numpy
from pygpukit.llm import detect_model_spec, load_model_from_safetensors, load_safetensors


def synchronize():
    """CUDA synchronize for accurate timing."""
    default_stream().synchronize()


def log_tensor(name, t):
    """Log tensor dtype/shape/device."""
    if hasattr(t, "dtype") and hasattr(t, "shape"):
        # GPUArray
        print(f"    {name}: dtype={t.dtype}, shape={t.shape}")
    elif hasattr(t, "dtype"):
        # numpy
        print(f"    {name}: dtype={t.dtype}, shape={t.shape} [NUMPY!]")


def profile_single_block(model, hidden, position_ids, past_kv, block_idx, verbose=False):
    """Profile a single block with per-operation timing."""
    block = model.blocks[block_idx]

    # Synchronize before starting
    synchronize()

    timings = {}

    # Attention norm
    start = time.perf_counter()
    residual = hidden
    x = block.attn_norm(hidden)
    synchronize()
    timings["attn_norm"] = (time.perf_counter() - start) * 1000

    # Attention (full)
    start = time.perf_counter()
    attn_out, present_kv = block.attn(x, position_ids, past_kv, use_cache=True)
    synchronize()
    timings["attention"] = (time.perf_counter() - start) * 1000

    # Residual add
    from pygpukit.ops import add

    start = time.perf_counter()
    x = add(residual, attn_out)
    synchronize()
    timings["attn_residual"] = (time.perf_counter() - start) * 1000

    # MLP norm
    start = time.perf_counter()
    residual = x
    x = block.mlp_norm(x)
    synchronize()
    timings["mlp_norm"] = (time.perf_counter() - start) * 1000

    # MLP
    start = time.perf_counter()
    x = block.mlp(x)
    synchronize()
    timings["mlp"] = (time.perf_counter() - start) * 1000

    # MLP residual add
    start = time.perf_counter()
    x = add(residual, x)
    synchronize()
    timings["mlp_residual"] = (time.perf_counter() - start) * 1000

    timings["total"] = sum(timings.values())

    return x, present_kv, timings


def profile_attention_breakdown(attn, x, position_ids, past_kv):
    """Profile attention sub-operations."""
    from pygpukit.ops import (
        concat_axis0,
        repeat_interleave_axis1,
        reshape_copy,
        rope_inplace,
        sdpa_causal,
        transpose_3d_021,
    )

    synchronize()
    timings = {}
    seq_len = x.shape[0]

    # Q, K, V projections
    start = time.perf_counter()
    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    synchronize()
    timings["qkv_proj"] = (time.perf_counter() - start) * 1000

    # Reshape
    start = time.perf_counter()
    q = reshape_copy(q, (seq_len, attn.num_heads, attn.head_dim))
    k = reshape_copy(k, (seq_len, attn.num_kv_heads, attn.head_dim))
    v = reshape_copy(v, (seq_len, attn.num_kv_heads, attn.head_dim))
    synchronize()
    timings["reshape"] = (time.perf_counter() - start) * 1000

    # QK Norm (if present)
    if attn.q_norm is not None:
        start = time.perf_counter()
        q_2d = reshape_copy(q, (seq_len * attn.num_heads, attn.head_dim))
        q_2d = attn.q_norm(q_2d)
        q = reshape_copy(q_2d, (seq_len, attn.num_heads, attn.head_dim))
        synchronize()
        timings["q_norm"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        k_2d = reshape_copy(k, (seq_len * attn.num_kv_heads, attn.head_dim))
        k_2d = attn.k_norm(k_2d)
        k = reshape_copy(k_2d, (seq_len, attn.num_kv_heads, attn.head_dim))
        synchronize()
        timings["k_norm"] = (time.perf_counter() - start) * 1000

    # RoPE
    if attn.config.use_rope and attn._cos is not None:
        start = time.perf_counter()
        q_dtype = q.dtype
        if q_dtype == "float16":
            cos = from_numpy(attn._cos[position_ids].astype(np.float16))
            sin = from_numpy(attn._sin[position_ids].astype(np.float16))
        else:
            cos = from_numpy(attn._cos[position_ids].astype(np.float32))
            sin = from_numpy(attn._sin[position_ids].astype(np.float32))
        synchronize()
        timings["rope_setup"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        if q_dtype in ("float32", "float16"):
            rope_inplace(q, k, cos, sin)
        synchronize()
        timings["rope_kernel"] = (time.perf_counter() - start) * 1000

    # KV cache concat
    if past_kv is not None:
        past_k, past_v = past_kv
        start = time.perf_counter()
        if isinstance(past_k, GPUArray):
            k = concat_axis0(past_k, k)
            v = concat_axis0(past_v, v)
        synchronize()
        timings["kv_concat"] = (time.perf_counter() - start) * 1000

    # GQA expand
    if attn.num_kv_groups > 1:
        start = time.perf_counter()
        k_expanded = repeat_interleave_axis1(k, attn.num_kv_groups)
        v_expanded = repeat_interleave_axis1(v, attn.num_kv_groups)
        synchronize()
        timings["gqa_expand"] = (time.perf_counter() - start) * 1000
    else:
        k_expanded = k
        v_expanded = v

    # Transpose
    start = time.perf_counter()
    q_t = transpose_3d_021(q)
    k_t = transpose_3d_021(k_expanded)
    v_t = transpose_3d_021(v_expanded)
    synchronize()
    timings["transpose"] = (time.perf_counter() - start) * 1000

    # SDPA
    start = time.perf_counter()
    attn_output = sdpa_causal(q_t, k_t, v_t)
    synchronize()
    timings["sdpa"] = (time.perf_counter() - start) * 1000

    # Output reshape
    start = time.perf_counter()
    attn_output = transpose_3d_021(attn_output)
    attn_output = reshape_copy(attn_output, (seq_len, attn.num_heads * attn.head_dim))
    synchronize()
    timings["output_reshape"] = (time.perf_counter() - start) * 1000

    # O projection
    start = time.perf_counter()
    _ = attn.o_proj(attn_output)
    synchronize()
    timings["o_proj"] = (time.perf_counter() - start) * 1000

    return timings


def get_ptr(arr):
    """Get memory pointer of GPUArray."""
    try:
        native = arr._get_native()
        ptr = native.data_ptr()
        return f"0x{ptr:x}" if isinstance(ptr, int) else str(ptr)
    except Exception as e:
        return f"N/A ({e})"


def profile_mlp_breakdown(mlp, x, verbose=True):
    """Profile MLP sub-operations with dtype/shape logging."""
    from pygpukit.ops import mul, silu

    synchronize()
    timings = {}

    if verbose:
        print(f"    Input: dtype={x.dtype}, shape={x.shape}")

    # gate_proj
    start = time.perf_counter()
    gate_out = mlp.gate_proj(x)
    synchronize()
    timings["gate_proj"] = (time.perf_counter() - start) * 1000
    if verbose:
        ptr = get_ptr(mlp.gate_proj.weight)
        print(
            f"    gate_proj weight: dtype={mlp.gate_proj.weight.dtype}, shape={mlp.gate_proj.weight.shape}, ptr={ptr}"
        )
        print(f"    gate_proj out: dtype={gate_out.dtype}, shape={gate_out.shape}")

    # silu
    start = time.perf_counter()
    gate = silu(gate_out)
    synchronize()
    timings["silu"] = (time.perf_counter() - start) * 1000
    if verbose:
        print(f"    silu out: dtype={gate.dtype}, shape={gate.shape}")

    # up_proj
    start = time.perf_counter()
    up = mlp.up_proj(x)
    synchronize()
    timings["up_proj"] = (time.perf_counter() - start) * 1000
    if verbose:
        ptr = get_ptr(mlp.up_proj.weight)
        print(
            f"    up_proj weight: dtype={mlp.up_proj.weight.dtype}, shape={mlp.up_proj.weight.shape}, ptr={ptr}"
        )
        print(f"    up_proj out: dtype={up.dtype}, shape={up.shape}")

    # mul
    start = time.perf_counter()
    gated = mul(gate, up)
    synchronize()
    timings["mul"] = (time.perf_counter() - start) * 1000

    # down_proj
    start = time.perf_counter()
    out = mlp.down_proj(gated)
    synchronize()
    timings["down_proj"] = (time.perf_counter() - start) * 1000
    if verbose:
        ptr = get_ptr(mlp.down_proj.weight)
        print(
            f"    down_proj weight: dtype={mlp.down_proj.weight.dtype}, shape={mlp.down_proj.weight.shape}, ptr={ptr}"
        )
        print(f"    down_proj out: dtype={out.dtype}, shape={out.shape}")

    return timings


def main():
    print("Loading Qwen3-8B-FP16...")

    # Use cached model path directly (no re-download)
    # Aratako/Qwen3-8B-ERP-v0.1 - already downloaded
    import os

    cache_base = os.path.expanduser("~/.cache/huggingface/hub")
    model_path = os.path.join(
        cache_base,
        "models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf",
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    print(f"Model path: {model_path}")

    # Detect model spec and load
    st = load_safetensors(index_path)
    spec = detect_model_spec(st.tensor_names)

    # Pre-load shards in REVERSE order to test memory layout theory
    print("Pre-loading shards in reverse order...")
    if hasattr(st, "_shard_files"):
        for shard in reversed(st._shard_files):
            print(f"  Loading {shard}...")
            st._get_shard(shard)

    model = load_model_from_safetensors(index_path, dtype="float16", spec=spec)
    print(f"Loaded {len(model.blocks)} blocks")

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        hidden, kv = model([1, 2, 3], use_cache=True)
        synchronize()

    # Test decode phase (single token)
    print("\n" + "=" * 60)
    print("DECODE PHASE PROFILING (single token, varying cache size)")
    print("=" * 60)

    # Build up cache with short prompt
    prompt_tokens = list(range(10, 50))  # 40 tokens
    hidden, past_kv = model(prompt_tokens, use_cache=True)
    synchronize()

    # Profile single token decode
    new_token = [100]
    position_ids = [len(prompt_tokens)]

    # Get embedding
    if not hasattr(model, "_embed_np_cache"):
        model._embed_np_cache = model.embed_tokens.to_numpy()
    hidden_np = model._embed_np_cache[new_token]
    hidden = from_numpy(hidden_np.astype(model._embed_np_cache.dtype))

    print(f"\nProfiling blocks with KV cache size = {len(prompt_tokens)}")
    print("-" * 60)

    block_times = []
    for i in range(len(model.blocks)):  # All blocks
        past = past_kv[i] if past_kv else None
        hidden, present, timings = profile_single_block(model, hidden, position_ids, past, i)
        past_kv[i] = present
        block_times.append(timings["total"])
        # Print progress every 10 blocks
        if i % 10 == 0:
            print(
                f"  Block {i}: {timings['total']:.2f}ms (attn={timings['attention']:.2f}, mlp={timings['mlp']:.2f})"
            )

    # Summary
    print("\n" + "-" * 60)
    print("BLOCK TIMING SUMMARY:")
    print(f"  Total: {sum(block_times):.1f}ms for {len(block_times)} blocks")
    print(f"  Avg:   {sum(block_times) / len(block_times):.2f}ms/block")
    print(f"  Min:   {min(block_times):.2f}ms (block {block_times.index(min(block_times))})")
    print(f"  Max:   {max(block_times):.2f}ms (block {block_times.index(max(block_times))})")
    print(f"  First 5 avg: {sum(block_times[:5]) / 5:.2f}ms")
    print(f"  Last 5 avg:  {sum(block_times[-5:]) / 5:.2f}ms")

    # Profile attention breakdown for block 0 and block 34
    print("\n" + "=" * 60)
    print("ATTENTION BREAKDOWN")
    print("=" * 60)

    # Reset and build cache again
    hidden, past_kv = model(prompt_tokens, use_cache=True)
    synchronize()

    hidden_np = model._embed_np_cache[new_token]
    hidden_decode = from_numpy(hidden_np.astype(model._embed_np_cache.dtype))

    for block_idx in [0, 17, 34]:  # Start, middle, end
        if block_idx >= len(model.blocks):
            continue
        print(f"\nBlock {block_idx} Attention Breakdown:")
        print("-" * 40)

        block = model.blocks[block_idx]
        x = block.attn_norm(hidden_decode)
        synchronize()

        past = past_kv[block_idx] if past_kv else None
        timings = profile_attention_breakdown(block.attn, x, position_ids, past)

        for op, t in timings.items():
            print(f"  {op:15s}: {t:6.2f} ms")

        total = sum(timings.values())
        print(f"  {'TOTAL':15s}: {total:6.2f} ms")

    # MLP breakdown for block 0 vs block 20
    print("\n" + "=" * 60)
    print("MLP BREAKDOWN (Block 0 vs Block 20)")
    print("=" * 60)

    # Reset
    hidden, past_kv = model(prompt_tokens, use_cache=True)
    synchronize()

    hidden_np = model._embed_np_cache[new_token]
    hidden_decode = from_numpy(hidden_np.astype(model._embed_np_cache.dtype))

    # Warmup ALL blocks by running matmul once
    print("\n  Warming up all block MLP weights (running matmul once each)...")
    dummy_input = from_numpy(np.zeros((1, 4096), dtype=np.float16))
    dummy_inter = from_numpy(np.zeros((1, 12288), dtype=np.float16))
    for _i, block in enumerate(model.blocks):
        # Run matmul to force CUDA kernel init, transpose cache, and memory access
        _ = block.mlp.gate_proj(dummy_input)
        _ = block.mlp.up_proj(dummy_input)
        _ = block.mlp.down_proj(dummy_inter)
    synchronize()
    print("  Done warming up all blocks.")

    # Check transpose cache addresses
    print("\n  Transpose cache (_weight_t) addresses:")
    for block_idx in [0, 10, 20, 30]:
        block = model.blocks[block_idx]
        gate_t = get_ptr(block.mlp.gate_proj._weight_t) if block.mlp.gate_proj._weight_t else "None"
        up_t = get_ptr(block.mlp.up_proj._weight_t) if block.mlp.up_proj._weight_t else "None"
        down_t = get_ptr(block.mlp.down_proj._weight_t) if block.mlp.down_proj._weight_t else "None"
        print(f"    Block {block_idx}: gate_t={gate_t}, up_t={up_t}, down_t={down_t}")

    # Test each block INDIVIDUALLY (fresh prefill each time)
    print("\n  Testing blocks individually (after weight warmup):")
    for block_idx in [0, 10, 20, 30]:
        # Fresh prefill
        hidden, past_kv = model(prompt_tokens, use_cache=True)
        synchronize()

        hidden_np = model._embed_np_cache[new_token]
        hidden_decode = from_numpy(hidden_np.astype(model._embed_np_cache.dtype))

        block = model.blocks[block_idx]
        x = block.attn_norm(hidden_decode)
        attn_out, _ = block.attn(x, position_ids, past_kv[block_idx], use_cache=True)
        from pygpukit.ops import add

        x = add(hidden_decode, attn_out)
        x = block.mlp_norm(x)
        synchronize()

        timings = profile_mlp_breakdown(block.mlp, x, verbose=False)
        print(
            f"  Block {block_idx} (fresh): gate={timings['gate_proj']:.2f}ms, up={timings['up_proj']:.2f}ms, down={timings['down_proj']:.2f}ms, TOTAL={sum(timings.values()):.2f}ms"
        )

    # Test: Use Block 0's weights with Block 20's input
    print("\n  Testing Block 20 input with BLOCK 0's weights:")
    hidden, past_kv = model(prompt_tokens, use_cache=True)
    synchronize()

    hidden_np = model._embed_np_cache[new_token]
    hidden_decode = from_numpy(hidden_np.astype(model._embed_np_cache.dtype))

    block20 = model.blocks[20]
    block0 = model.blocks[0]

    x = block20.attn_norm(hidden_decode)
    attn_out, _ = block20.attn(x, position_ids, past_kv[20], use_cache=True)
    x = add(hidden_decode, attn_out)
    x = block20.mlp_norm(x)
    synchronize()

    # Use Block 0's weights (which are fast)
    from pygpukit.ops import mul, silu

    synchronize()

    start = time.perf_counter()
    gate_out = block0.mlp.gate_proj(x)  # Block 0's weight!
    synchronize()
    t_gate = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    up_out = block0.mlp.up_proj(x)  # Block 0's weight!
    synchronize()
    t_up = (time.perf_counter() - start) * 1000

    gate = silu(gate_out)
    gated = mul(gate, up_out)

    start = time.perf_counter()
    _ = block0.mlp.down_proj(gated)  # Block 0's weight!
    synchronize()
    t_down = (time.perf_counter() - start) * 1000

    print(
        f"  Block 20 input + Block 0 weights: gate={t_gate:.2f}ms, up={t_up:.2f}ms, down={t_down:.2f}ms, TOTAL={t_gate + t_up + t_down:.2f}ms"
    )

    # Reverse test: Block 0's input with Block 20's weights
    print("\n  Testing Block 0 input with BLOCK 20's weights:")
    block0 = model.blocks[0]
    x = block0.attn_norm(hidden_decode)
    attn_out, _ = block0.attn(x, position_ids, past_kv[0], use_cache=True)
    x = add(hidden_decode, attn_out)
    x = block0.mlp_norm(x)
    synchronize()

    start = time.perf_counter()
    gate_out = block20.mlp.gate_proj(x)  # Block 20's weight!
    synchronize()
    t_gate = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    up_out = block20.mlp.up_proj(x)  # Block 20's weight!
    synchronize()
    t_up = (time.perf_counter() - start) * 1000

    gate = silu(gate_out)
    gated = mul(gate, up_out)

    start = time.perf_counter()
    _ = block20.mlp.down_proj(gated)  # Block 20's weight!
    synchronize()
    t_down = (time.perf_counter() - start) * 1000

    print(
        f"  Block 0 input + Block 20 weights: gate={t_gate:.2f}ms, up={t_up:.2f}ms, down={t_down:.2f}ms, TOTAL={t_gate + t_up + t_down:.2f}ms"
    )


if __name__ == "__main__":
    main()
