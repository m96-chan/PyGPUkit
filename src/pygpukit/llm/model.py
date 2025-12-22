"""CausalTransformerModel implementation for PyGPUkit.

Provides the unified Transformer runtime for GPT-2, LLaMA, and Qwen3 architectures.
Model-specific behavior is controlled by the ModelSpec configuration.

Key features:
- Hybrid Attention: CPU for seq_len=1 (decode), GPU for prefill
- GPU-native operations: RMSNorm, LayerNorm, SDPA, SiLU, GELU, RoPE
- CUDA Graph support for zero-allocation decode
- Speculative and Jacobi decoding modes
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Literal

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy

# Import from refactored modules
from pygpukit.llm.buffers import DecodeBuffers, PrefillBuffers
from pygpukit.llm.config import ModelSpec, TransformerConfig
from pygpukit.llm.layers import (
    MLP,
    Attention,
    Norm,
    TransformerBlock,
    precompute_freqs_cis,
)
from pygpukit.llm.sampling import sample_token
from pygpukit.ops.basic import (
    add,
    add_inplace,
    bias_add_inplace,
    copy_to,
    embedding_lookup,
    embedding_lookup_batch,
    embedding_lookup_ptr,
    gelu,
    kv_cache_prefill_gqa,
    kv_cache_update_gqa,
    kv_cache_update_gqa_ptr,
    matmul,
    mul_inplace,
    repeat_interleave_axis1,
    reshape_copy,
    rmsnorm,
    rope_inplace,
    sample_token_gpu,
    sample_topk_to_buf_ptr,
    sdpa_causal,
    sdpa_causal_fixed_cache,
    sdpa_causal_fixed_cache_ptr,
    silu,
    transpose,
    transpose_3d_021,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Unified CausalTransformerModel
# =============================================================================


class CausalTransformerModel:
    """Unified causal transformer model.

    The single runtime model for all architectures (GPT-2, LLaMA, Qwen3).
    Model-specific behavior is controlled by the spec attribute.
    """

    def __init__(
        self,
        config: TransformerConfig,
        embed_tokens: GPUArray,
        blocks: list[TransformerBlock],
        final_norm: Norm,
        lm_head: GPUArray | None = None,
        position_embed: GPUArray | None = None,  # For GPT-2 style
        spec: ModelSpec | None = None,
    ):
        self.config = config
        self.embed_tokens = embed_tokens
        self.blocks = blocks
        self.final_norm = final_norm
        self._lm_head = lm_head
        self.position_embed = position_embed
        self.spec = spec

    def __call__(
        self,
        input_ids: list[int],
        position_ids: list[int] | None = None,
        past_key_values: list[tuple | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, list[tuple | None] | None]:
        """Forward pass.

        Args:
            input_ids: Token IDs [seq_len]
            position_ids: Position IDs (auto-generated if None)
            past_key_values: List of (k, v) tuples per layer
            use_cache: Whether to return KV cache

        Returns:
            Tuple of (hidden_states, present_key_values)
        """
        seq_len = len(input_ids)

        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[0]
                position_ids = list(range(past_len, past_len + seq_len))
            else:
                position_ids = list(range(seq_len))

        # Token embeddings (cache numpy array to avoid repeated GPU->CPU transfer)
        if not hasattr(self, "_embed_np_cache"):
            self._embed_np_cache = self.embed_tokens.to_numpy()
        hidden_np = self._embed_np_cache[input_ids]

        # Add position embeddings (GPT-2 style)
        if self.position_embed is not None:
            if not hasattr(self, "_pos_embed_np_cache"):
                self._pos_embed_np_cache = self.position_embed.to_numpy()
            hidden_np = hidden_np + self._pos_embed_np_cache[position_ids]

        hidden: GPUArray = from_numpy(hidden_np.astype(self._embed_np_cache.dtype))

        # Transformer blocks
        present_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden, present_kv = block(hidden, position_ids, past_kv, use_cache)
            present_key_values.append(present_kv)

        # Final norm
        hidden = self.final_norm(hidden)

        if use_cache:
            return hidden, present_key_values
        return hidden, None

    @property
    def lm_head(self) -> GPUArray | None:
        """LM head weights (for backward compatibility)."""
        return self._lm_head

    def get_logits(self, hidden: GPUArray) -> GPUArray:
        """Compute logits from hidden states on GPU."""
        # Cache transposed lm_head to avoid repeated transpose
        if not hasattr(self, "_lm_head_t_cache"):
            lm_head = self._lm_head if self._lm_head is not None else self.embed_tokens
            self._lm_head_t_cache = transpose(lm_head)

        # GPU matmul: hidden @ lm_head.T
        # hidden: [seq_len, hidden_size], lm_head: [vocab_size, hidden_size]
        # Result: [seq_len, vocab_size]
        return matmul(hidden, self._lm_head_t_cache)

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        gpu_sampling: bool = False,
    ) -> list[int]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop at this token
            use_cache: Use KV cache
            gpu_sampling: Use GPU-based sampling (avoids full logits D2H transfer)

        Returns:
            List of all token IDs (input + generated)
        """
        tokens = list(input_ids)
        past_key_values = None

        if use_cache:
            # Prefill
            hidden, past_key_values = self(tokens, use_cache=True)
            logits = self.get_logits(hidden)

            if gpu_sampling:
                # GPU sampling: only transfer 1 int instead of full vocab logits
                next_token = sample_token_gpu(logits[-1], temperature, top_k, top_p)
            else:
                last_logits = logits.to_numpy()[-1]
                next_token = sample_token(last_logits, temperature, top_k, top_p)
            tokens.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                return tokens

            # Decode
            for _ in range(max_new_tokens - 1):
                hidden, past_key_values = self(
                    [next_token], past_key_values=past_key_values, use_cache=True
                )
                logits = self.get_logits(hidden)

                if gpu_sampling:
                    next_token = sample_token_gpu(logits[-1], temperature, top_k, top_p)
                else:
                    last_logits = logits.to_numpy()[-1]
                    next_token = sample_token(last_logits, temperature, top_k, top_p)
                tokens.append(next_token)

                if eos_token_id is not None and next_token == eos_token_id:
                    break
        else:
            for _ in range(max_new_tokens):
                hidden, _ = self(tokens, use_cache=False)
                logits = self.get_logits(hidden)

                if gpu_sampling:
                    next_token = sample_token_gpu(logits[-1], temperature, top_k, top_p)
                else:
                    last_logits = logits.to_numpy()[-1]
                    next_token = sample_token(last_logits, temperature, top_k, top_p)
                tokens.append(next_token)

                if eos_token_id is not None and next_token == eos_token_id:
                    break

        return tokens

    def generate_stream(
        self,
        input_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        gpu_sampling: bool = False,
    ) -> Generator[int, None, None]:
        """Generate tokens autoregressively with streaming.

        Yields tokens one at a time as they are generated, enabling
        real-time text display in chat applications.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop at this token
            gpu_sampling: Use GPU-based sampling (avoids full logits D2H transfer)

        Yields:
            Generated token IDs one at a time

        Example:
            >>> for token_id in model.generate_stream(input_ids, max_new_tokens=50):
            ...     token_str = tokenizer.decode([token_id])
            ...     print(token_str, end="", flush=True)
        """
        past_key_values = None

        # Prefill
        hidden, past_key_values = self(input_ids, use_cache=True)
        logits = self.get_logits(hidden)

        if gpu_sampling:
            next_token = sample_token_gpu(logits[-1], temperature, top_k, top_p)
        else:
            last_logits = logits.to_numpy()[-1]
            next_token = sample_token(last_logits, temperature, top_k, top_p)

        yield next_token

        if eos_token_id is not None and next_token == eos_token_id:
            return

        # Decode
        for _ in range(max_new_tokens - 1):
            hidden, past_key_values = self(
                [next_token], past_key_values=past_key_values, use_cache=True
            )
            logits = self.get_logits(hidden)

            if gpu_sampling:
                next_token = sample_token_gpu(logits[-1], temperature, top_k, top_p)
            else:
                last_logits = logits.to_numpy()[-1]
                next_token = sample_token(last_logits, temperature, top_k, top_p)

            yield next_token

            if eos_token_id is not None and next_token == eos_token_id:
                return

    def generate_cuda_graph(
        self,
        input_ids: list[int],
        max_new_tokens: int = 20,
        max_seq_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        use_graph: bool = False,
        gpu_sampling: bool = False,
    ) -> list[int]:
        """Generate tokens using fixed-length KV cache with optional CUDA Graph.

        This method uses fixed-length KV cache and pre-allocated decode buffers
        to eliminate all memory allocations during decode, enabling CUDA Graph capture.

        Flow:
            1. Prefill: Normal execution (no graph)
            2. Decode: Allocation-free execution with pre-allocated buffers
            3. (Optional) CUDA Graph: Capture first decode, replay for subsequent

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum new tokens to generate
            max_seq_len: Maximum sequence length (prefill + decode)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop at this token
            use_graph: Enable CUDA Graph capture/replay (experimental)
            gpu_sampling: Use GPU-based sampling (avoids full logits D2H transfer)

        Returns:
            List of all token IDs (input + generated)
        """
        prefill_len = len(input_ids)
        tokens = list(input_ids)

        # Ensure max_seq_len can hold prefill + max_new_tokens
        total_max = prefill_len + max_new_tokens
        if max_seq_len < total_max:
            max_seq_len = total_max

        # Get dtype from embed tokens
        dtype = str(self.embed_tokens.dtype)

        # Initialize fixed-length KV cache for all layers
        for block in self.blocks:
            block.attn.init_fixed_cache(max_seq_len, dtype=dtype)

        # ============================================================
        # Allocate decode buffers (zero allocations during decode)
        # ============================================================
        use_qk_norm = self.spec is not None and self.spec.use_qk_norm
        # Get vocab_size from lm_head or embed_tokens
        lm_head = self._lm_head if self._lm_head is not None else self.embed_tokens
        vocab_size = lm_head.shape[0]
        _decode_buffers = DecodeBuffers.allocate(
            self.config, dtype=dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
        )

        # Allocate prefill buffers (for reduced allocations during prefill)
        # NOTE: Full zero-allocation prefill requires kernel-level changes
        # to support variable seq_len within fixed buffers
        _prefill_buffers = PrefillBuffers.allocate(
            self.config, max_seq_len=prefill_len, dtype=dtype, use_qk_norm=use_qk_norm
        )

        # Pre-compute RoPE tables on GPU (full sequence)
        if self.config.use_rope:
            from pygpukit.ops.basic import cast_f32_to_bf16, cast_f32_to_f16

            cos_np, sin_np = precompute_freqs_cis(
                self.config.head_dim, max_seq_len, self.config.rope_theta
            )
            if dtype == "float16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                self._rope_cos_gpu = cast_f32_to_f16(cos_f32)
                self._rope_sin_gpu = cast_f32_to_f16(sin_f32)
            elif dtype == "bfloat16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                self._rope_cos_gpu = cast_f32_to_bf16(cos_f32)
                self._rope_sin_gpu = cast_f32_to_bf16(sin_f32)
            else:
                self._rope_cos_gpu = from_numpy(cos_np.astype(np.float32))
                self._rope_sin_gpu = from_numpy(sin_np.astype(np.float32))

        # ============================================================
        # Phase 1: Prefill (with reduced allocations)
        # ============================================================
        hidden, past_key_values = self._prefill_with_buffers(
            input_ids, _prefill_buffers, use_cache=True
        )

        # Copy prefill KV to fixed cache (GQA-expanded, transposed)
        for i, block in enumerate(self.blocks):
            past_k, past_v = past_key_values[i]
            # past_k/v shape: [prefill_len, num_kv_heads, head_dim]
            # cache shape: [num_heads, max_seq_len, head_dim]
            kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
            kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

        # Get first token (prefill - use CPU sampling since it's one-time)
        logits = self.get_logits(hidden)
        last_logits = logits.to_numpy()[-1]
        next_token = sample_token(last_logits, temperature, top_k, top_p)
        tokens.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            return tokens

        # ============================================================
        # Phase 2: Decode loop with zero allocations
        # ============================================================
        context_len = prefill_len + 1  # Current context length

        # Import CudaGraph for graph capture
        if use_graph:
            import gc

            from pygpukit._native_loader import get_native_module

            CudaGraph = getattr(get_native_module(), "CudaGraph")  # noqa: B009

            # Warm-up: Run _decode_step_zero_alloc a few times to initialize
            # all lazy state (method dispatch, CUDA kernel caching, etc.)
            for _ in range(3):
                _ = self._decode_step_zero_alloc(
                    next_token, context_len - 1, context_len, _decode_buffers
                )

            # Create inline decode function for graph capture
            # NOTE: Inline functions capture more reliably than method calls
            # due to apparent CUDA stream capture quirks
            buffers = _decode_buffers  # Closure capture
            model_self = self  # Closure capture

            def _inline_decode_step(tok_id: int, pos: int, ctx_len: int) -> None:
                """Inline decode step for reliable graph capture.

                Uses use_position_ptr=True so kernels read position from GPU buffer,
                allowing graph replay with different positions without recapture.
                """
                embedding_lookup(model_self.embed_tokens, buffers.hidden, tok_id)
                for block in model_self.blocks:
                    rmsnorm(
                        buffers.hidden,
                        block.attn_norm.weight,
                        block.attn_norm.eps,
                        out=buffers.norm_out,
                    )
                    copy_to(buffers.hidden, buffers.residual)
                    model_self._attention_forward_zero_alloc(
                        block.attn,
                        buffers.norm_out,
                        pos,
                        ctx_len,
                        buffers,
                        use_position_ptr=True,  # Read position from GPU buffer
                    )
                    add_inplace(buffers.hidden, buffers.residual)
                    copy_to(buffers.hidden, buffers.residual)
                    rmsnorm(
                        buffers.hidden,
                        block.mlp_norm.weight,
                        block.mlp_norm.eps,
                        out=buffers.norm_out,
                    )
                    model_self._mlp_forward_zero_alloc(block.mlp, buffers.norm_out, buffers)
                    add_inplace(buffers.hidden, buffers.residual)
                rmsnorm(
                    buffers.hidden,
                    model_self.final_norm.weight,
                    model_self.final_norm.eps,
                    out=buffers.norm_out,
                )
                copy_to(buffers.norm_out, buffers.hidden)

            graph = CudaGraph()
            graph_ready = False

            # Helper to update position buffer (outside graph capture/replay)
            # Use copy_from_numpy to avoid GPU allocation every call
            _pos_np = np.array([0], dtype=np.int32)  # Reusable numpy buffer

            def _update_position_buf(pos: int) -> None:
                """Write position to GPU buffer for _ptr kernels."""
                _pos_np[0] = pos
                _decode_buffers.position_buf._get_native().copy_from_numpy(_pos_np)

            # Helper to update random_val buffer (outside graph capture/replay)
            # Use copy_from_numpy to avoid GPU allocation every call
            import random

            _rand_np = np.array([0.0], dtype=np.float32)  # Reusable numpy buffer

            def _update_random_val_buf() -> None:
                """Write random value to GPU buffer for sampling kernel."""
                _rand_np[0] = random.random()
                _decode_buffers.random_val._get_native().copy_from_numpy(_rand_np)

            # Check if we can include sampling in Graph (top_k > 0 required)
            include_sampling_in_graph = gpu_sampling and top_k > 0

        for _step in range(max_new_tokens - 1):
            position = context_len - 1  # Position of current token

            if use_graph and not graph_ready:
                # First decode step: capture the graph
                # Write position and random_val to GPU buffers BEFORE capture
                _update_position_buf(position)
                if include_sampling_in_graph:
                    _update_random_val_buf()

                # Disable GC during capture to prevent allocations
                gc.disable()
                try:
                    graph.begin_capture()
                    _inline_decode_step(next_token, position, context_len)
                    # Include get_logits in graph (matmul to pre-allocated buffer)
                    matmul(
                        _decode_buffers.hidden,
                        self._lm_head_t_cache,
                        out=_decode_buffers.logits,
                    )
                    # Include sampling in graph (if top_k > 0)
                    if include_sampling_in_graph:
                        sample_topk_to_buf_ptr(
                            _decode_buffers.logits,
                            _decode_buffers.sampled_token,
                            _decode_buffers.random_val,
                            top_k,
                            temperature,
                        )
                    graph.end_capture()
                finally:
                    gc.enable()
                graph_ready = True
                sampling_str = "in graph" if include_sampling_in_graph else "outside"
                print(f"  [CUDA Graph] Captured {graph.num_nodes} nodes (sampling={sampling_str})")

                # Get result
                if include_sampling_in_graph:
                    graph.synchronize()
                    next_token = int(_decode_buffers.sampled_token.to_numpy()[0])
                else:
                    logits = _decode_buffers.logits
                    if gpu_sampling:
                        next_token = sample_token_gpu(logits, temperature, top_k, top_p)
                    else:
                        last_logits = logits.to_numpy()[0]
                        next_token = sample_token(last_logits, temperature, top_k, top_p)
            elif use_graph and graph_ready:
                # Subsequent steps: update position and random_val buffers, then replay
                _update_position_buf(position)
                if include_sampling_in_graph:
                    _update_random_val_buf()
                graph.replay()

                # Get result
                if include_sampling_in_graph:
                    graph.synchronize()
                    next_token = int(_decode_buffers.sampled_token.to_numpy()[0])
                else:
                    logits = _decode_buffers.logits
                    if gpu_sampling:
                        next_token = sample_token_gpu(logits, temperature, top_k, top_p)
                    else:
                        last_logits = logits.to_numpy()[0]
                        next_token = sample_token(last_logits, temperature, top_k, top_p)
            else:
                # No graph: use legacy decode step with allocations
                hidden = self._decode_step_fixed_cache(next_token, position, context_len)
                logits = self.get_logits(hidden)  # [1, vocab_size]
                if gpu_sampling:
                    next_token = sample_token_gpu(logits, temperature, top_k, top_p)
                else:
                    last_logits = logits.to_numpy()[0]
                    next_token = sample_token(last_logits, temperature, top_k, top_p)
            tokens.append(next_token)

            context_len += 1

            if eos_token_id is not None and next_token == eos_token_id:
                break

        return tokens

    def _decode_step_zero_alloc(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Single decode step with zero memory allocations.

        Uses pre-allocated DecodeBuffers for all intermediate computations.
        All operations write to pre-allocated buffers, no new GPU memory is allocated.

        Args:
            token_id: Current token ID
            position: Position in sequence
            context_len: Total context length
            buffers: Pre-allocated decode buffers

        Returns:
            Hidden states [1, hidden_size]
        """
        # Get token embedding directly to hidden (no copy needed)
        embedding_lookup(self.embed_tokens, buffers.hidden, token_id)

        # Transformer blocks with fixed cache
        for block in self.blocks:
            # Pre-norm: hidden -> norm_out
            rmsnorm(
                buffers.hidden, block.attn_norm.weight, block.attn_norm.eps, out=buffers.norm_out
            )

            # Save residual
            copy_to(buffers.hidden, buffers.residual)

            # Attention with fixed cache (writes to buffers.hidden)
            self._attention_forward_zero_alloc(
                block.attn, buffers.norm_out, position, context_len, buffers
            )

            # Add residual: hidden = residual + hidden
            add_inplace(buffers.hidden, buffers.residual)

            # MLP pre-norm
            copy_to(buffers.hidden, buffers.residual)
            rmsnorm(buffers.hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=buffers.norm_out)

            # MLP forward (SwiGLU)
            self._mlp_forward_zero_alloc(block.mlp, buffers.norm_out, buffers)

            # Add residual
            add_inplace(buffers.hidden, buffers.residual)

        # Final norm
        rmsnorm(buffers.hidden, self.final_norm.weight, self.final_norm.eps, out=buffers.norm_out)
        copy_to(buffers.norm_out, buffers.hidden)

        return buffers.hidden

    def _attention_forward_zero_alloc(
        self,
        attn: Attention,
        x: GPUArray,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
        use_position_ptr: bool = False,
        use_context_len_ptr: bool = False,
        max_kv_len: int | None = None,
    ) -> None:
        """Attention forward pass with zero allocations.

        Result is written to buffers.hidden.

        Args:
            use_position_ptr: If True, read position from buffers.position_buf
                              (for CUDA Graph replay without recapture).
            use_context_len_ptr: If True, read context_len from buffers.context_len_buf
                                 (for CUDA Graph replay without recapture).
            max_kv_len: Maximum KV length for CUDA Graph shared memory allocation.
                        Required if use_context_len_ptr=True.
        """
        # Fused QKV projection (1 matmul replaces 3, then zero-copy narrow views)
        # This is 4x faster for M=1 with cuBLASLt due to reduced kernel launch overhead
        attn.qkv_proj(x, out=buffers.qkv_proj_out)

        # Apply biases (fused projection has no bias)
        if attn.q_proj.bias is not None:
            bias_add_inplace(buffers.q_view, attn.q_proj.bias)
        if attn.k_proj.bias is not None:
            bias_add_inplace(buffers.k_view, attn.k_proj.bias)
        if attn.v_proj.bias is not None:
            bias_add_inplace(buffers.v_view, attn.v_proj.bias)

        # Reshape narrow views to 3D using pre-allocated buffers
        # q_view, k_view, v_view are pre-created zero-copy views of qkv_proj_out
        reshape_copy(buffers.q_view, (1, attn.num_heads, attn.head_dim), out=buffers.q)
        reshape_copy(buffers.k_view, (1, attn.num_kv_heads, attn.head_dim), out=buffers.k)
        reshape_copy(buffers.v_view, (1, attn.num_kv_heads, attn.head_dim), out=buffers.v)
        q, k, v = buffers.q, buffers.k, buffers.v

        # QK Norm (Qwen3) - zero allocation using pre-allocated buffers
        if attn.q_norm is not None and buffers.q_2d is not None and buffers.q_flat is not None:
            # Reshape q [1,H,D] -> q_flat [H,D], apply norm, reshape back to q [1,H,D]
            reshape_copy(q, (attn.num_heads, attn.head_dim), out=buffers.q_flat)
            rmsnorm(buffers.q_flat, attn.q_norm.weight, attn.q_norm.eps, out=buffers.q_2d)
            reshape_copy(buffers.q_2d, (1, attn.num_heads, attn.head_dim), out=buffers.q)
            q = buffers.q
        if attn.k_norm is not None and buffers.k_2d is not None and buffers.k_flat is not None:
            # Reshape k [1,H,D] -> k_flat [H,D], apply norm, reshape back to k [1,H,D]
            reshape_copy(k, (attn.num_kv_heads, attn.head_dim), out=buffers.k_flat)
            rmsnorm(buffers.k_flat, attn.k_norm.weight, attn.k_norm.eps, out=buffers.k_2d)
            reshape_copy(buffers.k_2d, (1, attn.num_kv_heads, attn.head_dim), out=buffers.k)
            k = buffers.k

        # Apply RoPE using pre-computed GPU tables (zero allocation)
        if self.config.use_rope and hasattr(self, "_rope_cos_gpu"):
            # Extract single row from pre-computed tables using GPU kernel
            if use_position_ptr and buffers.position_buf is not None:
                # Use _ptr variants for CUDA Graph replay
                embedding_lookup_ptr(self._rope_cos_gpu, buffers.cos, buffers.position_buf)
                embedding_lookup_ptr(self._rope_sin_gpu, buffers.sin, buffers.position_buf)
            else:
                embedding_lookup(self._rope_cos_gpu, buffers.cos, position)
                embedding_lookup(self._rope_sin_gpu, buffers.sin, position)
            # buffers.cos/sin are already [1, head_dim] - use directly
            rope_inplace(q, k, buffers.cos, buffers.sin)

        # Update KV cache at position (GQA-expanded, transposed)
        if use_position_ptr and buffers.position_buf is not None:
            # Use _ptr variants for CUDA Graph replay
            kv_cache_update_gqa_ptr(k, attn._k_cache, attn.num_heads, buffers.position_buf)
            kv_cache_update_gqa_ptr(v, attn._v_cache, attn.num_heads, buffers.position_buf)
        else:
            kv_cache_update_gqa(k, attn._k_cache, attn.num_heads, position)
            kv_cache_update_gqa(v, attn._v_cache, attn.num_heads, position)

        # Transpose Q for SDPA: [1, num_heads, head_dim] -> [num_heads, 1, head_dim]
        transpose_3d_021(q, out=buffers.q_t)

        # SDPA with fixed cache
        if use_context_len_ptr and buffers.context_len_buf is not None:
            # Use pointer-based SDPA for CUDA Graph replay
            assert max_kv_len is not None, "max_kv_len required for CUDA Graph mode"
            sdpa_causal_fixed_cache_ptr(
                buffers.q_t,
                attn._k_cache,
                attn._v_cache,
                buffers.attn_out,
                buffers.context_len_buf,
                max_kv_len,
            )
        else:
            sdpa_causal_fixed_cache(
                buffers.q_t, attn._k_cache, attn._v_cache, buffers.attn_out, context_len
            )

        # Transpose output: [num_heads, 1, head_dim] -> [1, num_heads, head_dim]
        transpose_3d_021(buffers.attn_out, out=buffers.q)  # Reuse q buffer for transposed output

        # Reshape to 2D: [1, hidden_size] - reuse q_proj_out buffer
        reshape_copy(buffers.q, (1, attn.num_heads * attn.head_dim), out=buffers.q_proj_out)

        # Output projection directly to hidden (eliminates copy)
        attn.o_proj(buffers.q_proj_out, out=buffers.hidden)

    def _mlp_forward_zero_alloc(
        self,
        mlp: MLP,
        x: GPUArray,
        buffers: DecodeBuffers,
    ) -> None:
        """MLP forward pass with zero allocations (SwiGLU).

        Result is written to buffers.hidden.
        """
        if mlp.activation == "silu":
            # Non-fused SwiGLU (2 separate matmuls) - for debugging
            mlp.gate_proj(x, out=buffers.mlp_gate)
            silu(buffers.mlp_gate, out=buffers.mlp_gate)

            mlp.up_proj(x, out=buffers.mlp_up)

            mul_inplace(buffers.mlp_gate, buffers.mlp_up)

            mlp.down_proj(buffers.mlp_gate, out=buffers.hidden)
        else:
            # GELU path (GPT-2) - still has allocations, rarely used
            fc1_out = mlp.fc1(x)
            gelu_out = gelu(fc1_out)
            fc2_out = mlp.fc2(gelu_out)
            copy_to(fc2_out, buffers.hidden)

    def _mlp_forward_batch_zero_alloc(
        self,
        mlp: MLP,
        x: GPUArray,
        buffers: DecodeBuffers,
        out: GPUArray,
    ) -> None:
        """Batch MLP forward pass with zero allocations (SwiGLU).

        Uses fused gate_up projection for efficiency.

        Args:
            mlp: MLP module
            x: Input tensor [seq_len, hidden_size]
            buffers: Pre-allocated decode buffers
            out: Output buffer [seq_len, hidden_size] to write result
        """
        seq_len = x.shape[0]

        if mlp.activation == "silu":
            # Fused gate_up projection
            gate_up_out = buffers.gate_up_out_batch.slice_rows(seq_len)
            mlp.gate_up_proj(x, out=gate_up_out)

            # Split into gate and up using narrow
            intermediate_size = mlp.intermediate_size
            gate = gate_up_out.narrow(0, intermediate_size)  # [seq_len, intermediate_size]
            up = gate_up_out.narrow(intermediate_size, intermediate_size)

            # SiLU in-place on gate
            silu(gate, out=gate)

            # Multiply gate * up in-place
            mul_inplace(gate, up)

            # Down projection to output buffer
            mlp.down_proj(gate, out=out)
        else:
            # GELU path - still has allocations (rarely used)
            fc1_out = mlp.fc1(x)
            gelu_out = gelu(fc1_out)
            mlp.fc2(gelu_out, out=out)

    def _prefill_with_buffers(
        self,
        input_ids: list[int],
        buffers: PrefillBuffers,
        use_cache: bool = True,
    ) -> tuple[GPUArray, list[tuple | None] | None]:
        """Prefill forward pass with reduced allocations using pre-allocated buffers.

        Uses PrefillBuffers for projection outputs, attention intermediates, and MLP
        to reduce memory allocations during prefill. Full zero-allocation requires
        kernel-level support for partial buffer operations.

        Args:
            input_ids: Token IDs [seq_len]
            buffers: Pre-allocated prefill buffers
            use_cache: Whether to return KV cache

        Returns:
            Tuple of (hidden_states, present_key_values)
        """
        seq_len = len(input_ids)
        assert seq_len <= buffers.max_seq_len, (
            f"seq_len {seq_len} > max_seq_len {buffers.max_seq_len}"
        )

        position_ids = list(range(seq_len))

        # Token embeddings - copy to pre-allocated buffer
        if not hasattr(self, "_embed_np_cache"):
            self._embed_np_cache = self.embed_tokens.to_numpy()
        hidden_np = self._embed_np_cache[input_ids]

        # Add position embeddings (GPT-2 style)
        if self.position_embed is not None:
            if not hasattr(self, "_pos_embed_np_cache"):
                self._pos_embed_np_cache = self.position_embed.to_numpy()
            hidden_np = hidden_np + self._pos_embed_np_cache[position_ids]

        # Copy to pre-allocated hidden buffer
        hidden = from_numpy(hidden_np.astype(self._embed_np_cache.dtype))
        copy_to(hidden, buffers.hidden)

        # Transformer blocks with buffer reuse
        present_key_values = []
        for block in self.blocks:
            # Process using buffers where possible
            hidden, present_kv = self._prefill_block_with_buffers(
                block, buffers.hidden, position_ids, buffers, use_cache
            )
            present_key_values.append(present_kv)

        # Final norm - reuse norm_out buffer
        rmsnorm(buffers.hidden, self.final_norm.weight, self.final_norm.eps, out=buffers.norm_out)
        copy_to(buffers.norm_out, buffers.hidden)

        if use_cache:
            return buffers.hidden, present_key_values
        return buffers.hidden, None

    def _prefill_block_with_buffers(
        self,
        block: TransformerBlock,
        hidden: GPUArray,
        position_ids: list[int],
        buffers: PrefillBuffers,
        use_cache: bool,
    ) -> tuple[GPUArray, tuple | None]:
        """Single transformer block forward with buffer reuse.

        Args:
            block: TransformerBlock to process
            hidden: Input hidden states [seq_len, hidden_size]
            position_ids: Position IDs for RoPE
            buffers: Pre-allocated prefill buffers
            use_cache: Whether to return KV cache

        Returns:
            Tuple of (output_hidden, present_kv)
        """
        # Attention block
        # Pre-norm -> norm_out
        rmsnorm(hidden, block.attn_norm.weight, block.attn_norm.eps, out=buffers.norm_out)

        # Save residual
        copy_to(hidden, buffers.residual)

        # Attention forward with buffers
        attn_out, present_kv = self._prefill_attention_with_buffers(
            block.attn, buffers.norm_out, position_ids, buffers, use_cache
        )

        # Residual connection: hidden = residual + attn_out
        add_inplace(attn_out, buffers.residual)
        copy_to(attn_out, buffers.hidden)

        # MLP block
        # Pre-norm
        copy_to(buffers.hidden, buffers.residual)
        rmsnorm(buffers.hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=buffers.norm_out)

        # MLP forward with buffers
        self._prefill_mlp_with_buffers(block.mlp, buffers.norm_out, buffers)

        # Residual connection
        add_inplace(buffers.hidden, buffers.residual)

        return buffers.hidden, present_kv

    def _prefill_attention_with_buffers(
        self,
        attn: Attention,
        x: GPUArray,
        position_ids: list[int],
        buffers: PrefillBuffers,
        use_cache: bool,
    ) -> tuple[GPUArray, tuple | None]:
        """Attention forward pass with buffer reuse during prefill.

        Args:
            attn: Attention layer
            x: Input [seq_len, hidden_size]
            position_ids: Position IDs for RoPE
            buffers: Pre-allocated prefill buffers
            use_cache: Whether to return KV cache

        Returns:
            Tuple of (output, present_kv)
        """
        seq_len = x.shape[0]

        # Project Q, K, V using pre-allocated buffers
        attn.q_proj(x, out=buffers.q_proj_out)
        attn.k_proj(x, out=buffers.k_proj_out)
        attn.v_proj(x, out=buffers.v_proj_out)

        # Reshape to 3D
        reshape_copy(buffers.q_proj_out, out=buffers.q)
        reshape_copy(buffers.k_proj_out, out=buffers.k)
        reshape_copy(buffers.v_proj_out, out=buffers.v)
        q, k, v = buffers.q, buffers.k, buffers.v

        # QK Norm (Qwen3 style)
        if attn.q_norm is not None and buffers.q_2d is not None:
            q_2d = reshape_copy(q, (seq_len * attn.num_heads, attn.head_dim))
            q_2d = attn.q_norm(q_2d)
            q = reshape_copy(q_2d, (seq_len, attn.num_heads, attn.head_dim))
        if attn.k_norm is not None and buffers.k_2d is not None:
            k_2d = reshape_copy(k, (seq_len * attn.num_kv_heads, attn.head_dim))
            k_2d = attn.k_norm(k_2d)
            k = reshape_copy(k_2d, (seq_len, attn.num_kv_heads, attn.head_dim))

        # Apply RoPE
        if self.config.use_rope and attn._cos is not None and attn._sin is not None:
            # Use Attention's precomputed cos/sin tables
            q_dtype = q.dtype
            if q_dtype == "float16":
                cos = from_numpy(attn._cos[position_ids].astype(np.float16))
                sin = from_numpy(attn._sin[position_ids].astype(np.float16))
            elif q_dtype == "bfloat16":
                # Fall back to float32 computation for bfloat16
                cos = from_numpy(attn._cos[position_ids].astype(np.float32))
                sin = from_numpy(attn._sin[position_ids].astype(np.float32))
            else:
                # FP32 path
                cos = from_numpy(attn._cos[position_ids].astype(np.float32))
                sin = from_numpy(attn._sin[position_ids].astype(np.float32))
            # Apply RoPE in-place (FP32 and FP16 have native kernel support)
            if q_dtype in ("float32", "float16"):
                rope_inplace(q, k, cos, sin)

        # Store for KV cache - MUST copy since buffers.k/v are reused across layers
        if use_cache:
            # Create copies of K, V to avoid aliasing
            # (shared buffers get overwritten by later layers)
            k_copy = reshape_copy(k, k.shape)
            v_copy = reshape_copy(v, v.shape)
            present_kv = (k_copy, v_copy)
        else:
            present_kv = None

        # Expand for GQA
        if attn.num_kv_groups > 1:
            k_expanded = repeat_interleave_axis1(k, attn.num_kv_groups)
            v_expanded = repeat_interleave_axis1(v, attn.num_kv_groups)
        else:
            k_expanded = k
            v_expanded = v

        # Transpose for SDPA: [seq, heads, dim] -> [heads, seq, dim]
        transpose_3d_021(q, out=buffers.q_t)
        k_t = transpose_3d_021(k_expanded)  # Can't use buffer due to GQA expansion
        v_t = transpose_3d_021(v_expanded)

        # SDPA with causal mask
        sdpa_causal(buffers.q_t, k_t, v_t, out=buffers.attn_out)

        # Transpose back and reshape
        transpose_3d_021(buffers.attn_out, out=buffers.attn_out_t)
        reshape_copy(buffers.attn_out_t, out=buffers.attn_out_2d)

        # Output projection
        attn.o_proj(buffers.attn_out_2d, out=buffers.o_proj_out)

        return buffers.o_proj_out, present_kv

    def _prefill_mlp_with_buffers(
        self,
        mlp: MLP,
        x: GPUArray,
        buffers: PrefillBuffers,
    ) -> None:
        """MLP forward pass with buffer reuse during prefill.

        Result is written to buffers.hidden.

        Args:
            mlp: MLP layer
            x: Input [seq_len, hidden_size]
            buffers: Pre-allocated prefill buffers
        """
        if mlp.activation == "silu":
            # SwiGLU: gate_proj -> SiLU -> * up_proj -> down_proj
            mlp.gate_proj(x, out=buffers.mlp_gate)
            silu(buffers.mlp_gate, out=buffers.mlp_gate)

            mlp.up_proj(x, out=buffers.mlp_up)

            # Element-wise multiply in-place
            mul_inplace(buffers.mlp_gate, buffers.mlp_up)

            # Down projection
            mlp.down_proj(buffers.mlp_gate, out=buffers.mlp_down)
            copy_to(buffers.mlp_down, buffers.hidden)
        else:
            # GELU path (GPT-2)
            fc1_out = mlp.fc1(x)
            gelu_out = gelu(fc1_out)
            fc2_out = mlp.fc2(gelu_out)
            copy_to(fc2_out, buffers.hidden)

    def _decode_step_fixed_cache(
        self,
        token_id: int,
        position: int,
        context_len: int,
    ) -> GPUArray:
        """Single decode step using fixed-length KV cache (legacy, with allocations).

        Args:
            token_id: Current token ID
            position: Position in sequence
            context_len: Total context length

        Returns:
            Hidden states [1, hidden_size]
        """
        # Get token embedding
        if not hasattr(self, "_embed_np_cache"):
            self._embed_np_cache = self.embed_tokens.to_numpy()
        hidden_np = self._embed_np_cache[token_id : token_id + 1]
        hidden = from_numpy(hidden_np.astype(self._embed_np_cache.dtype))

        # Transformer blocks with fixed cache
        for block in self.blocks:
            # Pre-norm
            residual = hidden
            hidden = block.attn_norm(hidden)

            # Attention with fixed cache
            hidden = block.attn.forward_fixed_cache(hidden, position, context_len)
            hidden = add(residual, hidden)

            # MLP
            residual = hidden
            hidden = block.mlp_norm(hidden)
            hidden = block.mlp(hidden)
            hidden = add(residual, hidden)

        # Final norm
        hidden = self.final_norm(hidden)

        return hidden

    def _decode_step_fixed_cache_batch(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
    ) -> GPUArray:
        """Batch decode step using fixed-length KV cache.

        Processes multiple tokens at once for speculative decoding verification.

        Args:
            token_ids: List of token IDs to decode [seq_len tokens]
            start_position: Starting position in sequence (first token's position)
            context_len: Total context length after adding this batch
                        (should equal start_position + len(token_ids))

        Returns:
            Hidden states [seq_len, hidden_size]
        """
        # Dispatch to optimized single-token path for M=1
        if len(token_ids) == 1:
            return self._decode_step_fixed_cache(token_ids[0], start_position, context_len)

        # M > 1: Batch decode path
        # Get token embeddings for batch
        if not hasattr(self, "_embed_np_cache"):
            self._embed_np_cache = self.embed_tokens.to_numpy()
        hidden_np = self._embed_np_cache[token_ids]  # [seq_len, hidden_size]
        hidden = from_numpy(hidden_np.astype(self._embed_np_cache.dtype))

        # Transformer blocks with fixed cache (batch)
        for block in self.blocks:
            # Pre-norm
            residual = hidden
            hidden = block.attn_norm(hidden)

            # Attention with fixed cache (batch)
            hidden = block.attn.forward_fixed_cache_batch(hidden, start_position, context_len)
            hidden = add(residual, hidden)

            # MLP
            residual = hidden
            hidden = block.mlp_norm(hidden)
            hidden = block.mlp(hidden)
            hidden = add(residual, hidden)

        # Final norm
        hidden = self.final_norm(hidden)

        return hidden

    def _decode_step_fixed_cache_batch_zero_alloc(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Batch decode step using pre-allocated buffers (zero-allocation).

        This function is designed to be CUDA Graph capture compatible.
        All intermediate buffers are pre-allocated in DecodeBuffers.

        Args:
            token_ids: List of token IDs to decode [seq_len tokens]
            start_position: Starting position in sequence (first token's position)
            context_len: Total context length after adding this batch
            buffers: Pre-allocated batch decode buffers

        Returns:
            Hidden states [seq_len, hidden_size] (view into buffers.hidden_batch)

        Note:
            Requires buffers.max_batch_size > 0 and len(token_ids) <= max_batch_size.
            TODO: CUDA Graph capture can be added once this path is validated.
        """
        seq_len = len(token_ids)

        if buffers.max_batch_size == 0:
            raise RuntimeError(
                "Batch buffers not allocated. Call DecodeBuffers.allocate(..., max_batch_size=8)"
            )
        if seq_len > buffers.max_batch_size:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_batch_size ({buffers.max_batch_size})"
            )

        # Get embeddings (still uses numpy - small one-time cost)
        if not hasattr(self, "_embed_np_cache"):
            self._embed_np_cache = self.embed_tokens.to_numpy()
        hidden_np = self._embed_np_cache[token_ids]  # [seq_len, hidden_size]

        # Copy to batch hidden buffer
        assert buffers.hidden_batch is not None
        buffers.hidden_batch._get_native().copy_from_numpy(
            hidden_np.astype(self._embed_np_cache.dtype)
        )

        # Use slice_rows for actual seq_len (logical batch size)
        # slice_rows creates a zero-copy view of the first N rows
        hidden = buffers.hidden_batch.slice_rows(seq_len)
        residual_buf = (
            buffers.residual_batch.slice_rows(seq_len) if buffers.residual_batch else None
        )
        norm_out_buf = (
            buffers.norm_out_batch.slice_rows(seq_len) if buffers.norm_out_batch else None
        )

        # Transformer blocks
        for block in self.blocks:
            # Pre-norm: attn_norm(hidden) -> norm_out
            if norm_out_buf is not None:
                rmsnorm(hidden, block.attn_norm.weight, block.attn_norm.eps, out=norm_out_buf)
            else:
                norm_out_buf = block.attn_norm(hidden)

            # Save residual
            if residual_buf is not None:
                copy_to(hidden, residual_buf)
            else:
                residual_buf = hidden

            # Attention with fixed cache (batch) - uses existing path for now
            # TODO: Add forward_fixed_cache_batch_zero_alloc to Attention class
            attn_out = block.attn.forward_fixed_cache_batch(
                norm_out_buf, start_position, context_len
            )

            # Residual connection: hidden = residual + attn_out
            add_inplace(residual_buf, attn_out)
            hidden = residual_buf

            # MLP norm
            if norm_out_buf is not None:
                rmsnorm(hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=norm_out_buf)
            else:
                norm_out_buf = block.mlp_norm(hidden)

            # Save residual for MLP
            if residual_buf is not hidden:
                copy_to(hidden, residual_buf)

            # MLP - uses existing path for now
            # TODO: Add zero-alloc MLP path
            mlp_out = block.mlp(norm_out_buf)

            # Residual connection
            add_inplace(residual_buf, mlp_out)
            hidden = residual_buf

        # Final norm
        if norm_out_buf is not None:
            rmsnorm(hidden, self.final_norm.weight, self.final_norm.eps, out=norm_out_buf)
            return norm_out_buf
        else:
            return self.final_norm(hidden)

    # =========================================================================
    # Self-Speculative Decoding
    # =========================================================================

    def snapshot_kv_cache(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Snapshot all layer KV caches to CPU memory.

        Returns:
            List of (k_cache_np, v_cache_np) tuples, one per layer.
            Each cache is numpy array of shape [num_heads, max_seq_len, head_dim].
        """
        snapshot = []
        for block in self.blocks:
            k_np = block.attn._k_cache.to_numpy().copy()
            v_np = block.attn._v_cache.to_numpy().copy()
            snapshot.append((k_np, v_np))
        return snapshot

    def restore_kv_cache(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Restore all layer KV caches from CPU snapshot.

        Args:
            snapshot: List of (k_cache_np, v_cache_np) tuples from snapshot_kv_cache().

        Note:
            This method copies data into existing arrays rather than replacing them.
            This is critical for CUDA Graph compatibility - the graph captures pointer
            addresses, so we must preserve the existing arrays.
        """
        for i, block in enumerate(self.blocks):
            k_np, v_np = snapshot[i]
            # Copy data into existing arrays (preserves pointers for CUDA Graph)
            k_np_typed: np.ndarray = k_np.astype(np.float16)
            v_np_typed: np.ndarray = v_np.astype(np.float16)
            block.attn._k_cache._get_native().copy_from_numpy(k_np_typed)
            block.attn._v_cache._get_native().copy_from_numpy(v_np_typed)

    def _draft_forward_early_layers(
        self,
        token_id: int,
        position: int,
        context_len: int,
        num_draft_layers: int,
    ) -> GPUArray:
        """Forward pass through only the first N layers (draft model).

        Uses the same KV cache as the full model but only updates early layers.
        After draft is done, the early layer KV entries need to be restored
        before running the full model verification.

        Args:
            token_id: Current token ID
            position: Position in sequence
            context_len: Total context length
            num_draft_layers: Number of early layers to use as draft

        Returns:
            Hidden states [1, hidden_size] after num_draft_layers
        """
        # Get token embedding
        if not hasattr(self, "_embed_np_cache"):
            self._embed_np_cache = self.embed_tokens.to_numpy()
        hidden_np = self._embed_np_cache[token_id : token_id + 1]
        hidden = from_numpy(hidden_np.astype(self._embed_np_cache.dtype))

        # Only run through first num_draft_layers blocks
        for i in range(min(num_draft_layers, len(self.blocks))):
            block = self.blocks[i]
            # Pre-norm
            residual = hidden
            hidden = block.attn_norm(hidden)

            # Attention with fixed cache
            hidden = block.attn.forward_fixed_cache(hidden, position, context_len)
            hidden = add(residual, hidden)

            # MLP
            residual = hidden
            hidden = block.mlp_norm(hidden)
            hidden = block.mlp(hidden)
            hidden = add(residual, hidden)

        # Note: We do NOT apply final_norm here since draft output
        # is only used for sampling, not for precise logits
        return hidden

    def _draft_get_logits(self, hidden: GPUArray) -> GPUArray:
        """Get logits from draft hidden states (after early layers).

        This applies final_norm and then computes logits.
        Note: The draft hidden states are from early layers, so the logits
        may not be identical to full model logits.
        """
        # Apply final norm (needed for proper logits computation)
        hidden_normed = self.final_norm(hidden)
        return self.get_logits(hidden_normed)

    def decode_step_self_speculative(
        self,
        token_id: int,
        position: int,
        context_len: int,
        max_draft_tokens: int = 4,
        draft_layers: int = 8,
    ) -> tuple[list[int], int, dict]:
        """Self-speculative decode step using early layers as draft.

        Algorithm:
        1. Snapshot KV cache state
        2. Generate max_draft_tokens using early layers (draft)
        3. Verify all draft tokens in one batch forward pass (full model)
        4. Accept tokens until first disagreement (greedy)
        5. Restore KV cache to snapshot
        6. Re-run single-token decode for accepted tokens to update KV properly

        Args:
            token_id: Current token ID (the last accepted token)
            position: Position in sequence (position of token_id)
            context_len: Total context length
            max_draft_tokens: Maximum number of draft tokens to generate
            draft_layers: Number of early layers to use as draft

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs (may be 1 to max_draft_tokens+1)
            - new_position: Updated position after accepting tokens
            - stats: Dict with 'draft_count', 'accepted_count' for analysis
        """
        # Snapshot KV cache before speculation
        kv_snapshot = self.snapshot_kv_cache()

        # === Step 1: Generate draft tokens using early layers ===
        draft_tokens = []
        draft_pos = position
        draft_ctx = context_len
        current_token = token_id

        for _ in range(max_draft_tokens):
            # Forward through early layers only
            hidden = self._draft_forward_early_layers(
                current_token, draft_pos, draft_ctx, draft_layers
            )
            # Get logits and sample (greedy for self-speculative)
            logits = self._draft_get_logits(hidden)
            logits_np = logits.to_numpy()[-1]  # [vocab_size]
            next_token = int(np.argmax(logits_np))  # Greedy sampling

            draft_tokens.append(next_token)
            current_token = next_token
            draft_pos += 1
            draft_ctx += 1

        # === Step 2: Restore KV cache for verification ===
        self.restore_kv_cache(kv_snapshot)

        # === Step 3: Verify with full model in batch ===
        # Input: [token_id, draft[0], draft[1], ..., draft[K-2]]
        # This gives logits for positions: [draft[0], draft[1], ..., draft[K-1]]
        verify_input = [token_id] + draft_tokens[:-1]
        # Context length should be: start_position + number of tokens being processed
        verify_ctx = position + len(verify_input)

        hidden_batch = self._decode_step_fixed_cache_batch(verify_input, position, verify_ctx)
        verify_logits = self.get_logits(hidden_batch)
        verify_logits_np = verify_logits.to_numpy()  # [K, vocab_size]

        # === Step 4: Accept/Reject tokens (greedy matching) ===
        accepted_tokens = []
        for i, draft_token in enumerate(draft_tokens):
            # Greedy: check if argmax matches draft
            target_token = int(np.argmax(verify_logits_np[i]))

            if target_token == draft_token:
                # Accept
                accepted_tokens.append(draft_token)
            else:
                # Reject: use target's token and stop
                accepted_tokens.append(target_token)
                break

        # If all draft tokens accepted, we can also take one bonus token
        # from the last position's distribution
        if len(accepted_tokens) == len(draft_tokens):
            # Need to run one more verify step to get the bonus token
            # For simplicity, we'll skip the bonus token in initial implementation
            pass

        # === Step 5: Restore KV cache and re-run accepted tokens ===
        self.restore_kv_cache(kv_snapshot)

        # Re-run full model single-token decode for each accepted token
        # This properly updates the KV cache
        new_pos = position
        new_ctx = context_len
        prev_token = token_id

        for acc_token in accepted_tokens:
            # Run full model decode (updates KV cache)
            self._decode_step_fixed_cache(prev_token, new_pos, new_ctx)
            prev_token = acc_token
            new_pos += 1
            new_ctx += 1

        # Stats for analysis
        stats = {
            "draft_count": len(draft_tokens),
            "accepted_count": len(
                [
                    t
                    for i, t in enumerate(accepted_tokens)
                    if i < len(draft_tokens) and t == draft_tokens[i]
                ]
            ),
        }

        return accepted_tokens, new_pos, stats

    def decode_step_self_speculative_lookahead(
        self,
        token_id: int,
        max_draft_tokens: int = 4,
        draft_layers: int = 8,
    ) -> tuple[list[int], dict]:
        """Self-speculative decode step with GPU-side lookahead KV (no CPU copies).

        Uses lookahead KV cache management to avoid CPU-GPU transfers.

        IMPORTANT: Before calling this method:
        1. Run prefill and store KV using kv_cache_prefill_gqa()
        2. Call set_lookahead_confirmed_pos(prefill_len) to mark prefill KV as committed

        Algorithm:
        1. Generate draft tokens using early layers (writes to speculative positions)
        2. Reset lookahead, verify with full model in batch
        3. Accept tokens until first disagreement
        4. Re-run for accepted tokens to ensure correct KV
        5. Commit accepted tokens

        Args:
            token_id: Current token ID (the last accepted token)
            max_draft_tokens: Maximum number of draft tokens to generate
            draft_layers: Number of early layers to use as draft

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs
            - stats: Dict with 'draft_count', 'accepted_count' for analysis
        """
        confirmed_pos = self.get_lookahead_confirmed_pos()

        # === Step 1: Generate draft tokens using early layers ===
        # Reset lookahead before draft phase
        self.reset_lookahead_all()

        draft_tokens = []
        current_token = token_id

        for i in range(max_draft_tokens):
            pos = confirmed_pos + i
            ctx = confirmed_pos + i + 1
            # Forward through early layers only
            hidden = self._draft_forward_early_layers(current_token, pos, ctx, draft_layers)
            logits = self._draft_get_logits(hidden)
            logits_np = logits.to_numpy()[-1]
            next_token = int(np.argmax(logits_np))

            draft_tokens.append(next_token)
            current_token = next_token

        # === Step 2: Reset and verify with full model in batch ===
        self.reset_lookahead_all()

        verify_input = [token_id] + draft_tokens[:-1]
        verify_ctx = confirmed_pos + len(verify_input)

        hidden_batch = self._decode_step_fixed_cache_batch(verify_input, confirmed_pos, verify_ctx)
        verify_logits = self.get_logits(hidden_batch)
        verify_logits_np = verify_logits.to_numpy()

        # === Step 3: Accept/Reject tokens ===
        accepted_tokens = []
        for i, draft_token in enumerate(draft_tokens):
            target_token = int(np.argmax(verify_logits_np[i]))

            if target_token == draft_token:
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(target_token)
                break

        # === Step 4: Re-run for accepted tokens if partial accept ===
        if len(accepted_tokens) < max_draft_tokens:
            self.reset_lookahead_all()
            # Use CUDA Graph if available
            use_graph = hasattr(self, "_decode_graph_ready") and self._decode_graph_ready
            current = token_id
            for i, acc_token in enumerate(accepted_tokens):
                pos = confirmed_pos + i
                ctx = confirmed_pos + i + 1
                if use_graph:
                    self._decode_step_graph_replay(current, pos, ctx)
                else:
                    self._decode_step_fixed_cache(current, pos, ctx)
                current = acc_token

        # === Step 5: Commit accepted tokens ===
        self.commit_lookahead_all(len(accepted_tokens))

        stats = {
            "draft_count": len(draft_tokens),
            "accepted_count": len(
                [
                    t
                    for i, t in enumerate(accepted_tokens)
                    if i < len(draft_tokens) and t == draft_tokens[i]
                ]
            ),
        }

        return accepted_tokens, stats

    # =========================================================================
    # Lookahead KV Cache Management (GPU-side, no CPU copies)
    # =========================================================================

    def set_lookahead_confirmed_pos(self, pos: int) -> None:
        """Set confirmed position for all layers (e.g., after prefill).

        Args:
            pos: Position where KV is finalized (tokens 0 to pos-1 are committed).
        """
        for block in self.blocks:
            block.attn.set_confirmed_pos(pos)

    def reset_lookahead_all(self) -> None:
        """Reset lookahead pointer to confirmed position for all layers.

        Called at the start of each Jacobi iteration. This resets the write
        pointer without modifying KV cache - speculative positions will be
        overwritten by the next forward pass.
        """
        for block in self.blocks:
            block.attn.reset_lookahead()

    def commit_lookahead_all(self, n_accepted: int) -> None:
        """Commit accepted tokens for all layers.

        Args:
            n_accepted: Number of accepted tokens to commit.
        """
        for block in self.blocks:
            block.attn.commit_lookahead(n_accepted)

    def get_lookahead_confirmed_pos(self) -> int:
        """Get current confirmed position (from first layer)."""
        return self.blocks[0].attn.get_confirmed_pos()

    # =========================================================================
    # CUDA Graph for Decode (seq_len=1)
    # =========================================================================

    def init_decode_graph(self, max_seq_len: int = 512) -> None:
        """Initialize CUDA Graph for single-token decode.

        .. deprecated:: 0.2.11
            Use :class:`DecodeM1` strategy instead::

                from pygpukit.llm import DecodeM1
                m1 = DecodeM1()
                m1.bind(model)
                m1.init_graph(max_seq_len=512)

            Will be removed in v0.3.0.

        Pre-allocates buffers, pre-computes RoPE, initializes KV cache,
        and captures the decode graph for replay.

        IMPORTANT: Call this AFTER prefill and KV cache initialization.

        Args:
            max_seq_len: Maximum sequence length for KV cache.
        """
        import gc
        import warnings

        warnings.warn(
            "init_decode_graph() is deprecated and will be removed in v0.3.0. "
            "Use DecodeM1 strategy instead: m1 = DecodeM1(); m1.bind(model); m1.init_graph()",
            DeprecationWarning,
            stacklevel=2,
        )

        from pygpukit._native_loader import get_native_module

        CudaGraph = getattr(get_native_module(), "CudaGraph")  # noqa: B009

        dtype = str(self.embed_tokens.dtype)
        use_qk_norm = self.spec is not None and self.spec.use_qk_norm
        lm_head = self._lm_head if self._lm_head is not None else self.embed_tokens
        vocab_size = lm_head.shape[0]

        # Allocate decode buffers with CUDA Graph support
        self._decode_buffers = DecodeBuffers.allocate(
            self.config, dtype=dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
        )

        # Pre-compute RoPE tables on GPU if not already done
        if self.config.use_rope and not hasattr(self, "_rope_cos_gpu"):
            from pygpukit.ops.basic import cast_f32_to_bf16, cast_f32_to_f16

            cos_np, sin_np = precompute_freqs_cis(
                self.config.head_dim, max_seq_len, self.config.rope_theta
            )
            if dtype == "float16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                self._rope_cos_gpu = cast_f32_to_f16(cos_f32)
                self._rope_sin_gpu = cast_f32_to_f16(sin_f32)
            elif dtype == "bfloat16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                self._rope_cos_gpu = cast_f32_to_bf16(cos_f32)
                self._rope_sin_gpu = cast_f32_to_bf16(sin_f32)
            else:
                self._rope_cos_gpu = from_numpy(cos_np.astype(np.float32))
                self._rope_sin_gpu = from_numpy(sin_np.astype(np.float32))

        # Cache transposed lm_head for graph (if not already done)
        if not hasattr(self, "_lm_head_t_cache"):
            lm_head_np = lm_head.to_numpy()
            self._lm_head_t_cache = from_numpy(lm_head_np.T.copy())

        # Numpy buffers for CPU-side updates (reusable, no allocation)
        self._pos_np = np.array([0], dtype=np.int32)
        self._tok_np = np.array([0], dtype=np.int32)
        self._ctx_np = np.array([0], dtype=np.int32)

        # Store max_seq_len for graph replay
        self._graph_max_seq_len = max_seq_len

        # Warmup before capture (with pointer-based SDPA)
        buffers = self._decode_buffers
        self._ctx_np[0] = 1
        buffers.context_len_buf._get_native().copy_from_numpy(self._ctx_np)
        for _ in range(3):
            self._decode_step_zero_alloc(0, 0, 1, buffers)

        # Capture the decode graph
        self._decode_graph = CudaGraph()

        # Write initial values to GPU buffers
        self._pos_np[0] = 0
        buffers.position_buf._get_native().copy_from_numpy(self._pos_np)
        self._tok_np[0] = 0
        buffers.token_id_buf._get_native().copy_from_numpy(self._tok_np)
        self._ctx_np[0] = max_seq_len  # Capture with max for shared memory
        buffers.context_len_buf._get_native().copy_from_numpy(self._ctx_np)

        gc.disable()
        try:
            self._decode_graph.begin_capture()

            # Embedding lookup from token_id_buf
            embedding_lookup_ptr(self.embed_tokens, buffers.hidden, buffers.token_id_buf)

            # Transformer blocks
            for block in self.blocks:
                rmsnorm(
                    buffers.hidden,
                    block.attn_norm.weight,
                    block.attn_norm.eps,
                    out=buffers.norm_out,
                )
                copy_to(buffers.hidden, buffers.residual)
                self._attention_forward_zero_alloc(
                    block.attn,
                    buffers.norm_out,
                    0,
                    max_seq_len,
                    buffers,
                    use_position_ptr=True,
                    use_context_len_ptr=True,
                    max_kv_len=max_seq_len,
                )
                add_inplace(buffers.hidden, buffers.residual)
                copy_to(buffers.hidden, buffers.residual)
                rmsnorm(
                    buffers.hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=buffers.norm_out
                )
                self._mlp_forward_zero_alloc(block.mlp, buffers.norm_out, buffers)
                add_inplace(buffers.hidden, buffers.residual)

            # Final norm
            rmsnorm(
                buffers.hidden, self.final_norm.weight, self.final_norm.eps, out=buffers.norm_out
            )
            copy_to(buffers.norm_out, buffers.hidden)

            # LM head projection to logits
            matmul(buffers.hidden, self._lm_head_t_cache, out=buffers.logits)

            self._decode_graph.end_capture()
        finally:
            gc.enable()

        self._decode_graph_ready = True
        print(f"  [CUDA Graph] Captured {self._decode_graph.num_nodes} nodes for decode")

    def _decode_step_graph_replay(self, token_id: int, position: int, context_len: int) -> GPUArray:
        """Execute decode step using CUDA Graph replay.

        .. deprecated:: 0.2.11
            Use :class:`DecodeM1` strategy instead::

                m1.step_graph(token_id, position, context_len)

            Will be removed in v0.3.0.

        Updates GPU buffers and replays the captured graph.
        Returns logits buffer.

        Args:
            token_id: Input token ID
            position: Position in sequence
            context_len: Total context length (for KV cache attention)

        Returns:
            Logits buffer [1, vocab_size]
        """
        import warnings

        warnings.warn(
            "_decode_step_graph_replay() is deprecated and will be removed in v0.3.0. "
            "Use DecodeM1.step_graph() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        assert hasattr(self, "_decode_graph_ready") and self._decode_graph_ready, (
            "Call init_decode_graph() first"
        )

        buffers = self._decode_buffers

        # Update GPU buffers (outside graph)
        try:
            self._tok_np[0] = token_id
            buffers.token_id_buf._get_native().copy_from_numpy(self._tok_np)
            self._pos_np[0] = position
            buffers.position_buf._get_native().copy_from_numpy(self._pos_np)
            self._ctx_np[0] = context_len
            buffers.context_len_buf._get_native().copy_from_numpy(self._ctx_np)
        except RuntimeError as e:
            raise RuntimeError(
                f"H2D copy failed: tok={token_id}, pos={position}, ctx={context_len}. Error: {e}"
            ) from e

        # Device synchronize to ensure H2D copies are visible to the graph
        # Using device sync (not just default stream sync) because the graph runs
        # on its own non-blocking capture stream, which may not see memory written
        # by the default stream without explicit device-level synchronization
        from pygpukit.core.backend import get_backend

        get_backend().synchronize()

        # Replay graph
        self._decode_graph.replay()

        # Synchronize graph's stream to ensure replay completes before reading results
        # IMPORTANT: Must use graph.synchronize(), not default_stream().synchronize()
        # because the graph runs on its own capture stream, not the default stream
        try:
            self._decode_graph.synchronize()
        except RuntimeError as e:
            raise RuntimeError(
                f"Graph replay sync failed: tok={token_id}, pos={position}, ctx={context_len}. "
                f"Error: {e}"
            ) from e

        return buffers.logits

    # =========================================================================
    # Batch CUDA Graph (seq_len > 1 only)
    # =========================================================================
    # CUDA Graph is applied only to batch decode where launch overhead is non-negligible.
    # M=1 decode remains non-graph because compute dominates.
    # This separation is intentional and performance-driven.

    def init_decode_graph_batch(
        self,
        batch_size: int,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize CUDA Graph for batch decode (seq_len > 1).

        .. deprecated:: 0.2.11
            Use :class:`DecodeBatch` strategy instead::

                from pygpukit.llm import DecodeBatch
                batch = DecodeBatch(batch_size=8)
                batch.bind(model)
                batch.init_graph(max_seq_len=512)

            Will be removed in v0.3.0.

        Captures a graph for batch verification decode. The graph is replayed
        with different token IDs and positions without recapturing.

        IMPORTANT: This is separate from M=1 CUDA Graph. M=1 uses non-graph path.

        Args:
            batch_size: Fixed batch size to capture (must match during replay)
            max_seq_len: Maximum sequence length for RoPE pre-computation
        """
        import gc
        import warnings

        warnings.warn(
            "init_decode_graph_batch() is deprecated and will be removed in v0.3.0. "
            "Use DecodeBatch strategy instead: batch = DecodeBatch(batch_size); batch.bind(model); batch.init_graph()",
            DeprecationWarning,
            stacklevel=2,
        )

        from pygpukit._native_loader import get_native_module

        CudaGraph = getattr(get_native_module(), "CudaGraph")  # noqa: B009

        dtype = str(self.embed_tokens.dtype)
        use_qk_norm = self.spec is not None and self.spec.use_qk_norm
        lm_head = self._lm_head if self._lm_head is not None else self.embed_tokens
        vocab_size = lm_head.shape[0]

        # Allocate batch decode buffers if not already done
        if not hasattr(self, "_batch_decode_buffers") or self._batch_decode_buffers is None:
            self._batch_decode_buffers = DecodeBuffers.allocate(
                self.config,
                dtype=dtype,
                use_qk_norm=use_qk_norm,
                vocab_size=vocab_size,
                max_batch_size=batch_size,
            )

        buffers = self._batch_decode_buffers

        if buffers.max_batch_size < batch_size:
            raise ValueError(
                f"Buffers max_batch_size ({buffers.max_batch_size}) < requested batch_size ({batch_size})"
            )

        # Pre-compute RoPE tables on GPU if not already done
        if self.config.use_rope and not hasattr(self, "_rope_cos_gpu"):
            from pygpukit.ops.basic import cast_f32_to_bf16, cast_f32_to_f16

            cos_np, sin_np = precompute_freqs_cis(
                self.config.head_dim, max_seq_len, self.config.rope_theta
            )
            if dtype == "float16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                self._rope_cos_gpu = cast_f32_to_f16(cos_f32)
                self._rope_sin_gpu = cast_f32_to_f16(sin_f32)
            elif dtype == "bfloat16":
                cos_f32 = from_numpy(cos_np.astype(np.float32))
                sin_f32 = from_numpy(sin_np.astype(np.float32))
                self._rope_cos_gpu = cast_f32_to_bf16(cos_f32)
                self._rope_sin_gpu = cast_f32_to_bf16(sin_f32)
            else:
                self._rope_cos_gpu = from_numpy(cos_np.astype(np.float32))
                self._rope_sin_gpu = from_numpy(sin_np.astype(np.float32))

        # Cache transposed lm_head for graph
        if not hasattr(self, "_lm_head_t_cache"):
            lm_head_np = lm_head.to_numpy()
            self._lm_head_t_cache = from_numpy(lm_head_np.T.copy())

        # Numpy buffers for CPU-side updates
        self._batch_token_ids_np = np.zeros(batch_size, dtype=np.int32)
        self._batch_start_pos_np = np.array([0], dtype=np.int32)
        self._batch_ctx_len_np = np.array([0], dtype=np.int32)

        # Store graph parameters
        self._batch_graph_size = batch_size
        self._batch_graph_max_seq_len = max_seq_len

        # Warmup before capture
        print(f"  [Batch CUDA Graph] Warming up with batch_size={batch_size}...")
        self._batch_ctx_len_np[0] = max_seq_len
        buffers.context_len_buf._get_native().copy_from_numpy(self._batch_ctx_len_np)
        for _ in range(3):
            self._decode_step_batch_for_graph(list(range(batch_size)), 0, batch_size, buffers)
        from pygpukit.core import default_stream

        default_stream().synchronize()

        # Capture the batch decode graph
        print("  [Batch CUDA Graph] Capturing graph...")
        self._batch_decode_graph = CudaGraph()

        # Write initial values to GPU buffers
        self._batch_token_ids_np[:] = list(range(batch_size))
        buffers.token_ids_batch_buf._get_native().copy_from_numpy(self._batch_token_ids_np)
        self._batch_start_pos_np[0] = 0
        buffers.start_position_batch_buf._get_native().copy_from_numpy(self._batch_start_pos_np)
        self._batch_ctx_len_np[0] = max_seq_len
        buffers.context_len_buf._get_native().copy_from_numpy(self._batch_ctx_len_np)

        gc.disable()
        try:
            self._batch_decode_graph.begin_capture()

            # Batch embedding lookup from GPU buffer
            embedding_lookup_batch(
                self.embed_tokens,
                buffers.hidden_batch,
                buffers.token_ids_batch_buf,
                batch_size,
            )

            # Use full max_batch_size views for graph (fixed size)
            hidden = buffers.hidden_batch.slice_rows(batch_size)
            residual_buf = buffers.residual_batch.slice_rows(batch_size)
            norm_out_buf = buffers.norm_out_batch.slice_rows(batch_size)
            mlp_out_buf = buffers.mlp_down_batch.slice_rows(batch_size)

            # Get RoPE tables (may be None if not using RoPE)
            rope_cos_gpu = getattr(self, "_rope_cos_gpu", None)
            rope_sin_gpu = getattr(self, "_rope_sin_gpu", None)
            start_pos_buf = buffers.start_position_batch_buf

            # Transformer blocks - capture forward pass with zero-alloc
            for block in self.blocks:
                # Pre-norm
                rmsnorm(hidden, block.attn_norm.weight, block.attn_norm.eps, out=norm_out_buf)
                copy_to(hidden, residual_buf)

                # Attention (zero-alloc path for CUDA Graph)
                attn_out = block.attn.forward_fixed_cache_batch_zero_alloc(
                    norm_out_buf, 0, max_seq_len, buffers, rope_cos_gpu, rope_sin_gpu, start_pos_buf
                )

                # Residual
                add_inplace(residual_buf, attn_out)
                copy_to(residual_buf, hidden)

                # MLP norm
                rmsnorm(hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=norm_out_buf)
                copy_to(hidden, residual_buf)

                # MLP (zero-alloc path for CUDA Graph)
                self._mlp_forward_batch_zero_alloc(block.mlp, norm_out_buf, buffers, mlp_out_buf)

                # Residual
                add_inplace(residual_buf, mlp_out_buf)
                copy_to(residual_buf, hidden)

            # Final norm
            rmsnorm(hidden, self.final_norm.weight, self.final_norm.eps, out=norm_out_buf)

            # LM head projection to logits
            matmul(norm_out_buf, self._lm_head_t_cache, out=buffers.logits_batch)

            self._batch_decode_graph.end_capture()
        finally:
            gc.enable()

        self._batch_decode_graph_ready = True
        print(f"  [Batch CUDA Graph] Captured {self._batch_decode_graph.num_nodes} nodes")

    def _decode_step_batch_for_graph(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Batch decode step for graph capture warmup.

        Uses zero-alloc attention and MLP to match graph capture code path.
        """
        seq_len = len(token_ids)

        # Copy token IDs to GPU buffer
        self._batch_token_ids_np[:seq_len] = token_ids
        buffers.token_ids_batch_buf._get_native().copy_from_numpy(self._batch_token_ids_np)

        # Update start position buffer
        self._batch_start_pos_np[0] = start_position
        buffers.start_position_batch_buf._get_native().copy_from_numpy(self._batch_start_pos_np)

        # Batch embedding lookup from GPU buffer
        embedding_lookup_batch(
            self.embed_tokens,
            buffers.hidden_batch,
            buffers.token_ids_batch_buf,
            seq_len,
        )

        # Use sliced views
        hidden = buffers.hidden_batch.slice_rows(seq_len)
        residual_buf = buffers.residual_batch.slice_rows(seq_len)
        norm_out_buf = buffers.norm_out_batch.slice_rows(seq_len)
        mlp_out_buf = buffers.mlp_down_batch.slice_rows(seq_len)

        # Get RoPE tables (may be None if not using RoPE)
        rope_cos_gpu = getattr(self, "_rope_cos_gpu", None)
        rope_sin_gpu = getattr(self, "_rope_sin_gpu", None)
        start_pos_buf = buffers.start_position_batch_buf

        # Transformer blocks with zero-alloc
        for block in self.blocks:
            rmsnorm(hidden, block.attn_norm.weight, block.attn_norm.eps, out=norm_out_buf)
            copy_to(hidden, residual_buf)

            # Zero-alloc attention
            attn_out = block.attn.forward_fixed_cache_batch_zero_alloc(
                norm_out_buf,
                start_position,
                context_len,
                buffers,
                rope_cos_gpu,
                rope_sin_gpu,
                start_pos_buf,
            )

            add_inplace(residual_buf, attn_out)
            copy_to(residual_buf, hidden)

            rmsnorm(hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=norm_out_buf)
            copy_to(hidden, residual_buf)

            # Zero-alloc MLP
            self._mlp_forward_batch_zero_alloc(block.mlp, norm_out_buf, buffers, mlp_out_buf)

            add_inplace(residual_buf, mlp_out_buf)
            copy_to(residual_buf, hidden)

        rmsnorm(hidden, self.final_norm.weight, self.final_norm.eps, out=norm_out_buf)
        return norm_out_buf

    def _decode_step_batch_graph_replay(
        self,
        token_ids: list[int],
        start_position: int,
        context_len: int,
    ) -> GPUArray:
        """Execute batch decode step using CUDA Graph replay.

        .. deprecated:: 0.2.11
            Use :class:`DecodeBatch` strategy instead::

                batch.step_graph(token_ids, start_position, context_len)

            Will be removed in v0.3.0.

        Updates GPU buffers and replays the captured batch graph.

        Args:
            token_ids: Batch of token IDs (must match captured batch_size)
            start_position: Starting position in sequence
            context_len: Total context length

        Returns:
            Logits buffer [batch_size, vocab_size]
        """
        import warnings

        warnings.warn(
            "_decode_step_batch_graph_replay() is deprecated and will be removed in v0.3.0. "
            "Use DecodeBatch.step_graph() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        assert hasattr(self, "_batch_decode_graph_ready") and self._batch_decode_graph_ready, (
            "Call init_decode_graph_batch() first"
        )

        batch_size = len(token_ids)
        if batch_size != self._batch_graph_size:
            raise ValueError(
                f"Batch size mismatch: got {batch_size}, expected {self._batch_graph_size}"
            )

        buffers = self._batch_decode_buffers

        # Update GPU buffers
        self._batch_token_ids_np[:batch_size] = token_ids
        buffers.token_ids_batch_buf._get_native().copy_from_numpy(self._batch_token_ids_np)
        self._batch_start_pos_np[0] = start_position
        buffers.start_position_batch_buf._get_native().copy_from_numpy(self._batch_start_pos_np)
        self._batch_ctx_len_np[0] = context_len
        buffers.context_len_buf._get_native().copy_from_numpy(self._batch_ctx_len_np)

        # Device synchronize to ensure H2D copies are visible to the graph
        from pygpukit.core.backend import get_backend

        get_backend().synchronize()

        # Replay graph
        self._batch_decode_graph.replay()

        # Synchronize graph's stream
        self._batch_decode_graph.synchronize()

        return buffers.logits_batch.slice_rows(batch_size)

    # =========================================================================
    # Jacobi Decoding
    # =========================================================================

    def _init_jacobi_guess(
        self,
        last_token: int,
        position: int,
        context_len: int,
        n_tokens: int,
        strategy: Literal["repeat", "ngram", "greedy"],
    ) -> list[int]:
        """Initialize guess tokens for Jacobi decoding.

        Args:
            last_token: The last accepted token
            position: Current position in sequence
            context_len: Current context length
            n_tokens: Number of tokens to guess
            strategy: Initialization strategy
                - "repeat": Repeat last_token n times
                - "ngram": Use n-gram cache (falls back to repeat if no match)
                - "greedy": Run greedy decode to get initial guess

        Returns:
            List of n_tokens guessed token IDs
        """
        if strategy == "repeat":
            return [last_token] * n_tokens

        elif strategy == "ngram":
            # N-gram cache lookup (simple implementation)
            # Check if we have this token in recent history
            if hasattr(self, "_ngram_cache") and last_token in self._ngram_cache:
                cached = self._ngram_cache[last_token]
                if len(cached) >= n_tokens:
                    return cached[:n_tokens]
            # Fallback to repeat
            return [last_token] * n_tokens

        elif strategy == "greedy":
            # Run greedy sequential decode to get initial guess
            # This is expensive but gives best initial guess
            kv_snapshot = self.snapshot_kv_cache()
            guess = []
            pos = position
            ctx = context_len
            current = last_token

            for _ in range(n_tokens):
                hidden = self._decode_step_fixed_cache(current, pos, ctx)
                logits = self.get_logits(hidden)
                next_token = int(np.argmax(logits.to_numpy()[-1]))
                guess.append(next_token)
                current = next_token
                pos += 1
                ctx += 1

            # Restore KV cache
            self.restore_kv_cache(kv_snapshot)
            return guess

        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def decode_step_jacobi(
        self,
        token_id: int,
        position: int,
        context_len: int,
        n_tokens: int = 8,
        max_iter: int = 3,
        init_strategy: Literal["repeat", "ngram", "greedy"] = "repeat",
    ) -> tuple[list[int], int, dict]:
        """Jacobi decoding step - parallel iterative decoding without draft model.

        Algorithm:
        1. Initialize N future positions with a guess
        2. Batch forward pass on all N positions
        3. Update each position with argmax(logits)
        4. Repeat until convergence or max_iter
        5. Accept converged tokens

        Args:
            token_id: Current token ID (the last accepted token)
            position: Position in sequence (position of token_id)
            context_len: Total context length
            n_tokens: Number of tokens to decode in parallel (default: 8)
            max_iter: Maximum iterations for convergence (default: 3)
            init_strategy: How to initialize guess tokens
                - "repeat": Repeat last token (fast, simple)
                - "ngram": Use n-gram cache if available
                - "greedy": Run greedy decode first (slow but accurate)

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs
            - new_position: Updated position after accepting tokens
            - stats: Dict with 'iterations', 'converged', 'accepted_count'
        """
        # Snapshot KV cache before iterations
        kv_snapshot = self.snapshot_kv_cache()

        # Initialize guess
        guess = self._init_jacobi_guess(token_id, position, context_len, n_tokens, init_strategy)

        iterations_used = 0
        converged = False

        # Track which positions have stabilized (same value for 2 consecutive iterations)
        prev_guess = None

        for iteration in range(max_iter):
            iterations_used = iteration + 1

            # Restore KV to clean state before each iteration
            self.restore_kv_cache(kv_snapshot)

            # Batch forward: input [last_token, guess[0], ..., guess[n-2]]
            # produces logits for [guess[0], guess[1], ..., guess[n-1]]
            input_tokens = [token_id] + guess[:-1]
            verify_ctx = position + len(input_tokens)

            hidden = self._decode_step_fixed_cache_batch(input_tokens, position, verify_ctx)
            logits = self.get_logits(hidden)
            logits_np = logits.to_numpy()  # [n_tokens, vocab_size]

            # Update guess with argmax
            new_guess = [int(np.argmax(logits_np[i])) for i in range(n_tokens)]

            # Check full convergence
            if new_guess == guess:
                converged = True
                break

            prev_guess = guess
            guess = new_guess

        # Find longest converged prefix
        # Position i is "stable" if it hasn't changed in the last iteration
        # AND all positions before it are also stable
        if converged:
            # All tokens converged
            accepted_tokens = guess
        else:
            # Find the longest prefix where tokens match between last two iterations
            # This indicates those positions have stabilized
            accepted_tokens = []
            if prev_guess is not None:
                for i in range(n_tokens):
                    if guess[i] == prev_guess[i]:
                        accepted_tokens.append(guess[i])
                    else:
                        break
            # If no convergence at all, take just the first token (safest)
            if len(accepted_tokens) == 0:
                # First position always sees correct context, so it's reliable
                accepted_tokens = [guess[0]]

        # Restore KV and re-run to properly update cache
        self.restore_kv_cache(kv_snapshot)

        new_pos = position
        new_ctx = context_len
        prev_token = token_id

        for acc_token in accepted_tokens:
            self._decode_step_fixed_cache(prev_token, new_pos, new_ctx)
            prev_token = acc_token
            new_pos += 1
            new_ctx += 1

        # Update n-gram cache for future use
        if not hasattr(self, "_ngram_cache"):
            self._ngram_cache: dict[int, list[int]] = {}
        self._ngram_cache[token_id] = accepted_tokens.copy()

        stats = {
            "iterations": iterations_used,
            "converged": converged,
            "accepted_count": len(accepted_tokens),
        }

        return accepted_tokens, new_pos, stats

    # =========================================================================
    # Jacobi Decoding with Lookahead KV (GPU-side, no CPU copies)
    # =========================================================================

    def _init_jacobi_guess_lookahead(
        self,
        last_token: int,
        n_tokens: int,
        strategy: Literal["repeat", "ngram", "greedy"],
    ) -> list[int]:
        """Initialize guess tokens for Jacobi lookahead (no CPU copies).

        Args:
            last_token: The last accepted token
            n_tokens: Number of tokens to guess
            strategy: Initialization strategy
                - "repeat": Repeat last_token n times
                - "ngram": Use n-gram cache (falls back to repeat)
                - "greedy": Run greedy decode (writes to lookahead positions)

        Returns:
            List of n_tokens guessed token IDs
        """
        if strategy == "repeat":
            return [last_token] * n_tokens

        elif strategy == "ngram":
            if hasattr(self, "_ngram_cache") and last_token in self._ngram_cache:
                cached = self._ngram_cache[last_token]
                if len(cached) >= n_tokens:
                    return cached[:n_tokens]
            return [last_token] * n_tokens

        elif strategy == "greedy":
            # Run greedy decode using lookahead positions
            # This writes KV at [confirmed_pos, confirmed_pos + n_tokens)
            confirmed_pos = self.get_lookahead_confirmed_pos()
            guess = []
            current = last_token

            for i in range(n_tokens):
                pos = confirmed_pos + i
                ctx = confirmed_pos + i + 1
                hidden = self._decode_step_fixed_cache(current, pos, ctx)
                logits = self.get_logits(hidden)
                next_token = int(np.argmax(logits.to_numpy()[-1]))
                guess.append(next_token)
                current = next_token

            # Reset lookahead after greedy init (KV will be overwritten)
            self.reset_lookahead_all()
            return guess

        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def decode_step_jacobi_lookahead(
        self,
        token_id: int,
        n_tokens: int = 8,
        max_iter: int = 3,
        init_strategy: Literal["repeat", "ngram", "greedy"] = "repeat",
    ) -> tuple[list[int], dict]:
        """Jacobi decoding step with GPU-side lookahead KV (no CPU copies).

        This method uses the lookahead KV cache management to avoid all
        CPU-GPU memory transfers during Jacobi iterations.

        IMPORTANT: Before calling this method:
        1. Run prefill and store KV using kv_cache_prefill_gqa()
        2. Call set_lookahead_confirmed_pos(prefill_len) to mark prefill KV as committed

        Algorithm:
        1. Initialize N future positions with a guess
        2. Reset lookahead pointer (no KV modification)
        3. Batch forward - writes KV at [confirmed_pos, confirmed_pos + n_tokens)
        4. Update guess with argmax(logits)
        5. Repeat until convergence or max_iter
        6. Commit accepted tokens by advancing confirmed_pos

        Args:
            token_id: Current token ID (the last accepted token)
            n_tokens: Number of tokens to decode in parallel (default: 8)
            max_iter: Maximum iterations for convergence (default: 3)
            init_strategy: How to initialize guess tokens
                - "repeat": Repeat last token (fast, simple)
                - "ngram": Use n-gram cache if available
                - "greedy": Run greedy decode first (slow but accurate)

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs
            - stats: Dict with 'iterations', 'converged', 'accepted_count'
        """
        # Get confirmed position (this is our starting point)
        confirmed_pos = self.get_lookahead_confirmed_pos()

        # Initialize guess (may use lookahead positions for greedy)
        guess = self._init_jacobi_guess_lookahead(token_id, n_tokens, init_strategy)

        iterations_used = 0
        converged = False
        prev_guess = None

        for iteration in range(max_iter):
            iterations_used = iteration + 1

            # Reset lookahead pointer (does NOT modify KV cache)
            self.reset_lookahead_all()

            # Batch forward: input [last_token, guess[0], ..., guess[n-2]]
            # produces logits for [guess[0], guess[1], ..., guess[n-1]]
            # Writes KV at [confirmed_pos, confirmed_pos + n_tokens)
            input_tokens = [token_id] + guess[:-1]
            start_pos = confirmed_pos
            ctx_len = confirmed_pos + len(input_tokens)

            hidden = self._decode_step_fixed_cache_batch(input_tokens, start_pos, ctx_len)
            logits = self.get_logits(hidden)
            logits_np = logits.to_numpy()  # [n_tokens, vocab_size]

            # Update guess with argmax
            new_guess = [int(np.argmax(logits_np[i])) for i in range(n_tokens)]

            # Check full convergence
            if new_guess == guess:
                converged = True
                break

            prev_guess = guess
            guess = new_guess

        # Find longest converged prefix
        if converged:
            accepted_tokens = guess
        else:
            accepted_tokens = []
            if prev_guess is not None:
                for i in range(n_tokens):
                    if guess[i] == prev_guess[i]:
                        accepted_tokens.append(guess[i])
                    else:
                        break
            if len(accepted_tokens) == 0:
                accepted_tokens = [guess[0]]

        # Commit accepted tokens - this is the ONLY state change
        # The KV for accepted tokens is already written from the last iteration
        # We just need to run one more forward to ensure KV is correct
        self.reset_lookahead_all()

        # Re-run with just the accepted tokens to ensure KV is correct
        if len(accepted_tokens) < n_tokens:
            # KV may have extra speculative entries - need to overwrite with correct values
            # Run sequential for accepted tokens only
            # Use CUDA Graph if available
            use_graph = hasattr(self, "_decode_graph_ready") and self._decode_graph_ready
            current = token_id
            for i, acc_token in enumerate(accepted_tokens):
                pos = confirmed_pos + i
                ctx = confirmed_pos + i + 1
                if use_graph:
                    self._decode_step_graph_replay(current, pos, ctx)
                else:
                    self._decode_step_fixed_cache(current, pos, ctx)
                current = acc_token
        # If all converged, KV is already correct from last batch forward

        # Commit the accepted tokens
        self.commit_lookahead_all(len(accepted_tokens))

        # Update n-gram cache for future use
        if not hasattr(self, "_ngram_cache"):
            self._ngram_cache: dict[int, list[int]] = {}
        self._ngram_cache[token_id] = accepted_tokens.copy()

        stats = {
            "iterations": iterations_used,
            "converged": converged,
            "accepted_count": len(accepted_tokens),
        }

        return accepted_tokens, stats


# =============================================================================
# Type Aliases
# =============================================================================

# GPT2Model and LlamaModel are now simple aliases for CausalTransformerModel.
# All models use CausalTransformerModel as the single runtime type.
GPT2Model = CausalTransformerModel
LlamaModel = CausalTransformerModel

# Legacy component aliases (import from layers module)
RMSNorm = Norm  # Use Norm with norm_type="rmsnorm"
LayerNorm = Norm  # Use Norm with norm_type="layernorm"
LlamaAttention = Attention
LlamaMLP = MLP
LlamaBlock = TransformerBlock
CausalSelfAttention = Attention
