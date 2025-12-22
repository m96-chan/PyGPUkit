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
)
from pygpukit.llm.sampling import sample_token
from pygpukit.ops.basic import (
    add,
    add_inplace,
    bias_add_inplace,
    copy_to,
    embedding_lookup,
    embedding_lookup_ptr,
    gelu,
    kv_cache_update_gqa,
    kv_cache_update_gqa_ptr,
    matmul,
    mul_inplace,
    repeat_interleave_axis1,
    reshape_copy,
    rmsnorm,
    rope_inplace,
    sample_token_gpu,
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

    # Type hints for dynamically added attributes
    _batch_decode_buffers: DecodeBuffers | None
    _batch_token_ids_np: np.ndarray

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
