"""Unified Transformer implementation for PyGPUkit.

Provides a common Transformer abstraction that supports both GPT-2 and LLaMA
architectures through configuration differences only.

Key features:
- Hybrid Attention: CPU for seq_len=1 (decode), GPU for prefill
- GPU-native operations: RMSNorm, LayerNorm, SDPA, SiLU, GELU, RoPE
- Unified TransformerConfig for all model variants
- Backward-compatible loaders for GPT-2 and LLaMA safetensors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.basic import (
    add,
    bias_add_inplace,
    gelu,
    layernorm,
    matmul,
    mul,
    reshape_copy,
    rmsnorm,
    rope_inplace,
    sdpa_causal,
    silu,
    transpose,
    transpose_3d_021,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Common Sampling Functions
# =============================================================================


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p.

    Args:
        logits: Logits array [vocab_size]
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative prob <= top_p (1.0 = disabled)

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # Convert to probabilities
    logits_max = logits.max()
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum()

    # Top-k filtering
    if top_k > 0 and top_k < len(probs):
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_k_indices] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        cutoff_idx = min(cutoff_idx, len(sorted_probs))
        mask = np.zeros_like(probs, dtype=bool)
        mask[sorted_indices[:cutoff_idx]] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()

    # Sample
    if temperature == 0:
        return int(np.argmax(probs))
    else:
        return int(np.random.choice(len(probs), p=probs))


# =============================================================================
# Unified Transformer Configuration
# =============================================================================


@dataclass
class TransformerConfig:
    """Unified configuration for Transformer models.

    Supports both GPT-2 and LLaMA style architectures through configuration.

    GPT-2 style:
        norm_type="layernorm", activation="gelu", use_rope=False

    LLaMA style:
        norm_type="rmsnorm", activation="silu", use_rope=True
    """

    # Core dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int | None = None  # None = MHA, int = GQA/MQA
    intermediate_size: int | None = None  # None = 4 * hidden_size

    # Architecture choices
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    activation: Literal["gelu", "silu"] = "silu"
    use_rope: bool = True
    causal: bool = True

    # Hyperparameters
    max_position_embeddings: int = 2048
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Weight tying
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_heads // self.num_kv_heads


# =============================================================================
# Legacy Config Classes (for backward compatibility)
# =============================================================================


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model (legacy, use TransformerConfig)."""

    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_positions: int = 1024
    layer_norm_eps: float = 1e-5

    @property
    def n_inner(self) -> int:
        return 4 * self.n_embd

    def to_transformer_config(self) -> TransformerConfig:
        """Convert to unified TransformerConfig."""
        return TransformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.n_embd,
            num_layers=self.n_layer,
            num_heads=self.n_head,
            num_kv_heads=self.n_head,  # MHA
            intermediate_size=self.n_inner,
            norm_type="layernorm",
            activation="gelu",
            use_rope=False,
            causal=True,
            max_position_embeddings=self.n_positions,
            norm_eps=self.layer_norm_eps,
        )


@dataclass
class LlamaConfig:
    """Configuration for Llama model (legacy, use TransformerConfig)."""

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_transformer_config(self) -> TransformerConfig:
        """Convert to unified TransformerConfig."""
        return TransformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            norm_type="rmsnorm",
            activation="silu",
            use_rope=True,
            causal=True,
            max_position_embeddings=self.max_position_embeddings,
            norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
        )


# =============================================================================
# Common Building Blocks
# =============================================================================


class Linear:
    """Linear layer: y = xW^T + b

    Weights are stored as [out_features, in_features] (PyTorch convention).
    """

    def __init__(self, weight: GPUArray, bias: GPUArray | None = None):
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.ndim}D")
        self.weight = weight
        self.bias = bias
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]
        self._weight_t: GPUArray | None = None

    def __call__(self, x: GPUArray) -> GPUArray:
        if x.ndim != 2:
            raise ValueError(f"input must be 2D [batch, in_features], got {x.ndim}D")
        if x.shape[1] != self.in_features:
            raise ValueError(f"input features {x.shape[1]} != weight {self.in_features}")

        if self._weight_t is None:
            self._weight_t = transpose(self.weight)

        y = matmul(x, self._weight_t)

        if self.bias is not None:
            bias_add_inplace(y, self.bias)

        return y


class Norm:
    """Unified normalization layer supporting RMSNorm and LayerNorm."""

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray | None = None,
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        eps: float = 1e-5,
    ):
        self.weight = weight
        self.bias = bias
        self.norm_type = norm_type
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        if self.norm_type == "rmsnorm":
            return rmsnorm(x, self.weight, self.eps)
        else:
            if self.bias is None:
                raise ValueError("LayerNorm requires bias")
            return layernorm(x, self.weight, self.bias, self.eps)


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute rotary embedding cos/sin tables."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    return cos, sin


def apply_rotary_pos_emb_numpy(
    q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rotary position embeddings to Q and K (numpy version)."""

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1)

    cos = cos[:, np.newaxis, :]
    sin = sin[:, np.newaxis, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Unified Attention
# =============================================================================


class Attention:
    """Unified attention with Hybrid CPU/GPU execution.

    Supports:
    - Multi-Head Attention (MHA): num_kv_heads == num_heads
    - Grouped Query Attention (GQA): num_kv_heads < num_heads
    - RoPE: enabled via config.use_rope
    - Hybrid execution: CPU for seq_len=1, GPU for longer sequences
    """

    def __init__(
        self,
        q_proj: GPUArray,
        k_proj: GPUArray,
        v_proj: GPUArray,
        o_proj: GPUArray,
        config: TransformerConfig,
        q_bias: GPUArray | None = None,
        k_bias: GPUArray | None = None,
        v_bias: GPUArray | None = None,
        o_bias: GPUArray | None = None,
    ):
        self.q_proj = Linear(q_proj, q_bias)
        self.k_proj = Linear(k_proj, k_bias)
        self.v_proj = Linear(v_proj, v_bias)
        self.o_proj = Linear(o_proj, o_bias)

        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups

        # Precompute RoPE if enabled
        if config.use_rope:
            self._cos, self._sin = precompute_freqs_cis(
                self.head_dim, config.max_position_embeddings, config.rope_theta
            )
        else:
            self._cos, self._sin = None, None

    def __call__(
        self,
        x: GPUArray,
        position_ids: list[int] | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, tuple | None]:
        """Forward pass with hybrid CPU/GPU attention.

        Args:
            x: Input tensor [seq_len, hidden_size]
            position_ids: Position IDs for RoPE (auto-generated if None)
            past_kv: Tuple of (past_k, past_v) numpy arrays
            use_cache: Whether to return KV cache

        Returns:
            Tuple of (output, present_kv)
        """
        seq_len = x.shape[0]

        if position_ids is None:
            position_ids = list(range(seq_len))

        # Hybrid routing: CPU for seq_len=1, GPU for prefill
        if seq_len > 1:
            return self._forward_gpu(x, position_ids, past_kv, use_cache)
        else:
            return self._forward_cpu(x, position_ids, past_kv, use_cache)

    def _forward_gpu(
        self,
        x: GPUArray,
        position_ids: list[int],
        past_kv: tuple | None,
        use_cache: bool,
    ) -> tuple[GPUArray, tuple | None]:
        """GPU path for long sequences (prefill)."""
        seq_len = x.shape[0]

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = reshape_copy(q, (seq_len, self.num_heads, self.head_dim))
        k = reshape_copy(k, (seq_len, self.num_kv_heads, self.head_dim))
        v = reshape_copy(v, (seq_len, self.num_kv_heads, self.head_dim))

        # Apply RoPE on GPU
        if self.config.use_rope:
            cos = from_numpy(self._cos[position_ids].astype(np.float32))
            sin = from_numpy(self._sin[position_ids].astype(np.float32))
            rope_inplace(q, k, cos, sin)

        # Convert to numpy for KV cache
        k_np = k.to_numpy()
        v_np = v.to_numpy()

        # Concatenate with past KV
        if past_kv is not None:
            past_k, past_v = past_kv
            k_np = np.concatenate([past_k, k_np], axis=0)
            v_np = np.concatenate([past_v, v_np], axis=0)

        present_kv = (k_np.copy(), v_np.copy()) if use_cache else None

        # Expand for GQA
        if self.num_kv_groups > 1:
            k_expanded = np.repeat(k_np, self.num_kv_groups, axis=1)
            v_expanded = np.repeat(v_np, self.num_kv_groups, axis=1)
        else:
            k_expanded = k_np
            v_expanded = v_np

        # GPU SDPA
        q_t = transpose_3d_021(q)
        k_t = from_numpy(k_expanded.transpose(1, 0, 2).astype(np.float32))
        v_t = from_numpy(v_expanded.transpose(1, 0, 2).astype(np.float32))

        attn_output = sdpa_causal(q_t, k_t, v_t)

        # Reshape output
        attn_output = transpose_3d_021(attn_output)
        attn_output = reshape_copy(attn_output, (seq_len, self.num_heads * self.head_dim))

        return self.o_proj(attn_output), present_kv

    def _forward_cpu(
        self,
        x: GPUArray,
        position_ids: list[int],
        past_kv: tuple | None,
        use_cache: bool,
    ) -> tuple[GPUArray, tuple | None]:
        """CPU path for seq_len=1 (decode) - minimal kernel overhead."""
        seq_len = x.shape[0]

        # Project Q, K, V (GPU matmul, then transfer)
        q = self.q_proj(x).to_numpy()
        k = self.k_proj(x).to_numpy()
        v = self.v_proj(x).to_numpy()

        # Reshape for multi-head
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE (CPU)
        if self.config.use_rope:
            cos = self._cos[position_ids]
            sin = self._sin[position_ids]
            q, k = apply_rotary_pos_emb_numpy(q, k, cos, sin)

        # Concatenate with past KV
        if past_kv is not None:
            past_k, past_v = past_kv
            k = np.concatenate([past_k, k], axis=0)
            v = np.concatenate([past_v, v], axis=0)

        present_kv = (k.copy(), v.copy()) if use_cache else None

        # Expand for GQA
        if self.num_kv_groups > 1:
            k_expanded = np.repeat(k, self.num_kv_groups, axis=1)
            v_expanded = np.repeat(v, self.num_kv_groups, axis=1)
        else:
            k_expanded = k
            v_expanded = v

        # CPU attention
        q = q.transpose(1, 0, 2)
        k_expanded = k_expanded.transpose(1, 0, 2)
        v_expanded = v_expanded.transpose(1, 0, 2)

        q_len = q.shape[1]
        kv_len = k_expanded.shape[1]
        scale = 1.0 / np.sqrt(self.head_dim)

        attn_scores = np.matmul(q, k_expanded.transpose(0, 2, 1)) * scale

        # Causal mask
        if self.config.causal:
            causal_mask = np.zeros((q_len, kv_len), dtype=bool)
            for i in range(q_len):
                start_mask = kv_len - q_len + i + 1
                if start_mask < kv_len:
                    causal_mask[i, start_mask:] = True
            attn_scores[:, causal_mask] = -1e9

        # Softmax
        attn_max = attn_scores.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn_scores - attn_max)
        attn_weights = attn_exp / attn_exp.sum(axis=-1, keepdims=True)

        # Attention output
        attn_output = np.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 0, 2)
        attn_output = attn_output.reshape(seq_len, self.num_heads * self.head_dim)

        # Output projection (GPU)
        out = from_numpy(attn_output.astype(np.float32))
        return self.o_proj(out), present_kv


# =============================================================================
# Unified MLP
# =============================================================================


class MLP:
    """Unified MLP supporting GELU and SwiGLU activations.

    GELU (GPT-2 style):
        fc1 -> GELU -> fc2

    SwiGLU (LLaMA style):
        gate_proj -> SiLU -> * up_proj -> down_proj
    """

    def __init__(
        self,
        config: TransformerConfig,
        # GELU path weights
        fc1_weight: GPUArray | None = None,
        fc1_bias: GPUArray | None = None,
        fc2_weight: GPUArray | None = None,
        fc2_bias: GPUArray | None = None,
        # SwiGLU path weights
        gate_proj: GPUArray | None = None,
        up_proj: GPUArray | None = None,
        down_proj: GPUArray | None = None,
    ):
        self.config = config
        self.activation = config.activation

        if config.activation == "gelu":
            if fc1_weight is None or fc2_weight is None:
                raise ValueError("GELU MLP requires fc1_weight and fc2_weight")
            self.fc1 = Linear(fc1_weight, fc1_bias)
            self.fc2 = Linear(fc2_weight, fc2_bias)
        else:  # silu (SwiGLU)
            if gate_proj is None or up_proj is None or down_proj is None:
                raise ValueError("SwiGLU MLP requires gate_proj, up_proj, down_proj")
            self.gate_proj = Linear(gate_proj)
            self.up_proj = Linear(up_proj)
            self.down_proj = Linear(down_proj)

    def __call__(self, x: GPUArray) -> GPUArray:
        if self.activation == "gelu":
            # GELU path: fc1 -> GELU -> fc2
            h = self.fc1(x)
            h = gelu(h)
            return self.fc2(h)
        else:
            # SwiGLU path: gate_proj -> SiLU -> * up_proj -> down_proj
            gate = silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(mul(gate, up))


# =============================================================================
# Unified TransformerBlock
# =============================================================================


class TransformerBlock:
    """Unified transformer block.

    Structure:
        Norm -> Attention -> Residual
        Norm -> MLP -> Residual
    """

    def __init__(
        self,
        attn_norm: Norm,
        attn: Attention,
        mlp_norm: Norm,
        mlp: MLP,
    ):
        self.attn_norm = attn_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def __call__(
        self,
        x: GPUArray,
        position_ids: list[int] | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, tuple | None]:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        attn_out, present_kv = self.attn(x, position_ids, past_kv, use_cache)
        x = add(residual, attn_out)

        # MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = add(residual, x)

        return x, present_kv


# =============================================================================
# Unified CausalTransformerModel
# =============================================================================


class CausalTransformerModel:
    """Unified causal transformer model.

    Supports GPT-2 and LLaMA architectures through configuration.
    """

    def __init__(
        self,
        config: TransformerConfig,
        embed_tokens: GPUArray,
        blocks: list[TransformerBlock],
        final_norm: Norm,
        lm_head: GPUArray | None = None,
        position_embed: GPUArray | None = None,  # For GPT-2 style
    ):
        self.config = config
        self.embed_tokens = embed_tokens
        self.blocks = blocks
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.position_embed = position_embed

    def __call__(
        self,
        input_ids: list[int],
        position_ids: list[int] | None = None,
        past_key_values: list[tuple] | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, list[tuple] | None]:
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

        # Token embeddings
        embed_np = self.embed_tokens.to_numpy()
        hidden = embed_np[input_ids].astype(np.float32)

        # Add position embeddings (GPT-2 style)
        if self.position_embed is not None:
            pos_embed_np = self.position_embed.to_numpy()
            hidden = hidden + pos_embed_np[position_ids]

        hidden = from_numpy(hidden)

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

    def get_logits(self, hidden: GPUArray) -> GPUArray:
        """Compute logits from hidden states."""
        hidden_np = hidden.to_numpy()

        if self.lm_head is not None:
            lm_head_np = self.lm_head.to_numpy()
        else:
            # Tied embeddings
            lm_head_np = self.embed_tokens.to_numpy()

        logits = hidden_np @ lm_head_np.T
        return from_numpy(logits.astype(np.float32))

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        use_cache: bool = True,
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

        Returns:
            List of all token IDs (input + generated)
        """
        tokens = list(input_ids)
        past_key_values = None

        if use_cache:
            # Prefill
            hidden, past_key_values = self(tokens, use_cache=True)
            logits = self.get_logits(hidden)
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
                last_logits = logits.to_numpy()[-1]
                next_token = sample_token(last_logits, temperature, top_k, top_p)
                tokens.append(next_token)

                if eos_token_id is not None and next_token == eos_token_id:
                    break
        else:
            for _ in range(max_new_tokens):
                hidden, _ = self(tokens, use_cache=False)
                logits = self.get_logits(hidden)
                last_logits = logits.to_numpy()[-1]
                next_token = sample_token(last_logits, temperature, top_k, top_p)
                tokens.append(next_token)

                if eos_token_id is not None and next_token == eos_token_id:
                    break

        return tokens


# =============================================================================
# Legacy Aliases (for backward compatibility)
# =============================================================================


# RMSNorm alias
class RMSNorm(Norm):
    """RMSNorm layer (legacy alias)."""

    def __init__(self, weight: GPUArray, eps: float = 1e-5):
        super().__init__(weight, None, "rmsnorm", eps)


# LayerNorm alias
class LayerNorm(Norm):
    """LayerNorm layer (legacy alias)."""

    def __init__(self, weight: GPUArray, bias: GPUArray, eps: float = 1e-5):
        super().__init__(weight, bias, "layernorm", eps)


# Legacy LlamaAttention alias
LlamaAttention = Attention


# Legacy LlamaMLP
class LlamaMLP(MLP):
    """LLaMA MLP (legacy alias)."""

    def __init__(
        self,
        gate_proj: GPUArray,
        up_proj: GPUArray,
        down_proj: GPUArray,
    ):
        # Create minimal config for SwiGLU
        config = TransformerConfig(activation="silu")
        super().__init__(config, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)


# Legacy LlamaBlock alias
LlamaBlock = TransformerBlock


# Legacy LlamaModel alias
class LlamaModel(CausalTransformerModel):
    """LLaMA model (legacy alias)."""

    pass


# Legacy GPT2Model alias
class GPT2Model(CausalTransformerModel):
    """GPT-2 model (legacy alias)."""

    def lm_head(self, hidden: GPUArray) -> GPUArray:
        """Legacy lm_head method."""
        return self.get_logits(hidden)


# =============================================================================
# Legacy Attention Classes (for backward compatibility)
# =============================================================================


class CausalSelfAttention(Attention):
    """GPT-2 style causal self-attention (legacy alias)."""

    def __init__(
        self,
        c_attn_weight: GPUArray,
        c_attn_bias: GPUArray | None,
        c_proj_weight: GPUArray,
        c_proj_bias: GPUArray | None,
        n_head: int,
        n_embd: int,
    ):
        # GPT-2 uses combined QKV projection
        # Split weights for unified Attention class
        # c_attn: [3*n_embd, n_embd] -> Q, K, V each [n_embd, n_embd]
        c_attn_np = c_attn_weight.to_numpy()
        q_weight = from_numpy(c_attn_np[:n_embd].copy())
        k_weight = from_numpy(c_attn_np[n_embd : 2 * n_embd].copy())
        v_weight = from_numpy(c_attn_np[2 * n_embd :].copy())

        q_bias, k_bias, v_bias = None, None, None
        if c_attn_bias is not None:
            c_attn_bias_np = c_attn_bias.to_numpy()
            q_bias = from_numpy(c_attn_bias_np[:n_embd].copy())
            k_bias = from_numpy(c_attn_bias_np[n_embd : 2 * n_embd].copy())
            v_bias = from_numpy(c_attn_bias_np[2 * n_embd :].copy())

        config = TransformerConfig(
            hidden_size=n_embd,
            num_heads=n_head,
            num_kv_heads=n_head,  # MHA
            norm_type="layernorm",
            activation="gelu",
            use_rope=False,
            causal=True,
        )

        super().__init__(
            q_weight,
            k_weight,
            v_weight,
            c_proj_weight,
            config,
            q_bias,
            k_bias,
            v_bias,
            c_proj_bias,
        )


# =============================================================================
# Legacy MLP Class (for backward compatibility)
# =============================================================================


class _LegacyMLP(MLP):
    """GPT-2 style MLP (legacy)."""

    def __init__(
        self,
        c_fc_weight: GPUArray,
        c_fc_bias: GPUArray | None,
        c_proj_weight: GPUArray,
        c_proj_bias: GPUArray | None,
    ):
        config = TransformerConfig(activation="gelu")
        super().__init__(
            config,
            fc1_weight=c_fc_weight,
            fc1_bias=c_fc_bias,
            fc2_weight=c_proj_weight,
            fc2_bias=c_proj_bias,
        )


# =============================================================================
# Safetensors Loaders
# =============================================================================


def load_gpt2_from_safetensors(
    model_path: str,
    config: GPT2Config | None = None,
    load_attention: bool = True,
) -> GPT2Model:
    """Load GPT-2 model from safetensors file.

    Args:
        model_path: Path to model.safetensors
        config: Model configuration (defaults to GPT-2 small)
        load_attention: Whether to load attention weights

    Returns:
        GPT2Model instance (CausalTransformerModel)
    """
    from pygpukit.llm import SafeTensorsFile

    if config is None:
        config = GPT2Config()

    transformer_config = config.to_transformer_config()
    st = SafeTensorsFile(model_path)

    def load_tensor(name: str, do_transpose: bool = False) -> GPUArray:
        data = st.tensor_bytes(name)
        info = st.tensor_info(name)
        dtype_map = {0: np.float32, 1: np.float16, 2: np.float32, 3: np.float64}
        np_dtype = dtype_map.get(info.dtype, np.float32)
        arr = np.frombuffer(data, dtype=np_dtype).reshape(info.shape)
        if do_transpose and arr.ndim == 2:
            arr = arr.T
        return from_numpy(arr.copy().astype(np.float32))

    def try_load(name: str, do_transpose: bool = False) -> GPUArray | None:
        if name in st.tensor_names:
            return load_tensor(name, do_transpose)
        return None

    # Embeddings
    wte = load_tensor("wte.weight")
    wpe = load_tensor("wpe.weight")

    # Blocks
    blocks = []
    for i in range(config.n_layer):
        prefix = f"h.{i}."

        # Attention norm
        ln_1_w = load_tensor(f"{prefix}ln_1.weight")
        ln_1_b = load_tensor(f"{prefix}ln_1.bias")
        attn_norm = Norm(ln_1_w, ln_1_b, "layernorm", config.layer_norm_eps)

        # Attention
        attn = None
        if load_attention:
            c_attn_w = load_tensor(f"{prefix}attn.c_attn.weight", do_transpose=True)
            c_attn_b = try_load(f"{prefix}attn.c_attn.bias")
            c_proj_w = load_tensor(f"{prefix}attn.c_proj.weight", do_transpose=True)
            c_proj_b = try_load(f"{prefix}attn.c_proj.bias")
            attn = CausalSelfAttention(
                c_attn_w, c_attn_b, c_proj_w, c_proj_b, config.n_head, config.n_embd
            )

        # MLP norm
        ln_2_w = load_tensor(f"{prefix}ln_2.weight")
        ln_2_b = load_tensor(f"{prefix}ln_2.bias")
        mlp_norm = Norm(ln_2_w, ln_2_b, "layernorm", config.layer_norm_eps)

        # MLP
        c_fc_w = load_tensor(f"{prefix}mlp.c_fc.weight", do_transpose=True)
        c_fc_b = try_load(f"{prefix}mlp.c_fc.bias")
        c_proj_w = load_tensor(f"{prefix}mlp.c_proj.weight", do_transpose=True)
        c_proj_b = try_load(f"{prefix}mlp.c_proj.bias")
        mlp = MLP(
            transformer_config,
            fc1_weight=c_fc_w,
            fc1_bias=c_fc_b,
            fc2_weight=c_proj_w,
            fc2_bias=c_proj_b,
        )

        if attn is not None:
            block = TransformerBlock(attn_norm, attn, mlp_norm, mlp)
            blocks.append(block)

    # Final norm
    ln_f_w = load_tensor("ln_f.weight")
    ln_f_b = load_tensor("ln_f.bias")
    final_norm = Norm(ln_f_w, ln_f_b, "layernorm", config.layer_norm_eps)

    return GPT2Model(transformer_config, wte, blocks, final_norm, None, wpe)


def load_llama_from_safetensors(
    model_path: str,
    config: LlamaConfig | None = None,
) -> LlamaModel:
    """Load Llama model from safetensors file.

    Args:
        model_path: Path to model.safetensors
        config: Model configuration (auto-detected if None)

    Returns:
        LlamaModel instance (CausalTransformerModel)
    """
    from pygpukit.llm import SafeTensorsFile

    st = SafeTensorsFile(model_path)

    # Auto-detect config
    if config is None:
        embed_info = st.tensor_info("model.embed_tokens.weight")
        vocab_size = embed_info.shape[0]
        hidden_size = embed_info.shape[1]

        num_layers = 0
        while f"model.layers.{num_layers}.self_attn.q_proj.weight" in st.tensor_names:
            num_layers += 1

        q_info = st.tensor_info("model.layers.0.self_attn.q_proj.weight")
        k_info = st.tensor_info("model.layers.0.self_attn.k_proj.weight")
        gate_info = st.tensor_info("model.layers.0.mlp.gate_proj.weight")

        head_dim = 64
        num_heads = q_info.shape[0] // head_dim
        num_kv_heads = k_info.shape[0] // head_dim

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=gate_info.shape[0],
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )

    transformer_config = config.to_transformer_config()

    def load_tensor(name: str) -> GPUArray:
        data = st.tensor_bytes(name)
        info = st.tensor_info(name)
        if info.dtype == 2:  # BFloat16
            arr = np.frombuffer(data, dtype=np.uint16).reshape(info.shape)
            arr_f32 = np.empty(arr.shape, dtype=np.float32)
            arr_f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
            return from_numpy(arr_f32)
        else:
            dtype_map = {0: np.float32, 1: np.float16, 3: np.float64}
            np_dtype = dtype_map.get(info.dtype, np.float32)
            arr = np.frombuffer(data, dtype=np_dtype).reshape(info.shape)
            return from_numpy(arr.copy().astype(np.float32))

    # Embeddings
    embed_tokens = load_tensor("model.embed_tokens.weight")

    # Blocks
    blocks = []
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."

        # Attention norm
        attn_norm = Norm(
            load_tensor(f"{prefix}input_layernorm.weight"),
            None,
            "rmsnorm",
            config.rms_norm_eps,
        )

        # Attention
        attn = Attention(
            load_tensor(f"{prefix}self_attn.q_proj.weight"),
            load_tensor(f"{prefix}self_attn.k_proj.weight"),
            load_tensor(f"{prefix}self_attn.v_proj.weight"),
            load_tensor(f"{prefix}self_attn.o_proj.weight"),
            transformer_config,
        )

        # MLP norm
        mlp_norm = Norm(
            load_tensor(f"{prefix}post_attention_layernorm.weight"),
            None,
            "rmsnorm",
            config.rms_norm_eps,
        )

        # MLP
        mlp = MLP(
            transformer_config,
            gate_proj=load_tensor(f"{prefix}mlp.gate_proj.weight"),
            up_proj=load_tensor(f"{prefix}mlp.up_proj.weight"),
            down_proj=load_tensor(f"{prefix}mlp.down_proj.weight"),
        )

        block = TransformerBlock(attn_norm, attn, mlp_norm, mlp)
        blocks.append(block)

    # Final norm
    final_norm = Norm(load_tensor("model.norm.weight"), None, "rmsnorm", config.rms_norm_eps)

    # LM head
    lm_head = None
    if "lm_head.weight" in st.tensor_names:
        lm_head = load_tensor("lm_head.weight")

    return LlamaModel(transformer_config, embed_tokens, blocks, final_norm, lm_head)


# =============================================================================
# Legacy apply_rotary_pos_emb (for backward compatibility)
# =============================================================================

apply_rotary_pos_emb = apply_rotary_pos_emb_numpy
