"""Unified Transformer implementation for PyGPUkit.

Provides a common Transformer abstraction that supports GPT-2, LLaMA, and Qwen3
architectures through ModelSpec configuration.

Key features:
- ModelSpec abstraction for model-specific differences
- Hybrid Attention: CPU for seq_len=1 (decode), GPU for prefill
- GPU-native operations: RMSNorm, LayerNorm, SDPA, SiLU, GELU, RoPE
- Unified TransformerConfig for all model variants
- Generic loader with automatic model detection
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
# ModelSpec - Data-only abstraction for model-specific differences
# =============================================================================


@dataclass(frozen=True)
class ModelSpec:
    """Model specification defining architecture-specific configurations.

    This is a data-only structure with no methods or behavior.
    All model-specific differences are expressed as configuration values.
    """

    # Model identifier
    name: str

    # Weight name patterns (HF name patterns for tensor lookup)
    # These are format strings with {layer} placeholder
    embed_tokens: str
    position_embed: str | None  # None if using RoPE
    lm_head: str | None  # None if tied embeddings
    final_norm: str
    final_norm_bias: str | None

    # Per-layer weight patterns
    attn_norm: str
    attn_norm_bias: str | None
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str
    q_bias: str | None
    k_bias: str | None
    v_bias: str | None
    o_bias: str | None
    q_norm: str | None  # QK Norm (Qwen3)
    k_norm: str | None

    mlp_norm: str
    mlp_norm_bias: str | None

    # MLP weights (GELU style)
    fc1: str | None
    fc1_bias: str | None
    fc2: str | None
    fc2_bias: str | None

    # MLP weights (SwiGLU style)
    gate_proj: str | None
    up_proj: str | None
    down_proj: str | None

    # Architecture flags
    norm_type: Literal["rmsnorm", "layernorm"]
    activation: Literal["gelu", "silu"]
    use_rope: bool
    use_qk_norm: bool
    use_position_embed: bool  # GPT-2 style absolute position embeddings
    qkv_combined: bool  # GPT-2 uses combined QKV projection
    weight_transpose: bool  # GPT-2 weights need transpose

    # Default hyperparameters
    default_norm_eps: float = 1e-5
    default_rope_theta: float = 10000.0

    # Config class name for detection
    hf_model_type: str = ""


# =============================================================================
# Concrete Model Specs
# =============================================================================


GPT2_SPEC = ModelSpec(
    name="gpt2",
    # Embeddings
    embed_tokens="wte.weight",
    position_embed="wpe.weight",
    lm_head=None,  # Tied to embed_tokens
    final_norm="ln_f.weight",
    final_norm_bias="ln_f.bias",
    # Attention (combined QKV)
    attn_norm="h.{layer}.ln_1.weight",
    attn_norm_bias="h.{layer}.ln_1.bias",
    q_proj="h.{layer}.attn.c_attn.weight",  # Combined QKV
    k_proj="h.{layer}.attn.c_attn.weight",  # Same tensor, split at load
    v_proj="h.{layer}.attn.c_attn.weight",
    o_proj="h.{layer}.attn.c_proj.weight",
    q_bias="h.{layer}.attn.c_attn.bias",
    k_bias="h.{layer}.attn.c_attn.bias",
    v_bias="h.{layer}.attn.c_attn.bias",
    o_bias="h.{layer}.attn.c_proj.bias",
    q_norm=None,
    k_norm=None,
    # MLP (GELU)
    mlp_norm="h.{layer}.ln_2.weight",
    mlp_norm_bias="h.{layer}.ln_2.bias",
    fc1="h.{layer}.mlp.c_fc.weight",
    fc1_bias="h.{layer}.mlp.c_fc.bias",
    fc2="h.{layer}.mlp.c_proj.weight",
    fc2_bias="h.{layer}.mlp.c_proj.bias",
    gate_proj=None,
    up_proj=None,
    down_proj=None,
    # Architecture
    norm_type="layernorm",
    activation="gelu",
    use_rope=False,
    use_qk_norm=False,
    use_position_embed=True,
    qkv_combined=True,
    weight_transpose=True,
    default_norm_eps=1e-5,
    default_rope_theta=10000.0,
    hf_model_type="gpt2",
)


LLAMA_SPEC = ModelSpec(
    name="llama",
    # Embeddings
    embed_tokens="model.embed_tokens.weight",
    position_embed=None,
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    final_norm_bias=None,
    # Attention
    attn_norm="model.layers.{layer}.input_layernorm.weight",
    attn_norm_bias=None,
    q_proj="model.layers.{layer}.self_attn.q_proj.weight",
    k_proj="model.layers.{layer}.self_attn.k_proj.weight",
    v_proj="model.layers.{layer}.self_attn.v_proj.weight",
    o_proj="model.layers.{layer}.self_attn.o_proj.weight",
    q_bias=None,
    k_bias=None,
    v_bias=None,
    o_bias=None,
    q_norm=None,
    k_norm=None,
    # MLP (SwiGLU)
    mlp_norm="model.layers.{layer}.post_attention_layernorm.weight",
    mlp_norm_bias=None,
    fc1=None,
    fc1_bias=None,
    fc2=None,
    fc2_bias=None,
    gate_proj="model.layers.{layer}.mlp.gate_proj.weight",
    up_proj="model.layers.{layer}.mlp.up_proj.weight",
    down_proj="model.layers.{layer}.mlp.down_proj.weight",
    # Architecture
    norm_type="rmsnorm",
    activation="silu",
    use_rope=True,
    use_qk_norm=False,
    use_position_embed=False,
    qkv_combined=False,
    weight_transpose=False,
    default_norm_eps=1e-5,
    default_rope_theta=10000.0,
    hf_model_type="llama",
)


QWEN3_SPEC = ModelSpec(
    name="qwen3",
    # Embeddings
    embed_tokens="model.embed_tokens.weight",
    position_embed=None,
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    final_norm_bias=None,
    # Attention
    attn_norm="model.layers.{layer}.input_layernorm.weight",
    attn_norm_bias=None,
    q_proj="model.layers.{layer}.self_attn.q_proj.weight",
    k_proj="model.layers.{layer}.self_attn.k_proj.weight",
    v_proj="model.layers.{layer}.self_attn.v_proj.weight",
    o_proj="model.layers.{layer}.self_attn.o_proj.weight",
    q_bias=None,
    k_bias=None,
    v_bias=None,
    o_bias=None,
    q_norm="model.layers.{layer}.self_attn.q_norm.weight",
    k_norm="model.layers.{layer}.self_attn.k_norm.weight",
    # MLP (SwiGLU)
    mlp_norm="model.layers.{layer}.post_attention_layernorm.weight",
    mlp_norm_bias=None,
    fc1=None,
    fc1_bias=None,
    fc2=None,
    fc2_bias=None,
    gate_proj="model.layers.{layer}.mlp.gate_proj.weight",
    up_proj="model.layers.{layer}.mlp.up_proj.weight",
    down_proj="model.layers.{layer}.mlp.down_proj.weight",
    # Architecture
    norm_type="rmsnorm",
    activation="silu",
    use_rope=True,
    use_qk_norm=True,
    use_position_embed=False,
    qkv_combined=False,
    weight_transpose=False,
    default_norm_eps=1e-6,
    default_rope_theta=1000000.0,
    hf_model_type="qwen3",
)


# Registry for model detection
MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt2": GPT2_SPEC,
    "llama": LLAMA_SPEC,
    "qwen3": QWEN3_SPEC,
    "qwen2": LLAMA_SPEC,  # Qwen2 uses same structure as LLaMA
}


def detect_model_spec(tensor_names: list[str]) -> ModelSpec:
    """Detect model type from tensor names.

    Args:
        tensor_names: List of tensor names from safetensors file

    Returns:
        ModelSpec for the detected model type

    Raises:
        ValueError: If model type cannot be detected
    """
    # Check for Qwen3-specific QK norm
    if any("q_norm" in name for name in tensor_names):
        return QWEN3_SPEC
    # Check for LLaMA-style structure
    if "model.embed_tokens.weight" in tensor_names:
        return LLAMA_SPEC
    # Check for GPT-2 structure
    if "wte.weight" in tensor_names:
        return GPT2_SPEC

    raise ValueError(
        f"Cannot detect model type from tensor names. First 10 names: {tensor_names[:10]}"
    )


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
        assert self.num_kv_heads is not None  # Set in __post_init__
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
    - QK Norm: optional normalization of Q and K (Qwen3 style)
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
        q_norm: Norm | None = None,
        k_norm: Norm | None = None,
    ):
        self.q_proj = Linear(q_proj, q_bias)
        self.k_proj = Linear(k_proj, k_bias)
        self.v_proj = Linear(v_proj, v_bias)
        self.o_proj = Linear(o_proj, o_bias)

        # QK Norm (Qwen3 style)
        self.q_norm = q_norm
        self.k_norm = k_norm

        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        assert config.num_kv_heads is not None  # Set in __post_init__
        self.num_kv_heads: int = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups

        # Precompute RoPE if enabled
        self._cos: np.ndarray | None
        self._sin: np.ndarray | None
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

        # QK Norm (Qwen3 style) - applied per head before RoPE
        # Reshape to 2D for norm, then back to 3D
        if self.q_norm is not None:
            q_shape = (seq_len, self.num_heads, self.head_dim)
            q_2d = reshape_copy(q, (seq_len * self.num_heads, self.head_dim))
            q_2d = self.q_norm(q_2d)
            q = reshape_copy(q_2d, q_shape)
        if self.k_norm is not None:
            k_shape = (seq_len, self.num_kv_heads, self.head_dim)
            k_2d = reshape_copy(k, (seq_len * self.num_kv_heads, self.head_dim))
            k_2d = self.k_norm(k_2d)
            k = reshape_copy(k_2d, k_shape)

        # Apply RoPE on GPU (native FP32/FP16/BF16 support)
        if self.config.use_rope:
            assert self._cos is not None and self._sin is not None
            # Match cos/sin dtype to q/k dtype for native kernel support
            q_dtype = q.dtype
            if q_dtype == "float16":
                cos = from_numpy(self._cos[position_ids].astype(np.float16))
                sin = from_numpy(self._sin[position_ids].astype(np.float16))
            elif q_dtype == "bfloat16":
                # NumPy doesn't support bfloat16, so use float32 -> convert on GPU
                cos = from_numpy(self._cos[position_ids].astype(np.float32))
                sin = from_numpy(self._sin[position_ids].astype(np.float32))
                # TODO: Add bfloat16 conversion when available
                # For now, fall back to float32 computation
                q_f32 = from_numpy(q.to_numpy().astype(np.float32))
                k_f32 = from_numpy(k.to_numpy().astype(np.float32))
                rope_inplace(q_f32, k_f32, cos, sin)
                # Convert back - using float16 as proxy since bfloat16 not in numpy
                q = from_numpy(q_f32.to_numpy().astype(np.float16))
                k = from_numpy(k_f32.to_numpy().astype(np.float16))
            else:
                # FP32 path
                cos = from_numpy(self._cos[position_ids].astype(np.float32))
                sin = from_numpy(self._sin[position_ids].astype(np.float32))
            # Apply RoPE in-place (FP32 and FP16 have native kernel support)
            if q_dtype in ("float32", "float16"):
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

        # GPU SDPA (use same dtype as q)
        q_t = transpose_3d_021(q)
        kv_dtype = k_np.dtype  # Preserve dtype from KV cache
        k_t = from_numpy(k_expanded.transpose(1, 0, 2).astype(kv_dtype))
        v_t = from_numpy(v_expanded.transpose(1, 0, 2).astype(kv_dtype))

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

        # QK Norm (Qwen3 style) - applied per head before RoPE
        # Reshape to 2D for norm, then back to 3D (preserve dtype)
        if self.q_norm is not None:
            q_shape = q.shape
            q_dtype = q.dtype
            q_2d = q.reshape(seq_len * self.num_heads, self.head_dim)
            q_2d = self.q_norm(from_numpy(q_2d)).to_numpy()
            q = q_2d.reshape(q_shape).astype(q_dtype)
        if self.k_norm is not None:
            k_shape = k.shape
            k_dtype = k.dtype
            k_2d = k.reshape(seq_len * self.num_kv_heads, self.head_dim)
            k_2d = self.k_norm(from_numpy(k_2d)).to_numpy()
            k = k_2d.reshape(k_shape).astype(k_dtype)

        # Apply RoPE (CPU)
        if self.config.use_rope:
            assert self._cos is not None and self._sin is not None
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

        # Output projection (GPU) - use same dtype as weights
        weight_dtype = str(self.o_proj.weight.dtype)
        out_dtype = np.float16 if weight_dtype == "float16" else np.float32
        out = from_numpy(attn_output.astype(out_dtype))
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

        # Token embeddings (preserve dtype)
        embed_np = self.embed_tokens.to_numpy()
        hidden_np = embed_np[input_ids]

        # Add position embeddings (GPT-2 style)
        if self.position_embed is not None:
            pos_embed_np = self.position_embed.to_numpy()
            hidden_np = hidden_np + pos_embed_np[position_ids]

        hidden: GPUArray = from_numpy(hidden_np.astype(embed_np.dtype))

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
        """Compute logits from hidden states."""
        hidden_np = hidden.to_numpy()

        if self._lm_head is not None:
            lm_head_np = self._lm_head.to_numpy()
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
# Type Aliases
# =============================================================================

# GPT2Model and LlamaModel are now simple aliases for CausalTransformerModel.
# All models use CausalTransformerModel as the single runtime type.
GPT2Model = CausalTransformerModel
LlamaModel = CausalTransformerModel

# Legacy component aliases
RMSNorm = Norm  # Use Norm with norm_type="rmsnorm"
LayerNorm = Norm  # Use Norm with norm_type="layernorm"
LlamaAttention = Attention
LlamaMLP = MLP
LlamaBlock = TransformerBlock
CausalSelfAttention = Attention


# =============================================================================
# Safetensors Loaders
# =============================================================================


def load_gpt2_from_safetensors(
    model_path: str,
    dtype: str = "float32",
) -> CausalTransformerModel:
    """Load GPT-2 model from safetensors file.

    Args:
        model_path: Path to model.safetensors
        dtype: Weight dtype ("float32" or "float16")

    Returns:
        CausalTransformerModel instance
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=GPT2_SPEC)


def load_llama_from_safetensors(
    model_path: str,
    dtype: str = "float32",
) -> CausalTransformerModel:
    """Load Llama model from safetensors file.

    Args:
        model_path: Path to model.safetensors
        dtype: Weight dtype ("float32" or "float16")

    Returns:
        CausalTransformerModel instance
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=LLAMA_SPEC)


# =============================================================================
# Qwen3 Configuration and Loader
# =============================================================================


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 model."""

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128  # Qwen3 uses 128, not hidden_size // num_heads
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

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


def load_qwen3_from_safetensors(
    model_path: str,
    dtype: str = "float32",
) -> CausalTransformerModel:
    """Load Qwen3 model from safetensors file.

    Args:
        model_path: Path to model.safetensors or model.safetensors.index.json
        dtype: Weight dtype ("float32" or "float16")

    Returns:
        CausalTransformerModel instance
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=QWEN3_SPEC)


# =============================================================================
# Legacy apply_rotary_pos_emb (for backward compatibility)
# =============================================================================

apply_rotary_pos_emb = apply_rotary_pos_emb_numpy


# =============================================================================
# Generic Model Loader using ModelSpec
# =============================================================================


def load_model_from_safetensors(
    model_path: str,
    dtype: str = "float32",
    spec: ModelSpec | None = None,
) -> CausalTransformerModel:
    """Load model from safetensors file using ModelSpec abstraction.

    Automatically detects model type (GPT-2, LLaMA, Qwen3) from tensor names
    and loads using the appropriate ModelSpec configuration.

    Args:
        model_path: Path to model.safetensors or model.safetensors.index.json
        dtype: Weight dtype ("float32" or "float16")
        spec: Optional ModelSpec to use (auto-detected if None)

    Returns:
        CausalTransformerModel instance

    Example:
        # Auto-detect model type
        model = load_model_from_safetensors("/path/to/model.safetensors")

        # Explicit model type
        model = load_model_from_safetensors("/path/to/model.safetensors", spec=LLAMA_SPEC)
    """
    from pygpukit.llm import load_safetensors

    st = load_safetensors(model_path)
    target_dtype = np.float16 if dtype == "float16" else np.float32

    # Detect model type if not specified
    if spec is None:
        spec = detect_model_spec(st.tensor_names)

    # Helper to load tensor with dtype conversion
    def load_tensor(name: str, do_transpose: bool = False) -> GPUArray:
        data = st.tensor_bytes(name)
        info = st.tensor_info(name)
        if info.dtype == 2:  # BFloat16
            arr = np.frombuffer(data, dtype=np.uint16).reshape(info.shape)
            arr_f32 = np.empty(arr.shape, dtype=np.float32)
            arr_f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
            arr = arr_f32
        else:
            dtype_map = {0: np.float32, 1: np.float16, 3: np.float64}
            np_dtype = dtype_map.get(info.dtype, np.float32)
            arr = np.frombuffer(data, dtype=np_dtype).reshape(info.shape).copy()
        if do_transpose and arr.ndim == 2:
            arr = arr.T
        return from_numpy(arr.astype(target_dtype))

    def try_load(name: str | None, do_transpose: bool = False) -> GPUArray | None:
        if name is None or name not in st.tensor_names:
            return None
        return load_tensor(name, do_transpose)

    def layer_name(pattern: str | None, layer: int) -> str | None:
        if pattern is None:
            return None
        return pattern.format(layer=layer)

    def required_name(pattern: str, layer: int) -> str:
        """Get layer name for a required pattern (never None)."""
        return pattern.format(layer=layer)

    # Auto-detect config from tensor shapes
    embed_info = st.tensor_info(spec.embed_tokens)
    vocab_size = embed_info.shape[0]
    hidden_size = embed_info.shape[1]

    # Count layers
    num_layers = 0
    while required_name(spec.q_proj, num_layers) in st.tensor_names:
        num_layers += 1

    # Detect num_heads and num_kv_heads from projection shapes
    q_info = st.tensor_info(required_name(spec.q_proj, 0))
    head_dim = 64  # Default

    # Try to get head_dim from q_norm if present (Qwen3)
    if spec.use_qk_norm and spec.q_norm is not None:
        q_norm_name = required_name(spec.q_norm, 0)
        if q_norm_name in st.tensor_names:
            q_norm_info = st.tensor_info(q_norm_name)
            head_dim = q_norm_info.shape[0]

    num_heads = q_info.shape[0] // head_dim

    # For GQA models, detect num_kv_heads
    num_kv_heads = num_heads
    if not spec.qkv_combined:
        k_info = st.tensor_info(required_name(spec.k_proj, 0))
        num_kv_heads = k_info.shape[0] // head_dim

    # Detect intermediate_size
    intermediate_size = 4 * hidden_size
    if spec.activation == "silu" and spec.gate_proj is not None:
        gate_info = st.tensor_info(required_name(spec.gate_proj, 0))
        intermediate_size = gate_info.shape[0]
    elif spec.activation == "gelu" and spec.fc1 is not None:
        fc1_info = st.tensor_info(required_name(spec.fc1, 0))
        intermediate_size = fc1_info.shape[0]

    # Build TransformerConfig
    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        norm_type=spec.norm_type,
        activation=spec.activation,
        use_rope=spec.use_rope,
        causal=True,
        norm_eps=spec.default_norm_eps,
        rope_theta=spec.default_rope_theta,
    )

    # Load embeddings
    embed_tokens = load_tensor(spec.embed_tokens)
    position_embed = try_load(spec.position_embed) if spec.use_position_embed else None

    # Load blocks
    blocks = []
    for layer_idx in range(num_layers):
        # Attention norm (required)
        attn_norm_weight = load_tensor(required_name(spec.attn_norm, layer_idx))
        attn_norm_bias = try_load(layer_name(spec.attn_norm_bias, layer_idx))
        attn_norm = Norm(attn_norm_weight, attn_norm_bias, spec.norm_type, spec.default_norm_eps)

        # QK Norm (Qwen3, optional)
        q_norm_layer = None
        k_norm_layer = None
        if spec.use_qk_norm:
            q_norm_weight = try_load(layer_name(spec.q_norm, layer_idx))
            k_norm_weight = try_load(layer_name(spec.k_norm, layer_idx))
            if q_norm_weight is not None:
                q_norm_layer = Norm(q_norm_weight, None, spec.norm_type, spec.default_norm_eps)
            if k_norm_weight is not None:
                k_norm_layer = Norm(k_norm_weight, None, spec.norm_type, spec.default_norm_eps)

        # Attention projections
        if spec.qkv_combined:
            # GPT-2 style: combined QKV tensor needs to be split
            c_attn_weight = load_tensor(
                required_name(spec.q_proj, layer_idx), do_transpose=spec.weight_transpose
            )
            c_attn_bias = try_load(layer_name(spec.q_bias, layer_idx))

            # Split combined QKV
            c_attn_np = c_attn_weight.to_numpy()
            q_weight = from_numpy(c_attn_np[:hidden_size].copy().astype(target_dtype))
            k_weight = from_numpy(
                c_attn_np[hidden_size : 2 * hidden_size].copy().astype(target_dtype)
            )
            v_weight = from_numpy(c_attn_np[2 * hidden_size :].copy().astype(target_dtype))

            q_bias, k_bias, v_bias = None, None, None
            if c_attn_bias is not None:
                c_attn_bias_np = c_attn_bias.to_numpy()
                q_bias = from_numpy(c_attn_bias_np[:hidden_size].copy().astype(target_dtype))
                k_bias = from_numpy(
                    c_attn_bias_np[hidden_size : 2 * hidden_size].copy().astype(target_dtype)
                )
                v_bias = from_numpy(c_attn_bias_np[2 * hidden_size :].copy().astype(target_dtype))

            o_weight = load_tensor(
                required_name(spec.o_proj, layer_idx), do_transpose=spec.weight_transpose
            )
            o_bias = try_load(layer_name(spec.o_bias, layer_idx))

            attn = Attention(
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                transformer_config,
                q_bias,
                k_bias,
                v_bias,
                o_bias,
                q_norm_layer,
                k_norm_layer,
            )
        else:
            # Separate Q, K, V projections (LLaMA/Qwen3 style)
            q_weight = load_tensor(required_name(spec.q_proj, layer_idx))
            k_weight = load_tensor(required_name(spec.k_proj, layer_idx))
            v_weight = load_tensor(required_name(spec.v_proj, layer_idx))
            o_weight = load_tensor(required_name(spec.o_proj, layer_idx))

            q_bias = try_load(layer_name(spec.q_bias, layer_idx))
            k_bias = try_load(layer_name(spec.k_bias, layer_idx))
            v_bias = try_load(layer_name(spec.v_bias, layer_idx))
            o_bias = try_load(layer_name(spec.o_bias, layer_idx))

            attn = Attention(
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                transformer_config,
                q_bias,
                k_bias,
                v_bias,
                o_bias,
                q_norm_layer,
                k_norm_layer,
            )

        # MLP norm (required)
        mlp_norm_weight = load_tensor(required_name(spec.mlp_norm, layer_idx))
        mlp_norm_bias = try_load(layer_name(spec.mlp_norm_bias, layer_idx))
        mlp_norm = Norm(mlp_norm_weight, mlp_norm_bias, spec.norm_type, spec.default_norm_eps)

        # MLP
        if spec.activation == "gelu" and spec.fc1 is not None and spec.fc2 is not None:
            fc1_weight = load_tensor(
                required_name(spec.fc1, layer_idx), do_transpose=spec.weight_transpose
            )
            fc1_bias = try_load(layer_name(spec.fc1_bias, layer_idx))
            fc2_weight = load_tensor(
                required_name(spec.fc2, layer_idx), do_transpose=spec.weight_transpose
            )
            fc2_bias = try_load(layer_name(spec.fc2_bias, layer_idx))
            mlp = MLP(
                transformer_config,
                fc1_weight=fc1_weight,
                fc1_bias=fc1_bias,
                fc2_weight=fc2_weight,
                fc2_bias=fc2_bias,
            )
        elif spec.gate_proj is not None and spec.up_proj is not None and spec.down_proj is not None:
            # SwiGLU
            gate_proj = load_tensor(required_name(spec.gate_proj, layer_idx))
            up_proj = load_tensor(required_name(spec.up_proj, layer_idx))
            down_proj = load_tensor(required_name(spec.down_proj, layer_idx))
            mlp = MLP(
                transformer_config,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
            )
        else:
            raise ValueError(f"ModelSpec {spec.name} has invalid MLP configuration")

        block = TransformerBlock(attn_norm, attn, mlp_norm, mlp)
        blocks.append(block)

    # Final norm
    final_norm_weight = load_tensor(spec.final_norm)
    final_norm_bias = try_load(spec.final_norm_bias)
    final_norm = Norm(final_norm_weight, final_norm_bias, spec.norm_type, spec.default_norm_eps)

    # LM head
    lm_head = None
    if spec.lm_head is not None and spec.lm_head in st.tensor_names:
        lm_head = load_tensor(spec.lm_head)

    return CausalTransformerModel(
        transformer_config, embed_tokens, blocks, final_norm, lm_head, position_embed, spec
    )
