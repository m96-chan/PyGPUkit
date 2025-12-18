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

from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy, zeros
from pygpukit.ops.basic import (
    add,
    add_inplace,
    bias_add_inplace,
    concat_axis0,
    copy_to,
    embedding_lookup,
    embedding_lookup_ptr,
    gelu,
    kv_cache_prefill_gqa,
    kv_cache_update_gqa,
    kv_cache_update_gqa_ptr,
    layernorm,
    matmul,
    mul,
    mul_inplace,
    repeat_interleave_axis1,
    reshape_copy,
    rmsnorm,
    rope_inplace,
    sample_token_gpu,
    sample_topk_to_buf_ptr,
    sdpa_causal,
    sdpa_causal_fixed_cache,
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
        probs_sum = probs.sum()
        probs = probs / probs_sum

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
        probs_sum = probs.sum()
        probs = probs / probs_sum

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
    _head_dim: int | None = None  # None = hidden_size // num_heads (default)

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
        if self._head_dim is not None:
            return self._head_dim
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
# Weight Repacking - Fix GPU memory placement for optimal performance
# =============================================================================


def repack_weight(weight: GPUArray) -> GPUArray:
    """Repack a weight tensor into a new contiguous GPU buffer.

    This fixes performance issues caused by fragmented GPU memory allocation.
    Weights allocated later during model loading may end up in suboptimal
    memory regions, causing 7x slower matmul performance.

    Args:
        weight: Original weight tensor on GPU

    Returns:
        New GPUArray with same data in freshly allocated contiguous memory
    """
    # Copy to CPU, then back to GPU to get fresh allocation
    # This ensures the new buffer is allocated contiguously
    weight_np = weight.to_numpy()
    return from_numpy(weight_np)


def repack_linear(linear: Linear) -> None:
    """Repack a Linear layer's weight in-place.

    Args:
        linear: Linear layer to repack
    """
    linear.weight = repack_weight(linear.weight)
    # Clear transpose cache - will be regenerated on first use
    linear._weight_t = None
    if linear.bias is not None:
        linear.bias = repack_weight(linear.bias)


def repack_norm(norm: Norm) -> None:
    """Repack a Norm layer's weight in-place.

    Args:
        norm: Norm layer to repack
    """
    norm.weight = repack_weight(norm.weight)
    if norm.bias is not None:
        norm.bias = repack_weight(norm.bias)


# =============================================================================
# Decode Buffers for CUDA Graph Support
# =============================================================================


@dataclass
class DecodeBuffers:
    """Pre-allocated buffers for allocation-free decode steps.

    These buffers are layer-shared (reused across all layers in a single decode step)
    since layers are processed sequentially. This eliminates all memory allocations
    during decode, enabling CUDA Graph capture.

    Buffer shapes (for Qwen3-8B example):
    - hidden: [1, 4096] - layer input/output
    - qkv_proj_out: [1, 6144] - Fused QKV projection output (q_dim + k_dim + v_dim)
    - q_proj_out: [1, 4096] - Q projection output (2D) - DEPRECATED, kept for compat
    - k_proj_out, v_proj_out: [1, 1024] - K/V projection outputs (2D) - DEPRECATED
    - o_proj_out: [1, 4096] - O projection output (2D)
    - q: [1, 32, 128] - query after reshape (3D)
    - k, v: [1, 8, 128] - key/value after reshape (3D)
    - attn_out: [32, 1, 128] - SDPA output (transposed format)
    - gate_up_out: [1, 24576] - Fused gate_up projection output (2 * intermediate_size)
    - mlp_gate, mlp_up: [1, 12288] - MLP intermediates (views into gate_up_out)
    - cos, sin: [1, 128] - RoPE tables
    - embed_out: [1, 4096] - embedding lookup output
    """

    # Main computation buffers
    hidden: GPUArray  # [1, hidden_size]
    q: GPUArray  # [1, num_heads, head_dim]
    k: GPUArray  # [1, num_kv_heads, head_dim]
    v: GPUArray  # [1, num_kv_heads, head_dim]
    attn_out: GPUArray  # [num_heads, 1, head_dim]
    mlp_gate: GPUArray  # [1, intermediate_size]
    mlp_up: GPUArray  # [1, intermediate_size]
    mlp_down: GPUArray  # [1, hidden_size] - down projection output

    # Projection output buffers (2D, for matmul out=)
    q_proj_out: GPUArray  # [1, num_heads * head_dim]
    k_proj_out: GPUArray  # [1, num_kv_heads * head_dim]
    v_proj_out: GPUArray  # [1, num_kv_heads * head_dim]
    o_proj_out: GPUArray  # [1, hidden_size]

    # Transposed Q buffer for SDPA
    q_t: GPUArray  # [num_heads, 1, head_dim]

    # RoPE buffers
    cos: GPUArray  # [1, head_dim]
    sin: GPUArray  # [1, head_dim]

    # Embedding output
    embed_out: GPUArray  # [1, hidden_size]

    # Temporary buffers for intermediate computations
    residual: GPUArray  # [1, hidden_size]
    norm_out: GPUArray  # [1, hidden_size]

    # For QK norm (Qwen3)
    q_2d: GPUArray | None = None  # [num_heads, head_dim] - rmsnorm output
    k_2d: GPUArray | None = None  # [num_kv_heads, head_dim] - rmsnorm output
    q_flat: GPUArray | None = None  # [num_heads, head_dim] - rmsnorm input
    k_flat: GPUArray | None = None  # [num_kv_heads, head_dim] - rmsnorm input

    # GPU position buffer for CUDA Graph replay (int32)
    position_buf: GPUArray | None = None  # [1] int32

    # Fused projection buffers (for reduced matmul count)
    # Used with GPUArray.narrow() for zero-copy splitting:
    # - qkv_proj_out: Single matmul replaces 3 (Q, K, V projections)
    # - gate_up_out: Single matmul replaces 2 (gate, up projections)
    qkv_proj_out: GPUArray | None = None  # [1, q_dim + k_dim + v_dim]
    gate_up_out: GPUArray | None = None  # [1, 2 * intermediate_size]

    # Pre-cached narrow views (created once, reused every forward to avoid object creation overhead)
    q_view: GPUArray | None = None  # view of qkv_proj_out[0:q_dim]
    k_view: GPUArray | None = None  # view of qkv_proj_out[q_dim:q_dim+k_dim]
    v_view: GPUArray | None = None  # view of qkv_proj_out[q_dim+k_dim:]
    gate_view: GPUArray | None = None  # view of gate_up_out[0:intermediate_size]
    up_view: GPUArray | None = None  # view of gate_up_out[intermediate_size:]

    # Logits buffer for CUDA Graph (lm_head projection output)
    logits: GPUArray | None = None  # [1, vocab_size]

    # Sampling buffers for CUDA Graph
    sampled_token: GPUArray | None = None  # [1] int32 - sampled token ID
    random_val: GPUArray | None = None  # [1] float32 - random value for sampling

    @classmethod
    def allocate(
        cls,
        config: TransformerConfig,
        dtype: str = "float16",
        use_qk_norm: bool = False,
        vocab_size: int | None = None,
    ) -> DecodeBuffers:
        """Allocate all decode buffers.

        Args:
            config: Model configuration
            dtype: Data type for buffers
            use_qk_norm: Whether to allocate QK norm buffers (Qwen3)
            vocab_size: Vocabulary size for logits buffer (optional, for CUDA Graph)
        """
        assert config.num_kv_heads is not None
        assert config.intermediate_size is not None

        hidden = zeros((1, config.hidden_size), dtype=dtype)
        q = zeros((1, config.num_heads, config.head_dim), dtype=dtype)
        k = zeros((1, config.num_kv_heads, config.head_dim), dtype=dtype)
        v = zeros((1, config.num_kv_heads, config.head_dim), dtype=dtype)
        attn_out = zeros((config.num_heads, 1, config.head_dim), dtype=dtype)
        mlp_gate = zeros((1, config.intermediate_size), dtype=dtype)
        mlp_up = zeros((1, config.intermediate_size), dtype=dtype)
        mlp_down = zeros((1, config.hidden_size), dtype=dtype)

        # Projection output buffers (2D for matmul out=)
        q_proj_out = zeros((1, config.num_heads * config.head_dim), dtype=dtype)
        k_proj_out = zeros((1, config.num_kv_heads * config.head_dim), dtype=dtype)
        v_proj_out = zeros((1, config.num_kv_heads * config.head_dim), dtype=dtype)
        o_proj_out = zeros((1, config.hidden_size), dtype=dtype)

        # Transposed Q buffer for SDPA
        q_t = zeros((config.num_heads, 1, config.head_dim), dtype=dtype)

        cos = zeros((1, config.head_dim), dtype=dtype)
        sin = zeros((1, config.head_dim), dtype=dtype)

        embed_out = zeros((1, config.hidden_size), dtype=dtype)
        residual = zeros((1, config.hidden_size), dtype=dtype)
        norm_out = zeros((1, config.hidden_size), dtype=dtype)

        # QK norm buffers
        q_2d = None
        k_2d = None
        q_flat = None
        k_flat = None
        if use_qk_norm:
            q_2d = zeros((config.num_heads, config.head_dim), dtype=dtype)
            k_2d = zeros((config.num_kv_heads, config.head_dim), dtype=dtype)
            q_flat = zeros((config.num_heads, config.head_dim), dtype=dtype)
            k_flat = zeros((config.num_kv_heads, config.head_dim), dtype=dtype)

        # GPU position buffer for CUDA Graph replay
        position_buf = zeros((1,), dtype="int32")

        # Fused projection buffers
        q_dim = config.num_heads * config.head_dim
        k_dim = config.num_kv_heads * config.head_dim
        v_dim = config.num_kv_heads * config.head_dim
        qkv_proj_out = zeros((1, q_dim + k_dim + v_dim), dtype=dtype)
        gate_up_out = zeros((1, 2 * config.intermediate_size), dtype=dtype)

        # Pre-create narrow views (avoids object creation overhead in forward loop)
        q_view = qkv_proj_out.narrow(0, q_dim)
        k_view = qkv_proj_out.narrow(q_dim, k_dim)
        v_view = qkv_proj_out.narrow(q_dim + k_dim, v_dim)
        gate_view = gate_up_out.narrow(0, config.intermediate_size)
        up_view = gate_up_out.narrow(config.intermediate_size, config.intermediate_size)

        # Logits buffer for CUDA Graph (optional)
        logits_buf = None
        sampled_token_buf = None
        random_val_buf = None
        if vocab_size is not None:
            logits_buf = zeros((1, vocab_size), dtype=dtype)
            sampled_token_buf = zeros((1,), dtype="int32")
            random_val_buf = zeros((1,), dtype="float32")

        return cls(
            hidden=hidden,
            q=q,
            k=k,
            v=v,
            attn_out=attn_out,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
            q_proj_out=q_proj_out,
            k_proj_out=k_proj_out,
            v_proj_out=v_proj_out,
            o_proj_out=o_proj_out,
            q_t=q_t,
            cos=cos,
            sin=sin,
            embed_out=embed_out,
            residual=residual,
            norm_out=norm_out,
            q_2d=q_2d,
            k_2d=k_2d,
            q_flat=q_flat,
            k_flat=k_flat,
            position_buf=position_buf,
            qkv_proj_out=qkv_proj_out,
            gate_up_out=gate_up_out,
            q_view=q_view,
            k_view=k_view,
            v_view=v_view,
            gate_view=gate_view,
            up_view=up_view,
            logits=logits_buf,
            sampled_token=sampled_token_buf,
            random_val=random_val_buf,
        )


@dataclass
class PrefillBuffers:
    """Pre-allocated buffers for allocation-free prefill phase.

    Unlike DecodeBuffers (seq_len=1), PrefillBuffers handles variable-length
    sequences up to max_seq_len. Buffers are allocated once and reused.

    Buffer shapes (for Qwen3-8B with max_seq_len=512):
    - hidden: [max_seq_len, hidden_size] - layer input/output
    - q_proj_out: [max_seq_len, num_heads * head_dim] - Q projection (2D)
    - k_proj_out: [max_seq_len, num_kv_heads * head_dim] - K projection (2D)
    - v_proj_out: [max_seq_len, num_kv_heads * head_dim] - V projection (2D)
    - o_proj_out: [max_seq_len, hidden_size] - O projection (2D)
    - q: [max_seq_len, num_heads, head_dim] - Q after reshape (3D)
    - k: [max_seq_len, num_kv_heads, head_dim] - K after reshape (3D)
    - v: [max_seq_len, num_kv_heads, head_dim] - V after reshape (3D)
    - q_t: [num_heads, max_seq_len, head_dim] - Q transposed for SDPA
    - k_t: [num_heads, max_seq_len, head_dim] - K transposed (GQA-expanded)
    - v_t: [num_heads, max_seq_len, head_dim] - V transposed (GQA-expanded)
    - attn_out: [num_heads, max_seq_len, head_dim] - SDPA output
    - attn_out_t: [max_seq_len, num_heads, head_dim] - attention transposed back
    - mlp_gate: [max_seq_len, intermediate_size] - MLP gate output
    - mlp_up: [max_seq_len, intermediate_size] - MLP up output
    - mlp_down: [max_seq_len, hidden_size] - MLP down output
    - residual: [max_seq_len, hidden_size] - residual connection
    - norm_out: [max_seq_len, hidden_size] - normalization output
    """

    max_seq_len: int

    # Main computation buffers
    hidden: GPUArray  # [max_seq_len, hidden_size]
    q: GPUArray  # [max_seq_len, num_heads, head_dim]
    k: GPUArray  # [max_seq_len, num_kv_heads, head_dim]
    v: GPUArray  # [max_seq_len, num_kv_heads, head_dim]

    # Projection outputs (2D for matmul)
    q_proj_out: GPUArray  # [max_seq_len, num_heads * head_dim]
    k_proj_out: GPUArray  # [max_seq_len, num_kv_heads * head_dim]
    v_proj_out: GPUArray  # [max_seq_len, num_kv_heads * head_dim]
    o_proj_out: GPUArray  # [max_seq_len, hidden_size]

    # Transposed buffers for SDPA (GQA-expanded for K, V)
    q_t: GPUArray  # [num_heads, max_seq_len, head_dim]
    k_t: GPUArray  # [num_heads, max_seq_len, head_dim]
    v_t: GPUArray  # [num_heads, max_seq_len, head_dim]

    # Attention output
    attn_out: GPUArray  # [num_heads, max_seq_len, head_dim]
    attn_out_t: GPUArray  # [max_seq_len, num_heads, head_dim]
    attn_out_2d: GPUArray  # [max_seq_len, num_heads * head_dim]

    # MLP buffers
    mlp_gate: GPUArray  # [max_seq_len, intermediate_size]
    mlp_up: GPUArray  # [max_seq_len, intermediate_size]
    mlp_down: GPUArray  # [max_seq_len, hidden_size]

    # RoPE buffers
    cos: GPUArray  # [max_seq_len, head_dim]
    sin: GPUArray  # [max_seq_len, head_dim]

    # Temporary buffers
    residual: GPUArray  # [max_seq_len, hidden_size]
    norm_out: GPUArray  # [max_seq_len, hidden_size]

    # QK Norm buffers (optional, for Qwen3)
    q_2d: GPUArray | None = None  # [max_seq_len * num_heads, head_dim]
    k_2d: GPUArray | None = None  # [max_seq_len * num_kv_heads, head_dim]

    @classmethod
    def allocate(
        cls,
        config: TransformerConfig,
        max_seq_len: int,
        dtype: str = "float16",
        use_qk_norm: bool = False,
    ) -> PrefillBuffers:
        """Allocate all prefill buffers.

        Args:
            config: Model configuration
            max_seq_len: Maximum sequence length for prefill
            dtype: Data type for buffers
            use_qk_norm: Whether to allocate QK norm buffers (Qwen3)
        """
        assert config.num_kv_heads is not None
        assert config.intermediate_size is not None

        # Main buffers
        hidden = zeros((max_seq_len, config.hidden_size), dtype=dtype)
        q = zeros((max_seq_len, config.num_heads, config.head_dim), dtype=dtype)
        k = zeros((max_seq_len, config.num_kv_heads, config.head_dim), dtype=dtype)
        v = zeros((max_seq_len, config.num_kv_heads, config.head_dim), dtype=dtype)

        # Projection outputs (2D)
        q_proj_out = zeros((max_seq_len, config.num_heads * config.head_dim), dtype=dtype)
        k_proj_out = zeros((max_seq_len, config.num_kv_heads * config.head_dim), dtype=dtype)
        v_proj_out = zeros((max_seq_len, config.num_kv_heads * config.head_dim), dtype=dtype)
        o_proj_out = zeros((max_seq_len, config.hidden_size), dtype=dtype)

        # Transposed buffers (GQA-expanded for K, V)
        q_t = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)
        k_t = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)
        v_t = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)

        # Attention output buffers
        attn_out = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)
        attn_out_t = zeros((max_seq_len, config.num_heads, config.head_dim), dtype=dtype)
        attn_out_2d = zeros((max_seq_len, config.num_heads * config.head_dim), dtype=dtype)

        # MLP buffers
        mlp_gate = zeros((max_seq_len, config.intermediate_size), dtype=dtype)
        mlp_up = zeros((max_seq_len, config.intermediate_size), dtype=dtype)
        mlp_down = zeros((max_seq_len, config.hidden_size), dtype=dtype)

        # RoPE buffers
        cos = zeros((max_seq_len, config.head_dim), dtype=dtype)
        sin = zeros((max_seq_len, config.head_dim), dtype=dtype)

        # Temporary buffers
        residual = zeros((max_seq_len, config.hidden_size), dtype=dtype)
        norm_out = zeros((max_seq_len, config.hidden_size), dtype=dtype)

        # QK Norm buffers (Qwen3)
        q_2d = None
        k_2d = None
        if use_qk_norm:
            q_2d = zeros((max_seq_len * config.num_heads, config.head_dim), dtype=dtype)
            k_2d = zeros((max_seq_len * config.num_kv_heads, config.head_dim), dtype=dtype)

        return cls(
            max_seq_len=max_seq_len,
            hidden=hidden,
            q=q,
            k=k,
            v=v,
            q_proj_out=q_proj_out,
            k_proj_out=k_proj_out,
            v_proj_out=v_proj_out,
            o_proj_out=o_proj_out,
            q_t=q_t,
            k_t=k_t,
            v_t=v_t,
            attn_out=attn_out,
            attn_out_t=attn_out_t,
            attn_out_2d=attn_out_2d,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
            cos=cos,
            sin=sin,
            residual=residual,
            norm_out=norm_out,
            q_2d=q_2d,
            k_2d=k_2d,
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

    def __call__(self, x: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
        """Forward pass: y = xW^T + b

        Args:
            x: Input tensor [batch, in_features]
            out: Optional output buffer [batch, out_features]. If provided,
                 result is written in-place (for CUDA Graph capture).
        """
        if x.ndim != 2:
            raise ValueError(f"input must be 2D [batch, in_features], got {x.ndim}D")
        if x.shape[1] != self.in_features:
            raise ValueError(f"input features {x.shape[1]} != weight {self.in_features}")

        if self._weight_t is None:
            self._weight_t = transpose(self.weight)

        y = matmul(x, self._weight_t, out=out)

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

        # Store dimensions for QKV split
        self.q_dim = self.num_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.v_dim = self.num_kv_heads * self.head_dim

        # Create fused QKV projection (reduces 3 matmuls to 1)
        # qkv_weight: [q_dim + k_dim + v_dim, hidden_size]
        # Used in decode path with GPUArray.narrow() for zero-copy splitting.
        qkv_weight = concat_axis0(concat_axis0(q_proj, k_proj), v_proj)
        self.qkv_proj = Linear(qkv_weight, None)  # No bias for fused (bias handled separately)

        # Precompute RoPE if enabled
        self._cos: np.ndarray | None
        self._sin: np.ndarray | None
        if config.use_rope:
            self._cos, self._sin = precompute_freqs_cis(
                self.head_dim, config.max_position_embeddings, config.rope_theta
            )
        else:
            self._cos, self._sin = None, None

        # Fixed-length KV cache for CUDA Graph (initialized on first use)
        self._k_cache: GPUArray | None = None
        self._v_cache: GPUArray | None = None
        self._max_cache_len: int = 0

        # Lookahead KV tracking for Jacobi decoding (GPU-side, no CPU copies)
        # confirmed_pos: KV at positions [0, confirmed_pos) is finalized
        # logical_pos: current write position during lookahead iterations
        self._confirmed_pos: int = 0
        self._logical_pos: int = 0

    def init_fixed_cache(self, max_seq_len: int, dtype: str = "float16") -> None:
        """Initialize fixed-length KV cache for CUDA Graph capture.

        Args:
            max_seq_len: Maximum sequence length to support.
            dtype: Data type for cache (float16/bfloat16).
        """
        # Cache shape: [num_heads, max_seq_len, head_dim] (transposed, GQA-expanded)
        # This eliminates per-step transpose and GQA expansion
        cache_shape = (self.num_heads, max_seq_len, self.head_dim)
        np_dtype = np.float16 if dtype == "float16" else np.float32
        self._k_cache = from_numpy(np.zeros(cache_shape, dtype=np_dtype))
        self._v_cache = from_numpy(np.zeros(cache_shape, dtype=np_dtype))
        self._max_cache_len = max_seq_len
        # Reset lookahead tracking
        self._confirmed_pos = 0
        self._logical_pos = 0

    # =========================================================================
    # Lookahead KV Cache Management (for Jacobi Decoding)
    # =========================================================================

    def set_confirmed_pos(self, pos: int) -> None:
        """Set the confirmed position (e.g., after prefill).

        Args:
            pos: Position where KV is finalized (0 to pos-1 are committed).
        """
        assert pos >= 0 and pos <= self._max_cache_len, f"Invalid pos {pos}"
        self._confirmed_pos = pos
        self._logical_pos = pos

    def reset_lookahead(self) -> None:
        """Reset lookahead pointer to confirmed position.

        Called at the start of each Jacobi iteration to reset speculative KV.
        This does NOT modify the KV cache - it just resets the write pointer.
        """
        self._logical_pos = self._confirmed_pos

    def commit_lookahead(self, n_accepted: int) -> None:
        """Commit accepted tokens by advancing confirmed_pos.

        Args:
            n_accepted: Number of accepted tokens to commit.
        """
        new_pos = self._confirmed_pos + n_accepted
        assert new_pos <= self._max_cache_len, f"Commit exceeds cache: {new_pos}"
        self._confirmed_pos = new_pos
        self._logical_pos = new_pos

    def get_confirmed_pos(self) -> int:
        """Get current confirmed position."""
        return self._confirmed_pos

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

        # Full GPU path for all sequence lengths (decode + prefill)
        # GPU KV Cache (#83) eliminates CPU-GPU transfer overhead
        return self._forward_gpu(x, position_ids, past_kv, use_cache)

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

        # GPU KV Cache - keep KV tensors on GPU to avoid CPU-GPU transfers
        # Concatenate with past KV on GPU
        if past_kv is not None:
            past_k, past_v = past_kv
            # past_kv can be GPUArray (from _forward_gpu) or numpy (from _forward_cpu)
            if isinstance(past_k, GPUArray):
                k = concat_axis0(past_k, k)
                v = concat_axis0(past_v, v)
            else:
                # Legacy numpy format - convert to GPU
                k_np = k.to_numpy()
                v_np = v.to_numpy()
                k_np = np.concatenate([past_k, k_np], axis=0)
                v_np = np.concatenate([past_v, v_np], axis=0)
                k = from_numpy(k_np)
                v = from_numpy(v_np)

        # Store KV cache as GPUArray for next iteration
        present_kv = (k, v) if use_cache else None

        # Expand for GQA on GPU
        if self.num_kv_groups > 1:
            k_expanded = repeat_interleave_axis1(k, self.num_kv_groups)
            v_expanded = repeat_interleave_axis1(v, self.num_kv_groups)
        else:
            k_expanded = k
            v_expanded = v

        # GPU SDPA - transpose [seq, heads, dim] -> [heads, seq, dim]
        q_t = transpose_3d_021(q)
        k_t = transpose_3d_021(k_expanded)
        v_t = transpose_3d_021(v_expanded)

        attn_output = sdpa_causal(q_t, k_t, v_t)

        # Reshape output
        attn_output = transpose_3d_021(attn_output)
        attn_output = reshape_copy(attn_output, (seq_len, self.num_heads * self.head_dim))

        return self.o_proj(attn_output), present_kv

    def forward_fixed_cache(
        self,
        x: GPUArray,
        position: int,
        context_len: int,
        *,
        out: GPUArray | None = None,
    ) -> GPUArray:
        """Forward pass using fixed-length KV cache (for CUDA Graph decode).

        Args:
            x: Input tensor [1, hidden_size] - single token
            position: Current position in sequence (for RoPE and cache update)
            context_len: Total context length (prefill + decoded so far)
            out: Optional pre-allocated output buffer

        Returns:
            Output tensor [1, hidden_size]
        """
        assert self._k_cache is not None, "Call init_fixed_cache first"
        assert x.shape[0] == 1, "forward_fixed_cache expects single token"

        # Fused QKV projection (1 matmul replaces 3, then zero-copy narrow views)
        qkv = self.qkv_proj(x)  # [1, q_dim + k_dim + v_dim]
        q_2d = qkv.narrow(0, self.q_dim)  # [1, q_dim]
        k_2d = qkv.narrow(self.q_dim, self.k_dim)  # [1, k_dim]
        v_2d = qkv.narrow(self.q_dim + self.k_dim, self.v_dim)  # [1, v_dim]

        # Zero-copy reshape for multi-head: [1, num_heads, head_dim]
        q = q_2d.view((1, self.num_heads, self.head_dim))
        k = k_2d.view((1, self.num_kv_heads, self.head_dim))
        v = v_2d.view((1, self.num_kv_heads, self.head_dim))

        # QK Norm (Qwen3 style) with zero-copy views
        if self.q_norm is not None:
            q_flat = q.view((self.num_heads, self.head_dim))
            q_normed = self.q_norm(q_flat)
            q = q_normed.view((1, self.num_heads, self.head_dim))
        if self.k_norm is not None:
            k_flat = k.view((self.num_kv_heads, self.head_dim))
            k_normed = self.k_norm(k_flat)
            k = k_normed.view((1, self.num_kv_heads, self.head_dim))

        # Apply RoPE
        if self.config.use_rope and self._cos is not None and self._sin is not None:
            q_dtype_name = q.dtype.name
            if q_dtype_name == "float16":
                cos = from_numpy(self._cos[position : position + 1].astype(np.float16))
                sin = from_numpy(self._sin[position : position + 1].astype(np.float16))
            else:
                cos = from_numpy(self._cos[position : position + 1].astype(np.float32))
                sin = from_numpy(self._sin[position : position + 1].astype(np.float32))
            rope_inplace(q, k, cos, sin)

        # Update fixed KV cache at current position (GQA-expanded, transposed)
        # k, v: [1, num_kv_heads, head_dim] -> cache: [num_heads, max_seq_len, head_dim]
        kv_cache_update_gqa(k, self._k_cache, self.num_heads, position)
        kv_cache_update_gqa(v, self._v_cache, self.num_heads, position)

        # Prepare for SDPA
        # Zero-copy view Q: [1, num_heads, head_dim] -> [num_heads, 1, head_dim]
        # (swapping dim 0 size=1 with dim 1 is a no-op in memory)
        q_t = q.view((self.num_heads, 1, self.head_dim))

        # Cache is already in SDPA-ready format: [num_heads, max_seq_len, head_dim]
        # No transpose or GQA expansion needed!

        # Allocate output buffer if needed
        if out is None:
            attn_out = from_numpy(np.zeros((self.num_heads, 1, self.head_dim), dtype=np.float16))
        else:
            attn_out = out

        # SDPA with fixed cache - only attend to context_len tokens
        sdpa_causal_fixed_cache(q_t, self._k_cache, self._v_cache, attn_out, context_len)

        # Zero-copy reshape output: [num_heads, 1, head_dim] -> [1, hidden_size]
        attn_output = attn_out.view((1, self.num_heads * self.head_dim))

        return self.o_proj(attn_output)

    def forward_fixed_cache_batch(
        self,
        x: GPUArray,
        start_position: int,
        context_len: int,
    ) -> GPUArray:
        """Forward pass for batch decode using fixed-length KV cache.

        Processes multiple tokens at once for speculative decoding verification.
        Each query token attends to all KV up to its position (causal masking).

        Args:
            x: Input tensor [seq_len, hidden_size] - multiple tokens
            start_position: Starting position for the batch (first token's position)
            context_len: Total context length after adding this batch
                        (should equal start_position + seq_len)

        Returns:
            Output tensor [seq_len, hidden_size]
        """
        assert self._k_cache is not None, "Call init_fixed_cache first"
        seq_len = x.shape[0]

        # Dispatch to optimized single-token path for M=1
        # (uses zero-copy view/narrow instead of numpy slicing)
        if seq_len == 1:
            return self.forward_fixed_cache(x, start_position, context_len)

        # M > 1: Batch decode path
        # Fused QKV projection
        qkv = self.qkv_proj(x)  # [seq_len, q_dim + k_dim + v_dim]

        # For seq_len > 1, we can't use narrow() because it doesn't handle
        # strided access for 2D arrays. Split QKV via numpy slicing.
        # TODO: Add a native batch_narrow kernel for better performance.
        qkv_np = qkv.to_numpy()  # [seq_len, total_qkv]
        q_np = qkv_np[:, :self.q_dim]  # [seq_len, q_dim]
        k_np = qkv_np[:, self.q_dim:self.q_dim + self.k_dim]  # [seq_len, k_dim]
        v_np = qkv_np[:, self.q_dim + self.k_dim:]  # [seq_len, v_dim]

        q_2d = from_numpy(q_np.astype(qkv_np.dtype))
        k_2d = from_numpy(k_np.astype(qkv_np.dtype))
        v_2d = from_numpy(v_np.astype(qkv_np.dtype))

        # Reshape for multi-head: [seq_len, num_heads, head_dim]
        q = reshape_copy(q_2d, (seq_len, self.num_heads, self.head_dim))
        k = reshape_copy(k_2d, (seq_len, self.num_kv_heads, self.head_dim))
        v = reshape_copy(v_2d, (seq_len, self.num_kv_heads, self.head_dim))

        # QK Norm (Qwen3 style)
        if self.q_norm is not None:
            q_flat = reshape_copy(q, (seq_len * self.num_heads, self.head_dim))
            q_normed = self.q_norm(q_flat)
            q = reshape_copy(q_normed, (seq_len, self.num_heads, self.head_dim))
        if self.k_norm is not None:
            k_flat = reshape_copy(k, (seq_len * self.num_kv_heads, self.head_dim))
            k_normed = self.k_norm(k_flat)
            k = reshape_copy(k_normed, (seq_len, self.num_kv_heads, self.head_dim))

        # Apply RoPE for multiple positions
        if self.config.use_rope and self._cos is not None and self._sin is not None:
            q_dtype_name = q.dtype.name
            end_pos = start_position + seq_len
            if q_dtype_name == "float16":
                cos = from_numpy(self._cos[start_position:end_pos].astype(np.float16))
                sin = from_numpy(self._sin[start_position:end_pos].astype(np.float16))
            else:
                cos = from_numpy(self._cos[start_position:end_pos].astype(np.float32))
                sin = from_numpy(self._sin[start_position:end_pos].astype(np.float32))
            rope_inplace(q, k, cos, sin)

        # Update KV cache with batch (use prefill kernel)
        # k, v: [seq_len, num_kv_heads, head_dim] -> cache: [num_heads, max_seq_len, head_dim]
        kv_cache_prefill_gqa(k, self._k_cache, self.num_heads, start_position)
        kv_cache_prefill_gqa(v, self._v_cache, self.num_heads, start_position)

        # Transpose Q for SDPA: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        q_t = transpose_3d_021(q)

        # Allocate output buffer
        attn_out = from_numpy(
            np.zeros((self.num_heads, seq_len, self.head_dim), dtype=np.float16)
        )

        # SDPA with causal masking - context_len should equal start_position + seq_len
        sdpa_causal_fixed_cache(q_t, self._k_cache, self._v_cache, attn_out, context_len)

        # Transpose and reshape output: [num_heads, seq_len, head_dim] -> [seq_len, hidden_size]
        attn_output = transpose_3d_021(attn_out)  # [seq_len, num_heads, head_dim]
        attn_output = reshape_copy(attn_output, (seq_len, self.num_heads * self.head_dim))

        return self.o_proj(attn_output)


# =============================================================================
# Unified MLP
# =============================================================================


class MLP:
    """Unified MLP supporting GELU and SwiGLU activations.

    GELU (GPT-2 style):
        fc1 -> GELU -> fc2

    SwiGLU (LLaMA style):
        gate_proj -> SiLU -> * up_proj -> down_proj

    With fusion optimization (SwiGLU):
        gate_up_proj (fused) -> split -> SiLU(gate) * up -> down_proj
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

            # Store intermediate size for split
            self.intermediate_size = gate_proj.shape[0]

            # Create fused gate_up projection (reduces 2 matmuls to 1)
            # gate_up_weight: [2 * intermediate_size, hidden_size]
            # Used in decode path with GPUArray.narrow() for zero-copy splitting.
            gate_up_weight = concat_axis0(gate_proj, up_proj)
            self.gate_up_proj = Linear(gate_up_weight, None)

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
            cos_np, sin_np = precompute_freqs_cis(
                self.config.head_dim, max_seq_len, self.config.rope_theta
            )
            np_dtype = np.float16 if dtype == "float16" else np.float32
            self._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
            self._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))

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

            from pygpukit._pygpukit_native import CudaGraph

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
                        block.attn, buffers.norm_out, pos, ctx_len, buffers,
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
                        _decode_buffers.hidden, self._lm_head_t_cache,
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
                print(f"  [CUDA Graph] Captured {graph.num_nodes} nodes "
                      f"(sampling={sampling_str})")

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
    ) -> None:
        """Attention forward pass with zero allocations.

        Result is written to buffers.hidden.

        Args:
            use_position_ptr: If True, read position from buffers.position_buf
                              (for CUDA Graph replay without recapture).
        """
        # Fused QKV projection (1 matmul replaces 3, then zero-copy narrow views)
        # This is 4x faster for M=1 with cuBLASLt due to reduced kernel launch overhead
        attn.qkv_proj(x, out=buffers.qkv_proj_out)

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
            return self._decode_step_fixed_cache(
                token_ids[0], start_position, context_len
            )

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
            hidden = block.attn.forward_fixed_cache_batch(
                hidden, start_position, context_len
            )
            hidden = add(residual, hidden)

            # MLP
            residual = hidden
            hidden = block.mlp_norm(hidden)
            hidden = block.mlp(hidden)
            hidden = add(residual, hidden)

        # Final norm
        hidden = self.final_norm(hidden)

        return hidden

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
        """
        for i, block in enumerate(self.blocks):
            k_np, v_np = snapshot[i]
            block.attn._k_cache = from_numpy(k_np.astype(np.float16))
            block.attn._v_cache = from_numpy(v_np.astype(np.float16))

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

        hidden_batch = self._decode_step_fixed_cache_batch(
            verify_input, position, verify_ctx
        )
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
                [t for i, t in enumerate(accepted_tokens)
                 if i < len(draft_tokens) and t == draft_tokens[i]]
            ),
        }

        return accepted_tokens, new_pos, stats

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
        guess = self._init_jacobi_guess(
            token_id, position, context_len, n_tokens, init_strategy
        )

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

            hidden = self._decode_step_fixed_cache_batch(
                input_tokens, position, verify_ctx
            )
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
        guess = self._init_jacobi_guess_lookahead(
            token_id, n_tokens, init_strategy
        )

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

            hidden = self._decode_step_fixed_cache_batch(
                input_tokens, start_pos, ctx_len
            )
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
            current = token_id
            for i, acc_token in enumerate(accepted_tokens):
                pos = confirmed_pos + i
                ctx = confirmed_pos + i + 1
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
# Model Weight Repacking
# =============================================================================


def repack_model_weights(model: CausalTransformerModel) -> None:
    """Repack all model weights into contiguous GPU memory.

    This fixes severe performance regression (7x slowdown) caused by
    fragmented GPU memory allocation during model loading. Weights
    allocated later end up in suboptimal memory regions.

    The repacking is done in two phases:
    1. Convert ALL weights to numpy (freeing GPU memory)
    2. Reallocate ALL weights fresh in contiguous memory

    After repacking:
    - All blocks should have similar matmul latency
    - No per-layer performance degradation

    Args:
        model: CausalTransformerModel to repack in-place
    """
    import gc

    # Phase 1: Collect all weights as numpy arrays
    # This frees GPU memory as we go
    numpy_cache: dict[int, dict] = {}

    # Keep track of dummy allocations to shift allocation base
    dummy_arrays: list[GPUArray] = []

    # Embedding
    embed_np = model.embed_tokens.to_numpy()
    model.embed_tokens = None  # type: ignore

    # Position embedding
    pos_embed_np = None
    if model.position_embed is not None:
        pos_embed_np = model.position_embed.to_numpy()
        model.position_embed = None

    # lm_head
    lm_head_np = None
    if model._lm_head is not None:
        lm_head_np = model._lm_head.to_numpy()
        model._lm_head = None

    # Final norm
    final_norm_weight_np = model.final_norm.weight.to_numpy()
    final_norm_bias_np = None
    if model.final_norm.bias is not None:
        final_norm_bias_np = model.final_norm.bias.to_numpy()
    model.final_norm.weight = None  # type: ignore
    model.final_norm.bias = None

    # All blocks
    for i, block in enumerate(model.blocks):
        numpy_cache[i] = {}

        # Attention norms
        numpy_cache[i]["attn_norm_w"] = block.attn_norm.weight.to_numpy()
        numpy_cache[i]["attn_norm_b"] = (
            block.attn_norm.bias.to_numpy() if block.attn_norm.bias is not None else None
        )
        block.attn_norm.weight = None  # type: ignore
        block.attn_norm.bias = None

        numpy_cache[i]["mlp_norm_w"] = block.mlp_norm.weight.to_numpy()
        numpy_cache[i]["mlp_norm_b"] = (
            block.mlp_norm.bias.to_numpy() if block.mlp_norm.bias is not None else None
        )
        block.mlp_norm.weight = None  # type: ignore
        block.mlp_norm.bias = None

        # Attention projections
        attn = block.attn
        numpy_cache[i]["q_w"] = attn.q_proj.weight.to_numpy()
        numpy_cache[i]["q_b"] = (
            attn.q_proj.bias.to_numpy() if attn.q_proj.bias is not None else None
        )
        attn.q_proj.weight = None  # type: ignore
        attn.q_proj.bias = None
        attn.q_proj._weight_t = None

        numpy_cache[i]["k_w"] = attn.k_proj.weight.to_numpy()
        numpy_cache[i]["k_b"] = (
            attn.k_proj.bias.to_numpy() if attn.k_proj.bias is not None else None
        )
        attn.k_proj.weight = None  # type: ignore
        attn.k_proj.bias = None
        attn.k_proj._weight_t = None

        numpy_cache[i]["v_w"] = attn.v_proj.weight.to_numpy()
        numpy_cache[i]["v_b"] = (
            attn.v_proj.bias.to_numpy() if attn.v_proj.bias is not None else None
        )
        attn.v_proj.weight = None  # type: ignore
        attn.v_proj.bias = None
        attn.v_proj._weight_t = None

        numpy_cache[i]["o_w"] = attn.o_proj.weight.to_numpy()
        numpy_cache[i]["o_b"] = (
            attn.o_proj.bias.to_numpy() if attn.o_proj.bias is not None else None
        )
        attn.o_proj.weight = None  # type: ignore
        attn.o_proj.bias = None
        attn.o_proj._weight_t = None

        # QK norms
        if attn.q_norm is not None:
            numpy_cache[i]["q_norm_w"] = attn.q_norm.weight.to_numpy()
            numpy_cache[i]["q_norm_b"] = (
                attn.q_norm.bias.to_numpy() if attn.q_norm.bias is not None else None
            )
            attn.q_norm.weight = None  # type: ignore
            attn.q_norm.bias = None
        if attn.k_norm is not None:
            numpy_cache[i]["k_norm_w"] = attn.k_norm.weight.to_numpy()
            numpy_cache[i]["k_norm_b"] = (
                attn.k_norm.bias.to_numpy() if attn.k_norm.bias is not None else None
            )
            attn.k_norm.weight = None  # type: ignore
            attn.k_norm.bias = None

        # MLP projections
        mlp = block.mlp
        if mlp.activation == "gelu":
            numpy_cache[i]["fc1_w"] = mlp.fc1.weight.to_numpy()
            numpy_cache[i]["fc1_b"] = mlp.fc1.bias.to_numpy() if mlp.fc1.bias is not None else None
            mlp.fc1.weight = None  # type: ignore
            mlp.fc1.bias = None
            mlp.fc1._weight_t = None

            numpy_cache[i]["fc2_w"] = mlp.fc2.weight.to_numpy()
            numpy_cache[i]["fc2_b"] = mlp.fc2.bias.to_numpy() if mlp.fc2.bias is not None else None
            mlp.fc2.weight = None  # type: ignore
            mlp.fc2.bias = None
            mlp.fc2._weight_t = None
        else:  # SwiGLU
            numpy_cache[i]["gate_w"] = mlp.gate_proj.weight.to_numpy()
            numpy_cache[i]["gate_b"] = (
                mlp.gate_proj.bias.to_numpy() if mlp.gate_proj.bias is not None else None
            )
            mlp.gate_proj.weight = None  # type: ignore
            mlp.gate_proj.bias = None
            mlp.gate_proj._weight_t = None

            numpy_cache[i]["up_w"] = mlp.up_proj.weight.to_numpy()
            numpy_cache[i]["up_b"] = (
                mlp.up_proj.bias.to_numpy() if mlp.up_proj.bias is not None else None
            )
            mlp.up_proj.weight = None  # type: ignore
            mlp.up_proj.bias = None
            mlp.up_proj._weight_t = None

            numpy_cache[i]["down_w"] = mlp.down_proj.weight.to_numpy()
            numpy_cache[i]["down_b"] = (
                mlp.down_proj.bias.to_numpy() if mlp.down_proj.bias is not None else None
            )
            mlp.down_proj.weight = None  # type: ignore
            mlp.down_proj.bias = None
            mlp.down_proj._weight_t = None

    # Force garbage collection to free GPU memory
    gc.collect()

    # Allocate dummy arrays to fill the freed memory space
    # This forces new allocations to go into fresh memory regions
    import numpy as np

    dummy_size = 1024 * 1024 * 512  # 512M elements = 1GB for FP16
    try:
        for _ in range(16):  # Allocate ~16GB of dummy memory
            dummy = from_numpy(np.zeros(dummy_size, dtype=np.float16))
            dummy_arrays.append(dummy)
    except Exception:
        pass  # Continue with whatever dummy memory we could allocate

    # Phase 2: Reallocate all weights fresh
    # Allocate blocks in REVERSE order so later blocks get the "fast" memory first
    # This is critical - CUDA memory allocation order affects matmul performance
    for i in reversed(range(len(model.blocks))):
        block = model.blocks[i]
        cache = numpy_cache[i]

        # Attention norms
        block.attn_norm.weight = from_numpy(cache["attn_norm_w"])
        if cache["attn_norm_b"] is not None:
            block.attn_norm.bias = from_numpy(cache["attn_norm_b"])

        block.mlp_norm.weight = from_numpy(cache["mlp_norm_w"])
        if cache["mlp_norm_b"] is not None:
            block.mlp_norm.bias = from_numpy(cache["mlp_norm_b"])

        # Attention projections
        attn = block.attn
        attn.q_proj.weight = from_numpy(cache["q_w"])
        if cache["q_b"] is not None:
            attn.q_proj.bias = from_numpy(cache["q_b"])

        attn.k_proj.weight = from_numpy(cache["k_w"])
        if cache["k_b"] is not None:
            attn.k_proj.bias = from_numpy(cache["k_b"])

        attn.v_proj.weight = from_numpy(cache["v_w"])
        if cache["v_b"] is not None:
            attn.v_proj.bias = from_numpy(cache["v_b"])

        attn.o_proj.weight = from_numpy(cache["o_w"])
        if cache["o_b"] is not None:
            attn.o_proj.bias = from_numpy(cache["o_b"])

        # QK norms
        if "q_norm_w" in cache:
            attn.q_norm.weight = from_numpy(cache["q_norm_w"])
            if cache["q_norm_b"] is not None:
                attn.q_norm.bias = from_numpy(cache["q_norm_b"])
        if "k_norm_w" in cache:
            attn.k_norm.weight = from_numpy(cache["k_norm_w"])
            if cache["k_norm_b"] is not None:
                attn.k_norm.bias = from_numpy(cache["k_norm_b"])

        # MLP projections
        mlp = block.mlp
        if mlp.activation == "gelu":
            mlp.fc1.weight = from_numpy(cache["fc1_w"])
            if cache["fc1_b"] is not None:
                mlp.fc1.bias = from_numpy(cache["fc1_b"])

            mlp.fc2.weight = from_numpy(cache["fc2_w"])
            if cache["fc2_b"] is not None:
                mlp.fc2.bias = from_numpy(cache["fc2_b"])
        else:  # SwiGLU
            mlp.gate_proj.weight = from_numpy(cache["gate_w"])
            if cache["gate_b"] is not None:
                mlp.gate_proj.bias = from_numpy(cache["gate_b"])

            mlp.up_proj.weight = from_numpy(cache["up_w"])
            if cache["up_b"] is not None:
                mlp.up_proj.bias = from_numpy(cache["up_b"])

            mlp.down_proj.weight = from_numpy(cache["down_w"])
            if cache["down_b"] is not None:
                mlp.down_proj.bias = from_numpy(cache["down_b"])

        # Clear this block's cache immediately to reduce memory
        del numpy_cache[i]

    # Final norm
    model.final_norm.weight = from_numpy(final_norm_weight_np)
    if final_norm_bias_np is not None:
        model.final_norm.bias = from_numpy(final_norm_bias_np)

    # lm_head
    if lm_head_np is not None:
        model._lm_head = from_numpy(lm_head_np)

    # Embedding and position embedding last (after all blocks)
    model.embed_tokens = from_numpy(embed_np)
    del embed_np

    if pos_embed_np is not None:
        model.position_embed = from_numpy(pos_embed_np)
        del pos_embed_np

    # Clear any cached transposes
    if hasattr(model, "_lm_head_t_cache"):
        delattr(model, "_lm_head_t_cache")

    # Free dummy arrays now that weights are in fresh memory
    del dummy_arrays
    gc.collect()


# =============================================================================
# Generic Model Loader using ModelSpec
# =============================================================================


def load_model_from_safetensors(
    model_path: str,
    dtype: str = "float32",
    spec: ModelSpec | None = None,
    repack_weights: bool = True,
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
    # Pass head_dim explicitly if it differs from hidden_size // num_heads
    explicit_head_dim = None
    if head_dim != hidden_size // num_heads:
        explicit_head_dim = head_dim

    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        _head_dim=explicit_head_dim,
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

    model = CausalTransformerModel(
        transformer_config, embed_tokens, blocks, final_norm, lm_head, position_embed, spec
    )
    if repack_weights:
        repack_model_weights(model)
    return model
