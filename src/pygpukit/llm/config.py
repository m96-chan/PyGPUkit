"""Model configuration classes for PyGPUkit LLM.

Provides:
- ModelSpec: Data-only abstraction for model-specific differences
- TransformerConfig: Unified configuration for all model variants
- Legacy config classes: GPT2Config, LlamaConfig, Qwen3Config
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

    # MoE weights (format strings with {layer} and {expert} placeholders)
    moe_gate: str | None = None  # Router: [hidden, num_experts]
    expert_gate_proj: str | None = None  # Expert gate/w1
    expert_up_proj: str | None = None  # Expert up/w3
    expert_down_proj: str | None = None  # Expert down/w2

    # Architecture flags
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    activation: Literal["gelu", "silu"] = "silu"
    use_rope: bool = True
    use_qk_norm: bool = False
    use_position_embed: bool = False  # GPT-2 style absolute position embeddings
    qkv_combined: bool = False  # GPT-2 uses combined QKV projection
    weight_transpose: bool = False  # GPT-2 weights need transpose
    is_moe: bool = False  # MoE model flag

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


# Qwen3 MoE spec - Qwen3 attention + MoE FFN
QWEN3_MOE_SPEC = ModelSpec(
    name="qwen3_moe",
    # Embeddings
    embed_tokens="model.embed_tokens.weight",
    position_embed=None,
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    final_norm_bias=None,
    # Attention (same as Qwen3 with QK norm)
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
    # MLP norm (used before MoE)
    mlp_norm="model.layers.{layer}.post_attention_layernorm.weight",
    mlp_norm_bias=None,
    # Standard MLP weights (not used for MoE)
    fc1=None,
    fc1_bias=None,
    fc2=None,
    fc2_bias=None,
    gate_proj=None,
    up_proj=None,
    down_proj=None,
    # MoE weights (Qwen3 MoE uses mlp.gate and mlp.experts.{expert}.{gate,up,down}_proj)
    moe_gate="model.layers.{layer}.mlp.gate.weight",
    expert_gate_proj="model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
    expert_up_proj="model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
    expert_down_proj="model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
    # Architecture
    norm_type="rmsnorm",
    activation="silu",
    use_rope=True,
    use_qk_norm=True,
    use_position_embed=False,
    qkv_combined=False,
    weight_transpose=False,
    is_moe=True,
    default_norm_eps=1e-6,
    default_rope_theta=10000000.0,  # Qwen3-MoE uses 10M rope_theta
    hf_model_type="qwen3_moe",
)


# Qwen2 spec - like LLaMA but with QKV biases
QWEN2_SPEC = ModelSpec(
    name="qwen2",
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
    q_bias="model.layers.{layer}.self_attn.q_proj.bias",
    k_bias="model.layers.{layer}.self_attn.k_proj.bias",
    v_bias="model.layers.{layer}.self_attn.v_proj.bias",
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
    default_norm_eps=1e-6,
    default_rope_theta=1000000.0,
    hf_model_type="qwen2",
)


# Mixtral MoE spec - like LLaMA attention + MoE FFN
MIXTRAL_SPEC = ModelSpec(
    name="mixtral",
    # Embeddings
    embed_tokens="model.embed_tokens.weight",
    position_embed=None,
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    final_norm_bias=None,
    # Attention (same as LLaMA)
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
    # MLP norm (used before MoE)
    mlp_norm="model.layers.{layer}.post_attention_layernorm.weight",
    mlp_norm_bias=None,
    # Standard MLP weights (not used for MoE)
    fc1=None,
    fc1_bias=None,
    fc2=None,
    fc2_bias=None,
    gate_proj=None,
    up_proj=None,
    down_proj=None,
    # MoE weights
    moe_gate="model.layers.{layer}.block_sparse_moe.gate.weight",
    expert_gate_proj="model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight",
    expert_up_proj="model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight",
    expert_down_proj="model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight",
    # Architecture
    norm_type="rmsnorm",
    activation="silu",
    use_rope=True,
    use_qk_norm=False,
    use_position_embed=False,
    qkv_combined=False,
    weight_transpose=False,
    is_moe=True,
    default_norm_eps=1e-5,
    default_rope_theta=1000000.0,
    hf_model_type="mixtral",
)


# Registry for model detection
MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt2": GPT2_SPEC,
    "llama": LLAMA_SPEC,
    "qwen3": QWEN3_SPEC,
    "qwen3_moe": QWEN3_MOE_SPEC,
    "qwen2": QWEN2_SPEC,
    "mixtral": MIXTRAL_SPEC,
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
    # Check for Mixtral MoE (has block_sparse_moe)
    if any("block_sparse_moe" in name for name in tensor_names):
        return MIXTRAL_SPEC
    # Check for Qwen3 MoE (has mlp.experts and q_norm)
    has_qwen3_moe = any("mlp.experts" in name for name in tensor_names)
    has_qk_norm = any("q_norm" in name for name in tensor_names)
    if has_qwen3_moe and has_qk_norm:
        return QWEN3_MOE_SPEC
    # Check for Qwen3-specific QK norm (dense model)
    if has_qk_norm:
        return QWEN3_SPEC
    # Check for Qwen2-style structure (has QKV biases)
    if (
        "model.embed_tokens.weight" in tensor_names
        and "model.layers.0.self_attn.q_proj.bias" in tensor_names
    ):
        return QWEN2_SPEC
    # Check for LLaMA-style structure (no QKV biases)
    if "model.embed_tokens.weight" in tensor_names:
        return LLAMA_SPEC
    # Check for GPT-2 structure
    if "wte.weight" in tensor_names:
        return GPT2_SPEC

    raise ValueError(
        f"Cannot detect model type from tensor names. First 10 names: {tensor_names[:10]}"
    )


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

    MoE style (Mixtral):
        num_experts=8, num_experts_per_tok=2
    """

    # Core dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int | None = None  # None = MHA, int = GQA/MQA
    intermediate_size: int | None = None  # None = 4 * hidden_size
    _head_dim: int | None = None  # None = hidden_size // num_heads (default)

    # MoE configuration
    num_experts: int | None = None  # None = standard MLP, int = MoE
    num_experts_per_tok: int = 2  # Top-K experts per token
    moe_intermediate_size: int | None = None  # Expert FFN size (default: intermediate_size)

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
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size

    @property
    def is_moe(self) -> bool:
        """Check if this is an MoE model."""
        return self.num_experts is not None and self.num_experts > 1

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
