"""LLM support module for PyGPUkit.

Provides:
- SafeTensors file loading with memory mapping
- Tensor metadata and data access
- GPU tensor allocation helpers
- LLM model implementations (CausalTransformerModel)
- Layer implementations (Attention, MLP, etc.)
- Decode strategies (M1, Batch, Jacobi, Speculative)
"""

from __future__ import annotations

# Buffers (refactored v0.2.11)
from pygpukit.llm.buffers import (
    DecodeBuffers,
    PrefillBuffers,
)

# Chat template support (v0.2.10)
from pygpukit.llm.chat import (
    ChatMessage,
    apply_chat_template,
    create_chat_prompt,
    format_chat_messages,
)

# Config classes and ModelSpec (refactored v0.2.11)
from pygpukit.llm.config import (
    GPT2_SPEC,
    LLAMA_SPEC,
    MIXTRAL_SPEC,
    MODEL_SPECS,
    QWEN2_SPEC,
    QWEN3_MOE_SPEC,
    QWEN3_SPEC,
    GPT2Config,
    LlamaConfig,
    ModelSpec,
    Qwen3Config,
    TransformerConfig,
    detect_model_spec,
)

# Decode strategies (refactored v0.2.11)
from pygpukit.llm.decode import (
    DecodeBatch,
    DecodeJacobi,
    DecodeM1,
    DecodeM1Graph,
    DecodeSpeculative,
    DecodeStrategy,
)

# Layers (refactored v0.2.18)
from pygpukit.llm.layers import (
    MLP,
    Attention,
    Linear,
    LinearBF16,
    LinearFP8,
    MoELayer,
    Norm,
    TransformerBlock,
    apply_rotary_pos_emb_numpy,
    precompute_freqs_cis,
    repack_linear,
    repack_norm,
    repack_weight,
)

# Loaders (refactored v0.2.11)
# Quantization/Optimization configs (v0.2.18 - Issue #115)
from pygpukit.llm.loader import (
    FP8QuantConfig,
    ModelOptimizationInfo,
    PruningConfig,
    QATQuantConfig,
    SparsityConfig,
    load_gpt2_from_safetensors,
    load_llama_from_safetensors,
    load_mixtral_from_safetensors,
    load_model_from_safetensors,
    load_qwen3_from_safetensors,
    repack_model_weights,
)

# Model (refactored v0.2.18)
from pygpukit.llm.model import (
    CausalSelfAttention,
    CausalTransformerModel,
    GPT2Model,
    LayerNorm,
    LlamaAttention,
    LlamaBlock,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
)

# SafeTensors (extracted v0.2.18)
from pygpukit.llm.safetensors import (
    Dtype,
    SafeTensorsFile,
    ShardedSafeTensorsFile,
    TensorInfo,
    load_safetensors,
)

# Sampling (refactored v0.2.11)
from pygpukit.llm.sampling import sample_token

# Tokenizer (extracted v0.2.18)
from pygpukit.llm.tokenizer import Tokenizer

__all__ = [
    # SafeTensors
    "Dtype",
    "TensorInfo",
    "SafeTensorsFile",
    "ShardedSafeTensorsFile",
    "load_safetensors",
    # Tokenizer
    "Tokenizer",
    # Core Transformer (v0.2.9)
    "CausalTransformerModel",
    "TransformerConfig",
    "Attention",
    "MLP",
    "MoELayer",
    "Norm",
    "TransformerBlock",
    "Linear",  # Backward compatibility alias
    "LinearBF16",
    "LinearFP8",
    # ModelSpec (v0.2.9)
    "ModelSpec",
    "GPT2_SPEC",
    "LLAMA_SPEC",
    "MIXTRAL_SPEC",
    "QWEN2_SPEC",
    "QWEN3_MOE_SPEC",
    "QWEN3_SPEC",
    "MODEL_SPECS",
    "detect_model_spec",
    # Loaders
    "load_model_from_safetensors",
    "load_gpt2_from_safetensors",
    "load_llama_from_safetensors",
    "load_mixtral_from_safetensors",
    "load_qwen3_from_safetensors",
    # Legacy config classes
    "GPT2Config",
    "LlamaConfig",
    "Qwen3Config",
    # Type aliases (all point to unified types)
    "GPT2Model",
    "LlamaModel",
    "CausalSelfAttention",
    "LayerNorm",
    "LlamaAttention",
    "LlamaBlock",
    "LlamaMLP",
    "RMSNorm",
    # Chat template support (v0.2.10)
    "ChatMessage",
    "apply_chat_template",
    "format_chat_messages",
    "create_chat_prompt",
    # Buffers (v0.2.11)
    "DecodeBuffers",
    "PrefillBuffers",
    # RoPE utilities (v0.2.11)
    "apply_rotary_pos_emb_numpy",
    "precompute_freqs_cis",
    # Weight repacking (v0.2.11)
    "repack_linear",
    "repack_norm",
    "repack_weight",
    "repack_model_weights",
    # Sampling (v0.2.11)
    "sample_token",
    # Decode strategies (v0.2.11)
    "DecodeStrategy",
    "DecodeM1",
    "DecodeM1Graph",
    "DecodeBatch",
    "DecodeSpeculative",
    "DecodeJacobi",
    # Quantization/Optimization configs (v0.2.18 - Issue #115)
    "FP8QuantConfig",
    "QATQuantConfig",
    "PruningConfig",
    "SparsityConfig",
    "ModelOptimizationInfo",
]
