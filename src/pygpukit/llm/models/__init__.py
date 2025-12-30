"""LLM model implementations.

This module provides unified transformer runtime implementations.
"""

from __future__ import annotations

# Legacy component aliases (for backward compatibility)
from pygpukit.llm.models.causal import (
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

__all__ = [
    # Primary model class
    "CausalTransformerModel",
    # Architecture aliases
    "GPT2Model",
    "LlamaModel",
    # Legacy aliases
    "RMSNorm",
    "LayerNorm",
    "LlamaAttention",
    "LlamaMLP",
    "LlamaBlock",
    "CausalSelfAttention",
]
