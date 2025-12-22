"""Test unified LLM interface (v0.2.9 ModelSpec refactor)."""

import pytest


def test_type_aliases():
    """Test that GPT2Model and LlamaModel are aliases for CausalTransformerModel."""
    from pygpukit.llm import (
        CausalTransformerModel,
        GPT2Model,
        LlamaModel,
    )

    # All should be the exact same class
    assert GPT2Model is CausalTransformerModel
    assert LlamaModel is CausalTransformerModel


def test_component_aliases():
    """Test that component aliases point to unified classes."""
    from pygpukit.llm import (
        MLP,
        Attention,
        CausalSelfAttention,
        LlamaAttention,
        LlamaBlock,
        LlamaMLP,
        Norm,
        RMSNorm,
        TransformerBlock,
    )

    # Attention aliases
    assert CausalSelfAttention is Attention
    assert LlamaAttention is Attention

    # MLP aliases
    assert LlamaMLP is MLP

    # Block aliases
    assert LlamaBlock is TransformerBlock

    # Norm aliases (RMSNorm = Norm now)
    assert RMSNorm is Norm


def test_model_specs_exist():
    """Test that ModelSpec instances are defined correctly."""
    from pygpukit.llm import (
        GPT2_SPEC,
        LLAMA_SPEC,
        MODEL_SPECS,
        QWEN2_SPEC,
        QWEN3_SPEC,
        ModelSpec,
    )

    # All specs should be ModelSpec instances
    assert isinstance(GPT2_SPEC, ModelSpec)
    assert isinstance(LLAMA_SPEC, ModelSpec)
    assert isinstance(QWEN2_SPEC, ModelSpec)
    assert isinstance(QWEN3_SPEC, ModelSpec)

    # Check names
    assert GPT2_SPEC.name == "gpt2"
    assert LLAMA_SPEC.name == "llama"
    assert QWEN2_SPEC.name == "qwen2"
    assert QWEN3_SPEC.name == "qwen3"

    # Check architecture flags
    assert GPT2_SPEC.norm_type == "layernorm"
    assert GPT2_SPEC.activation == "gelu"
    assert GPT2_SPEC.use_rope is False
    assert GPT2_SPEC.use_position_embed is True

    assert LLAMA_SPEC.norm_type == "rmsnorm"
    assert LLAMA_SPEC.activation == "silu"
    assert LLAMA_SPEC.use_rope is True
    assert LLAMA_SPEC.use_qk_norm is False

    assert QWEN2_SPEC.norm_type == "rmsnorm"
    assert QWEN2_SPEC.activation == "silu"
    assert QWEN2_SPEC.use_rope is True
    assert QWEN2_SPEC.use_qk_norm is False
    assert QWEN2_SPEC.default_norm_eps == 1e-6
    assert QWEN2_SPEC.default_rope_theta == 1000000.0

    assert QWEN3_SPEC.norm_type == "rmsnorm"
    assert QWEN3_SPEC.activation == "silu"
    assert QWEN3_SPEC.use_rope is True
    assert QWEN3_SPEC.use_qk_norm is True
    assert QWEN3_SPEC.default_norm_eps == 1e-6
    assert QWEN3_SPEC.default_rope_theta == 1000000.0

    # Check MODEL_SPECS registry
    assert MODEL_SPECS["gpt2"] is GPT2_SPEC
    assert MODEL_SPECS["llama"] is LLAMA_SPEC
    assert MODEL_SPECS["qwen2"] is QWEN2_SPEC
    assert MODEL_SPECS["qwen3"] is QWEN3_SPEC


def test_detect_model_spec():
    """Test automatic model detection from tensor names."""
    from pygpukit.llm import (
        GPT2_SPEC,
        LLAMA_SPEC,
        QWEN3_SPEC,
        detect_model_spec,
    )

    # GPT-2 detection
    gpt2_tensors = ["wte.weight", "wpe.weight", "h.0.ln_1.weight"]
    assert detect_model_spec(gpt2_tensors) is GPT2_SPEC

    # LLaMA detection
    llama_tensors = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]
    assert detect_model_spec(llama_tensors) is LLAMA_SPEC

    # Qwen3 detection (has q_norm)
    qwen3_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert detect_model_spec(qwen3_tensors) is QWEN3_SPEC


def test_detect_model_spec_unknown():
    """Test that unknown tensor names raise ValueError."""
    from pygpukit.llm import detect_model_spec

    with pytest.raises(ValueError, match="Cannot detect model type"):
        detect_model_spec(["unknown.tensor.weight"])


def test_transformer_config():
    """Test TransformerConfig creation and properties."""
    from pygpukit.llm import TransformerConfig

    # Test default values
    config = TransformerConfig()
    assert config.num_kv_heads == config.num_heads  # MHA default
    assert config.intermediate_size == 4 * config.hidden_size

    # Test GQA config
    gqa_config = TransformerConfig(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
    )
    assert gqa_config.head_dim == 128  # 4096 / 32
    assert gqa_config.num_kv_groups == 4  # 32 / 8


def test_loader_signatures():
    """Test that loader functions have correct signatures."""
    import inspect

    from pygpukit.llm import (
        load_gpt2_from_safetensors,
        load_llama_from_safetensors,
        load_model_from_safetensors,
        load_qwen3_from_safetensors,
    )

    # Check load_model_from_safetensors signature
    sig = inspect.signature(load_model_from_safetensors)
    params = list(sig.parameters.keys())
    assert "model_path" in params
    assert "dtype" in params
    assert "spec" in params

    # Check simplified loader signatures (no config parameter)
    for loader in [
        load_gpt2_from_safetensors,
        load_llama_from_safetensors,
        load_qwen3_from_safetensors,
    ]:
        sig = inspect.signature(loader)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "dtype" in params
        # Should NOT have config parameter anymore
        assert "config" not in params


def test_causal_transformer_model_has_spec():
    """Test that CausalTransformerModel accepts spec parameter."""
    import inspect

    from pygpukit.llm import CausalTransformerModel

    sig = inspect.signature(CausalTransformerModel.__init__)
    params = list(sig.parameters.keys())
    assert "spec" in params


def test_exports():
    """Test that all expected symbols are exported from pygpukit.llm."""
    from pygpukit import llm

    # Core classes
    assert hasattr(llm, "CausalTransformerModel")
    assert hasattr(llm, "TransformerConfig")
    assert hasattr(llm, "Attention")
    assert hasattr(llm, "MLP")
    assert hasattr(llm, "Norm")
    assert hasattr(llm, "TransformerBlock")
    assert hasattr(llm, "Linear")

    # ModelSpec
    assert hasattr(llm, "ModelSpec")
    assert hasattr(llm, "GPT2_SPEC")
    assert hasattr(llm, "LLAMA_SPEC")
    assert hasattr(llm, "QWEN3_SPEC")
    assert hasattr(llm, "MODEL_SPECS")
    assert hasattr(llm, "detect_model_spec")

    # Loaders
    assert hasattr(llm, "load_model_from_safetensors")
    assert hasattr(llm, "load_gpt2_from_safetensors")
    assert hasattr(llm, "load_llama_from_safetensors")
    assert hasattr(llm, "load_qwen3_from_safetensors")

    # Type aliases
    assert hasattr(llm, "GPT2Model")
    assert hasattr(llm, "LlamaModel")
    assert hasattr(llm, "CausalSelfAttention")
    assert hasattr(llm, "LlamaAttention")
    assert hasattr(llm, "LlamaBlock")
    assert hasattr(llm, "LlamaMLP")
    assert hasattr(llm, "RMSNorm")
    assert hasattr(llm, "LayerNorm")

    # Legacy configs (still exported for reference)
    assert hasattr(llm, "GPT2Config")
    assert hasattr(llm, "LlamaConfig")
    assert hasattr(llm, "Qwen3Config")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
