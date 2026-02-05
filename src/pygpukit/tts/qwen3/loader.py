"""Model loader for Qwen3-TTS.

Loads model weights from SafeTensors format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.llm.config import TransformerConfig
from pygpukit.llm.layers import MLP, Attention, Norm, TransformerBlock
from pygpukit.llm.layers.linear import LinearBF16

from .model import (
    CodePredictor,
    Qwen3TTSConfig,
    Qwen3TTSModel,
    TalkerModel,
    TextProjection,
)
from .speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig
from .speech_tokenizer import SpeechTokenizer, SpeechTokenizerConfig


def _bf16_to_f32(tensor: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    if tensor.dtype == np.uint16:
        # bfloat16 stored as uint16
        return (tensor.astype(np.uint32) << 16).view(np.float32)
    return tensor.astype(np.float32)


def load_safetensors_weights(
    path: Path | str,
    device: str = "cuda",
) -> dict[str, GPUArray]:
    """Load weights from safetensors file.

    Args:
        path: Path to .safetensors file
        device: Target device (cuda/cpu)

    Returns:
        Dictionary of weight name -> GPUArray
    """
    try:
        import torch
        from safetensors import safe_open

        weights = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Convert to numpy, handling bfloat16
                if tensor.dtype == torch.bfloat16:
                    np_array = tensor.float().cpu().numpy()
                else:
                    np_array = tensor.cpu().numpy()
                weights[key] = from_numpy(np_array.astype(np.float32))

        return weights
    except ImportError:
        raise ImportError("PyTorch and safetensors are required for loading weights")


def load_config(model_path: Path | str) -> dict[str, Any]:
    """Load model configuration from config.json."""
    path = Path(model_path)
    config_path = path / "config.json"

    with open(config_path) as f:
        return json.load(f)


def build_qwen3_tts_config(raw_config: dict[str, Any]) -> Qwen3TTSConfig:
    """Build Qwen3TTSConfig from raw config dict."""
    talker = raw_config.get("talker_config", {})
    rope = talker.get("rope_scaling", {})

    return Qwen3TTSConfig(
        model_type=raw_config.get("tts_model_type", "custom_voice"),
        model_size=raw_config.get("tts_model_size", "0.6B"),
        # Talker
        vocab_size=talker.get("vocab_size", 3072),
        hidden_size=talker.get("hidden_size", 1024),
        num_hidden_layers=talker.get("num_hidden_layers", 28),
        num_attention_heads=talker.get("num_attention_heads", 16),
        num_key_value_heads=talker.get("num_key_value_heads", 8),
        head_dim=talker.get("head_dim", 128),
        intermediate_size=talker.get("intermediate_size", 3072),
        max_position_embeddings=talker.get("max_position_embeddings", 32768),
        # RoPE
        mrope_section=tuple(rope.get("mrope_section", [24, 20, 20])),
        rope_theta=talker.get("rope_theta", 1000000.0),
        # Codec
        codec_vocab_size=talker.get("vocab_size", 3072),
        num_codebooks=talker.get("num_code_groups", 16),
        codec_bos_token_id=talker.get("codec_bos_id", 2149),
        codec_eos_token_id=talker.get("codec_eos_token_id", 2150),
        codec_pad_token_id=talker.get("codec_pad_id", 2148),
        # Speakers
        supported_speakers=tuple(talker.get("spk_id", {}).keys()),
    )


def build_talker_model(
    config: Qwen3TTSConfig,
    weights: dict[str, GPUArray],
    prefix: str = "talker",
) -> TalkerModel:
    """Build TalkerModel from weights."""

    def get_weight(name: str) -> GPUArray:
        full_name = f"{prefix}.{name}"
        if full_name not in weights:
            raise KeyError(f"Weight '{full_name}' not found")
        return weights[full_name]

    def get_weight_optional(name: str) -> GPUArray | None:
        full_name = f"{prefix}.{name}"
        return weights.get(full_name)

    # Text embeddings
    embed_tokens = get_weight("model.text_embedding.weight")

    # Codec embedding
    codec_embed = get_weight("model.codec_embedding.weight")

    # Create TransformerConfig for Attention layers
    transformer_config = TransformerConfig()
    transformer_config.hidden_size = config.hidden_size
    transformer_config.num_layers = config.num_hidden_layers
    transformer_config.num_heads = config.num_attention_heads
    transformer_config.num_kv_heads = config.num_key_value_heads
    transformer_config.intermediate_size = config.intermediate_size
    transformer_config._head_dim = config.head_dim
    transformer_config.max_position_embeddings = config.max_position_embeddings
    transformer_config.norm_eps = 1e-6
    transformer_config.rope_theta = config.rope_theta

    # Build transformer blocks
    blocks = []
    for i in range(config.num_hidden_layers):
        layer_prefix = f"model.layers.{i}"

        # Separate Q, K, V projections
        q_weight = get_weight(f"{layer_prefix}.self_attn.q_proj.weight")
        k_weight = get_weight(f"{layer_prefix}.self_attn.k_proj.weight")
        v_weight = get_weight(f"{layer_prefix}.self_attn.v_proj.weight")
        o_weight = get_weight(f"{layer_prefix}.self_attn.o_proj.weight")

        # QK normalization weights (optional)
        q_norm_weight = get_weight_optional(f"{layer_prefix}.self_attn.q_norm.weight")
        k_norm_weight = get_weight_optional(f"{layer_prefix}.self_attn.k_norm.weight")
        use_qk_norm = q_norm_weight is not None and k_norm_weight is not None

        # Build QK norm if available
        q_norm = None
        k_norm = None
        if use_qk_norm:
            q_norm = Norm(
                weight=q_norm_weight,
                bias=None,
                norm_type="rmsnorm",
                eps=1e-6,
            )
            k_norm = Norm(
                weight=k_norm_weight,
                bias=None,
                norm_type="rmsnorm",
                eps=1e-6,
            )

        # Attention
        attn = Attention(
            LinearBF16(q_weight),
            LinearBF16(k_weight),
            LinearBF16(v_weight),
            LinearBF16(o_weight),
            transformer_config,
            q_norm=q_norm,
            k_norm=k_norm,
        )

        # MLP
        gate_weight = get_weight(f"{layer_prefix}.mlp.gate_proj.weight")
        up_weight = get_weight(f"{layer_prefix}.mlp.up_proj.weight")
        down_weight = get_weight(f"{layer_prefix}.mlp.down_proj.weight")

        mlp = MLP(
            transformer_config,
            gate_proj=LinearBF16(gate_weight),
            up_proj=LinearBF16(up_weight),
            down_proj=LinearBF16(down_weight),
        )

        # Norms
        attn_norm = Norm(
            weight=get_weight(f"{layer_prefix}.input_layernorm.weight"),
            bias=None,
            norm_type="rmsnorm",
            eps=1e-6,
        )
        mlp_norm = Norm(
            weight=get_weight(f"{layer_prefix}.post_attention_layernorm.weight"),
            bias=None,
            norm_type="rmsnorm",
            eps=1e-6,
        )

        block = TransformerBlock(
            attn_norm=attn_norm,
            attn=attn,
            mlp_norm=mlp_norm,
            mlp=mlp,
        )
        blocks.append(block)

    # Final norm
    final_norm = Norm(
        weight=get_weight("model.norm.weight"),
        bias=None,
        norm_type="rmsnorm",
        eps=1e-6,
    )

    # Codec head
    codec_head = get_weight("codec_head.weight")

    # Text projection (2-layer MLP, if exists)
    text_projection = None
    text_proj_fc1_key = f"{prefix}.text_projection.linear_fc1.weight"
    if text_proj_fc1_key in weights:
        text_projection = TextProjection(
            fc1_weight=weights[text_proj_fc1_key],
            fc1_bias=weights.get(f"{prefix}.text_projection.linear_fc1.bias"),
            fc2_weight=weights[f"{prefix}.text_projection.linear_fc2.weight"],
            fc2_bias=weights.get(f"{prefix}.text_projection.linear_fc2.bias"),
        )

    return TalkerModel(
        config=config,
        embed_tokens=embed_tokens,
        codec_embed=codec_embed,
        blocks=blocks,
        final_norm=final_norm,
        codec_head=codec_head,
        code_predictor=None,  # TODO: build code predictor
        text_projection=text_projection,
    )


def load_qwen3_tts(
    model_path: str | Path,
    device: str = "cuda",
    load_speech_tokenizer: bool = True,
) -> Qwen3TTSModel:
    """Load Qwen3-TTS model from pretrained weights.

    Args:
        model_path: Path to model directory
        device: Target device
        load_speech_tokenizer: Whether to load speech tokenizer

    Returns:
        Qwen3TTSModel instance
    """
    path = Path(model_path)

    # Load config
    raw_config = load_config(path)
    config = build_qwen3_tts_config(raw_config)

    # Load main model weights
    print(f"Loading main model from {path / 'model.safetensors'}...")
    weights = load_safetensors_weights(path / "model.safetensors", device)

    # Build talker
    print("Building talker model...")
    talker = build_talker_model(config, weights)

    # Load speech tokenizer
    speech_tokenizer = None
    if load_speech_tokenizer:
        tokenizer_path = path / "speech_tokenizer"
        if tokenizer_path.exists():
            print(f"Loading speech tokenizer from {tokenizer_path}...")
            # TODO: Implement speech tokenizer loading
            # speech_tokenizer = load_speech_tokenizer(tokenizer_path)

    return Qwen3TTSModel(
        config=config,
        talker=talker,
        speaker_encoder=None,  # TODO: load if available
        speech_tokenizer=speech_tokenizer,
    )


__all__ = [
    "load_qwen3_tts",
    "load_safetensors_weights",
    "load_config",
    "build_qwen3_tts_config",
    "build_talker_model",
]
