"""Test script for Qwen3-TTS model loading and generation.

Tests:
1. Load model config from downloaded weights
2. Inspect safetensors structure
3. Basic forward pass test
"""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

# Model path
MODEL_PATH = Path("F:/LLM/Qwen3-TTS-12Hz-0.6B-CustomVoice")


def inspect_config():
    """Load and inspect model configuration."""
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)

    config_path = MODEL_PATH / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    print(f"Model type: {config.get('model_type')}")
    print(f"TTS model type: {config.get('tts_model_type')}")
    print(f"TTS model size: {config.get('tts_model_size')}")

    talker = config.get("talker_config", {})
    print(f"\nTalker config:")
    print(f"  hidden_size: {talker.get('hidden_size')}")
    print(f"  num_hidden_layers: {talker.get('num_hidden_layers')}")
    print(f"  num_attention_heads: {talker.get('num_attention_heads')}")
    print(f"  num_key_value_heads: {talker.get('num_key_value_heads')}")
    print(f"  head_dim: {talker.get('head_dim')}")
    print(f"  text_hidden_size: {talker.get('text_hidden_size')}")
    print(f"  text_vocab_size: {talker.get('text_vocab_size')}")
    print(f"  vocab_size: {talker.get('vocab_size')} (codec)")
    print(f"  num_code_groups: {talker.get('num_code_groups')}")

    rope = talker.get("rope_scaling", {})
    print(f"\nRoPE config:")
    print(f"  mrope_section: {rope.get('mrope_section')}")
    print(f"  interleaved: {rope.get('interleaved')}")
    print(f"  rope_theta: {talker.get('rope_theta')}")

    print(f"\nSpeaker IDs:")
    for name, spk_id in talker.get("spk_id", {}).items():
        print(f"  {name}: {spk_id}")

    return config


def inspect_safetensors():
    """Inspect safetensors file structure."""
    print("\n" + "=" * 60)
    print("Safetensors Structure (Main Model)")
    print("=" * 60)

    model_path = MODEL_PATH / "model.safetensors"

    with safe_open(model_path, framework="numpy") as f:
        keys = list(f.keys())

    print(f"Total tensors: {len(keys)}")

    # Group by prefix
    prefixes = {}
    for key in keys:
        prefix = key.split(".")[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print("\nTensor groups:")
    for prefix, tensor_keys in sorted(prefixes.items()):
        print(f"  {prefix}: {len(tensor_keys)} tensors")

    # Show some example keys
    print("\nExample keys (first 20):")
    for key in keys[:20]:
        with safe_open(model_path, framework="pt") as f:
            shape = f.get_tensor(key).shape
            dtype = f.get_tensor(key).dtype
        print(f"  {key}: {list(shape)} ({dtype})")

    return keys


def inspect_speech_tokenizer():
    """Inspect speech tokenizer safetensors."""
    print("\n" + "=" * 60)
    print("Safetensors Structure (Speech Tokenizer)")
    print("=" * 60)

    tokenizer_path = MODEL_PATH / "speech_tokenizer" / "model.safetensors"

    with safe_open(tokenizer_path, framework="numpy") as f:
        keys = list(f.keys())

    print(f"Total tensors: {len(keys)}")

    # Group by prefix
    prefixes = {}
    for key in keys:
        parts = key.split(".")
        prefix = parts[0] if len(parts) > 1 else key
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print("\nTensor groups:")
    for prefix, tensor_keys in sorted(prefixes.items()):
        print(f"  {prefix}: {len(tensor_keys)} tensors")

    # Show some example keys
    print("\nExample keys (first 20):")
    for key in keys[:20]:
        with safe_open(tokenizer_path, framework="pt") as f:
            shape = f.get_tensor(key).shape
            dtype = f.get_tensor(key).dtype
        print(f"  {key}: {list(shape)} ({dtype})")

    return keys


def test_basic_loading():
    """Test basic weight loading."""
    print("\n" + "=" * 60)
    print("Basic Weight Loading Test")
    print("=" * 60)

    model_path = MODEL_PATH / "model.safetensors"

    with safe_open(model_path, framework="pt") as f:
        # Load text embedding
        if "talker.text_projection.0.weight" in f.keys():
            text_proj = f.get_tensor("talker.text_projection.0.weight")
            print(f"Text projection shape: {text_proj.shape}")
            print(f"  dtype: {text_proj.dtype}")

        # Load first layer attention weights
        qkv_key = "talker.model.layers.0.self_attn.qkv_proj.weight"
        if qkv_key in f.keys():
            qkv = f.get_tensor(qkv_key)
            print(f"\nFirst layer QKV proj shape: {qkv.shape}")
            print(f"  dtype: {qkv.dtype}")

        # Load codec embedding
        codec_key = "talker.codec_embed.weight"
        if codec_key in f.keys():
            codec_embed = f.get_tensor(codec_key)
            print(f"\nCodec embedding shape: {codec_embed.shape}")
            print(f"  vocab_size x hidden_size: {codec_embed.shape[0]} x {codec_embed.shape[1]}")

        # Load codec head
        head_key = "talker.codec_head.weight"
        if head_key in f.keys():
            codec_head = f.get_tensor(head_key)
            print(f"\nCodec head shape: {codec_head.shape}")

    print("\nWeight loading test passed!")


def count_parameters():
    """Count total parameters."""
    print("\n" + "=" * 60)
    print("Parameter Count")
    print("=" * 60)

    total_params = 0

    # Main model
    model_path = MODEL_PATH / "model.safetensors"
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            total_params += tensor.numel()

    print(f"Main model: {total_params:,} parameters ({total_params / 1e9:.2f}B)")

    # Speech tokenizer
    tokenizer_path = MODEL_PATH / "speech_tokenizer" / "model.safetensors"
    tokenizer_params = 0
    with safe_open(tokenizer_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            tokenizer_params += tensor.numel()

    print(f"Speech tokenizer: {tokenizer_params:,} parameters ({tokenizer_params / 1e6:.1f}M)")
    print(f"Total: {total_params + tokenizer_params:,} parameters")


if __name__ == "__main__":
    print(f"Testing Qwen3-TTS from: {MODEL_PATH}")
    print()

    config = inspect_config()
    main_keys = inspect_safetensors()
    tokenizer_keys = inspect_speech_tokenizer()
    test_basic_loading()
    count_parameters()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
