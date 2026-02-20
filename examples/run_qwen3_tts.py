"""Qwen3-TTS inference test script.

Tests loading and running the Qwen3-TTS model with PyGPUkit.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Model path
MODEL_PATH = Path("F:/LLM/Qwen3-TTS-12Hz-0.6B-CustomVoice")


def load_tokenizer():
    """Load the Qwen tokenizer using transformers."""
    try:
        from transformers import AutoTokenizer

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        print(f"  vocab_size: {tokenizer.vocab_size}")
        print(f"  eos_token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
        print(f"  pad_token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        return tokenizer
    except ImportError:
        print("ERROR: transformers package required for tokenizer")
        print("Install with: pip install transformers")
        sys.exit(1)


def test_weight_loading():
    """Test loading weights from safetensors."""
    print("\n" + "=" * 60)
    print("Testing Weight Loading")
    print("=" * 60)

    from pygpukit.tts.qwen3 import load_safetensors_weights, load_config, build_qwen3_tts_config

    # Load config
    print("Loading config...")
    raw_config = load_config(MODEL_PATH)
    config = build_qwen3_tts_config(raw_config)
    print(f"  model_type: {config.model_type}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")

    # Load weights
    print("\nLoading weights (this may take a moment)...")
    weights = load_safetensors_weights(MODEL_PATH / "model.safetensors")
    print(f"  Loaded {len(weights)} tensors")

    # Check some key weights
    key_weights = [
        "talker.model.embed_tokens.weight",
        "talker.codec_embed.weight",
        "talker.codec_head.weight",
        "talker.model.layers.0.self_attn.qkv_proj.weight",
        "talker.model.norm.weight",
    ]

    print("\nKey weight shapes:")
    for key in key_weights:
        if key in weights:
            shape = weights[key].shape
            print(f"  {key}: {shape}")
        else:
            # Try without talker prefix
            alt_key = key.replace("talker.", "")
            if alt_key in weights:
                print(f"  {alt_key}: {weights[alt_key].shape}")
            else:
                print(f"  {key}: NOT FOUND")

    return config, weights


def test_model_building():
    """Test building the TalkerModel from weights."""
    print("\n" + "=" * 60)
    print("Testing Model Building")
    print("=" * 60)

    from pygpukit.tts.qwen3 import (
        load_qwen3_tts,
        load_config,
        build_qwen3_tts_config,
    )

    # Load config first to inspect
    raw_config = load_config(MODEL_PATH)
    config = build_qwen3_tts_config(raw_config)

    print(f"Building model with config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  intermediate_size: {config.intermediate_size}")

    # Load model
    print("\nLoading Qwen3-TTS model...")
    try:
        model = load_qwen3_tts(MODEL_PATH, load_speech_tokenizer=False)
        print("Model loaded successfully!")
        print(f"  Talker blocks: {len(model.talker.blocks)}")
        print(f"  embed_tokens shape: {model.talker.embed_tokens.shape}")
        print(f"  codec_embed shape: {model.talker.codec_embed.shape}")
        print(f"  codec_head shape: {model.talker.codec_head.shape}")
        return model
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, tokenizer):
    """Test a simple forward pass."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    # Create a simple test input
    # For CustomVoice model, the input format is specific
    # Let's first test with raw text

    text = "Hello world"
    print(f"Input text: '{text}'")

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Convert to numpy array
    input_ids = np.array(tokens, dtype=np.int64)

    # Forward pass
    print("\nRunning forward pass...")
    try:
        logits, _ = model.talker.forward(input_ids)
        print(f"Output logits shape: {logits.shape}")
        print(f"Output logits dtype: {logits.dtype}")

        # Check the logits
        print(f"Logits stats:")
        print(f"  min: {logits.min():.4f}")
        print(f"  max: {logits.max():.4f}")
        print(f"  mean: {logits.mean():.4f}")

        # Sample from last token
        last_logits = logits[-1]
        top_5_ids = np.argsort(last_logits)[-5:][::-1]
        print(f"\nTop 5 predicted tokens:")
        for idx in top_5_ids:
            score = last_logits[idx]
            print(f"  {idx}: {score:.4f}")

        print("\nForward pass successful!")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model, tokenizer):
    """Test codec generation."""
    print("\n" + "=" * 60)
    print("Testing Codec Generation")
    print("=" * 60)

    # Load raw config to get special tokens
    with open(MODEL_PATH / "config.json") as f:
        config = json.load(f)

    talker_config = config.get("talker_config", {})
    spk_ids = talker_config.get("spk_id", {})
    codec_bos_id = talker_config.get("codec_bos_id", 2149)
    codec_eos_id = talker_config.get("codec_eos_token_id", 2150)
    codec_nothink_id = talker_config.get("codec_nothink_id", 2155)

    print(f"Codec special tokens:")
    print(f"  BOS: {codec_bos_id}")
    print(f"  EOS: {codec_eos_id}")
    print(f"  NOTHINK: {codec_nothink_id}")
    print(f"Available speakers: {list(spk_ids.keys())}")

    # Build input for CustomVoice generation
    # The format is: <tts_text_bos> [SPEAKER_ID] [NOTHINK_ID] <text tokens> <tts_text_eod>
    speaker = "vivian"
    spk_id = spk_ids.get(speaker, 3065)
    text = "Hello, this is a test."

    print(f"\nGenerating with speaker: {speaker} (id={spk_id})")
    print(f"Text: {text}")

    # Tokenize text
    text_tokens = tokenizer.encode(text, add_special_tokens=False)

    # Build full input sequence
    # Format: [tts_text_bos, spk_id, nothink_id, text_tokens..., tts_text_eod]
    tts_text_bos = 151672
    tts_text_eod = 151673

    input_ids = [tts_text_bos, spk_id, codec_nothink_id] + text_tokens + [tts_text_eod]
    print(f"Input sequence length: {len(input_ids)}")

    # Generate a few tokens to test
    print("\nGenerating codec tokens (max 10 for quick test)...")
    try:
        input_array = np.array(input_ids, dtype=np.int64)
        codes = model.talker.generate(
            input_array,
            max_new_tokens=10,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            eos_token_id=codec_eos_id,
        )
        print(f"Generated codes shape: {codes.shape}")
        print(f"Generated codes: {codes}")
        print("\nGeneration test passed!")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"Qwen3-TTS Inference Test")
    print(f"Model path: {MODEL_PATH}")
    print()

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Test weight loading
    test_weight_loading()

    # Test model building
    model = test_model_building()

    if model is not None:
        # Test forward pass
        test_forward_pass(model, tokenizer)

        # Test generation
        test_generation(model, tokenizer)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
