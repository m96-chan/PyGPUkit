"""Qwen3-TTS Demo Script.

Demonstrates text-to-speech generation using PyGPUkit's Qwen3-TTS implementation.

Usage:
    python examples/demo_qwen3_tts.py
    python examples/demo_qwen3_tts.py --text "Hello world" --speaker vivian
    python examples/demo_qwen3_tts.py --output output.wav
    python examples/demo_qwen3_tts.py --no-audio  # Skip audio decoding
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Model path
MODEL_PATH = Path("F:/LLM/Qwen3-TTS-12Hz-0.6B-CustomVoice")
SPEECH_TOKENIZER_PATH = MODEL_PATH / "speech_tokenizer"


def load_tokenizer():
    """Load the Qwen tokenizer."""
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    return tokenizer


def load_model():
    """Load the Qwen3-TTS model."""
    from pygpukit.tts.qwen3 import load_qwen3_tts

    print("Loading Qwen3-TTS model...")
    start = time.time()
    model = load_qwen3_tts(MODEL_PATH, load_speech_tokenizer=False)
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    print(f"  Blocks: {len(model.talker.blocks)}")
    print(f"  Hidden size: {model.config.hidden_size}")
    return model


def load_speech_decoder():
    """Load the speech tokenizer decoder (vocoder)."""
    from pygpukit.tts.qwen3.speech_decoder import load_speech_decoder as _load

    print("Loading speech decoder (vocoder)...")
    start = time.time()
    decoder = _load(SPEECH_TOKENIZER_PATH)
    elapsed = time.time() - start
    print(f"Speech decoder loaded in {elapsed:.2f}s")
    return decoder


def get_speaker_config():
    """Get speaker configuration from model config."""
    with open(MODEL_PATH / "config.json") as f:
        config = json.load(f)

    talker = config.get("talker_config", {})
    return {
        "spk_ids": talker.get("spk_id", {}),
        "codec_bos_id": talker.get("codec_bos_id", 2149),
        "codec_eos_id": talker.get("codec_eos_token_id", 2150),
        "codec_nothink_id": talker.get("codec_nothink_id", 2155),
        "tts_text_bos": 151672,
        "tts_text_eod": 151673,
    }


def generate_codec_tokens(
    model,
    tokenizer,
    text: str,
    speaker: str = "vivian",
    max_tokens: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
) -> np.ndarray:
    """Generate codec tokens from text.

    Args:
        model: Qwen3TTSModel
        tokenizer: HuggingFace tokenizer
        text: Text to synthesize
        speaker: Speaker name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        Generated codec tokens [num_codebooks, seq_len]
    """
    config = get_speaker_config()
    spk_ids = config["spk_ids"]

    if speaker not in spk_ids:
        available = list(spk_ids.keys())
        raise ValueError(f"Unknown speaker '{speaker}'. Available: {available}")

    spk_id = spk_ids[speaker]

    # Tokenize text
    text_tokens = tokenizer.encode(text, add_special_tokens=False)

    # Build input sequence
    # Format: [tts_text_bos, spk_id, nothink_id, text_tokens..., tts_text_eod]
    input_ids = (
        [
            config["tts_text_bos"],
            spk_id,
            config["codec_nothink_id"],
        ]
        + text_tokens
        + [config["tts_text_eod"]]
    )

    print("\nInput sequence:")
    print(f"  Text: '{text}'")
    print(f"  Speaker: {speaker} (id={spk_id})")
    print(f"  Text tokens: {len(text_tokens)}")
    print(f"  Total input: {len(input_ids)} tokens")

    # Generate
    print(f"\nGenerating codec tokens (max {max_tokens})...")
    start = time.time()

    input_array = np.array(input_ids, dtype=np.int64)
    codes = model.talker.generate(
        input_array,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=1.0,
        eos_token_id=config["codec_eos_id"],
    )

    elapsed = time.time() - start
    num_tokens = codes.shape[-1]
    tokens_per_sec = num_tokens / elapsed

    print(f"Generated {num_tokens} codec tokens in {elapsed:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Codes shape: {codes.shape}")

    return codes


def save_codes_as_debug(codes: np.ndarray, output_path: Path):
    """Save codec codes as numpy file for debugging."""
    np.save(output_path.with_suffix(".npy"), codes)
    print(f"Saved codes to {output_path.with_suffix('.npy')}")


def decode_codes_to_audio(decoder, codes: np.ndarray) -> np.ndarray:
    """Decode codec tokens to audio waveform using speech decoder.

    Args:
        decoder: SpeechDecoder instance
        codes: Codec tokens [1, seq_len] or [seq_len]

    Returns:
        Audio waveform [samples]
    """
    # Prepare codes in expected format [num_quantizers, seq_len]
    if codes.ndim == 1:
        codes = codes[np.newaxis, :]

    # Flatten and filter out special tokens (>= 2048)
    # Special tokens: BOS=2149, EOS=2150, PAD=2148, NOTHINK=2155
    codes_flat = codes.flatten()
    valid_mask = codes_flat < 2048
    valid_codes = codes_flat[valid_mask]

    if len(valid_codes) == 0:
        print("  Warning: No valid codec tokens to decode")
        return np.zeros(4800, dtype=np.float32)  # Return 200ms of silence

    print(f"  Filtered {len(codes_flat)} tokens -> {len(valid_codes)} valid codes")

    # Reshape to [num_quantizers, seq_len]
    # Single codebook - replicate to 16 quantizers
    valid_codes = valid_codes[np.newaxis, :]
    codes = np.tile(valid_codes, (16, 1))

    print(f"\nDecoding {codes.shape[1]} codec frames to audio...")
    start = time.time()
    audio = decoder.decode(codes)
    elapsed = time.time() - start

    duration = len(audio) / 24000
    rtf = duration / elapsed
    print(f"  Audio samples: {len(audio)}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Decode time: {elapsed:.3f}s")
    print(f"  Real-time factor: {rtf:.1f}x")

    return audio


def save_audio(audio: np.ndarray, output_path: Path, sample_rate: int = 24000):
    """Save audio to WAV file."""
    try:
        import scipy.io.wavfile as wavfile

        # Normalize to int16 range
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16)
        print(f"Saved audio to {output_path}")
        print(f"  Duration: {len(audio) / sample_rate:.2f}s")
        print(f"  Sample rate: {sample_rate} Hz")
    except ImportError:
        print("scipy not available, saving as raw numpy instead")
        np.save(output_path.with_suffix(".npy"), audio)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Demo")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Qwen3 text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="vivian",
        help="Speaker name (vivian, ryan, serena, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qwen3_tts_output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum codec tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speakers and exit",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio decoding (only generate codec tokens)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-TTS Demo")
    print("=" * 60)

    # List speakers if requested
    if args.list_speakers:
        config = get_speaker_config()
        print("\nAvailable speakers:")
        for name, spk_id in config["spk_ids"].items():
            print(f"  {name}: {spk_id}")
        return

    # Load tokenizer and model
    tokenizer = load_tokenizer()
    model = load_model()

    # Load speech decoder if needed
    speech_decoder = None
    if not args.no_audio:
        speech_decoder = load_speech_decoder()

    # Generate codec tokens
    codes = generate_codec_tokens(
        model,
        tokenizer,
        text=args.text,
        speaker=args.speaker,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Save codes for debugging
    output_path = Path(args.output)
    save_codes_as_debug(codes, output_path)

    # Decode to audio
    if speech_decoder is not None:
        audio = decode_codes_to_audio(speech_decoder, codes)
        # Normalize audio to prevent clipping
        audio = audio / (np.abs(audio).max() + 1e-6) * 0.9
        save_audio(audio, output_path)
    else:
        print("\nSkipping audio decoding (--no-audio flag)")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
