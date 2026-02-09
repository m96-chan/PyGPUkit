"""Qwen3-TTS Voice Clone Demo.

Demonstrates voice cloning using PyGPUkit's Qwen3-TTS Base model implementation.

Supports two modes:
- x_vector_only_mode (default): Uses speaker embedding only - simpler but lower quality
- ICL mode (--icl): Uses ref_audio codec codes + ref_text for higher quality

Usage:
    # x_vector_only mode (speaker embedding only)
    python examples/demo_qwen3_tts_voice_clone.py --ref-audio path/to/reference.wav

    # ICL mode (reference audio + reference text)
    python examples/demo_qwen3_tts_voice_clone.py --ref-audio path/to/ref.wav --ref-text "Hello" --icl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Model paths
BASE_MODEL_PATH = Path("F:/LLM/Qwen3-TTS-12Hz-1.7B-Base")
SPEECH_TOKENIZER_PATH = BASE_MODEL_PATH / "speech_tokenizer"


def load_audio(audio_path: str | Path, target_sr: int = 24000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import scipy.io.wavfile as wavfile

    sr, audio = wavfile.read(str(audio_path))

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        # Simple linear interpolation resampling
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        old_idx = np.linspace(0, len(audio) - 1, new_len)
        old_idx_floor = np.floor(old_idx).astype(int)
        old_idx_ceil = np.minimum(old_idx_floor + 1, len(audio) - 1)
        frac = old_idx - old_idx_floor
        audio = audio[old_idx_floor] * (1 - frac) + audio[old_idx_ceil] * frac
        audio = audio.astype(np.float32)

    return audio


def load_tokenizer():
    """Load the Qwen tokenizer."""
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)
    return tokenizer


def load_model():
    """Load the Qwen3-TTS Base model."""
    from pygpukit.tts.qwen3 import load_qwen3_tts

    print("Loading Qwen3-TTS Base model...")
    start = time.time()
    model = load_qwen3_tts(BASE_MODEL_PATH, load_speech_tokenizer=False)
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    print(f"  Blocks: {len(model.talker.blocks)}")
    print(f"  Hidden size: {model.config.hidden_size}")
    return model


def load_speaker_encoder():
    """Load the speaker encoder for voice cloning."""
    from pygpukit.tts.qwen3 import load_speaker_encoder as _load

    print("Loading speaker encoder...")
    start = time.time()
    encoder = _load(BASE_MODEL_PATH)
    elapsed = time.time() - start
    print(f"Speaker encoder loaded in {elapsed:.2f}s")
    return encoder


def load_speech_encoder():
    """Load the speech encoder for ICL mode (audio -> codec codes)."""
    from pygpukit.tts.qwen3.speech_encoder import load_speech_encoder as _load

    print("Loading speech encoder (for ICL mode)...")
    start = time.time()
    encoder = _load(BASE_MODEL_PATH)
    elapsed = time.time() - start
    print(f"Speech encoder loaded in {elapsed:.2f}s")
    return encoder


def load_speech_decoder():
    """Load the speech decoder (vocoder)."""
    from pygpukit.tts.qwen3.speech_decoder import load_speech_decoder as _load

    # Check if speech_tokenizer exists in Base model or use CustomVoice
    tokenizer_path = SPEECH_TOKENIZER_PATH
    if not tokenizer_path.exists():
        # Try CustomVoice model's speech tokenizer
        tokenizer_path = Path("F:/LLM/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer")

    print(f"Loading speech decoder from {tokenizer_path}...")
    start = time.time()
    decoder = _load(tokenizer_path)
    elapsed = time.time() - start
    print(f"Speech decoder loaded in {elapsed:.2f}s")
    return decoder


def extract_speaker_embedding(
    encoder,
    audio: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Extract speaker embedding from reference audio."""
    print("Extracting speaker embedding...")
    start = time.time()
    embedding = encoder.extract_embedding(audio, sample_rate=sample_rate)
    elapsed = time.time() - start
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Extraction time: {elapsed:.3f}s")
    return embedding


def encode_audio_to_codes(
    encoder,
    audio: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Encode audio to codec codes for ICL mode."""
    print("Encoding reference audio to codec codes...")
    start = time.time()
    codes = encoder.encode(audio, sample_rate=sample_rate)
    elapsed = time.time() - start
    print(f"  Codes shape: {codes.shape}")
    print(f"  Encoding time: {elapsed:.3f}s")
    return codes


def generate_with_icl(
    model,
    tokenizer,
    ref_codes: np.ndarray,
    ref_text: str,
    target_text: str,
    speaker_embedding: np.ndarray | None = None,
    max_tokens: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
) -> np.ndarray:
    """Generate codec tokens using ICL (In-Context Learning) mode.

    ICL mode provides higher quality voice cloning by using both:
    - Reference audio encoded as codec codes
    - Reference text to associate the voice with language

    Args:
        model: Qwen3TTSModel
        tokenizer: HuggingFace tokenizer
        ref_codes: Reference audio codec codes [32, seq_len]
        ref_text: Reference text (what the reference audio says)
        target_text: Text to synthesize
        speaker_embedding: Optional speaker embedding for additional conditioning
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        Generated codec tokens
    """
    # Special token IDs for Base model
    codec_bos_id = 2149
    codec_eos_id = 2150
    codec_nothink_id = 2155
    tts_text_bos = 151672
    tts_text_eod = 151673

    # Tokenize texts
    ref_text_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
    target_text_tokens = tokenizer.encode(target_text, add_special_tokens=False)

    # Use first codebook for ICL (semantic codes)
    # ref_codes shape: [32, seq_len] -> use first row
    ref_semantic_codes = ref_codes[0].tolist()

    # Build ICL input sequence
    # Format: [tts_text_bos, ref_codec_codes..., ref_text_tokens..., tts_text_eod,
    #          nothink_id, target_text_tokens..., tts_text_eod]
    input_ids = (
        [tts_text_bos]
        + ref_semantic_codes  # Reference audio codes
        + ref_text_tokens      # Reference text
        + [tts_text_eod]
        + [codec_nothink_id]   # Separator
        + target_text_tokens   # Target text
        + [tts_text_eod]
    )

    print("\nICL Input sequence:")
    print(f"  Reference codes: {len(ref_semantic_codes)} frames")
    print(f"  Reference text: '{ref_text}' ({len(ref_text_tokens)} tokens)")
    print(f"  Target text: '{target_text}' ({len(target_text_tokens)} tokens)")
    print(f"  Total input: {len(input_ids)} tokens")

    # Generate with model
    print(f"\nGenerating codec tokens with ICL mode (max {max_tokens})...")
    start = time.time()

    input_array = np.array(input_ids, dtype=np.int64)

    # ICL mode can optionally use speaker embedding for additional conditioning
    codes = model.talker.generate(
        input_array,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=1.0,
        eos_token_id=codec_eos_id,
        speaker_embedding=speaker_embedding,
    )

    elapsed = time.time() - start
    num_tokens = codes.shape[-1]
    tokens_per_sec = num_tokens / elapsed

    print(f"Generated {num_tokens} codec tokens in {elapsed:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Codes shape: {codes.shape}")

    return codes


def generate_with_voice_clone(
    model,
    tokenizer,
    speaker_embedding: np.ndarray,
    text: str,
    max_tokens: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
) -> np.ndarray:
    """Generate codec tokens using voice cloning.

    For x_vector_only_mode, we use the speaker embedding directly.
    The embedding is added to the first token's hidden state.

    Args:
        model: Qwen3TTSModel
        tokenizer: HuggingFace tokenizer
        speaker_embedding: Speaker embedding [2048]
        text: Text to synthesize
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        Generated codec tokens
    """
    # Special token IDs for Base model
    # Base model doesn't have spk_id, uses x_vector instead
    codec_bos_id = 2149
    codec_eos_id = 2150
    codec_nothink_id = 2155
    tts_text_bos = 151672
    tts_text_eod = 151673

    # Tokenize text
    text_tokens = tokenizer.encode(text, add_special_tokens=False)

    # Build input sequence for x_vector_only_mode
    # Format: [tts_text_bos, nothink_id, text_tokens..., tts_text_eod]
    # (No speaker ID since we use x_vector)
    input_ids = (
        [tts_text_bos, codec_nothink_id]
        + text_tokens
        + [tts_text_eod]
    )

    print("\nInput sequence:")
    print(f"  Text: '{text}'")
    print(f"  Text tokens: {len(text_tokens)}")
    print(f"  Total input: {len(input_ids)} tokens")

    # Generate with speaker embedding injection
    print(f"\nGenerating codec tokens with voice clone (max {max_tokens})...")
    start = time.time()

    input_array = np.array(input_ids, dtype=np.int64)

    # Voice clone: inject speaker embedding into first hidden state
    codes = model.talker.generate(
        input_array,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=1.0,
        eos_token_id=codec_eos_id,
        speaker_embedding=speaker_embedding,
    )

    elapsed = time.time() - start
    num_tokens = codes.shape[-1]
    tokens_per_sec = num_tokens / elapsed

    print(f"Generated {num_tokens} codec tokens in {elapsed:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Codes shape: {codes.shape}")

    return codes


def decode_codes_to_audio(decoder, codes: np.ndarray) -> np.ndarray:
    """Decode codec tokens to audio waveform."""
    # Filter special tokens
    codes_flat = codes.flatten()
    valid_mask = codes_flat < 2048
    valid_codes = codes_flat[valid_mask]

    if len(valid_codes) == 0:
        print("  Warning: No valid codec tokens to decode")
        return np.zeros(4800, dtype=np.float32)

    print(f"  Filtered {len(codes_flat)} tokens -> {len(valid_codes)} valid codes")

    # Reshape to [num_quantizers, seq_len]
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

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-6) * 0.9
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16)
        print(f"Saved audio to {output_path}")
        print(f"  Duration: {len(audio) / sample_rate:.2f}s")
    except ImportError:
        print("scipy not available, saving as numpy")
        np.save(output_path.with_suffix(".npy"), audio)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Clone Demo")
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Path to reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference text (what the reference audio says). Required for ICL mode.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of voice cloning with Qwen3 TTS.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qwen3_voice_clone_output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum codec tokens to generate",
    )
    parser.add_argument(
        "--icl",
        action="store_true",
        help="Use ICL (In-Context Learning) mode for higher quality voice cloning",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio decoding",
    )
    args = parser.parse_args()

    # Validate ICL mode requirements
    if args.icl and not args.ref_text:
        parser.error("--ref-text is required when using --icl mode")

    print("=" * 60)
    print("Qwen3-TTS Voice Clone Demo")
    print("=" * 60)

    # Check reference audio exists
    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        print(f"Error: Reference audio not found: {ref_audio_path}")
        return

    # Load reference audio
    print(f"\nLoading reference audio: {ref_audio_path}")
    ref_audio = load_audio(ref_audio_path)
    print(f"  Duration: {len(ref_audio) / 24000:.2f}s")
    print(f"  Samples: {len(ref_audio)}")

    # Load models
    tokenizer = load_tokenizer()
    model = load_model()
    speaker_encoder = load_speaker_encoder()

    # Load speech encoder for ICL mode
    speech_encoder = None
    if args.icl:
        speech_encoder = load_speech_encoder()

    # Load speech decoder if needed
    speech_decoder = None
    if not args.no_audio:
        speech_decoder = load_speech_decoder()

    # Extract speaker embedding
    speaker_embedding = extract_speaker_embedding(speaker_encoder, ref_audio)

    # Generate codec tokens
    if args.icl:
        # ICL mode: use reference audio codes + reference text
        print("\n" + "-" * 40)
        print("Using ICL (In-Context Learning) mode")
        print("-" * 40)

        # Encode reference audio to codes
        ref_codes = encode_audio_to_codes(speech_encoder, ref_audio)

        codes = generate_with_icl(
            model,
            tokenizer,
            ref_codes=ref_codes,
            ref_text=args.ref_text,
            target_text=args.text,
            speaker_embedding=speaker_embedding,  # Optional additional conditioning
            max_tokens=args.max_tokens,
        )
    else:
        # x_vector_only mode: use speaker embedding only
        print("\n" + "-" * 40)
        print("Using x_vector_only mode (speaker embedding only)")
        print("-" * 40)

        codes = generate_with_voice_clone(
            model,
            tokenizer,
            speaker_embedding,
            text=args.text,
            max_tokens=args.max_tokens,
        )

    # Save codes
    output_path = Path(args.output)
    np.save(output_path.with_suffix(".npy"), codes)
    print(f"Saved codes to {output_path.with_suffix('.npy')}")

    # Decode to audio
    if speech_decoder is not None:
        audio = decode_codes_to_audio(speech_decoder, codes)
        save_audio(audio, output_path)
    else:
        print("\nSkipping audio decoding (--no-audio flag)")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
