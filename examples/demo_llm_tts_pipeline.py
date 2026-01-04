"""Demo: LLM to TTS Pipeline.

This example demonstrates connecting LLM text generation directly to TTS synthesis
with streaming audio output.

Requirements:
    - LLM model (e.g., Qwen2.5-7B-Instruct)
    - TTS model (Kokoro)
    - Audio playback library (optional, for real-time playback)

Usage:
    python examples/demo_llm_tts_pipeline.py

"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np


def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save audio to WAV file."""
    # Normalize to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM to TTS Pipeline Demo")
    parser.add_argument(
        "--llm-path",
        type=str,
        default="F:/LLM/Qwen2.5-7B-Instruct",
        help="Path to LLM model",
    )
    parser.add_argument(
        "--tts-path",
        type=str,
        default="F:/TTS/kokoro-v1.0",
        help="Path to TTS model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Input prompt for the LLM",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_speech.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="TTS voice to use",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (save chunks separately)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LLM to TTS Pipeline Demo")
    print("=" * 60)

    # Check if models exist
    llm_path = Path(args.llm_path)
    tts_path = Path(args.tts_path)

    if not llm_path.exists():
        print(f"LLM model not found: {llm_path}")
        print("Please provide a valid path with --llm-path")
        return

    if not tts_path.exists():
        print(f"TTS model not found: {tts_path}")
        print("Please provide a valid path with --tts-path")
        return

    # Import models
    print("\nLoading models...")
    start = time.perf_counter()

    from pygpukit.llm import QwenModel
    from pygpukit.pipeline import LLMToTTSPipeline
    from pygpukit.tts.kokoro import KokoroModel

    # Load LLM
    print(f"  Loading LLM from {llm_path}...")
    llm = QwenModel.from_safetensors(str(llm_path))

    # Load TTS
    print(f"  Loading TTS from {tts_path}...")
    tts = KokoroModel.from_pretrained(str(tts_path))

    load_time = time.perf_counter() - start
    print(f"  Models loaded in {load_time:.2f}s")

    # Create pipeline
    print("\nCreating LLM-TTS pipeline...")
    pipeline = LLMToTTSPipeline(
        llm_model=llm,
        tts_model=tts,
        voice=args.voice,
    )

    # Generate speech
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    print("Generating speech...")

    start = time.perf_counter()
    audio_chunks = []
    sample_rate = None

    for i, chunk in enumerate(
        pipeline.generate_speech(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            stream_sentences=True,
        )
    ):
        sample_rate = chunk.sample_rate
        audio_chunks.append(chunk.audio)

        print(f"  Chunk {i + 1}: '{chunk.text[:50]}...' ({chunk.duration_ms:.1f}ms audio)")

        if args.stream:
            # Save each chunk separately
            chunk_path = f"chunk_{i:03d}.wav"
            save_wav(chunk.audio, chunk.sample_rate, chunk_path)
            print(f"    Saved: {chunk_path}")

    total_time = time.perf_counter() - start

    # Concatenate and save
    if audio_chunks and sample_rate:
        final_audio = np.concatenate(audio_chunks)
        save_wav(final_audio, sample_rate, args.output)
        print(f"\nSaved: {args.output}")

    # Print statistics
    stats = pipeline.stats
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"  Total tokens generated: {stats.total_tokens}")
    print(f"  Total sentences: {stats.total_sentences}")
    print(f"  Total audio duration: {stats.total_audio_duration_ms:.1f}ms")
    print(f"  Total synthesis time: {stats.total_synthesis_time_ms:.1f}ms")
    print(f"  First audio latency: {stats.first_audio_latency_ms:.1f}ms")
    print(f"  Realtime factor: {stats.realtime_factor:.2f}x")
    print(f"  Total pipeline time: {total_time * 1000:.1f}ms")


if __name__ == "__main__":
    main()
