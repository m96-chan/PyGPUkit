"""Demo: Full Voice Pipeline.

Tests each component of the voice pipeline:
1. TTS synthesis (text to speech)
2. LLM to TTS pipeline (text generation to speech)
3. Full voice pipeline (ASR to LLM to TTS)

Usage:
    # Test TTS only
    python examples/demo_full_voice_pipeline.py --tts-only

    # Test LLM to TTS
    python examples/demo_full_voice_pipeline.py --llm-tts

    # Full pipeline with audio file
    python examples/demo_full_voice_pipeline.py --full --audio input.wav

    # Full pipeline simulation (no real audio)
    python examples/demo_full_voice_pipeline.py --simulate
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np


def resolve_model_path(path: str) -> str:
    """Resolve model directory to safetensors file path.

    If the path is a directory, look for model.safetensors or .index.json.
    """
    p = Path(path)
    if p.is_dir():
        # Check for sharded model first
        index_file = p / "model.safetensors.index.json"
        if index_file.exists():
            return str(index_file)
        # Check for single file model
        single_file = p / "model.safetensors"
        if single_file.exists():
            return str(single_file)
        # Fallback to directory (will fail, but with clear error)
        return str(p)
    return str(p)


def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save audio to WAV file."""
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"Saved: {path} ({len(audio)/sample_rate:.2f}s)")


def demo_tts_only(tts_path: str, output_dir: str) -> None:
    """Demo 1: TTS synthesis only."""
    print("=" * 60)
    print("Demo 1: TTS Synthesis")
    print("=" * 60)

    from pygpukit.tts.kokoro import KokoroModel

    print(f"\nLoading TTS from {tts_path}...")
    start = time.perf_counter()
    tts = KokoroModel.from_pretrained(tts_path)
    print(f"  Loaded in {time.perf_counter() - start:.2f}s")

    # Test sentences
    sentences = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "PyGPUkit provides GPU-accelerated speech synthesis.",
    ]

    Path(output_dir).mkdir(exist_ok=True)

    for i, text in enumerate(sentences):
        print(f"\n[{i+1}/{len(sentences)}] Synthesizing: '{text[:40]}...'")

        start = time.perf_counter()
        result = tts.synthesize(text)
        elapsed = time.perf_counter() - start

        # Get audio from AudioBuffer
        if hasattr(result.audio, "data"):
            # AudioBuffer: data is GPUArray
            audio = result.audio.data.to_numpy()
            sample_rate = result.audio.sample_rate
        elif hasattr(result.audio, "to_numpy"):
            audio = result.audio.to_numpy()
            sample_rate = 24000  # Kokoro default
        else:
            audio = np.array(result.audio, dtype=np.float32)
            sample_rate = 24000

        duration = len(audio) / sample_rate
        rtf = elapsed / duration

        print(f"  Duration: {duration:.2f}s, Time: {elapsed:.2f}s, RTF: {rtf:.2f}x")

        output_path = f"{output_dir}/tts_demo_{i+1}.wav"
        save_wav(audio, sample_rate, output_path)

    print("\nTTS demo complete!")


def demo_llm_tts(llm_path: str, tts_path: str, output_dir: str) -> None:
    """Demo 2: LLM to TTS pipeline."""
    print("=" * 60)
    print("Demo 2: LLM to TTS Pipeline")
    print("=" * 60)

    from tokenizers import Tokenizer

    from pygpukit.llm import load_model_from_safetensors
    from pygpukit.pipeline import LLMToTTSPipeline
    from pygpukit.tts.kokoro import KokoroModel

    print(f"\nLoading LLM from {llm_path}...")
    start = time.perf_counter()
    llm = load_model_from_safetensors(resolve_model_path(llm_path))
    print(f"  LLM loaded in {time.perf_counter() - start:.2f}s")

    # Load tokenizer
    llm_dir = Path(llm_path)
    tokenizer_path = llm_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"  Tokenizer loaded from {tokenizer_path}")
    else:
        print(f"  Warning: tokenizer.json not found in {llm_dir}")
        tokenizer = None

    print(f"\nLoading TTS from {tts_path}...")
    start = time.perf_counter()
    tts = KokoroModel.from_pretrained(tts_path)
    print(f"  TTS loaded in {time.perf_counter() - start:.2f}s")

    # Create pipeline with tokenizer
    pipeline = LLMToTTSPipeline(llm, tts, llm_tokenizer=tokenizer)

    # Test prompt
    prompt = "What is the capital of Japan? Answer briefly."
    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    Path(output_dir).mkdir(exist_ok=True)

    start = time.perf_counter()
    audio_chunks = []
    sample_rate = 24000

    print("Generating response and synthesizing speech...")
    for i, chunk in enumerate(pipeline.generate_speech(prompt, max_new_tokens=64)):
        sample_rate = chunk.sample_rate
        audio_chunks.append(chunk.audio)
        print(f"  Chunk {i+1}: '{chunk.text[:30]}...' ({chunk.duration_ms:.0f}ms)")

    total_time = time.perf_counter() - start

    if audio_chunks:
        audio = np.concatenate(audio_chunks)
        output_path = f"{output_dir}/llm_tts_demo.wav"
        save_wav(audio, sample_rate, output_path)

        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Audio duration: {len(audio)/sample_rate:.2f}s")

    # Print stats
    stats = pipeline.stats
    print("\nStatistics:")
    print(f"  Tokens: {stats.total_tokens}")
    print(f"  Sentences: {stats.total_sentences}")
    print(f"  First audio latency: {stats.first_audio_latency_ms:.0f}ms")

    print("\nLLM-TTS demo complete!")


def demo_vad() -> None:
    """Demo 3: VAD (Voice Activity Detection) only."""
    print("=" * 60)
    print("Demo 3: VAD Test")
    print("=" * 60)

    from pygpukit.pipeline import VADConfig, VoiceActivityDetector

    # Create VAD
    config = VADConfig(
        energy_threshold=0.02,
        silence_threshold=0.01,
        min_silence_duration=0.3,
    )
    vad = VoiceActivityDetector(config)

    # Simulate audio with speech and silence
    sample_rate = 16000
    duration = 3.0  # seconds

    print("\nSimulating audio with speech patterns...")

    # Create test audio: silence -> speech -> silence -> speech -> silence
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate pattern: speech from 0.5-1.5s and 2.0-2.5s
    audio = np.zeros_like(t)
    speech_mask1 = (t >= 0.5) & (t <= 1.5)
    speech_mask2 = (t >= 2.0) & (t <= 2.5)

    # Add speech-like signal (random noise with envelope)
    audio[speech_mask1] = np.random.randn(np.sum(speech_mask1)) * 0.1
    audio[speech_mask2] = np.random.randn(np.sum(speech_mask2)) * 0.1

    # Process in chunks
    chunk_size = int(0.1 * sample_rate)  # 100ms chunks
    print(f"Processing {len(audio)/sample_rate:.1f}s audio in {chunk_size/sample_rate*1000:.0f}ms chunks...")

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size].astype(np.float32)
        events = vad.process_audio(chunk)

        chunk_time = i / sample_rate
        for event in events:
            print(f"  [{chunk_time:.2f}s] Event: {event.event_type}")
            if event.event_type == "speech_end" and event.audio is not None:
                print(f"    Speech duration: {event.duration:.2f}s")

    print("\nVAD demo complete!")


def demo_full_pipeline_simulate(
    llm_path: str, tts_path: str, output_dir: str
) -> None:
    """Demo 4: Full pipeline with simulated speech input."""
    print("=" * 60)
    print("Demo 4: Full Voice Pipeline (Simulated)")
    print("=" * 60)

    from pygpukit.llm import load_model_from_safetensors
    from pygpukit.pipeline import VADConfig, VoicePipeline, VoicePipelineCallback
    from pygpukit.tts.kokoro import KokoroModel

    # Since we don't have Whisper loaded, we'll create a mock
    # that returns predefined text
    class MockWhisper:
        """Mock Whisper model for testing."""

        def transcribe(self, audio: np.ndarray) -> object:
            """Return mock transcription."""

            class Result:
                text = "What is artificial intelligence?"

            return Result()

    class DemoCallback(VoicePipelineCallback):
        """Callback that prints events."""

        def on_listening_start(self) -> None:
            print("\n[Pipeline] Listening...")

        def on_speech_detected(self) -> None:
            print("[Pipeline] Speech detected")

        def on_transcription_start(self) -> None:
            print("[Pipeline] Transcribing...")

        def on_transcription_complete(self, text: str) -> None:
            print(f"[User] {text}")

        def on_generation_start(self) -> None:
            print("[Pipeline] Generating response...")
            print("[Assistant] ", end="", flush=True)

        def on_generation_token(self, token: str) -> None:
            print(token, end="", flush=True)

        def on_generation_complete(self, text: str) -> None:
            print()

        def on_tts_start(self) -> None:
            print("[Pipeline] Synthesizing speech...")

        def on_audio_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
            duration = len(audio) / sample_rate
            print(f"[Pipeline] Audio chunk: {duration:.2f}s")

        def on_tts_complete(self) -> None:
            pass

        def on_error(self, error: Exception) -> None:
            print(f"[Error] {error}")

    print(f"\nLoading LLM from {llm_path}...")
    start = time.perf_counter()
    llm = load_model_from_safetensors(resolve_model_path(llm_path))
    print(f"  LLM loaded in {time.perf_counter() - start:.2f}s")

    print(f"\nLoading TTS from {tts_path}...")
    start = time.perf_counter()
    tts = KokoroModel.from_pretrained(tts_path)
    print(f"  TTS loaded in {time.perf_counter() - start:.2f}s")

    # Create pipeline with mock Whisper
    whisper = MockWhisper()
    callback = DemoCallback()

    vad_config = VADConfig(
        energy_threshold=0.02,
        min_silence_duration=0.3,
    )

    pipeline = VoicePipeline(
        whisper_model=whisper,
        llm_model=llm,
        tts_model=tts,
        vad_config=vad_config,
        callback=callback,
        system_prompt="You are a helpful AI assistant. Keep responses brief and clear.",
    )

    # Simulate user speech
    print("\n--- Simulating voice conversation ---")

    # Create simulated audio: noise (speech) followed by silence
    sample_rate = 16000
    speech_duration = 1.5
    silence_duration = 0.5

    speech = np.random.randn(int(speech_duration * sample_rate)).astype(np.float32) * 0.1
    silence = np.zeros(int(silence_duration * sample_rate), dtype=np.float32)
    simulated_audio = np.concatenate([speech, silence])

    Path(output_dir).mkdir(exist_ok=True)

    # Process through pipeline
    pipeline.start_listening()

    output_audio_chunks = []
    output_sample_rate = 24000

    # Feed audio in chunks
    chunk_size = int(0.1 * sample_rate)
    start_time = time.perf_counter()

    for i in range(0, len(simulated_audio), chunk_size):
        chunk = simulated_audio[i : i + chunk_size]

        for output_chunk in pipeline.process_audio(chunk):
            output_audio_chunks.append(output_chunk)
            output_sample_rate = 24000

    total_time = time.perf_counter() - start_time

    # Save output
    if output_audio_chunks:
        output_audio = np.concatenate(output_audio_chunks)
        output_path = f"{output_dir}/voice_pipeline_demo.wav"
        save_wav(output_audio, output_sample_rate, output_path)

    # Print stats
    stats = pipeline.stats
    print("\n" + "=" * 40)
    print("Pipeline Statistics:")
    print(f"  Total turns: {stats.total_turns}")
    print(f"  User speech: {stats.total_user_speech_duration:.2f}s")
    print(f"  Assistant speech: {stats.total_assistant_speech_duration:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    print("\nFull pipeline demo complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Pipeline Demo")
    parser.add_argument("--tts-only", action="store_true", help="Test TTS only")
    parser.add_argument("--llm-tts", action="store_true", help="Test LLM to TTS pipeline")
    parser.add_argument("--vad", action="store_true", help="Test VAD only")
    parser.add_argument("--simulate", action="store_true", help="Simulate full pipeline")
    parser.add_argument(
        "--llm-path",
        type=str,
        default="F:/LLM/Qwen2.5-7B-Instruct",
        help="Path to LLM model",
    )
    parser.add_argument(
        "--tts-path",
        type=str,
        default="F:/LLM/Kokoro-82M",
        help="Path to TTS model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/voice_demo",
        help="Output directory",
    )
    args = parser.parse_args()

    # Run requested demos
    if args.tts_only:
        demo_tts_only(args.tts_path, args.output_dir)
    elif args.llm_tts:
        demo_llm_tts(args.llm_path, args.tts_path, args.output_dir)
    elif args.vad:
        demo_vad()
    elif args.simulate:
        demo_full_pipeline_simulate(args.llm_path, args.tts_path, args.output_dir)
    else:
        # Run all demos in sequence
        print("Running all demos...\n")
        demo_vad()
        print("\n")
        demo_tts_only(args.tts_path, args.output_dir)
        print("\n")
        demo_llm_tts(args.llm_path, args.tts_path, args.output_dir)


if __name__ == "__main__":
    main()
