"""Demo: Full Voice Pipeline.

This example demonstrates the complete voice conversation pipeline:
Audio Input → Whisper (ASR) → LLM → TTS → Audio Output

Requirements:
    - Whisper model (e.g., kotoba-whisper-v2.0)
    - LLM model (e.g., Qwen2.5-7B-Instruct)
    - TTS model (Kokoro)
    - PyAudio for audio capture (optional)

Usage:
    # With audio file
    python examples/demo_voice_pipeline.py --audio input.wav

    # With microphone (requires PyAudio)
    python examples/demo_voice_pipeline.py --mic

"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np


def resolve_model_path(path: str) -> str:
    """Resolve model directory to safetensors file path."""
    p = Path(path)
    if p.is_dir():
        index_file = p / "model.safetensors.index.json"
        if index_file.exists():
            return str(index_file)
        single_file = p / "model.safetensors"
        if single_file.exists():
            return str(single_file)
    return str(p)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import scipy.io.wavfile as wav

    sr, audio = wav.read(path)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        from scipy import signal

        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_samples)

    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save audio to WAV file."""
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


class ConsolePipelineCallback:
    """Callback that prints pipeline events to console."""

    def on_listening_start(self) -> None:
        print("\n[Listening...]")

    def on_speech_detected(self) -> None:
        print("[Speech detected]")

    def on_transcription_start(self) -> None:
        print("[Transcribing...]")

    def on_transcription_complete(self, text: str) -> None:
        print(f"\nUser: {text}")

    def on_generation_start(self) -> None:
        print("\nAssistant: ", end="", flush=True)

    def on_generation_token(self, token: str) -> None:
        print(token, end="", flush=True)

    def on_generation_complete(self, text: str) -> None:
        print()  # Newline

    def on_tts_start(self) -> None:
        pass

    def on_audio_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        duration = len(audio) / sample_rate
        print(f"[Audio chunk: {duration:.2f}s]")

    def on_tts_complete(self) -> None:
        pass

    def on_interruption(self) -> None:
        print("\n[Interrupted by user]")

    def on_error(self, error: Exception) -> None:
        print(f"\n[Error: {error}]")


def demo_with_audio_file(
    audio_path: str,
    whisper_path: str,
    llm_path: str,
    tts_path: str,
    output_path: str,
    voice: str | None = None,
) -> None:
    """Demo with pre-recorded audio file."""
    print("=" * 60)
    print("Voice Pipeline Demo (Audio File)")
    print("=" * 60)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    print(f"  Duration: {len(audio) / 16000:.2f}s")

    # Load models
    print("\nLoading models...")
    start = time.perf_counter()

    from pygpukit.asr import WhisperModel
    from pygpukit.llm import load_model_from_safetensors
    from pygpukit.pipeline import VoicePipeline
    from pygpukit.tts.kokoro import KokoroModel

    print(f"  Loading Whisper from {whisper_path}...")
    whisper = WhisperModel.from_pretrained(whisper_path)

    print(f"  Loading LLM from {llm_path}...")
    llm = load_model_from_safetensors(resolve_model_path(llm_path))

    print(f"  Loading TTS from {tts_path}...")
    tts = KokoroModel.from_pretrained(tts_path)

    load_time = time.perf_counter() - start
    print(f"  Models loaded in {load_time:.2f}s")

    # Create pipeline
    callback = ConsolePipelineCallback()
    pipeline = VoicePipeline(
        whisper_model=whisper,
        llm_model=llm,
        tts_model=tts,
        voice=voice,
        callback=callback,
        system_prompt="You are a helpful voice assistant. Keep responses concise.",
    )

    # Process audio
    print("\nProcessing audio...")
    pipeline.start_listening()

    output_audio_chunks = []
    sample_rate = 24000  # TTS sample rate

    # Simulate real-time by feeding audio in chunks
    chunk_size = int(0.1 * 16000)  # 100ms chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]

        for output_chunk in pipeline.process_audio(chunk):
            output_audio_chunks.append(output_chunk)
            sample_rate = 24000  # Assume Kokoro sample rate

    # Save output
    if output_audio_chunks:
        output_audio = np.concatenate(output_audio_chunks)
        save_wav(output_audio, sample_rate, output_path)
        print(f"\nSaved output: {output_path}")
        print(f"  Duration: {len(output_audio) / sample_rate:.2f}s")

    # Print stats
    stats = pipeline.stats
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"  Total turns: {stats.total_turns}")
    print(f"  User speech duration: {stats.total_user_speech_duration:.2f}s")
    print(f"  Assistant speech duration: {stats.total_assistant_speech_duration:.2f}s")


def demo_transcribe_only(
    audio_path: str,
    whisper_path: str,
) -> None:
    """Demo: Transcribe only (no LLM/TTS)."""
    print("=" * 60)
    print("Whisper Transcription Demo")
    print("=" * 60)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    print(f"  Duration: {len(audio) / 16000:.2f}s")

    # Load Whisper
    print(f"\nLoading Whisper from {whisper_path}...")
    from pygpukit.asr import WhisperModel

    whisper = WhisperModel.from_pretrained(whisper_path)

    # Transcribe
    print("\nTranscribing...")
    start = time.perf_counter()
    result = whisper.transcribe(audio)
    elapsed = time.perf_counter() - start

    print(f"\nTranscription ({elapsed:.2f}s):")
    print("-" * 40)
    print(result.text)
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Pipeline Demo")
    parser.add_argument("--audio", type=str, help="Path to input audio file")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument(
        "--whisper-path",
        type=str,
        default="kotoba-tech/kotoba-whisper-v2.0",
        help="Path to Whisper model",
    )
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
        "--output",
        type=str,
        default="voice_output.wav",
        help="Output audio file path",
    )
    parser.add_argument("--voice", type=str, default=None, help="TTS voice")
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only transcribe (no LLM/TTS)",
    )
    args = parser.parse_args()

    if args.transcribe_only and args.audio:
        demo_transcribe_only(args.audio, args.whisper_path)
    elif args.audio:
        demo_with_audio_file(
            audio_path=args.audio,
            whisper_path=args.whisper_path,
            llm_path=args.llm_path,
            tts_path=args.tts_path,
            output_path=args.output,
            voice=args.voice,
        )
    elif args.mic:
        print("Microphone input not yet implemented.")
        print("Please provide an audio file with --audio")
    else:
        parser.print_help()
        print("\nExample:")
        print("  python examples/demo_voice_pipeline.py --audio input.wav")
        print("  python examples/demo_voice_pipeline.py --audio input.wav --transcribe-only")


if __name__ == "__main__":
    main()
