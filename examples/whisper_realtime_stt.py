#!/usr/bin/env python3
"""Real-time Speech-to-Text Demo using Whisper.

This demo shows how to use PyGPUkit's Whisper implementation for
real-time speech recognition from any PCM audio source.

Supported input sources:
- Microphone (requires sounddevice)
- PCM file (raw audio)
- WAV file

Usage:
    # From microphone (default)
    python whisper_realtime_stt.py

    # From WAV file
    python whisper_realtime_stt.py --input audio.wav

    # From raw PCM file (16kHz, mono, float32)
    python whisper_realtime_stt.py --input audio.pcm --pcm

    # Specify model
    python whisper_realtime_stt.py --model kotoba-tech/kotoba-whisper-v2.0

    # Adjust chunk size (seconds)
    python whisper_realtime_stt.py --chunk-size 5.0

Requirements:
    pip install sounddevice soundfile numpy
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Audio constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1  # Mono


@dataclass
class TranscriptionEvent:
    """Event for transcription results."""

    text: str
    start_time: float
    end_time: float
    is_partial: bool = False


class AudioBuffer:
    """Thread-safe audio buffer for real-time processing."""

    def __init__(self, chunk_duration: float = 5.0, overlap: float = 0.5):
        """Initialize audio buffer.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
        """
        self.chunk_samples = int(chunk_duration * SAMPLE_RATE)
        self.overlap_samples = int(overlap * SAMPLE_RATE)
        self.stride_samples = self.chunk_samples - self.overlap_samples

        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._total_samples = 0

    def write(self, audio: np.ndarray) -> None:
        """Write audio samples to buffer."""
        with self._lock:
            self._buffer.extend(audio.flatten())
            self._total_samples += len(audio.flatten())

    def read_chunk(self) -> tuple[np.ndarray, float] | None:
        """Read a chunk of audio if available.

        Returns:
            Tuple of (audio_chunk, start_time) or None if not enough data
        """
        with self._lock:
            if len(self._buffer) < self.chunk_samples:
                return None

            # Extract chunk
            chunk = np.array([self._buffer[i] for i in range(self.chunk_samples)])

            # Calculate start time
            consumed = self._total_samples - len(self._buffer)
            start_time = consumed / SAMPLE_RATE

            # Remove processed samples (keeping overlap)
            for _ in range(self.stride_samples):
                if self._buffer:
                    self._buffer.popleft()

            return chunk.astype(np.float32), start_time

    @property
    def buffered_duration(self) -> float:
        """Get buffered duration in seconds."""
        with self._lock:
            return len(self._buffer) / SAMPLE_RATE


class RealtimeSTT:
    """Real-time Speech-to-Text engine using Whisper."""

    def __init__(
        self,
        model_id: str = "kotoba-tech/kotoba-whisper-v2.0",
        chunk_duration: float = 5.0,
        language: str | None = None,
        on_transcription: Callable[[TranscriptionEvent], None] | None = None,
    ):
        """Initialize real-time STT.

        Args:
            model_id: Whisper model ID or path
            chunk_duration: Duration of each chunk in seconds
            language: Language code (e.g., "ja", "en")
            on_transcription: Callback for transcription events
        """
        self.model_id = model_id
        self.chunk_duration = chunk_duration
        self.language = language
        self.on_transcription = on_transcription

        self._model = None
        self._buffer = AudioBuffer(chunk_duration=chunk_duration)
        self._running = False
        self._thread: threading.Thread | None = None

    def load_model(self) -> None:
        """Load Whisper model."""
        print(f"Loading model: {self.model_id}...")
        from pygpukit.asr import WhisperModel

        self._model = WhisperModel.from_pretrained(self.model_id)
        print("Model loaded successfully!")

    def start(self) -> None:
        """Start the transcription thread."""
        if self._model is None:
            self.load_model()

        self._running = True
        self._thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the transcription thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def feed_audio(self, audio: np.ndarray) -> None:
        """Feed audio samples to the STT engine.

        Args:
            audio: Audio samples (float32, -1.0 to 1.0)
        """
        self._buffer.write(audio)

    def _transcription_loop(self) -> None:
        """Background loop for processing audio chunks."""
        while self._running:
            chunk_data = self._buffer.read_chunk()

            if chunk_data is None:
                time.sleep(0.1)
                continue

            audio_chunk, start_time = chunk_data

            try:
                # Transcribe chunk
                result = self._model.transcribe(
                    audio_chunk,
                    language=self.language,
                    temperature=0.0,
                )

                # Create event
                event = TranscriptionEvent(
                    text=result.text.strip(),
                    start_time=start_time,
                    end_time=start_time + len(audio_chunk) / SAMPLE_RATE,
                )

                # Callback
                if self.on_transcription and event.text:
                    self.on_transcription(event)

            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)


def read_pcm_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read raw PCM file.

    Args:
        path: Path to PCM file
        sample_rate: Expected sample rate

    Returns:
        Audio array (float32)
    """
    # Try to read as float32 first, then int16
    try:
        audio = np.fromfile(path, dtype=np.float32)
        if np.abs(audio).max() > 10:  # Probably int16
            raise ValueError("Not float32")
    except (ValueError, Exception):
        audio = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0

    return audio


def read_wav_file(path: str) -> tuple[np.ndarray, int]:
    """Read WAV file.

    Args:
        path: Path to WAV file

    Returns:
        Tuple of (audio, sample_rate)
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr
    except ImportError as err:
        raise ImportError("soundfile is required: pip install soundfile") from err


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    try:
        import resampy

        return resampy.resample(audio, orig_sr, target_sr)
    except ImportError:
        # Simple linear interpolation fallback
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class MicrophoneStream:
    """Microphone audio stream."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = 1024,
        device: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._stream = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start microphone stream.

        Args:
            callback: Function to call with audio chunks
        """
        try:
            import sounddevice as sd
        except ImportError as err:
            raise ImportError(
                "sounddevice is required for microphone: pip install sounddevice"
            ) from err

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            callback(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=np.float32,
            blocksize=self.chunk_size,
            device=self.device,
            callback=audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop microphone stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()


def print_transcription(event: TranscriptionEvent) -> None:
    """Print transcription event to console."""
    timestamp = f"[{event.start_time:6.1f}s - {event.end_time:6.1f}s]"
    print(f"{timestamp} {event.text}")


def list_audio_devices() -> list[dict]:
    """List available audio input devices.

    Returns:
        List of device info dicts with 'index', 'name', 'channels', 'sample_rate'
    """
    try:
        import sounddevice as sd
    except ImportError as err:
        raise ImportError("sounddevice is required: pip install sounddevice") from err

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:  # Input device
            devices.append(
                {
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                }
            )
    return devices


def print_audio_devices() -> None:
    """Print available audio input devices."""
    devices = list_audio_devices()
    print("\nAvailable audio input devices:")
    print("-" * 60)
    for dev in devices:
        print(f"  [{dev['index']:2d}] {dev['name']}")
        print(f"       Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']:.0f} Hz")
    print("-" * 60)


def select_audio_device() -> int | None:
    """Interactively select an audio input device.

    Returns:
        Selected device index or None for default
    """
    devices = list_audio_devices()

    if not devices:
        print("No audio input devices found!")
        return None

    if len(devices) == 1:
        print(f"Using audio device: {devices[0]['name']}")
        return devices[0]["index"]

    print("\nAvailable audio input devices:")
    print("-" * 60)
    for dev in devices:
        print(f"  [{dev['index']:2d}] {dev['name']}")
    print("-" * 60)

    while True:
        try:
            choice = input(
                f"Select device [0-{max(d['index'] for d in devices)}, Enter=default]: "
            ).strip()
            if choice == "":
                return None
            idx = int(choice)
            if any(d["index"] == idx for d in devices):
                return idx
            print(f"Invalid device index: {idx}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(0)


def demo_microphone(args: argparse.Namespace) -> None:
    """Run demo with microphone input."""
    # Select device if not specified
    device = args.device
    if device is None and args.select_device:
        device = select_audio_device()

    print("=" * 60)
    print("Real-time Speech-to-Text Demo (Microphone)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto'}")
    print(f"Chunk size: {args.chunk_size}s")
    if device is not None:
        print(f"Device: {device}")
    print("-" * 60)
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("-" * 60)

    # Initialize STT
    stt = RealtimeSTT(
        model_id=args.model,
        chunk_duration=args.chunk_size,
        language=args.language,
        on_transcription=print_transcription,
    )
    stt.load_model()

    # Start microphone
    mic = MicrophoneStream(device=device)

    try:
        stt.start()
        mic.start(stt.feed_audio)

        # Keep running until Ctrl+C
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        stt.stop()


def demo_file(args: argparse.Namespace) -> None:
    """Run demo with file input."""
    print("=" * 60)
    print("Real-time Speech-to-Text Demo (File)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Language: {args.language or 'auto'}")
    print(f"Chunk size: {args.chunk_size}s")
    print("-" * 60)

    # Load audio
    if args.pcm:
        print("Loading PCM file...")
        audio = read_pcm_file(args.input)
        sr = args.sample_rate
    else:
        print("Loading audio file...")
        audio, sr = read_wav_file(args.input)

    # Resample if needed
    if sr != SAMPLE_RATE:
        print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
        audio = resample_audio(audio, sr, SAMPLE_RATE)

    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.1f}s")
    print("-" * 60)

    # Initialize STT
    stt = RealtimeSTT(
        model_id=args.model,
        chunk_duration=args.chunk_size,
        language=args.language,
        on_transcription=print_transcription,
    )
    stt.load_model()

    # Process audio in real-time simulation
    stt.start()

    # Feed audio in chunks (simulating real-time)
    chunk_samples = int(0.1 * SAMPLE_RATE)  # 100ms chunks
    try:
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            stt.feed_audio(chunk)

            # Simulate real-time by sleeping
            if not args.fast:
                time.sleep(len(chunk) / SAMPLE_RATE)

        # Wait for processing to complete
        print("\nProcessing remaining audio...")
        time.sleep(args.chunk_size + 1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stt.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Speech-to-Text Demo using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available microphones
    python whisper_realtime_stt.py --list-devices

    # Select microphone interactively
    python whisper_realtime_stt.py --select-device

    # Use specific microphone by index
    python whisper_realtime_stt.py --device 2

    # WAV file input
    python whisper_realtime_stt.py --input recording.wav

    # Raw PCM file (16kHz, mono, float32)
    python whisper_realtime_stt.py --input audio.pcm --pcm

    # Japanese model with 3-second chunks
    python whisper_realtime_stt.py --model kotoba-tech/kotoba-whisper-v2.0 \\
                                   --language ja --chunk-size 3.0
""",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input audio file (WAV or PCM). If not specified, uses microphone.",
    )
    parser.add_argument(
        "--pcm",
        action="store_true",
        help="Treat input as raw PCM file",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Sample rate for PCM input (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="kotoba-tech/kotoba-whisper-v2.0",
        help="Whisper model ID or path",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language code (e.g., 'ja', 'en'). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=5.0,
        help="Chunk duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        default=None,
        help="Audio input device index (for microphone)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--select-device",
        "-s",
        action="store_true",
        help="Interactively select audio input device at startup",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Process file as fast as possible (no real-time simulation)",
    )

    args = parser.parse_args()

    # List devices and exit
    if args.list_devices:
        print_audio_devices()
        return

    if args.input:
        demo_file(args)
    else:
        demo_microphone(args)


if __name__ == "__main__":
    main()
