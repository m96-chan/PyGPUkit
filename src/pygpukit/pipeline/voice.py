"""Full Voice Pipeline: Audio → Whisper → VAD → LLM → TTS.

Real-time voice conversation pipeline with streaming support.

Architecture:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Audio In   │───▶│     VAD     │───▶│   Whisper   │
│  (Buffer)   │    │  (Detect)   │    │   (ASR)     │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │ text
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Audio Out  │◀───│     TTS     │◀───│     LLM     │
│  (Stream)   │    │  (Synth)    │    │ (Generate)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

Key Design Decisions:
1. VAD runs continuously on audio input to detect speech endpoints
2. Whisper processes audio chunks when VAD detects end-of-utterance
3. LLM receives transcribed text and generates response (streaming)
4. TTS converts LLM output to speech (streaming, sentence-by-sentence)
5. Interruption: User speech cancels current TTS playback
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pygpukit.asr.whisper import WhisperModel
    from pygpukit.llm.models.causal import CausalTransformerModel
    from pygpukit.tts.kokoro.model import KokoroModel


# =============================================================================
# Voice Activity Detection (VAD)
# =============================================================================


class VADState(Enum):
    """VAD state machine states."""

    SILENCE = auto()  # No speech detected
    SPEECH = auto()  # Speech in progress
    TRAILING = auto()  # Post-speech silence (waiting for endpoint)


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""

    # Energy-based detection thresholds
    energy_threshold: float = 0.01  # RMS energy threshold for speech
    silence_threshold: float = 0.005  # RMS energy threshold for silence

    # Timing parameters (in seconds)
    min_speech_duration: float = 0.1  # Minimum speech duration to trigger
    min_silence_duration: float = 0.5  # Silence duration to detect endpoint
    max_speech_duration: float = 30.0  # Maximum speech duration (force endpoint)

    # Frame parameters
    frame_duration_ms: int = 30  # Frame size for VAD processing (ms)
    sample_rate: int = 16000  # Expected sample rate


@dataclass
class VADEvent:
    """Event from VAD processing."""

    event_type: str  # "speech_start", "speech_end", "audio_chunk"
    timestamp: float  # Wall clock time
    audio: np.ndarray | None = None  # Audio data (for speech_end event)
    duration: float = 0.0  # Duration of speech (for speech_end)


class VoiceActivityDetector:
    """Energy-based Voice Activity Detection.

    Detects speech segments in audio stream and triggers events:
    - speech_start: User started speaking
    - speech_end: User stopped speaking (with accumulated audio)
    """

    def __init__(self, config: VADConfig | None = None):
        """Initialize VAD.

        Args:
            config: VAD configuration. If None, uses defaults.
        """
        self.config = config or VADConfig()
        self._state = VADState.SILENCE
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_time: float | None = None
        self._silence_start_time: float | None = None
        self._frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)

    def reset(self) -> None:
        """Reset VAD state."""
        self._state = VADState.SILENCE
        self._speech_buffer = []
        self._speech_start_time = None
        self._silence_start_time = None

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio frame."""
        return float(np.sqrt(np.mean(audio**2)))

    def process_audio(self, audio: np.ndarray) -> list[VADEvent]:
        """Process audio chunk and return VAD events.

        Args:
            audio: Audio samples (float32, mono, at config.sample_rate).

        Returns:
            List of VAD events (may be empty).
        """
        events: list[VADEvent] = []
        current_time = time.time()

        # Process in frames
        for i in range(0, len(audio), self._frame_size):
            frame = audio[i : i + self._frame_size]
            if len(frame) < self._frame_size // 2:
                continue

            energy = self._compute_energy(frame)
            is_speech = energy > self.config.energy_threshold
            is_silence = energy < self.config.silence_threshold

            if self._state == VADState.SILENCE:
                if is_speech:
                    self._state = VADState.SPEECH
                    self._speech_start_time = current_time
                    self._speech_buffer = [frame]
                    events.append(VADEvent(event_type="speech_start", timestamp=current_time))

            elif self._state == VADState.SPEECH:
                self._speech_buffer.append(frame)

                # Check for max speech duration
                if self._speech_start_time:
                    speech_duration = current_time - self._speech_start_time
                    if speech_duration >= self.config.max_speech_duration:
                        # Force endpoint
                        events.append(self._create_speech_end_event(current_time))
                        self.reset()
                        continue

                if is_silence:
                    self._state = VADState.TRAILING
                    self._silence_start_time = current_time

            elif self._state == VADState.TRAILING:
                self._speech_buffer.append(frame)

                if is_speech:
                    # Resume speech
                    self._state = VADState.SPEECH
                    self._silence_start_time = None
                elif is_silence and self._silence_start_time:
                    # Check if silence duration exceeded
                    silence_duration = current_time - self._silence_start_time
                    if silence_duration >= self.config.min_silence_duration:
                        # Check minimum speech duration
                        if self._speech_start_time:
                            speech_duration = current_time - self._speech_start_time
                            if speech_duration >= self.config.min_speech_duration:
                                events.append(self._create_speech_end_event(current_time))
                        self.reset()

        return events

    def _create_speech_end_event(self, timestamp: float) -> VADEvent:
        """Create speech_end event with accumulated audio."""
        audio = (
            np.concatenate(self._speech_buffer)
            if self._speech_buffer
            else np.array([], dtype=np.float32)
        )
        duration = len(audio) / self.config.sample_rate
        return VADEvent(
            event_type="speech_end",
            timestamp=timestamp,
            audio=audio,
            duration=duration,
        )

    @property
    def is_speaking(self) -> bool:
        """Check if user is currently speaking."""
        return self._state in (VADState.SPEECH, VADState.TRAILING)


# =============================================================================
# Audio Buffer
# =============================================================================


@dataclass
class AudioBuffer:
    """Circular buffer for audio streaming."""

    sample_rate: int = 16000
    max_duration: float = 60.0  # Maximum buffer duration in seconds
    _buffer: deque = field(default_factory=deque)

    def __post_init__(self):
        self._max_samples = int(self.sample_rate * self.max_duration)

    def append(self, audio: np.ndarray) -> None:
        """Append audio samples to buffer."""
        for sample in audio:
            self._buffer.append(sample)
            if len(self._buffer) > self._max_samples:
                self._buffer.popleft()

    def get_audio(self, duration: float | None = None) -> np.ndarray:
        """Get audio from buffer.

        Args:
            duration: Duration in seconds. If None, returns all audio.

        Returns:
            Audio samples as numpy array.
        """
        if duration is None:
            return np.array(list(self._buffer), dtype=np.float32)

        num_samples = int(duration * self.sample_rate)
        samples = list(self._buffer)[-num_samples:]
        return np.array(samples, dtype=np.float32)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# =============================================================================
# Pipeline State
# =============================================================================


class PipelineState(Enum):
    """Voice pipeline state."""

    IDLE = auto()  # Waiting for user input
    LISTENING = auto()  # Receiving audio, VAD active
    TRANSCRIBING = auto()  # Whisper processing
    GENERATING = auto()  # LLM generating response
    SPEAKING = auto()  # TTS output in progress


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    user_audio: np.ndarray | None = None  # User's speech audio
    user_text: str = ""  # Transcribed user speech
    assistant_text: str = ""  # LLM response
    assistant_audio: np.ndarray | None = None  # TTS audio
    start_time: float = 0.0
    end_time: float = 0.0
    was_interrupted: bool = False


@dataclass
class PipelineStats:
    """Statistics for the voice pipeline."""

    total_turns: int = 0
    total_user_speech_duration: float = 0.0
    total_assistant_speech_duration: float = 0.0
    avg_transcription_latency_ms: float = 0.0
    avg_llm_first_token_latency_ms: float = 0.0
    avg_tts_first_audio_latency_ms: float = 0.0
    interruptions: int = 0


# =============================================================================
# Callbacks
# =============================================================================


class VoicePipelineCallback:
    """Callback interface for voice pipeline events.

    Override methods to handle events. All methods have default no-op implementations.
    """

    def on_listening_start(self) -> None:
        """Called when pipeline starts listening for speech."""
        pass

    def on_speech_detected(self) -> None:
        """Called when user starts speaking."""
        pass

    def on_transcription_start(self) -> None:
        """Called when transcription begins."""
        pass

    def on_transcription_complete(self, text: str) -> None:
        """Called when transcription is complete.

        Args:
            text: Transcribed text.
        """
        pass

    def on_generation_start(self) -> None:
        """Called when LLM generation starts."""
        pass

    def on_generation_token(self, token: str) -> None:
        """Called for each generated token.

        Args:
            token: Generated token text.
        """
        pass

    def on_generation_complete(self, text: str) -> None:
        """Called when LLM generation is complete.

        Args:
            text: Complete generated text.
        """
        pass

    def on_tts_start(self) -> None:
        """Called when TTS synthesis starts."""
        pass

    def on_audio_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        """Called when an audio chunk is ready for playback.

        Args:
            audio: Audio samples (float32).
            sample_rate: Sample rate in Hz.
        """
        pass

    def on_tts_complete(self) -> None:
        """Called when TTS synthesis is complete."""
        pass

    def on_interruption(self) -> None:
        """Called when user interrupts the assistant."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs.

        Args:
            error: The exception.
        """
        pass


# =============================================================================
# Voice Pipeline
# =============================================================================


class VoicePipeline:
    """Full voice conversation pipeline.

    Connects audio input through ASR, LLM, and TTS with streaming support.

    Example:
        ```python
        from pygpukit.asr import WhisperModel
        from pygpukit.llm import QwenModel
        from pygpukit.tts.kokoro import KokoroModel
        from pygpukit.pipeline import VoicePipeline

        whisper = WhisperModel.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")
        llm = QwenModel.from_safetensors("path/to/model")
        tts = KokoroModel.from_pretrained("path/to/kokoro")

        pipeline = VoicePipeline(whisper, llm, tts)

        # Process audio in real-time
        for audio_chunk in audio_stream:
            for output_audio in pipeline.process_audio(audio_chunk):
                play_audio(output_audio)
        ```
    """

    def __init__(
        self,
        whisper_model: WhisperModel,
        llm_model: CausalTransformerModel,
        tts_model: KokoroModel,
        llm_tokenizer: object | None = None,
        vad_config: VADConfig | None = None,
        system_prompt: str | None = None,
        voice: str | None = None,
        callback: VoicePipelineCallback | None = None,
    ):
        """Initialize voice pipeline.

        Args:
            whisper_model: Whisper ASR model.
            llm_model: LLM for response generation.
            tts_model: TTS model for speech synthesis.
            llm_tokenizer: Tokenizer for LLM (if None, uses model's tokenizer).
            vad_config: VAD configuration.
            system_prompt: System prompt for LLM.
            voice: TTS voice to use.
            callback: Event callback.
        """
        self.whisper = whisper_model
        self.llm = llm_model
        self.tts = tts_model
        self.tokenizer = llm_tokenizer or getattr(llm_model, "tokenizer", None)
        self.system_prompt = system_prompt or "You are a helpful voice assistant."
        self.voice = voice
        self.callback = callback or VoicePipelineCallback()

        self._vad = VoiceActivityDetector(vad_config)
        self._state = PipelineState.IDLE
        self._conversation_history: list[ConversationTurn] = []
        self._stats = PipelineStats()
        self._current_turn: ConversationTurn | None = None

        # Interruption handling
        self._interrupt_flag = threading.Event()
        self._speaking_lock = threading.Lock()

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats

    @property
    def conversation_history(self) -> list[ConversationTurn]:
        """Get conversation history."""
        return self._conversation_history

    def start_listening(self) -> None:
        """Start listening for audio input."""
        self._state = PipelineState.LISTENING
        self._vad.reset()
        self.callback.on_listening_start()

    def stop_listening(self) -> None:
        """Stop listening for audio input."""
        self._state = PipelineState.IDLE

    def interrupt(self) -> None:
        """Interrupt current TTS playback."""
        if self._state == PipelineState.SPEAKING:
            self._interrupt_flag.set()
            self._stats.interruptions += 1
            self.callback.on_interruption()
            if self._current_turn:
                self._current_turn.was_interrupted = True

    def process_audio(self, audio: np.ndarray) -> Iterator[np.ndarray]:
        """Process audio input and yield output audio.

        This is the main entry point for real-time audio processing.

        Args:
            audio: Audio samples (float32, mono, 16kHz).

        Yields:
            Output audio chunks for playback.
        """
        if self._state == PipelineState.IDLE:
            return

        # Check for interruption during TTS
        if self._state == PipelineState.SPEAKING and self._vad.is_speaking:
            self.interrupt()

        # Process through VAD
        events = self._vad.process_audio(audio)

        for event in events:
            if event.event_type == "speech_start":
                self.callback.on_speech_detected()

                # Interrupt if speaking
                if self._state == PipelineState.SPEAKING:
                    self.interrupt()

            elif event.event_type == "speech_end" and event.audio is not None:
                # Process complete utterance
                yield from self._process_utterance(event.audio)

    def _process_utterance(self, audio: np.ndarray) -> Iterator[np.ndarray]:
        """Process a complete user utterance through the pipeline.

        Args:
            audio: User's speech audio.

        Yields:
            TTS audio chunks.
        """
        turn = ConversationTurn(user_audio=audio, start_time=time.time())
        self._current_turn = turn

        try:
            # Step 1: Transcribe with Whisper
            self._state = PipelineState.TRANSCRIBING
            self.callback.on_transcription_start()

            transcribe_start = time.perf_counter()
            result = self.whisper.transcribe(audio)
            _transcribe_time = (time.perf_counter() - transcribe_start) * 1000  # noqa: F841

            turn.user_text = result.text.strip()
            self.callback.on_transcription_complete(turn.user_text)

            if not turn.user_text:
                return

            # Step 2: Generate LLM response
            self._state = PipelineState.GENERATING
            self.callback.on_generation_start()

            # Build prompt with conversation history
            prompt = self._build_prompt(turn.user_text)

            # Generate response (streaming)
            generate_start = time.perf_counter()
            first_token_time: float | None = None

            response_tokens: list[str] = []
            sentence_buffer = ""

            for token_text in self._generate_streaming(prompt):
                if first_token_time is None:
                    first_token_time = (time.perf_counter() - generate_start) * 1000

                response_tokens.append(token_text)
                self.callback.on_generation_token(token_text)

                # Buffer for sentence detection
                sentence_buffer += token_text

                # Check for sentence boundary for streaming TTS
                if self._is_sentence_end(sentence_buffer):
                    # Start TTS for this sentence
                    yield from self._synthesize_sentence(sentence_buffer.strip())
                    sentence_buffer = ""

                # Check for interruption
                if self._interrupt_flag.is_set():
                    break

            # Handle remaining text
            if sentence_buffer.strip() and not self._interrupt_flag.is_set():
                yield from self._synthesize_sentence(sentence_buffer.strip())

            turn.assistant_text = "".join(response_tokens)
            self.callback.on_generation_complete(turn.assistant_text)

            # Update stats
            self._stats.total_turns += 1
            self._stats.total_user_speech_duration += len(audio) / 16000

        except Exception as e:
            self.callback.on_error(e)
            raise
        finally:
            turn.end_time = time.time()
            self._conversation_history.append(turn)
            self._current_turn = None
            self._state = PipelineState.LISTENING
            self._interrupt_flag.clear()

    def _build_prompt(self, user_text: str) -> str:
        """Build prompt with system message and conversation history."""
        messages = [f"System: {self.system_prompt}"]

        # Add recent history (last 5 turns)
        for turn in self._conversation_history[-5:]:
            if turn.user_text:
                messages.append(f"User: {turn.user_text}")
            if turn.assistant_text:
                messages.append(f"Assistant: {turn.assistant_text}")

        messages.append(f"User: {user_text}")
        messages.append("Assistant:")

        return "\n".join(messages)

    def _generate_streaming(self, prompt: str) -> Iterator[str]:
        """Generate LLM response with streaming.

        Args:
            prompt: Input prompt.

        Yields:
            Generated token text.
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if hasattr(input_ids, "ids"):
            input_ids = input_ids.ids

        # Check if model has streaming support
        if hasattr(self.llm, "generate_stream"):
            for token_id in self.llm.generate_stream(
                input_ids=list(input_ids),
                max_new_tokens=256,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            ):
                yield self.tokenizer.decode([token_id])
        else:
            # Fall back to non-streaming
            all_tokens = self.llm.generate(
                input_ids=list(input_ids),
                max_new_tokens=256,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )
            new_tokens = all_tokens[len(input_ids) :]

            for token_id in new_tokens:
                yield self.tokenizer.decode([token_id])

                if self._interrupt_flag.is_set():
                    break

    def _is_sentence_end(self, text: str) -> bool:
        """Check if text ends with a sentence boundary."""
        text = text.rstrip()
        return text.endswith((".", "!", "?", "。", "！", "？"))

    def _synthesize_sentence(self, text: str) -> Iterator[np.ndarray]:
        """Synthesize a sentence with TTS.

        Args:
            text: Text to synthesize.

        Yields:
            Audio chunks.
        """
        if not text:
            return

        self._state = PipelineState.SPEAKING
        self.callback.on_tts_start()

        try:
            result = self.tts.synthesize(text=text, voice=self.voice)

            # Get audio
            if hasattr(result.audio, "to_numpy"):
                audio = result.audio.to_numpy()
            else:
                audio = result.audio

            self.callback.on_audio_chunk(audio, result.sample_rate)
            yield audio

            self._stats.total_assistant_speech_duration += len(audio) / result.sample_rate

        finally:
            self.callback.on_tts_complete()

    def reset(self) -> None:
        """Reset pipeline state and conversation history."""
        self._state = PipelineState.IDLE
        self._vad.reset()
        self._conversation_history = []
        self._current_turn = None
        self._interrupt_flag.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_voice_pipeline(
    whisper_path: str,
    llm_path: str,
    tts_path: str,
    system_prompt: str | None = None,
    voice: str | None = None,
) -> VoicePipeline:
    """Create a voice pipeline from model paths.

    Args:
        whisper_path: Path to Whisper model.
        llm_path: Path to LLM model.
        tts_path: Path to TTS model.
        system_prompt: System prompt for LLM.
        voice: TTS voice to use.

    Returns:
        Configured VoicePipeline.
    """
    from pygpukit.asr import WhisperModel
    from pygpukit.llm import QwenModel
    from pygpukit.tts.kokoro import KokoroModel

    whisper = WhisperModel.from_pretrained(whisper_path)
    llm = QwenModel.from_safetensors(llm_path)
    tts = KokoroModel.from_pretrained(tts_path)

    return VoicePipeline(
        whisper_model=whisper,
        llm_model=llm,
        tts_model=tts,
        system_prompt=system_prompt,
        voice=voice,
    )
