"""LLM to TTS Pipeline.

Connects LLM text generation directly to TTS synthesis with minimal latency.
Supports streaming output with sentence boundary detection.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray

if TYPE_CHECKING:
    from pygpukit.llm.models.causal import CausalTransformerModel
    from pygpukit.tts.kokoro.model import KokoroModel


@dataclass
class TTSChunk:
    """A chunk of synthesized audio."""

    audio: np.ndarray  # Audio samples (float32, mono)
    sample_rate: int  # Sample rate in Hz
    text: str  # Source text for this chunk
    start_time: float  # Wall clock time when synthesis started
    end_time: float  # Wall clock time when synthesis completed
    is_final: bool = False  # True if this is the last chunk

    @property
    def duration_ms(self) -> float:
        """Audio duration in milliseconds."""
        return len(self.audio) / self.sample_rate * 1000

    @property
    def latency_ms(self) -> float:
        """Synthesis latency in milliseconds."""
        return (self.end_time - self.start_time) * 1000


class StreamingTTSCallback:
    """Callback for streaming TTS output.

    Override methods to handle events. All methods have default no-op implementations.
    """

    def on_audio_chunk(self, chunk: TTSChunk) -> None:
        """Called when a new audio chunk is available.

        Args:
            chunk: The synthesized audio chunk.
        """
        pass

    def on_complete(self) -> None:
        """Called when all audio has been generated."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs.

        Args:
            error: The exception that occurred.
        """
        pass


@dataclass
class SentenceBuffer:
    """Buffer for accumulating text and detecting sentence boundaries."""

    text: str = ""
    _sentence_end_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r"[.!?]+[\s\n]+|[.!?]+$")
    )

    def append(self, text: str) -> None:
        """Append text to the buffer."""
        self.text += text

    def extract_sentences(self) -> list[str]:
        """Extract complete sentences from the buffer.

        Returns:
            List of complete sentences. Incomplete sentence remains in buffer.
        """
        sentences = []

        while True:
            match = self._sentence_end_pattern.search(self.text)
            if match is None:
                break

            # Extract the sentence including the punctuation
            end_pos = match.end()
            sentence = self.text[:end_pos].strip()
            self.text = self.text[end_pos:].lstrip()

            if sentence:
                sentences.append(sentence)

        return sentences

    def flush(self) -> str | None:
        """Flush any remaining text in the buffer.

        Returns:
            Remaining text, or None if buffer is empty.
        """
        if self.text.strip():
            result = self.text.strip()
            self.text = ""
            return result
        return None

    def clear(self) -> None:
        """Clear the buffer."""
        self.text = ""


@dataclass
class PipelineStats:
    """Statistics for the LLM-TTS pipeline."""

    total_tokens: int = 0
    total_sentences: int = 0
    total_audio_duration_ms: float = 0.0
    total_synthesis_time_ms: float = 0.0
    first_audio_latency_ms: float | None = None
    token_to_audio_latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_synthesis_time_per_sentence_ms(self) -> float:
        """Average synthesis time per sentence."""
        if self.total_sentences == 0:
            return 0.0
        return self.total_synthesis_time_ms / self.total_sentences

    @property
    def realtime_factor(self) -> float:
        """Realtime factor (synthesis time / audio duration)."""
        if self.total_audio_duration_ms == 0:
            return 0.0
        return self.total_synthesis_time_ms / self.total_audio_duration_ms


class LLMToTTSPipeline:
    """Pipeline connecting LLM output to TTS synthesis.

    This pipeline:
    1. Takes token IDs from an LLM
    2. Decodes them to text
    3. Detects sentence boundaries for streaming
    4. Synthesizes speech with minimal latency

    Example:
        ```python
        from pygpukit.llm import QwenModel
        from pygpukit.tts.kokoro import KokoroModel
        from pygpukit.pipeline import LLMToTTSPipeline

        llm = QwenModel.from_safetensors("path/to/model")
        tts = KokoroModel.from_pretrained("path/to/kokoro")

        pipeline = LLMToTTSPipeline(llm, tts)

        # Stream audio while generating
        for chunk in pipeline.generate_speech("What is the meaning of life?"):
            play_audio(chunk.audio, chunk.sample_rate)
        ```
    """

    def __init__(
        self,
        llm_model: CausalTransformerModel,
        tts_model: KokoroModel,
        llm_tokenizer: object | None = None,
        voice: str | None = None,
        speed: float = 1.0,
    ):
        """Initialize the LLM-TTS pipeline.

        Args:
            llm_model: The LLM model for text generation.
            tts_model: The TTS model for speech synthesis.
            llm_tokenizer: Tokenizer for the LLM (if None, uses model's tokenizer).
            voice: Voice to use for TTS (None for default).
            speed: Speech speed multiplier.
        """
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.llm_tokenizer = llm_tokenizer or getattr(llm_model, "tokenizer", None)
        self.voice = voice
        self.speed = speed

        self._sentence_buffer = SentenceBuffer()
        self._stats = PipelineStats()
        self._generation_start_time: float | None = None

    @property
    def stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._stats = PipelineStats()

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text.
        """
        if self.llm_tokenizer is None:
            raise ValueError("No tokenizer available for decoding")

        return self.llm_tokenizer.decode(token_ids)

    def _synthesize_text(self, text: str) -> TTSChunk:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.

        Returns:
            Audio chunk with synthesis results.
        """
        start_time = time.perf_counter()

        result = self.tts_model.synthesize(
            text=text,
            voice=self.voice,
            speed=self.speed,
        )

        end_time = time.perf_counter()

        # Get audio from result (AudioBuffer or GPUArray)
        if hasattr(result.audio, "data"):
            # AudioBuffer: data is GPUArray
            audio = result.audio.data.to_numpy()
            sample_rate = result.audio.sample_rate
        elif isinstance(result.audio, GPUArray):
            audio = result.audio.to_numpy()
            sample_rate = 24000  # Kokoro default
        else:
            audio = np.asarray(result.audio, dtype=np.float32)
            sample_rate = 24000

        return TTSChunk(
            audio=audio,
            sample_rate=sample_rate,
            text=text,
            start_time=start_time,
            end_time=end_time,
        )

    def generate_speech(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stream_sentences: bool = True,
    ) -> Iterator[TTSChunk]:
        """Generate speech from a prompt.

        This method streams audio chunks as sentences are completed.

        Args:
            prompt: Input prompt for the LLM.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            stream_sentences: If True, yield audio after each sentence.
                If False, yield all audio at the end.

        Yields:
            TTSChunk objects containing synthesized audio.
        """
        self._generation_start_time = time.perf_counter()
        self._sentence_buffer.clear()
        self.reset_stats()

        if self.llm_tokenizer is None:
            raise ValueError("No tokenizer available")

        # Encode prompt
        input_ids = self.llm_tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            pass
        elif hasattr(input_ids, "ids"):
            input_ids = input_ids.ids
        else:
            input_ids = list(input_ids)

        # Generate tokens
        generated_tokens: list[int] = []
        prev_text = ""

        # Use streaming generation if available
        if hasattr(self.llm_model, "generate_streaming"):
            token_iterator = self.llm_model.generate_streaming(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            # Fall back to non-streaming generation
            all_tokens = self.llm_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            # Extract only new tokens
            new_tokens = all_tokens[len(input_ids) :]
            token_iterator = iter(new_tokens)

        for token_id in token_iterator:
            generated_tokens.append(token_id)
            self._stats.total_tokens += 1

            # Decode current text
            current_text = self._decode_tokens(generated_tokens)

            # Find new text
            if len(current_text) > len(prev_text):
                new_text = current_text[len(prev_text) :]
                self._sentence_buffer.append(new_text)
                prev_text = current_text

                if stream_sentences:
                    # Check for complete sentences
                    sentences = self._sentence_buffer.extract_sentences()
                    for sentence in sentences:
                        chunk = self._synthesize_text(sentence)
                        self._update_stats(chunk)
                        yield chunk

        # Flush remaining text
        remaining = self._sentence_buffer.flush()
        if remaining:
            chunk = self._synthesize_text(remaining)
            chunk.is_final = True
            self._update_stats(chunk)
            yield chunk

    def _update_stats(self, chunk: TTSChunk) -> None:
        """Update statistics with a new chunk."""
        self._stats.total_sentences += 1
        self._stats.total_audio_duration_ms += chunk.duration_ms
        self._stats.total_synthesis_time_ms += chunk.latency_ms

        if self._stats.first_audio_latency_ms is None and self._generation_start_time:
            self._stats.first_audio_latency_ms = (
                chunk.end_time - self._generation_start_time
            ) * 1000

        self._stats.token_to_audio_latencies_ms.append(chunk.latency_ms)

    def synthesize_text(self, text: str) -> TTSChunk:
        """Directly synthesize text without LLM generation.

        Useful for testing TTS in isolation.

        Args:
            text: Text to synthesize.

        Returns:
            Audio chunk with synthesis results.
        """
        return self._synthesize_text(text)

    def generate_speech_with_callback(
        self,
        prompt: str,
        callback: StreamingTTSCallback,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> PipelineStats:
        """Generate speech with a callback for streaming output.

        Args:
            prompt: Input prompt for the LLM.
            callback: Callback for receiving audio chunks.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.

        Returns:
            Pipeline statistics.
        """
        try:
            for chunk in self.generate_speech(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ):
                callback.on_audio_chunk(chunk)

            callback.on_complete()
        except Exception as e:
            callback.on_error(e)
            raise

        return self._stats


# Convenience function for simple usage
def speak(
    llm_model: CausalTransformerModel,
    tts_model: KokoroModel,
    prompt: str,
    voice: str | None = None,
    max_new_tokens: int = 256,
) -> np.ndarray:
    """Generate speech from a prompt (non-streaming).

    This is a convenience function that concatenates all audio chunks.

    Args:
        llm_model: The LLM model.
        tts_model: The TTS model.
        prompt: Input prompt.
        voice: Voice to use (None for default).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Concatenated audio as numpy array.
    """
    pipeline = LLMToTTSPipeline(llm_model, tts_model, voice=voice)

    audio_chunks = []
    for chunk in pipeline.generate_speech(prompt, max_new_tokens=max_new_tokens):
        audio_chunks.append(chunk.audio)

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(audio_chunks)
