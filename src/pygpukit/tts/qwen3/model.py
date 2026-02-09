"""Qwen3-TTS Model for PyGPUkit.

Main TTS model combining:
- Talker Model: Transformer LLM for text-to-codec generation
- Speaker Encoder: ECAPA-TDNN for speaker embedding extraction
- Speech Tokenizer: Audio codec for encode/decode

Supports three generation modes:
- Voice Design: Generate speech with natural language voice descriptions
- Voice Clone: Clone voice from reference audio
- CustomVoice: Use predefined speakers with optional style instructions
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.llm.layers import Norm, TransformerBlock
from pygpukit.llm.layers.rope import (
    precompute_freqs_cis,
)
from pygpukit.llm.sampling import sample_token

from .speaker_encoder import SpeakerEncoder, compute_mel_spectrogram
from .speech_tokenizer import SpeechTokenizer


class TextProjection:
    """2-layer MLP for projecting text embeddings to talker hidden size."""

    def __init__(
        self,
        fc1_weight: GPUArray,
        fc1_bias: GPUArray | None,
        fc2_weight: GPUArray,
        fc2_bias: GPUArray | None,
    ):
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        """Project hidden states."""
        # FC1: [batch, text_hidden] -> [batch, intermediate]
        fc1_np = self.fc1_weight.to_numpy()
        out = hidden @ fc1_np.T
        if self.fc1_bias is not None:
            out = out + self.fc1_bias.to_numpy()

        # GELU activation
        out = out * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (out + 0.044715 * out**3)))

        # FC2: [batch, intermediate] -> [batch, hidden_size]
        fc2_np = self.fc2_weight.to_numpy()
        out = out @ fc2_np.T
        if self.fc2_bias is not None:
            out = out + self.fc2_bias.to_numpy()

        return out


@dataclass
class Qwen3TTSConfig:
    """Configuration for Qwen3-TTS model."""

    # Model size
    model_type: str = "custom_voice"  # "custom_voice", "voice_design", "base"
    model_size: str = "1.7B"  # "0.6B" or "1.7B"

    # Talker model
    vocab_size: int = 152064
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768

    # Multimodal RoPE
    mrope_section: Sequence[int] = field(default_factory=lambda: [16, 24, 24])
    rope_theta: float = 1000000.0

    # Codec
    codec_vocab_size: int = 2048
    num_codebooks: int = 8
    codec_bos_token_id: int = 151936
    codec_eos_token_id: int = 151937
    codec_pad_token_id: int = 151938

    # Speaker encoder
    speaker_encoder_sample_rate: int = 24000
    speaker_embed_dim: int = 192

    # Speech tokenizer
    speech_tokenizer_sample_rate: int = 24000
    speech_tokenizer_frame_rate: int = 12

    # Supported speakers (for CustomVoice mode)
    supported_speakers: Sequence[str] = field(
        default_factory=lambda: [
            "aiden",
            "dylan",
            "eric",
            "ono_anna",
            "ryan",
            "serena",
            "sohee",
            "uncle_fu",
            "vivian",
        ]
    )

    # Supported languages
    supported_languages: Sequence[str] = field(
        default_factory=lambda: ["auto", "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru"]
    )


class VoiceClonePromptItem(NamedTuple):
    """Voice clone prompt data."""

    ref_code: np.ndarray | None  # [num_q, seq_len] reference codec codes
    ref_spk_embedding: np.ndarray  # [embed_dim] speaker embedding
    x_vector_only_mode: bool  # Use only x-vector (no ICL)
    icl_mode: bool  # In-context learning mode
    ref_text: str | None  # Reference text (required for ICL mode)


class GenerationOutput(NamedTuple):
    """Output from TTS generation."""

    audio: list[np.ndarray]  # List of audio waveforms [samples]
    sample_rate: int


# =============================================================================
# Talker Model (Transformer for codec generation)
# =============================================================================


class CodePredictor:
    """Predicts remaining codec codes given the first codebook.

    Small transformer that takes the first codec code and predicts
    the remaining num_codebooks-1 codes sequentially.
    """

    def __init__(
        self,
        config: Qwen3TTSConfig,
        embedding: GPUArray,
        blocks: list[TransformerBlock],
        lm_heads: list[GPUArray],
    ):
        self.config = config
        self.embedding = embedding
        self.blocks = blocks
        self.lm_heads = lm_heads
        self.num_predictions = len(lm_heads)

    def __call__(
        self,
        first_code: np.ndarray,  # [batch, seq_len]
        hidden_state: np.ndarray,  # [batch, seq_len, hidden_size] from talker
    ) -> np.ndarray:
        """Predict remaining codec codes.

        Args:
            first_code: First codebook codes [batch, seq_len]
            hidden_state: Hidden state from talker [batch, seq_len, hidden]

        Returns:
            All codes [batch, num_codebooks, seq_len]
        """
        batch, seq_len = first_code.shape
        codes = [first_code]

        # Embed first code
        embed_np = self.embedding.to_numpy()
        hidden = hidden_state.copy()

        for _i, lm_head in enumerate(self.lm_heads):
            # Add code embedding
            code_embed = embed_np[codes[-1].flatten()].reshape(batch, seq_len, -1)
            hidden = hidden + code_embed

            # Forward through blocks
            hidden_gpu = from_numpy(hidden.astype(np.float32))
            for block in self.blocks:
                hidden_gpu = block(hidden_gpu, position_ids=list(range(seq_len)))
            hidden = hidden_gpu.to_numpy()

            # Predict next code
            lm_head_np = lm_head.to_numpy()
            logits = hidden @ lm_head_np.T
            next_code = np.argmax(logits, axis=-1)
            codes.append(next_code)

        return np.stack(codes, axis=1)


class TalkerModel:
    """Transformer model for text-to-codec generation.

    Takes text tokens and generates codec tokens autoregressively.
    """

    def __init__(
        self,
        config: Qwen3TTSConfig,
        embed_tokens: GPUArray,
        codec_embed: GPUArray,
        blocks: list[TransformerBlock],
        final_norm: Norm,
        codec_head: GPUArray,
        code_predictor: CodePredictor | None = None,
        text_projection: TextProjection | None = None,
    ):
        self.config = config
        self.embed_tokens = embed_tokens
        self.codec_embed = codec_embed
        self.blocks = blocks
        self.final_norm = final_norm
        self.codec_head = codec_head
        self.code_predictor = code_predictor
        self.text_projection = text_projection

        # Precompute RoPE tables
        self._init_rope()

    def _init_rope(self):
        """Initialize RoPE cos/sin tables."""
        head_dim = self.config.head_dim
        max_len = self.config.max_position_embeddings
        theta = self.config.rope_theta

        # Standard 1D RoPE for text
        self.cos_table, self.sin_table = precompute_freqs_cis(
            head_dim=head_dim,
            max_seq_len=max_len,
            theta=theta,
        )

    def forward(
        self,
        input_ids: np.ndarray,  # [seq_len] text token IDs
        position_ids: np.ndarray | None = None,
        past_key_values: list | None = None,
        speaker_embedding: np.ndarray | None = None,  # [hidden_size] for voice clone
    ) -> tuple[np.ndarray, list]:
        """Forward pass through talker model.

        Args:
            input_ids: Text token IDs [seq_len]
            position_ids: Position IDs
            past_key_values: KV cache
            speaker_embedding: Speaker embedding for voice cloning [hidden_size]

        Returns:
            (logits, present_key_values)
        """
        seq_len = len(input_ids)

        if position_ids is None:
            position_ids = np.arange(seq_len)

        # Token embeddings
        embed_np = self.embed_tokens.to_numpy()
        hidden = embed_np[input_ids]

        # Text projection if available (2-layer MLP)
        if self.text_projection is not None:
            hidden = self.text_projection(hidden)

        # Add speaker embedding to first token (x_vector injection)
        if speaker_embedding is not None:
            hidden[0] = hidden[0] + speaker_embedding

        hidden = from_numpy(hidden.astype(np.float32))

        # Transformer blocks
        present_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden, present_kv = block(
                hidden,
                position_ids=position_ids.tolist(),
                past_kv=past_kv,
                use_cache=past_key_values is not None,
            )
            present_key_values.append(present_kv)

        # Final norm
        hidden = self.final_norm(hidden)

        # Codec head
        hidden_np = hidden.to_numpy()
        codec_head_np = self.codec_head.to_numpy()
        logits = hidden_np @ codec_head_np.T

        return logits, present_key_values

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        speaker_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate codec tokens autoregressively.

        Args:
            input_ids: Input text token IDs [seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            eos_token_id: Stop token
            speaker_embedding: Speaker embedding for voice cloning [hidden_size]

        Returns:
            Generated codec codes [num_codebooks, gen_len]
        """
        if eos_token_id is None:
            eos_token_id = self.config.codec_eos_token_id

        # Prefill (with speaker embedding injection)
        logits, past_kv = self.forward(input_ids, speaker_embedding=speaker_embedding)

        # Sample first token
        first_logits = logits[-1]
        first_token = sample_token(first_logits, temperature=temperature, top_k=top_k, top_p=top_p)

        generated = [first_token]
        current_pos = len(input_ids)

        # Autoregressive generation
        for _ in range(max_new_tokens - 1):
            # Single token forward
            new_input = np.array([generated[-1]])
            logits, past_kv = self.forward(
                new_input,
                position_ids=np.array([current_pos]),
                past_key_values=past_kv,
            )

            # Sample next token
            next_logits = logits[-1]
            next_token = sample_token(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            if next_token == eos_token_id:
                break

            generated.append(next_token)
            current_pos += 1

        # Generate remaining codebooks with code predictor
        first_codes = np.array(generated)[np.newaxis, :]  # [1, gen_len]

        if self.code_predictor is not None:
            # Get hidden states for code predictor
            # (Simplified: use last hidden state)
            all_codes = self.code_predictor(first_codes, np.zeros_like(first_codes))
        else:
            all_codes = first_codes[np.newaxis, :, :]

        return all_codes[0]  # [num_codebooks, gen_len]


# =============================================================================
# Main Qwen3-TTS Model
# =============================================================================


class Qwen3TTSModel:
    """Qwen3-TTS model for text-to-speech synthesis.

    Combines:
    - Talker: Text-to-codec transformer
    - Speaker Encoder: ECAPA-TDNN for voice cloning
    - Speech Tokenizer: Codec for audio encode/decode

    Supports three modes:
    - generate_voice_design(): Voice from natural language description
    - generate_voice_clone(): Clone voice from reference audio
    - generate_custom_voice(): Use predefined speaker with style
    """

    def __init__(
        self,
        config: Qwen3TTSConfig,
        talker: TalkerModel,
        speaker_encoder: SpeakerEncoder | None = None,
        speech_tokenizer: SpeechTokenizer | None = None,
    ):
        self.config = config
        self.talker = talker
        self.speaker_encoder = speaker_encoder
        self.speech_tokenizer = speech_tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> Qwen3TTSModel:
        """Load Qwen3-TTS model from pretrained weights.

        Args:
            model_path: Path to model directory
            device: Device to load model on
            dtype: Model dtype

        Returns:
            Qwen3TTSModel instance
        """
        # TODO: Implement weight loading from safetensors
        raise NotImplementedError(
            "from_pretrained not yet implemented. Use from_weights() with pre-loaded weights."
        )

    def _validate_language(self, language: str) -> str:
        """Validate and normalize language code."""
        lang = language.lower()
        if lang not in self.config.supported_languages:
            raise ValueError(
                f"Unsupported language '{language}'. Supported: {self.config.supported_languages}"
            )
        return lang

    def _validate_speaker(self, speaker: str) -> str:
        """Validate speaker name (for CustomVoice mode)."""
        spk = speaker.lower().replace(" ", "_")
        if spk not in self.config.supported_speakers:
            raise ValueError(
                f"Unsupported speaker '{speaker}'. Supported: {self.config.supported_speakers}"
            )
        return spk

    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text for talker input.

        Note: Uses placeholder tokenization. In production, use
        HuggingFace tokenizers with the model's tokenizer.json.
        """
        # Placeholder: simple character-level tokenization
        # In practice, use the model's actual tokenizer
        tokens = [ord(c) % self.config.vocab_size for c in text]
        return np.array(tokens, dtype=np.int64)

    def _extract_speaker_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract speaker embedding from reference audio."""
        if self.speaker_encoder is None:
            raise RuntimeError("Speaker encoder not loaded")

        # Resample to 24kHz if needed
        if sample_rate != self.config.speaker_encoder_sample_rate:
            ratio = self.config.speaker_encoder_sample_rate / sample_rate
            new_len = int(len(audio) * ratio)
            old_idx = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(old_idx, np.arange(len(audio)), audio)

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(
            audio,
            sample_rate=self.config.speaker_encoder_sample_rate,
        )

        # Extract embedding
        mel_gpu = from_numpy(mel[np.newaxis, :, :].astype(np.float32))
        embedding = self.speaker_encoder(mel_gpu)

        return embedding.to_numpy()[0]

    def create_voice_clone_prompt(
        self,
        ref_audio: np.ndarray,
        ref_text: str | None = None,
        sample_rate: int = 24000,
        x_vector_only_mode: bool = False,
    ) -> VoiceClonePromptItem:
        """Create voice clone prompt from reference audio.

        Args:
            ref_audio: Reference audio waveform
            ref_text: Reference text (required for ICL mode)
            sample_rate: Audio sample rate
            x_vector_only_mode: Use only x-vector (no ICL)

        Returns:
            VoiceClonePromptItem for generation
        """
        if not x_vector_only_mode and (ref_text is None or ref_text == ""):
            raise ValueError("ref_text required when x_vector_only_mode=False")

        # Extract speaker embedding
        spk_embedding = self._extract_speaker_embedding(ref_audio, sample_rate)

        # Encode reference audio to codes (for ICL mode)
        ref_code = None
        if not x_vector_only_mode and self.speech_tokenizer is not None:
            output = self.speech_tokenizer.encode(ref_audio, sr=sample_rate)
            ref_code = output.audio_codes[0]

        return VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=not x_vector_only_mode,
            ref_text=ref_text,
        )

    def generate_voice_design(
        self,
        text: str | list[str],
        instruct: str | list[str],
        language: str | list[str] = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        """Generate speech with voice design from natural language.

        Args:
            text: Text to synthesize
            instruct: Voice description (e.g., "A warm female voice")
            language: Language code
            max_new_tokens: Max generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            GenerationOutput with audio and sample_rate
        """
        if self.config.model_type != "voice_design":
            raise ValueError(
                f"This model ({self.config.model_type}) does not support voice design. "
                "Use a VoiceDesign model."
            )

        # Handle batched input
        texts = [text] if isinstance(text, str) else text
        instructs = [instruct] if isinstance(instruct, str) else instruct
        languages = [language] * len(texts) if isinstance(language, str) else language

        audio_list = []
        for t, ins, lang in zip(texts, instructs, languages):
            lang = self._validate_language(lang)

            # Build prompt (simplified)
            prompt = f"[INST]{ins}[/INST][LANG]{lang}[TEXT]{t}"
            input_ids = self._tokenize_text(prompt)

            # Generate codec codes
            codes = self.talker.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Decode to audio
            if self.speech_tokenizer is not None:
                output = self.speech_tokenizer.decode([codes])
                audio_list.extend(output.audio)
            else:
                # Return codes as placeholder
                audio_list.append(codes.flatten().astype(np.float32))

        return GenerationOutput(
            audio=audio_list,
            sample_rate=self.config.speech_tokenizer_sample_rate,
        )

    def generate_voice_clone(
        self,
        text: str | list[str],
        ref_audio: np.ndarray | list[np.ndarray],
        ref_text: str | list[str] | None = None,
        language: str | list[str] = "auto",
        x_vector_only_mode: bool = False,
        sample_rate: int = 24000,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        """Generate speech by cloning voice from reference audio.

        Args:
            text: Text to synthesize
            ref_audio: Reference audio for voice cloning
            ref_text: Reference text (required for ICL mode)
            language: Language code
            x_vector_only_mode: Use only speaker embedding
            sample_rate: Reference audio sample rate
            max_new_tokens: Max generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            GenerationOutput with audio and sample_rate
        """
        if self.config.model_type != "base":
            raise ValueError(
                f"This model ({self.config.model_type}) does not support voice cloning. "
                "Use a Base model."
            )

        # Handle batched input
        texts = [text] if isinstance(text, str) else text
        ref_audios = [ref_audio] if isinstance(ref_audio, np.ndarray) else ref_audio
        ref_texts = [ref_text] * len(texts) if isinstance(ref_text, (str, type(None))) else ref_text
        languages = [language] * len(texts) if isinstance(language, str) else language

        audio_list = []
        for t, ref_aud, ref_txt, lang in zip(texts, ref_audios, ref_texts, languages):
            lang = self._validate_language(lang)

            # Create voice clone prompt (for future use with speaker conditioning)
            _prompt_item = self.create_voice_clone_prompt(
                ref_audio=ref_aud,
                ref_text=ref_txt,
                sample_rate=sample_rate,
                x_vector_only_mode=x_vector_only_mode,
            )

            # Build prompt (simplified)
            prompt = f"[LANG]{lang}[TEXT]{t}"
            input_ids = self._tokenize_text(prompt)

            # Generate codec codes
            codes = self.talker.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Decode to audio
            if self.speech_tokenizer is not None:
                output = self.speech_tokenizer.decode([codes])
                audio_list.extend(output.audio)
            else:
                audio_list.append(codes.flatten().astype(np.float32))

        return GenerationOutput(
            audio=audio_list,
            sample_rate=self.config.speech_tokenizer_sample_rate,
        )

    def generate_custom_voice(
        self,
        text: str | list[str],
        speaker: str | list[str],
        language: str | list[str] = "auto",
        instruct: str | list[str] | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        """Generate speech with predefined speaker.

        Args:
            text: Text to synthesize
            speaker: Speaker name (e.g., "vivian", "ryan")
            language: Language code
            instruct: Optional style instruction
            max_new_tokens: Max generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            GenerationOutput with audio and sample_rate
        """
        if self.config.model_type != "custom_voice":
            raise ValueError(
                f"This model ({self.config.model_type}) does not support custom voice. "
                "Use a CustomVoice model."
            )

        # Handle batched input
        texts = [text] if isinstance(text, str) else text
        speakers = [speaker] * len(texts) if isinstance(speaker, str) else speaker
        languages = [language] * len(texts) if isinstance(language, str) else language
        instructs = [instruct] * len(texts) if isinstance(instruct, (str, type(None))) else instruct

        audio_list = []
        for t, spk, lang, ins in zip(texts, speakers, languages, instructs):
            lang = self._validate_language(lang)
            spk = self._validate_speaker(spk)

            # Build prompt (simplified)
            prompt = f"[SPEAKER]{spk}[LANG]{lang}"
            if ins:
                prompt += f"[INST]{ins}[/INST]"
            prompt += f"[TEXT]{t}"

            input_ids = self._tokenize_text(prompt)

            # Generate codec codes
            codes = self.talker.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Decode to audio
            if self.speech_tokenizer is not None:
                output = self.speech_tokenizer.decode([codes])
                audio_list.extend(output.audio)
            else:
                audio_list.append(codes.flatten().astype(np.float32))

        return GenerationOutput(
            audio=audio_list,
            sample_rate=self.config.speech_tokenizer_sample_rate,
        )

    def get_supported_speakers(self) -> list[str]:
        """Get list of supported speaker names."""
        return list(self.config.supported_speakers)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return list(self.config.supported_languages)


__all__ = [
    "Qwen3TTSModel",
    "Qwen3TTSConfig",
    "TalkerModel",
    "TextProjection",
    "CodePredictor",
    "VoiceClonePromptItem",
    "GenerationOutput",
]
