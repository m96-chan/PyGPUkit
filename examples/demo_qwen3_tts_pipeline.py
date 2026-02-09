"""Qwen3-TTS Pipeline Demo.

Demonstrates the full media pipeline using Qwen3-TTS:
1. TTS-only: Text to speech synthesis
2. LLM to TTS: LLM text generation to speech
3. Full voice pipeline: ASR -> LLM -> TTS

Supports two voice cloning modes:
- x_vector_only: Speaker embedding only (simpler)
- ICL: Reference audio + text for higher quality

Usage:
    # TTS only (with predefined speaker)
    python examples/demo_qwen3_tts_pipeline.py --tts-only --text "Hello world"

    # TTS with voice cloning
    python examples/demo_qwen3_tts_pipeline.py --tts-only --ref-audio ref.wav --text "Hello"

    # LLM to TTS pipeline
    python examples/demo_qwen3_tts_pipeline.py --llm-tts --prompt "Explain AI briefly"

    # Full voice pipeline (ASR -> LLM -> TTS)
    python examples/demo_qwen3_tts_pipeline.py --full --audio input.wav
"""

from __future__ import annotations

import argparse
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Model paths
QWEN3_TTS_PATH = Path("F:/LLM/Qwen3-TTS-12Hz-1.7B-Base")
LLM_PATH = Path("F:/LLM/Qwen2.5-7B-Instruct")


# =============================================================================
# Utility Functions
# =============================================================================


def load_audio(path: str | Path, target_sr: int = 24000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import scipy.io.wavfile as wav

    sr, audio = wav.read(str(path))

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
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        old_idx = np.linspace(0, len(audio) - 1, new_len)
        old_idx_floor = np.floor(old_idx).astype(int)
        old_idx_ceil = np.minimum(old_idx_floor + 1, len(audio) - 1)
        frac = old_idx - old_idx_floor
        audio = audio[old_idx_floor] * (1 - frac) + audio[old_idx_ceil] * frac

    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, sample_rate: int, path: str | Path) -> None:
    """Save audio to WAV file."""
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"Saved: {path} ({len(audio) / sample_rate:.2f}s)")


def resolve_model_path(path: str | Path) -> str:
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


# =============================================================================
# Qwen3-TTS Wrapper
# =============================================================================


@dataclass
class Qwen3TTSResult:
    """Result from Qwen3-TTS synthesis."""

    audio: np.ndarray
    sample_rate: int
    codes: np.ndarray
    duration_sec: float
    synthesis_time_sec: float


class Qwen3TTSWrapper:
    """Wrapper for Qwen3-TTS model with voice cloning support."""

    def __init__(
        self,
        model_path: str | Path = QWEN3_TTS_PATH,
        load_decoder: bool = True,
    ):
        """Initialize Qwen3-TTS wrapper.

        Args:
            model_path: Path to Qwen3-TTS model directory
            load_decoder: Whether to load speech decoder
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.speaker_encoder = None
        self.speech_encoder = None
        self.speech_decoder = None

        self._load_models(load_decoder)

    def _load_models(self, load_decoder: bool) -> None:
        """Load all required models."""
        from transformers import AutoTokenizer

        from pygpukit.tts.qwen3 import load_qwen3_tts
        from pygpukit.tts.qwen3 import load_speaker_encoder as load_spk_enc

        print(f"Loading Qwen3-TTS from {self.model_path}...")
        start = time.perf_counter()

        # Main model
        self.model = load_qwen3_tts(self.model_path, load_speech_tokenizer=False)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        # Speaker encoder (for voice cloning)
        self.speaker_encoder = load_spk_enc(self.model_path)

        # Speech decoder (vocoder)
        if load_decoder:
            from pygpukit.tts.qwen3.speech_decoder import load_speech_decoder

            tokenizer_path = self.model_path / "speech_tokenizer"
            if not tokenizer_path.exists():
                # Fallback to CustomVoice model's tokenizer
                tokenizer_path = Path("F:/LLM/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer")
            self.speech_decoder = load_speech_decoder(tokenizer_path)

        elapsed = time.perf_counter() - start
        print(f"  Models loaded in {elapsed:.2f}s")

    def load_speech_encoder_for_icl(self) -> None:
        """Load speech encoder for ICL mode (lazy loading)."""
        if self.speech_encoder is None:
            from pygpukit.tts.qwen3.speech_encoder import load_speech_encoder

            print("Loading speech encoder for ICL mode...")
            self.speech_encoder = load_speech_encoder(self.model_path)

    def extract_speaker_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Extract speaker embedding from reference audio."""
        return self.speaker_encoder.extract_embedding(audio, sample_rate=sample_rate)

    def encode_audio_to_codes(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Encode audio to codec codes for ICL mode."""
        self.load_speech_encoder_for_icl()
        return self.speech_encoder.encode(audio, sample_rate=sample_rate)

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray | None = None,
        ref_codes: np.ndarray | None = None,
        ref_text: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> Qwen3TTSResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding for voice cloning
            ref_codes: Reference audio codec codes for ICL mode
            ref_text: Reference text for ICL mode
            max_tokens: Maximum codec tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Qwen3TTSResult with audio and metadata
        """
        # Special token IDs
        codec_bos_id = 2149
        codec_eos_id = 2150
        codec_nothink_id = 2155
        tts_text_bos = 151672
        tts_text_eod = 151673

        # Tokenize text
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Build input sequence
        if ref_codes is not None and ref_text is not None:
            # ICL mode
            ref_text_tokens = self.tokenizer.encode(ref_text, add_special_tokens=False)
            ref_semantic_codes = ref_codes[0].tolist()

            input_ids = (
                [tts_text_bos]
                + ref_semantic_codes
                + ref_text_tokens
                + [tts_text_eod]
                + [codec_nothink_id]
                + text_tokens
                + [tts_text_eod]
            )
        else:
            # x_vector_only mode
            input_ids = (
                [tts_text_bos, codec_nothink_id]
                + text_tokens
                + [tts_text_eod]
            )

        input_array = np.array(input_ids, dtype=np.int64)

        # Generate codec tokens
        start = time.perf_counter()

        codes = self.model.talker.generate(
            input_array,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=1.0,
            eos_token_id=codec_eos_id,
            speaker_embedding=speaker_embedding,
        )

        synthesis_time = time.perf_counter() - start

        # Decode to audio
        audio = self._decode_codes(codes)

        return Qwen3TTSResult(
            audio=audio,
            sample_rate=24000,
            codes=codes,
            duration_sec=len(audio) / 24000,
            synthesis_time_sec=synthesis_time,
        )

    def _decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Decode codec codes to audio."""
        if self.speech_decoder is None:
            # Return empty audio if no decoder
            return np.zeros(4800, dtype=np.float32)

        # Filter special tokens
        codes_flat = codes.flatten()
        valid_mask = codes_flat < 2048
        valid_codes = codes_flat[valid_mask]

        if len(valid_codes) == 0:
            return np.zeros(4800, dtype=np.float32)

        # Reshape to [num_quantizers, seq_len]
        valid_codes = valid_codes[np.newaxis, :]
        codes_multi = np.tile(valid_codes, (16, 1))

        audio = self.speech_decoder.decode(codes_multi)
        return audio


# =============================================================================
# Demo Functions
# =============================================================================


def demo_tts_only(
    tts: Qwen3TTSWrapper,
    text: str,
    ref_audio: np.ndarray | None = None,
    ref_text: str | None = None,
    use_icl: bool = False,
    output_dir: str = "output/qwen3_tts",
) -> None:
    """Demo 1: TTS synthesis only."""
    print("=" * 60)
    print("Demo: Qwen3-TTS Synthesis")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract speaker embedding if reference audio provided
    speaker_embedding = None
    ref_codes = None

    if ref_audio is not None:
        print("\nExtracting speaker embedding...")
        speaker_embedding = tts.extract_speaker_embedding(ref_audio)
        print(f"  Embedding shape: {speaker_embedding.shape}")

        if use_icl and ref_text:
            print("\nEncoding reference audio for ICL mode...")
            ref_codes = tts.encode_audio_to_codes(ref_audio)
            print(f"  Reference codes shape: {ref_codes.shape}")

    print(f"\nSynthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")

    result = tts.synthesize(
        text=text,
        speaker_embedding=speaker_embedding,
        ref_codes=ref_codes,
        ref_text=ref_text,
    )

    rtf = result.synthesis_time_sec / result.duration_sec if result.duration_sec > 0 else 0
    print(f"  Duration: {result.duration_sec:.2f}s")
    print(f"  Synthesis time: {result.synthesis_time_sec:.2f}s")
    print(f"  RTF: {rtf:.2f}x")
    print(f"  Codes shape: {result.codes.shape}")

    output_path = f"{output_dir}/qwen3_tts_output.wav"
    save_wav(result.audio, result.sample_rate, output_path)

    print("\nTTS demo complete!")


def demo_llm_tts(
    tts: Qwen3TTSWrapper,
    llm_path: str | Path,
    prompt: str,
    ref_audio: np.ndarray | None = None,
    max_llm_tokens: int = 128,
    output_dir: str = "output/qwen3_tts",
) -> None:
    """Demo 2: LLM to TTS pipeline."""
    print("=" * 60)
    print("Demo: LLM to Qwen3-TTS Pipeline")
    print("=" * 60)

    from tokenizers import Tokenizer

    from pygpukit.llm import load_model_from_safetensors

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load LLM
    print(f"\nLoading LLM from {llm_path}...")
    start = time.perf_counter()
    llm = load_model_from_safetensors(resolve_model_path(str(llm_path)))
    print(f"  LLM loaded in {time.perf_counter() - start:.2f}s")

    # Load tokenizer
    llm_dir = Path(llm_path)
    tokenizer_path = llm_dir / "tokenizer.json"
    if tokenizer_path.exists():
        llm_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    # Extract speaker embedding if reference audio provided
    speaker_embedding = None
    if ref_audio is not None:
        print("\nExtracting speaker embedding...")
        speaker_embedding = tts.extract_speaker_embedding(ref_audio)

    # Generate text with LLM
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    print("Generating response...")

    input_ids = llm_tokenizer.encode(prompt).ids

    start = time.perf_counter()
    output_ids = llm.generate(
        input_ids=input_ids,
        max_new_tokens=max_llm_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    llm_time = time.perf_counter() - start

    # Decode response
    new_tokens = output_ids[len(input_ids):]
    response_text = llm_tokenizer.decode(new_tokens)
    print(f"\nResponse ({llm_time:.2f}s):")
    print(response_text)

    # Synthesize speech
    print("\n" + "-" * 40)
    print("Synthesizing speech...")

    result = tts.synthesize(
        text=response_text,
        speaker_embedding=speaker_embedding,
    )

    print(f"  Duration: {result.duration_sec:.2f}s")
    print(f"  Synthesis time: {result.synthesis_time_sec:.2f}s")

    output_path = f"{output_dir}/llm_tts_output.wav"
    save_wav(result.audio, result.sample_rate, output_path)

    # Statistics
    total_time = llm_time + result.synthesis_time_sec
    print("\n" + "=" * 40)
    print("Statistics:")
    print(f"  LLM tokens: {len(new_tokens)}")
    print(f"  LLM time: {llm_time:.2f}s")
    print(f"  TTS time: {result.synthesis_time_sec:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Audio duration: {result.duration_sec:.2f}s")

    print("\nLLM-TTS demo complete!")


def demo_full_pipeline(
    tts: Qwen3TTSWrapper,
    llm_path: str | Path,
    audio_path: str | Path,
    ref_audio: np.ndarray | None = None,
    output_dir: str = "output/qwen3_tts",
) -> None:
    """Demo 3: Full voice pipeline (ASR -> LLM -> TTS)."""
    print("=" * 60)
    print("Demo: Full Voice Pipeline (ASR -> LLM -> Qwen3-TTS)")
    print("=" * 60)

    from tokenizers import Tokenizer

    from pygpukit.asr import WhisperModel
    from pygpukit.llm import load_model_from_safetensors

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load input audio
    print(f"\nLoading input audio: {audio_path}")
    input_audio = load_audio(audio_path, target_sr=16000)  # Whisper uses 16kHz
    print(f"  Duration: {len(input_audio) / 16000:.2f}s")

    # Load Whisper
    whisper_path = "kotoba-tech/kotoba-whisper-v2.0"
    print(f"\nLoading Whisper from {whisper_path}...")
    start = time.perf_counter()
    whisper = WhisperModel.from_pretrained(whisper_path)
    print(f"  Whisper loaded in {time.perf_counter() - start:.2f}s")

    # Load LLM
    print(f"\nLoading LLM from {llm_path}...")
    start = time.perf_counter()
    llm = load_model_from_safetensors(resolve_model_path(str(llm_path)))
    print(f"  LLM loaded in {time.perf_counter() - start:.2f}s")

    # Load tokenizer
    llm_dir = Path(llm_path)
    tokenizer_path = llm_dir / "tokenizer.json"
    llm_tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Extract speaker embedding if reference audio provided
    speaker_embedding = None
    if ref_audio is not None:
        print("\nExtracting speaker embedding...")
        speaker_embedding = tts.extract_speaker_embedding(ref_audio)

    # Step 1: ASR
    print("\n[Step 1] Transcribing audio...")
    start = time.perf_counter()
    asr_result = whisper.transcribe(input_audio)
    asr_time = time.perf_counter() - start
    print(f"  Transcription ({asr_time:.2f}s): {asr_result.text}")

    # Step 2: LLM
    print("\n[Step 2] Generating response...")
    system_prompt = "You are a helpful voice assistant. Keep responses concise."
    full_prompt = f"{system_prompt}\n\nUser: {asr_result.text}\n\nAssistant:"

    input_ids = llm_tokenizer.encode(full_prompt).ids

    start = time.perf_counter()
    output_ids = llm.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    llm_time = time.perf_counter() - start

    new_tokens = output_ids[len(input_ids):]
    response_text = llm_tokenizer.decode(new_tokens)
    print(f"  Response ({llm_time:.2f}s): {response_text}")

    # Step 3: TTS
    print("\n[Step 3] Synthesizing speech...")
    result = tts.synthesize(
        text=response_text,
        speaker_embedding=speaker_embedding,
    )

    output_path = f"{output_dir}/voice_pipeline_output.wav"
    save_wav(result.audio, result.sample_rate, output_path)

    # Statistics
    total_time = asr_time + llm_time + result.synthesis_time_sec
    print("\n" + "=" * 40)
    print("Pipeline Statistics:")
    print(f"  ASR time: {asr_time:.2f}s")
    print(f"  LLM time: {llm_time:.2f}s")
    print(f"  TTS time: {result.synthesis_time_sec:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output audio: {result.duration_sec:.2f}s")

    print("\nFull pipeline demo complete!")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Pipeline Demo")

    # Mode selection
    parser.add_argument("--tts-only", action="store_true", help="TTS synthesis only")
    parser.add_argument("--llm-tts", action="store_true", help="LLM to TTS pipeline")
    parser.add_argument("--full", action="store_true", help="Full voice pipeline")

    # Input
    parser.add_argument("--text", type=str, default="Hello, this is Qwen3 TTS.", help="Text to synthesize")
    parser.add_argument("--prompt", type=str, default="Explain AI in one sentence.", help="LLM prompt")
    parser.add_argument("--audio", type=str, help="Input audio file for ASR")

    # Voice cloning
    parser.add_argument("--ref-audio", type=str, help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", type=str, help="Reference text (for ICL mode)")
    parser.add_argument("--icl", action="store_true", help="Use ICL mode for voice cloning")

    # Model paths
    parser.add_argument("--tts-path", type=str, default=str(QWEN3_TTS_PATH), help="Qwen3-TTS model path")
    parser.add_argument("--llm-path", type=str, default=str(LLM_PATH), help="LLM model path")

    # Output
    parser.add_argument("--output-dir", type=str, default="output/qwen3_tts", help="Output directory")

    args = parser.parse_args()

    # Validate ICL mode
    if args.icl and not args.ref_text:
        parser.error("--ref-text is required when using --icl mode")

    # Check model paths
    if not Path(args.tts_path).exists():
        print(f"Error: Qwen3-TTS model not found: {args.tts_path}")
        return

    # Load reference audio if provided
    ref_audio = None
    if args.ref_audio:
        if not Path(args.ref_audio).exists():
            print(f"Error: Reference audio not found: {args.ref_audio}")
            return
        ref_audio = load_audio(args.ref_audio, target_sr=24000)
        print(f"Loaded reference audio: {len(ref_audio) / 24000:.2f}s")

    # Load TTS model
    tts = Qwen3TTSWrapper(args.tts_path)

    # Run selected demo
    if args.tts_only:
        demo_tts_only(
            tts=tts,
            text=args.text,
            ref_audio=ref_audio,
            ref_text=args.ref_text,
            use_icl=args.icl,
            output_dir=args.output_dir,
        )
    elif args.llm_tts:
        if not Path(args.llm_path).exists():
            print(f"Error: LLM model not found: {args.llm_path}")
            return
        demo_llm_tts(
            tts=tts,
            llm_path=args.llm_path,
            prompt=args.prompt,
            ref_audio=ref_audio,
            output_dir=args.output_dir,
        )
    elif args.full:
        if not args.audio:
            parser.error("--audio is required for full pipeline mode")
        if not Path(args.audio).exists():
            print(f"Error: Input audio not found: {args.audio}")
            return
        if not Path(args.llm_path).exists():
            print(f"Error: LLM model not found: {args.llm_path}")
            return
        demo_full_pipeline(
            tts=tts,
            llm_path=args.llm_path,
            audio_path=args.audio,
            ref_audio=ref_audio,
            output_dir=args.output_dir,
        )
    else:
        # Default: TTS only
        demo_tts_only(
            tts=tts,
            text=args.text,
            ref_audio=ref_audio,
            ref_text=args.ref_text,
            use_icl=args.icl,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
