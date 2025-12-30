"""Text processing for Kokoro TTS.

Handles grapheme-to-phoneme (G2P) conversion and tokenization.

Kokoro uses phoneme-based input with a vocabulary of 178 tokens.
For best quality, use the misaki G2P library. A basic fallback is provided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygpukit.tts.kokoro.config import KokoroConfig


# Default phoneme vocabulary from Kokoro config.json
# This maps phoneme symbols to token IDs
DEFAULT_VOCAB: dict[str, int] = {
    # Padding
    "$": 0,
    # Punctuation (1-15)
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "\u2014": 9,  # em-dash
    "\u2026": 10,  # ellipsis
    '"': 11,
    "(": 12,
    ")": 13,
    "\u201c": 14,  # left double quote
    "\u201d": 15,  # right double quote
    # Space
    " ": 16,
    # IPA vowels and consonants (17-177)
    # This is a subset - full vocab loaded from config.json
    "a": 17,
    "b": 18,
    "d": 19,
    "e": 20,
    "f": 21,
    "h": 22,
    "i": 23,
    "j": 24,
    "k": 25,
    "l": 26,
    "m": 27,
    "n": 28,
    "o": 29,
    "p": 30,
    "s": 31,
    "t": 32,
    "u": 33,
    "v": 34,
    "w": 35,
    "z": 36,
    # IPA special characters
    "\u0251": 69,  # open back unrounded vowel
    "\u0259": 83,  # schwa
    "\u014b": 112,  # eng (ng)
    "\u03b8": 119,  # theta (th)
    "\u0283": 131,  # esh (sh)
    "\u02c8": 145,  # primary stress
    "\u02cc": 146,  # secondary stress
}


@dataclass
class TokenizerOutput:
    """Output from tokenizer.

    Attributes:
        tokens: List of token IDs
        phonemes: Phoneme string (for debugging)
        text: Original text
    """

    tokens: list[int]
    phonemes: str
    text: str

    def __len__(self) -> int:
        return len(self.tokens)


class KokoroTokenizer:
    """Tokenizer for Kokoro TTS model.

    Converts text to phoneme token sequences for the model.

    Args:
        vocab: Phoneme to token ID mapping (from config.json)
        lang: Language code for G2P ('a' for American English, etc.)
        use_misaki: Whether to use misaki G2P library (recommended)

    Example:
        >>> tokenizer = KokoroTokenizer.from_config(config)
        >>> output = tokenizer.encode("Hello, world!")
        >>> print(output.tokens)  # [22, 83, 26, 29, 33, 3, 16, 35, ...]
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        lang: str = "a",
        use_misaki: bool = True,
    ):
        self.vocab = vocab or DEFAULT_VOCAB
        self.lang = lang
        self.use_misaki = use_misaki
        self._misaki_pipeline = None

        # Create reverse mapping for decoding
        self.id_to_phoneme = {v: k for k, v in self.vocab.items()}

        # Padding token
        self.pad_token = "$"
        self.pad_id = self.vocab.get(self.pad_token, 0)

        # Try to initialize misaki if requested
        if use_misaki:
            self._init_misaki()

    def _init_misaki(self) -> bool:
        """Initialize misaki G2P pipeline."""
        try:
            from misaki import en

            # Create pipeline for the specified language
            if self.lang in ("a", "en-us"):
                self._misaki_pipeline = en.G2P(trf=False)  # Fast mode
            else:
                # Fallback to basic
                self._misaki_pipeline = None
            return self._misaki_pipeline is not None
        except ImportError:
            self._misaki_pipeline = None
            return False

    @classmethod
    def from_config(cls, config: KokoroConfig, **kwargs) -> KokoroTokenizer:
        """Create tokenizer from KokoroConfig.

        Args:
            config: KokoroConfig with vocab mapping
            **kwargs: Additional arguments for tokenizer

        Returns:
            KokoroTokenizer instance
        """
        vocab = config.vocab if config.vocab else DEFAULT_VOCAB
        return cls(vocab=vocab, **kwargs)

    def _text_to_phonemes_basic(self, text: str) -> str:
        """Basic text to phoneme conversion (fallback).

        This is a simple character-level conversion for testing.
        For production, use misaki G2P.
        """
        # Normalize text
        text = text.lower()

        # Simple replacements for common patterns
        replacements = [
            (r"th", "\u03b8"),  # theta
            (r"sh", "\u0283"),  # esh
            (r"ng", "\u014b"),  # eng
            (r"ch", "t\u0283"),  # ch
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        return text

    def _text_to_phonemes_misaki(self, text: str) -> str:
        """Convert text to phonemes using misaki G2P."""
        if self._misaki_pipeline is None:
            return self._text_to_phonemes_basic(text)

        try:
            # misaki returns a generator of (grapheme, phoneme) tuples
            result = self._misaki_pipeline(text)

            # Collect all phonemes from the generator
            phoneme_parts = []
            for item in result:
                if isinstance(item, tuple) and len(item) >= 2:
                    # (grapheme, phoneme) tuple
                    phoneme_parts.append(str(item[1]) if item[1] else "")
                elif isinstance(item, str):
                    phoneme_parts.append(item)

            # Join phonemes with space separator
            phonemes = " ".join(p for p in phoneme_parts if p)
            return phonemes if phonemes else self._text_to_phonemes_basic(text)
        except Exception:
            # Fallback on error
            return self._text_to_phonemes_basic(text)

    def text_to_phonemes(self, text: str) -> str:
        """Convert text to phoneme string.

        Args:
            text: Input text

        Returns:
            Phoneme string
        """
        if self.use_misaki and self._misaki_pipeline is not None:
            return self._text_to_phonemes_misaki(text)
        return self._text_to_phonemes_basic(text)

    def phonemes_to_tokens(self, phonemes: str) -> list[int]:
        """Convert phoneme string to token IDs.

        Args:
            phonemes: Phoneme string

        Returns:
            List of token IDs
        """
        tokens = []
        i = 0
        while i < len(phonemes):
            # Try to match longest sequence first
            matched = False

            # Check for multi-character phonemes (up to 3 chars)
            for length in [3, 2, 1]:
                if i + length <= len(phonemes):
                    substr = phonemes[i : i + length]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        i += length
                        matched = True
                        break

            if not matched:
                # Unknown character - skip or use padding
                i += 1

        return tokens

    def encode(self, text: str) -> TokenizerOutput:
        """Encode text to token sequence.

        Args:
            text: Input text

        Returns:
            TokenizerOutput with tokens, phonemes, and original text
        """
        phonemes = self.text_to_phonemes(text)
        tokens = self.phonemes_to_tokens(phonemes)

        return TokenizerOutput(
            tokens=tokens,
            phonemes=phonemes,
            text=text,
        )

    def decode(self, tokens: list[int]) -> str:
        """Decode token sequence to phoneme string.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        phonemes = []
        for token_id in tokens:
            if token_id in self.id_to_phoneme:
                phonemes.append(self.id_to_phoneme[token_id])
        return "".join(phonemes)

    def __call__(self, text: str) -> TokenizerOutput:
        """Encode text (callable interface)."""
        return self.encode(text)

    def __repr__(self) -> str:
        return (
            f"KokoroTokenizer(vocab_size={len(self.vocab)}, "
            f"lang='{self.lang}', misaki={self._misaki_pipeline is not None})"
        )


def normalize_text(text: str) -> str:
    """Normalize text for TTS processing.

    - Converts to lowercase where appropriate
    - Normalizes whitespace
    - Expands common abbreviations

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Expand common abbreviations
    abbreviations = {
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Dr.": "Doctor",
        "Jr.": "Junior",
        "Sr.": "Senior",
        "vs.": "versus",
        "etc.": "etcetera",
        "e.g.": "for example",
        "i.e.": "that is",
    }

    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)

    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked processing.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


__all__ = [
    "KokoroTokenizer",
    "TokenizerOutput",
    "normalize_text",
    "split_sentences",
    "DEFAULT_VOCAB",
]
