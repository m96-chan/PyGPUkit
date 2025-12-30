"""BPE Tokenizer for PyGPUkit LLM.

**Note:** This tokenizer is experimental and intended for demos/testing only.
For production use, we recommend HuggingFace tokenizers:
- https://github.com/huggingface/tokenizers
- pip install tokenizers

PyGPUkit's core responsibility is GPU execution, not tokenization.
The model API expects token IDs as input - use your preferred tokenizer
to convert text to token IDs before passing to PyGPUkit models.
"""

from __future__ import annotations

from pygpukit.core.backend import get_rust_module

# Get the Rust llm module
_rust = get_rust_module()
_llm = _rust.llm if _rust else None


class Tokenizer:
    """BPE Tokenizer for GPT-2 style models.

    **EXPERIMENTAL: This tokenizer is intended for demos and testing only.**

    For production use, we recommend HuggingFace tokenizers:
    - https://github.com/huggingface/tokenizers
    - pip install tokenizers

    PyGPUkit's core responsibility is GPU execution, not tokenization.
    The model API expects token IDs as input - use your preferred tokenizer
    to convert text to token IDs before passing to PyGPUkit models.

    Limitations:
    - Only supports a subset of HuggingFace tokenizer.json formats
    - May not work with all models (e.g., Qwen3 uses unsupported format)
    - No chat template support
    - No special token handling beyond BOS/EOS/PAD

    Example:
        >>> # For demos/testing only
        >>> tok = Tokenizer("tokenizer.json")
        >>> ids = tok.encode("Hello, world!")
        >>> text = tok.decode(ids)

        >>> # For production, use HuggingFace tokenizers:
        >>> from tokenizers import Tokenizer as HFTokenizer
        >>> hf_tok = HFTokenizer.from_file("tokenizer.json")
        >>> ids = hf_tok.encode("Hello, world!").ids
    """

    def __init__(self, path: str):
        """Load tokenizer from tokenizer.json file.

        Args:
            path: Path to the tokenizer.json file
        """
        if _llm is None:
            raise RuntimeError("Rust LLM module not available")
        self._inner = _llm.Tokenizer(path)

    @classmethod
    def from_json(cls, json_str: str) -> Tokenizer:
        """Load tokenizer from JSON string.

        Args:
            json_str: JSON string containing tokenizer config

        Returns:
            Tokenizer instance
        """
        if _llm is None:
            raise RuntimeError("Rust LLM module not available")
        instance = cls.__new__(cls)
        instance._inner = _llm.Tokenizer.from_json(json_str)
        return instance

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._inner.vocab_size

    @property
    def bos_token_id(self) -> int | None:
        """Get BOS (beginning of sequence) token ID if available."""
        return self._inner.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """Get EOS (end of sequence) token ID if available."""
        return self._inner.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Get PAD token ID if available."""
        return self._inner.pad_token_id

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        return list(self._inner.encode(text))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self._inner.decode(token_ids)

    def id_to_token(self, token_id: int) -> str | None:
        """Get token string for an ID.

        Args:
            token_id: Token ID

        Returns:
            Token string if ID is valid, None otherwise
        """
        return self._inner.id_to_token(token_id)

    def token_to_id(self, token: str) -> int | None:
        """Get ID for a token string.

        Args:
            token: Token string

        Returns:
            Token ID if token exists, None otherwise
        """
        return self._inner.token_to_id(token)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"


__all__ = [
    "Tokenizer",
]
