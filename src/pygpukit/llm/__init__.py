"""LLM support module for PyGPUkit.

Provides:
- SafeTensors file loading with memory mapping
- Tensor metadata and data access
- GPU tensor allocation helpers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.backend import get_rust_module

if TYPE_CHECKING:
    from collections.abc import Sequence

# Get the Rust llm module
_rust = get_rust_module()
_llm = _rust.llm if _rust else None


class Dtype:
    """Tensor data type enumeration."""

    Float32 = 0
    Float16 = 1
    BFloat16 = 2
    Float64 = 3
    Int32 = 4
    Int64 = 5
    Int16 = 6
    Int8 = 7
    UInt8 = 8
    Bool = 9

    _NAMES = {
        0: "float32",
        1: "float16",
        2: "bfloat16",
        3: "float64",
        4: "int32",
        5: "int64",
        6: "int16",
        7: "int8",
        8: "uint8",
        9: "bool",
    }

    _SIZES = {
        0: 4,  # float32
        1: 2,  # float16
        2: 2,  # bfloat16
        3: 8,  # float64
        4: 4,  # int32
        5: 8,  # int64
        6: 2,  # int16
        7: 1,  # int8
        8: 1,  # uint8
        9: 1,  # bool
    }

    @classmethod
    def element_size(cls, dtype: int) -> int:
        """Get the size in bytes of a single element."""
        return cls._SIZES.get(dtype, 0)

    @classmethod
    def name(cls, dtype: int) -> str:
        """Get the string name of a dtype."""
        return cls._NAMES.get(dtype, "unknown")


class TensorInfo:
    """Metadata for a single tensor in a safetensors file."""

    def __init__(
        self,
        name: str,
        dtype: int,
        shape: Sequence[int],
        offset: int,
        size_bytes: int,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape)
        self.offset = offset
        self.size_bytes = size_bytes

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def dtype_name(self) -> str:
        """String name of the dtype."""
        return Dtype.name(self.dtype)

    def __repr__(self) -> str:
        return (
            f"TensorInfo(name='{self.name}', dtype={self.dtype_name}, "
            f"shape={self.shape}, size_bytes={self.size_bytes})"
        )


class SafeTensorsFile:
    """Memory-mapped SafeTensors file.

    Provides efficient access to tensor metadata and data from a .safetensors file
    using memory mapping for zero-copy data access.

    Example:
        >>> st = SafeTensorsFile("model.safetensors")
        >>> print(st.tensor_names)
        ['weight', 'bias']
        >>> info = st.tensor_info('weight')
        >>> print(info.shape, info.dtype_name)
        [768, 768] float16
        >>> data = st.tensor_bytes('weight')
    """

    def __init__(self, path: str):
        """Open a safetensors file.

        Args:
            path: Path to the .safetensors file
        """
        if _llm is None:
            raise RuntimeError("Rust LLM module not available")
        self._inner = _llm.SafeTensorsFile(path)

    @property
    def tensor_names(self) -> list[str]:
        """Get list of all tensor names."""
        return self._inner.tensor_names

    @property
    def file_size(self) -> int:
        """Total file size in bytes."""
        return self._inner.file_size

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the file."""
        return self._inner.num_tensors

    def tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a tensor by name.

        Args:
            name: Tensor name

        Returns:
            TensorInfo with dtype, shape, offset, and size

        Raises:
            KeyError: If tensor name not found
        """
        info = self._inner.tensor_info(name)
        return TensorInfo(
            name=info.name,
            dtype=int(info.dtype),
            shape=info.shape,
            offset=info.offset,
            size_bytes=info.size_bytes,
        )

    def tensor_bytes(self, name: str) -> bytes:
        """Get raw tensor data as bytes.

        Args:
            name: Tensor name

        Returns:
            Raw bytes of the tensor data

        Raises:
            KeyError: If tensor name not found
        """
        return bytes(self._inner.tensor_bytes(name))

    def tensor_as_f32(self, name: str):
        """Get tensor data as numpy float32 array.

        Args:
            name: Tensor name

        Returns:
            1D numpy array of float32 values

        Raises:
            KeyError: If tensor name not found
            ValueError: If tensor dtype is not Float32
        """
        return self._inner.tensor_as_f32(name)

    def __len__(self) -> int:
        return self.num_tensors

    def __contains__(self, name: str) -> bool:
        return name in self._inner

    def __repr__(self) -> str:
        return f"SafeTensorsFile(num_tensors={self.num_tensors}, file_size={self.file_size})"


def load_safetensors(path: str) -> SafeTensorsFile:
    """Load a safetensors file.

    Args:
        path: Path to the .safetensors file

    Returns:
        SafeTensorsFile object for accessing tensor data
    """
    return SafeTensorsFile(path)


class Tokenizer:
    """BPE Tokenizer for GPT-2 style models.

    Loads tokenizer.json format and provides basic encode/decode functionality.

    Example:
        >>> tok = Tokenizer("tokenizer.json")
        >>> ids = tok.encode("Hello, world!")
        >>> text = tok.decode(ids)
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


from pygpukit.llm.model import (  # noqa: E402
    # Unified Transformer classes
    TransformerConfig,
    CausalTransformerModel,
    Attention,
    MLP,
    Norm,
    TransformerBlock,
    Linear,
    # Legacy GPT-2 aliases
    GPT2Config,
    GPT2Model,
    CausalSelfAttention,
    LayerNorm,
    load_gpt2_from_safetensors,
    # Legacy Llama aliases
    LlamaConfig,
    LlamaModel,
    LlamaAttention,
    LlamaBlock,
    LlamaMLP,
    RMSNorm,
    load_llama_from_safetensors,
)

__all__ = [
    # SafeTensors
    "Dtype",
    "TensorInfo",
    "SafeTensorsFile",
    "load_safetensors",
    # Tokenizer
    "Tokenizer",
    # Unified Transformer (v0.2.9)
    "TransformerConfig",
    "CausalTransformerModel",
    "Attention",
    "MLP",
    "Norm",
    "TransformerBlock",
    "Linear",
    # Legacy GPT-2 aliases
    "GPT2Config",
    "GPT2Model",
    "CausalSelfAttention",
    "LayerNorm",
    "load_gpt2_from_safetensors",
    # Legacy Llama aliases
    "LlamaConfig",
    "LlamaModel",
    "LlamaAttention",
    "LlamaBlock",
    "LlamaMLP",
    "RMSNorm",
    "load_llama_from_safetensors",
]
