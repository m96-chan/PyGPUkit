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
    Float8E4M3 = 4  # FP8 E4M3 (1 sign, 4 exponent, 3 mantissa)
    Float8E5M2 = 5  # FP8 E5M2 (1 sign, 5 exponent, 2 mantissa)
    Int32 = 6
    Int64 = 7
    Int16 = 8
    Int8 = 9
    UInt8 = 10
    Bool = 11

    _NAMES = {
        0: "float32",
        1: "float16",
        2: "bfloat16",
        3: "float64",
        4: "float8_e4m3",
        5: "float8_e5m2",
        6: "int32",
        7: "int64",
        8: "int16",
        9: "int8",
        10: "uint8",
        11: "bool",
    }

    _SIZES = {
        0: 4,  # float32
        1: 2,  # float16
        2: 2,  # bfloat16
        3: 8,  # float64
        4: 1,  # float8_e4m3
        5: 1,  # float8_e5m2
        6: 4,  # int32
        7: 8,  # int64
        8: 2,  # int16
        9: 1,  # int8
        10: 1,  # uint8
        11: 1,  # bool
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

    def tensor_data_ptr(self, name: str) -> tuple[int, int]:
        """Get raw mmap pointer for direct GPU transfer.

        Args:
            name: Tensor name

        Returns:
            Tuple of (ptr, size_bytes) where ptr is the raw mmap address

        Raises:
            KeyError: If tensor name not found
        """
        return self._inner.tensor_data_ptr(name)

    def __len__(self) -> int:
        return self.num_tensors

    def __contains__(self, name: str) -> bool:
        return name in self._inner

    def __repr__(self) -> str:
        return f"SafeTensorsFile(num_tensors={self.num_tensors}, file_size={self.file_size})"


class ShardedSafeTensorsFile:
    """Sharded SafeTensors file loader.

    Handles models split across multiple .safetensors files with an index.json.
    Lazily opens shards on demand to minimize memory usage.

    Example:
        >>> st = ShardedSafeTensorsFile("model.safetensors.index.json")
        >>> print(st.tensor_names[:5])
        ['lm_head.weight', 'model.embed_tokens.weight', ...]
        >>> info = st.tensor_info('model.embed_tokens.weight')
        >>> data = st.tensor_bytes('model.embed_tokens.weight')
    """

    def __init__(self, index_json_path: str):
        """Open a sharded safetensors model.

        Args:
            index_json_path: Path to model.safetensors.index.json
        """
        import json
        from pathlib import Path

        self._index_path = Path(index_json_path)
        self._base_dir = self._index_path.parent

        with open(index_json_path, encoding="utf-8") as f:
            index = json.load(f)

        # weight_map: { tensor_name: shard_filename }
        self._weight_map: dict[str, str] = index.get("weight_map", {})
        self._metadata = index.get("metadata", {})

        # Lazy-loaded shard files
        self._shards: dict[str, SafeTensorsFile] = {}

        # Unique shard files
        self._shard_files = list(set(self._weight_map.values()))

    def _get_shard(self, shard_file: str) -> SafeTensorsFile:
        """Lazily open a shard file."""
        if shard_file not in self._shards:
            shard_path = self._base_dir / shard_file
            self._shards[shard_file] = SafeTensorsFile(str(shard_path))
        return self._shards[shard_file]

    @property
    def tensor_names(self) -> list[str]:
        """Get list of all tensor names across all shards."""
        return list(self._weight_map.keys())

    @property
    def file_size(self) -> int:
        """Total file size across all shards (lazy, opens all shards)."""
        total = 0
        for shard_file in self._shard_files:
            total += self._get_shard(shard_file).file_size
        return total

    @property
    def num_tensors(self) -> int:
        """Number of tensors across all shards."""
        return len(self._weight_map)

    def tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a tensor by name.

        Args:
            name: Tensor name

        Returns:
            TensorInfo with dtype, shape, offset, and size

        Raises:
            KeyError: If tensor name not found
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_info(name)

    def tensor_bytes(self, name: str) -> bytes:
        """Get raw tensor data as bytes.

        Args:
            name: Tensor name

        Returns:
            Raw bytes of the tensor data

        Raises:
            KeyError: If tensor name not found
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_bytes(name)

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
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_as_f32(name)

    def tensor_data_ptr(self, name: str) -> tuple[int, int]:
        """Get raw mmap pointer for direct GPU transfer.

        Args:
            name: Tensor name

        Returns:
            Tuple of (ptr, size_bytes) where ptr is the raw mmap address

        Raises:
            KeyError: If tensor name not found
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_data_ptr(name)

    def __len__(self) -> int:
        return self.num_tensors

    def __contains__(self, name: str) -> bool:
        return name in self._weight_map

    def __repr__(self) -> str:
        return (
            f"ShardedSafeTensorsFile(num_tensors={self.num_tensors}, "
            f"num_shards={len(self._shard_files)})"
        )


def load_safetensors(path: str) -> SafeTensorsFile | ShardedSafeTensorsFile:
    """Load a safetensors file (single or sharded).

    Automatically detects sharded models by .index.json extension.

    Args:
        path: Path to .safetensors file or .safetensors.index.json

    Returns:
        SafeTensorsFile or ShardedSafeTensorsFile for accessing tensor data

    Example:
        # Single file
        st = load_safetensors("model.safetensors")

        # Sharded model
        st = load_safetensors("model.safetensors.index.json")
    """
    if path.endswith(".index.json"):
        return ShardedSafeTensorsFile(path)
    else:
        return SafeTensorsFile(path)


class Tokenizer:
    """BPE Tokenizer for GPT-2 style models.

    **⚠️ EXPERIMENTAL: This tokenizer is intended for demos and testing only.**

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


# Chat template support (v0.2.10)
# Buffers (refactored v0.2.11)
from pygpukit.llm.buffers import (  # noqa: E402
    DecodeBuffers,
    PrefillBuffers,
)
from pygpukit.llm.chat import (  # noqa: E402
    ChatMessage,
    apply_chat_template,
    create_chat_prompt,
    format_chat_messages,
)

# Config classes and ModelSpec (refactored v0.2.11)
from pygpukit.llm.config import (  # noqa: E402
    GPT2_SPEC,
    LLAMA_SPEC,
    MIXTRAL_SPEC,
    MODEL_SPECS,
    QWEN2_SPEC,
    QWEN3_MOE_SPEC,
    QWEN3_SPEC,
    GPT2Config,
    LlamaConfig,
    ModelSpec,
    Qwen3Config,
    TransformerConfig,
    detect_model_spec,
)

# Decode strategies (refactored v0.2.11)
from pygpukit.llm.decode import (  # noqa: E402
    DecodeBatch,
    DecodeJacobi,
    DecodeM1,
    DecodeM1Graph,
    DecodeSpeculative,
    DecodeStrategy,
)

# Layers (refactored v0.2.11)
from pygpukit.llm.layers import (  # noqa: E402
    MLP,
    Attention,
    Linear,  # Backward compatibility alias
    LinearBF16,
    LinearFP8,
    MoELayer,
    Norm,
    TransformerBlock,
    apply_rotary_pos_emb_numpy,
    precompute_freqs_cis,
    repack_linear,
    repack_norm,
    repack_weight,
)

# Loaders (refactored v0.2.11)
from pygpukit.llm.loader import (  # noqa: E402
    load_gpt2_from_safetensors,
    load_llama_from_safetensors,
    load_mixtral_from_safetensors,
    load_model_from_safetensors,
    load_qwen3_from_safetensors,
    repack_model_weights,
)

# Model (refactored v0.2.11)
from pygpukit.llm.model import (  # noqa: E402
    # Type aliases
    CausalSelfAttention,
    CausalTransformerModel,
    GPT2Model,
    LayerNorm,
    LlamaAttention,
    LlamaBlock,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
)

# Sampling (refactored v0.2.11)
from pygpukit.llm.sampling import sample_token  # noqa: E402

__all__ = [
    # SafeTensors
    "Dtype",
    "TensorInfo",
    "SafeTensorsFile",
    "ShardedSafeTensorsFile",
    "load_safetensors",
    # Tokenizer
    "Tokenizer",
    # Core Transformer (v0.2.9)
    "CausalTransformerModel",
    "TransformerConfig",
    "Attention",
    "MLP",
    "MoELayer",
    "Norm",
    "TransformerBlock",
    "Linear",  # Backward compatibility alias
    "LinearBF16",
    "LinearFP8",
    # ModelSpec (v0.2.9)
    "ModelSpec",
    "GPT2_SPEC",
    "LLAMA_SPEC",
    "MIXTRAL_SPEC",
    "QWEN2_SPEC",
    "QWEN3_MOE_SPEC",
    "QWEN3_SPEC",
    "MODEL_SPECS",
    "detect_model_spec",
    # Loaders
    "load_model_from_safetensors",
    "load_gpt2_from_safetensors",
    "load_llama_from_safetensors",
    "load_mixtral_from_safetensors",
    "load_qwen3_from_safetensors",
    # Legacy config classes
    "GPT2Config",
    "LlamaConfig",
    "Qwen3Config",
    # Type aliases (all point to unified types)
    "GPT2Model",
    "LlamaModel",
    "CausalSelfAttention",
    "LayerNorm",
    "LlamaAttention",
    "LlamaBlock",
    "LlamaMLP",
    "RMSNorm",
    # Chat template support (v0.2.10)
    "ChatMessage",
    "apply_chat_template",
    "format_chat_messages",
    "create_chat_prompt",
    # Buffers (v0.2.11)
    "DecodeBuffers",
    "PrefillBuffers",
    # RoPE utilities (v0.2.11)
    "apply_rotary_pos_emb_numpy",
    "precompute_freqs_cis",
    # Weight repacking (v0.2.11)
    "repack_linear",
    "repack_norm",
    "repack_weight",
    "repack_model_weights",
    # Sampling (v0.2.11)
    "sample_token",
    # Decode strategies (v0.2.11)
    "DecodeStrategy",
    "DecodeM1",
    "DecodeM1Graph",
    "DecodeBatch",
    "DecodeSpeculative",
    "DecodeJacobi",
]
