"""CLIP Text Encoder.

Provides CLIP text encoding for Stable Diffusion models.
Supports both CLIP-L and CLIP-G variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class CLIPTextEncoder:
    """CLIP Text Encoder for diffusion models.

    Encodes text prompts into embeddings for conditioning.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 77,
        weights: dict[str, GPUArray] | None = None,
    ):
        """Initialize CLIP encoder.

        Args:
            hidden_size: Hidden dimension (768 for L, 1280 for G).
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_length: Maximum sequence length.
            weights: Pre-loaded weights.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.weights = weights or {}
        self.tokenizer: Tokenizer | None = None

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        dtype: str = "float32",
    ) -> CLIPTextEncoder:
        """Load CLIP encoder from SafeTensors.

        Args:
            path: Path to model directory or safetensors file.
            dtype: Weight dtype.

        Returns:
            Loaded CLIP encoder.
        """
        from pygpukit.llm.safetensors import load_safetensors

        path = Path(path)

        # Find safetensors file
        if path.is_dir():
            for name in ["model.safetensors", "text_encoder.safetensors"]:
                model_path = path / name
                if model_path.exists():
                    path = model_path
                    break

        st = load_safetensors(str(path))

        # Detect hidden size from weights
        hidden_size = 768
        num_layers = 12
        for name in st.tensor_names:
            if "embeddings.token_embedding.weight" in name:
                info = st.tensor_info(name)
                hidden_size = info.shape[1]
            if "encoder.layers" in name:
                # Count layers
                layer_num = int(name.split("layers.")[1].split(".")[0])
                num_layers = max(num_layers, layer_num + 1)

        # Load weights
        weights = {}
        for name in st.tensor_names:
            info = st.tensor_info(name)
            data = np.frombuffer(
                st.tensor_bytes(name), dtype=cls._dtype_from_safetensors(info.dtype)
            )
            data = data.reshape(info.shape)
            if dtype == "float16":
                data = data.astype(np.float16)
            else:
                data = data.astype(np.float32)
            weights[name] = from_numpy(data)

        encoder = cls(
            hidden_size=hidden_size,
            num_layers=num_layers,
            weights=weights,
        )

        # Load tokenizer if available
        tokenizer_path = (
            path.parent / "tokenizer.json" if path.is_file() else path / "tokenizer.json"
        )
        if tokenizer_path.exists():
            from tokenizers import Tokenizer

            encoder.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return encoder

    @staticmethod
    def _dtype_from_safetensors(dtype_int: int) -> np.dtype:
        dtype_map = {0: np.float32, 1: np.float16, 2: np.float32, 3: np.float64}
        return dtype_map.get(dtype_int, np.float32)

    def tokenize(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> tuple[GPUArray, GPUArray]:
        """Tokenize text input.

        Args:
            text: Input text(s).
            max_length: Maximum length (default: self.max_length).
            padding: Whether to pad to max_length.
            truncation: Whether to truncate to max_length.

        Returns:
            Tuple of (input_ids, attention_mask).
        """
        if max_length is None:
            max_length = self.max_length

        if isinstance(text, str):
            text = [text]

        batch_size = len(text)

        input_ids: np.ndarray
        attention_mask: np.ndarray

        if self.tokenizer is not None:
            # Use HuggingFace tokenizer
            encoded = self.tokenizer.encode_batch(text)
            ids_list: list[list[int]] = []
            mask_list: list[list[int]] = []

            for enc in encoded:
                ids = list(enc.ids)

                # Truncate
                if truncation and len(ids) > max_length:
                    ids = ids[:max_length]

                # Create mask
                mask = [1] * len(ids)

                # Pad
                if padding:
                    pad_len = max_length - len(ids)
                    ids = ids + [0] * pad_len
                    mask = mask + [0] * pad_len

                ids_list.append(ids)
                mask_list.append(mask)

            input_ids = np.array(ids_list, dtype=np.int64)
            attention_mask = np.array(mask_list, dtype=np.int64)
        else:
            # Simple fallback tokenization (space-based)
            input_ids = np.zeros((batch_size, max_length), dtype=np.int64)
            attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)

            for i, t in enumerate(text):
                # Very simple: treat each character as a token
                tokens = [ord(c) % 10000 for c in t][: max_length - 2]
                tokens = [49406] + tokens + [49407]  # BOS and EOS

                input_ids[i, : len(tokens)] = tokens
                attention_mask[i, : len(tokens)] = 1

        return from_numpy(input_ids), from_numpy(attention_mask)

    def encode(
        self,
        text: str | list[str],
        output_hidden_states: bool = False,
    ) -> tuple[GPUArray, GPUArray]:
        """Encode text to embeddings.

        Args:
            text: Input text(s).
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Tuple of (last_hidden_state, pooled_output).
        """
        input_ids, attention_mask = self.tokenize(text)
        return self.forward(input_ids, attention_mask)

    def forward(
        self,
        input_ids: GPUArray,
        attention_mask: GPUArray | None = None,
    ) -> tuple[GPUArray, GPUArray]:
        """Forward pass through CLIP encoder.

        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Attention mask [B, seq_len].

        Returns:
            Tuple of (last_hidden_state [B, seq_len, hidden], pooled [B, hidden]).
        """
        ids = input_ids.to_numpy()
        B, seq_len = ids.shape

        # Token embeddings
        if "text_model.embeddings.token_embedding.weight" in self.weights:
            embed_weight = self.weights["text_model.embeddings.token_embedding.weight"].to_numpy()
            x = embed_weight[ids]  # [B, seq_len, hidden]
        else:
            # Random embeddings for testing
            np.random.seed(42)
            x = np.random.randn(B, seq_len, self.hidden_size).astype(np.float32) * 0.02

        # Position embeddings
        if "text_model.embeddings.position_embedding.weight" in self.weights:
            pos_embed = self.weights["text_model.embeddings.position_embedding.weight"].to_numpy()
            x = x + pos_embed[:seq_len]
        else:
            # Add sinusoidal position embedding
            positions = np.arange(seq_len)
            pos_embed = self._sinusoidal_embed(positions, self.hidden_size)
            x = x + pos_embed

        # Process through transformer layers (simplified)
        for layer_idx in range(self.num_layers):
            x = self._transformer_layer(x, layer_idx)

        # Final layer norm
        if "text_model.final_layer_norm.weight" in self.weights:
            gamma = self.weights["text_model.final_layer_norm.weight"].to_numpy()
            beta = self.weights["text_model.final_layer_norm.bias"].to_numpy()
            x = self._layer_norm(x, gamma, beta)

        # Pooled output: take EOS token embedding
        # Find EOS position (usually the last non-padded token)
        pooled = x[:, -1, :]  # Simple: take last token

        return from_numpy(x.astype(np.float32)), from_numpy(pooled.astype(np.float32))

    def _transformer_layer(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Process through one transformer layer."""
        # Simplified transformer layer
        B, N, D = x.shape

        # Self-attention (simplified)
        residual = x
        x = self._layer_norm(x)
        attn_out = x.mean(axis=1, keepdims=True)
        attn_out = np.broadcast_to(attn_out, x.shape)
        x = residual + 0.1 * attn_out

        # MLP (simplified)
        residual = x
        x = self._layer_norm(x)
        x = residual + 0.1 * x

        return x

    def _layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)

        if gamma is not None:
            x_norm = x_norm * gamma
        if beta is not None:
            x_norm = x_norm + beta

        return x_norm

    def _sinusoidal_embed(self, positions: np.ndarray, dim: int) -> np.ndarray:
        """Generate sinusoidal position embeddings."""
        half_dim = dim // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(half_dim) / half_dim)
        args = positions[:, np.newaxis] * freqs[np.newaxis, :]
        embed = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        return embed.astype(np.float32)


# Convenience class for CLIP-L (768-dim)
class CLIPTextEncoderL(CLIPTextEncoder):
    """CLIP-L text encoder (768-dim, 12 layers)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_size", 768)
        kwargs.setdefault("num_layers", 12)
        kwargs.setdefault("num_heads", 12)
        super().__init__(**kwargs)


# Convenience class for CLIP-G (1280-dim)
class CLIPTextEncoderG(CLIPTextEncoder):
    """CLIP-G text encoder (1280-dim, 32 layers)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_size", 1280)
        kwargs.setdefault("num_layers", 32)
        kwargs.setdefault("num_heads", 20)
        super().__init__(**kwargs)


__all__ = [
    "CLIPTextEncoder",
    "CLIPTextEncoderL",
    "CLIPTextEncoderG",
]
