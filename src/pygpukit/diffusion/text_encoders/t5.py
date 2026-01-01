"""T5 Text Encoder.

Provides T5 text encoding for SD3 and Flux models.
Uses the encoder-only variant (T5EncoderModel).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class T5Encoder:
    """T5 Text Encoder for diffusion models.

    Encoder-only T5 for generating text embeddings.
    Used by SD3 (T5-XXL) and Flux (T5-XXL).
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 24,
        num_heads: int = 64,
        d_ff: int = 10240,
        max_length: int = 512,
        weights: dict[str, GPUArray] | None = None,
    ):
        """Initialize T5 encoder.

        Args:
            hidden_size: Model dimension (4096 for T5-XXL).
            num_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            max_length: Maximum sequence length.
            weights: Pre-loaded weights.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_length = max_length
        self.weights = weights or {}
        self.tokenizer: Tokenizer | None = None

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        dtype: str = "float32",
    ) -> T5Encoder:
        """Load T5 encoder from SafeTensors.

        Args:
            path: Path to model directory or safetensors file.
            dtype: Weight dtype.

        Returns:
            Loaded T5 encoder.
        """

        path = Path(path)
        base_dir = path if path.is_dir() else path.parent

        # Check for sharded index file first
        index_path = None
        for name in [
            "model.safetensors.index.fp16.json",
            "model.safetensors.index.json",
        ]:
            candidate = base_dir / name
            if candidate.exists():
                index_path = candidate
                break

        if index_path is not None:
            # Load sharded model using Python safetensors library
            return cls._load_sharded(index_path, dtype)

        # Single file loading (fallback to Rust loader)
        if path.is_dir():
            for name in ["model.safetensors", "text_encoder_2.safetensors"]:
                model_path = path / name
                if model_path.exists():
                    path = model_path
                    break

        from pygpukit.llm.safetensors import load_safetensors

        st = load_safetensors(str(path))

        # Detect config from weights
        hidden_size = 4096
        num_layers = 24
        for name in st.tensor_names:
            if "embed_tokens.weight" in name:
                info = st.tensor_info(name)
                hidden_size = info.shape[1]
            if "block" in name or "layer" in name:
                try:
                    layer_num = int(name.split("block.")[1].split(".")[0])
                    num_layers = max(num_layers, layer_num + 1)
                except (IndexError, ValueError):
                    pass

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

        # Load tokenizer
        tokenizer_path = (
            path.parent / "tokenizer.json" if path.is_file() else path / "tokenizer.json"
        )
        if tokenizer_path.exists():
            from tokenizers import Tokenizer

            encoder.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return encoder

    @classmethod
    def _load_sharded(cls, index_path: Path, dtype: str) -> T5Encoder:
        """Load T5 encoder from sharded SafeTensors using Python library."""
        import json

        from safetensors import safe_open

        base_dir = index_path.parent

        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})

        # Get unique shard files
        shard_files = sorted(set(weight_map.values()))

        # Detect config from weight names
        hidden_size = 4096
        num_layers = 24
        for name in weight_map.keys():
            if "block" in name:
                try:
                    layer_num = int(name.split("block.")[1].split(".")[0])
                    num_layers = max(num_layers, layer_num + 1)
                except (IndexError, ValueError):
                    pass

        print(f"Loading T5 encoder from {len(shard_files)} shards...")

        # Load weights from each shard
        weights = {}
        np_dtype = np.float16 if dtype == "float16" else np.float32

        for shard_file in shard_files:
            shard_path = base_dir / shard_file
            print(f"  Loading {shard_file}...")

            with safe_open(str(shard_path), framework="numpy") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    # Convert to target dtype
                    if tensor.dtype != np_dtype:
                        tensor = tensor.astype(np_dtype)
                    weights[name] = from_numpy(tensor)

                    # Detect hidden size from embed_tokens
                    if "embed_tokens.weight" in name:
                        hidden_size = tensor.shape[1]

        print(f"Loaded {len(weights)} weights (hidden_size={hidden_size}, layers={num_layers})")

        encoder = cls(
            hidden_size=hidden_size,
            num_layers=num_layers,
            weights=weights,
        )

        # Load tokenizer
        tokenizer_path = base_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = base_dir.parent / "tokenizer" / "tokenizer.json"
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
            max_length: Maximum length.
            padding: Whether to pad.
            truncation: Whether to truncate.

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
            encoded = self.tokenizer.encode_batch(text)
            ids_list: list[list[int]] = []
            mask_list: list[list[int]] = []

            for enc in encoded:
                ids = list(enc.ids)
                if truncation and len(ids) > max_length:
                    ids = ids[:max_length]
                mask = [1] * len(ids)
                if padding:
                    pad_len = max_length - len(ids)
                    ids = ids + [0] * pad_len
                    mask = mask + [0] * pad_len
                ids_list.append(ids)
                mask_list.append(mask)

            input_ids = np.array(ids_list, dtype=np.int64)
            attention_mask = np.array(mask_list, dtype=np.int64)
        else:
            # Fallback tokenization
            input_ids = np.zeros((batch_size, max_length), dtype=np.int64)
            attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)

            for i, t in enumerate(text):
                tokens = [ord(c) % 32000 for c in t][: max_length - 1]
                tokens = tokens + [1]  # EOS token
                input_ids[i, : len(tokens)] = tokens
                attention_mask[i, : len(tokens)] = 1

        return from_numpy(input_ids), from_numpy(attention_mask)

    def encode(
        self,
        text: str | list[str],
    ) -> GPUArray:
        """Encode text to embeddings.

        Args:
            text: Input text(s).

        Returns:
            Hidden states [B, seq_len, hidden_size].
        """
        input_ids, attention_mask = self.tokenize(text)
        return self.forward(input_ids, attention_mask)

    def forward(
        self,
        input_ids: GPUArray,
        attention_mask: GPUArray | None = None,
    ) -> GPUArray:
        """Forward pass through T5 encoder (GPU-accelerated).

        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Attention mask [B, seq_len].

        Returns:
            Hidden states [B, seq_len, hidden_size].
        """

        ids = input_ids.to_numpy()
        B, seq_len = ids.shape

        # Token embeddings (CPU - indexing only)
        if "encoder.embed_tokens.weight" in self.weights:
            embed_weight = self.weights["encoder.embed_tokens.weight"].to_numpy()
            x_np = embed_weight[ids]
        elif "shared.weight" in self.weights:
            embed_weight = self.weights["shared.weight"].to_numpy()
            x_np = embed_weight[ids]
        else:
            np.random.seed(42)
            x_np = np.random.randn(B, seq_len, self.hidden_size).astype(np.float32) * 0.02

        # Move to GPU
        x = from_numpy(x_np.astype(np.float32))

        # Compute relative position bias (CPU, cached)
        rel_pos_bias = self._compute_relative_position_bias(seq_len)

        # Process through encoder layers (GPU)
        for layer_idx in range(self.num_layers):
            x = self._encoder_layer_gpu(x, layer_idx, rel_pos_bias, attention_mask)

        # Final layer norm
        if "encoder.final_layer_norm.weight" in self.weights:
            gamma = self.weights["encoder.final_layer_norm.weight"]
            x = self._rms_norm_gpu(x, gamma)
        else:
            x = self._rms_norm_gpu(x, None)

        return x

    def _compute_relative_position_bias(self, seq_len: int) -> np.ndarray | None:
        """Compute relative position bias for attention.

        T5 uses bucketed relative position bias.

        Returns:
            Bias tensor [1, num_heads, seq_len, seq_len] or None.
        """
        key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        if key not in self.weights:
            return None

        # rel_pos_bias: [num_buckets, num_heads]
        rel_pos_weight = self.weights[key].to_numpy()
        num_buckets, num_heads = rel_pos_weight.shape

        # Compute relative positions
        context_pos = np.arange(seq_len)[:, None]
        memory_pos = np.arange(seq_len)[None, :]
        relative_position = memory_pos - context_pos  # [seq_len, seq_len]

        # Bucket relative positions (T5 bucketing scheme)
        rel_buckets = self._relative_position_bucket(
            relative_position, bidirectional=True, num_buckets=num_buckets
        )

        # Lookup bias: [seq_len, seq_len, num_heads]
        bias = rel_pos_weight[rel_buckets]

        # Reshape to [1, num_heads, seq_len, seq_len]
        bias = bias.transpose(2, 0, 1)[None, :, :, :]

        return bias.astype(np.float32)

    def _relative_position_bucket(
        self,
        relative_position: np.ndarray,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> np.ndarray:
        """T5 relative position bucketing."""
        ret = 0
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(np.int32) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            np.log(n.astype(np.float32) / max_exact + 1e-6)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)

        ret += np.where(is_small, n, val_if_large)
        return ret

    def _encoder_layer_gpu(
        self,
        x: GPUArray,
        layer_idx: int,
        rel_pos_bias: np.ndarray | None,
        attention_mask: GPUArray | None,
    ) -> GPUArray:
        """Process through one T5 encoder layer (GPU)."""
        prefix = f"encoder.block.{layer_idx}"

        # Self-attention block
        residual = x

        # Pre-LN
        attn_ln_key = f"{prefix}.layer.0.layer_norm.weight"
        gamma = self.weights.get(attn_ln_key)
        x = self._rms_norm_gpu(x, gamma)

        # Self-attention (GPU)
        x = self._self_attention_gpu(x, layer_idx, rel_pos_bias, attention_mask)

        # Residual
        x_np = x.to_numpy() + residual.to_numpy()
        x = from_numpy(x_np)

        # Feed-forward block
        residual = x

        # Pre-LN
        ffn_ln_key = f"{prefix}.layer.1.layer_norm.weight"
        gamma = self.weights.get(ffn_ln_key)
        x = self._rms_norm_gpu(x, gamma)

        # FFN (GPU)
        x = self._feed_forward_gpu(x, layer_idx)

        # Residual
        x_np = x.to_numpy() + residual.to_numpy()
        x = from_numpy(x_np)

        return x

    def _self_attention_gpu(
        self,
        x: GPUArray,
        layer_idx: int,
        rel_pos_bias: np.ndarray | None,
        attention_mask: GPUArray | None,
    ) -> GPUArray:
        """T5 self-attention with GPU batched matmul."""
        from pygpukit.ops.matmul.generic import batched_matmul, matmul

        x_np = x.to_numpy()
        B, N, D = x_np.shape
        prefix = f"encoder.block.{layer_idx}.layer.0.SelfAttention"

        # Get Q, K, V, O weights
        q_w = self.weights.get(f"{prefix}.q.weight")
        k_w = self.weights.get(f"{prefix}.k.weight")
        v_w = self.weights.get(f"{prefix}.v.weight")
        o_w = self.weights.get(f"{prefix}.o.weight")

        if q_w is None:
            return from_numpy(x_np * 0.1)

        # Project Q, K, V using GPU matmul
        # x: [B, N, D] -> reshape to [B*N, D] for matmul
        inner_dim = q_w.shape[0]
        head_dim = inner_dim // self.num_heads

        x_2d = from_numpy(x_np.reshape(B * N, D).astype(np.float32))

        # Transpose weights: [inner_dim, D] -> [D, inner_dim]
        q_wt = from_numpy(q_w.to_numpy().T.astype(np.float32))
        k_wt = from_numpy(k_w.to_numpy().T.astype(np.float32))
        v_wt = from_numpy(v_w.to_numpy().T.astype(np.float32))

        q = matmul(x_2d, q_wt).to_numpy().reshape(B, N, inner_dim)
        k = matmul(x_2d, k_wt).to_numpy().reshape(B, N, inner_dim)
        v = matmul(x_2d, v_wt).to_numpy().reshape(B, N, inner_dim)

        # Reshape to [B, num_heads, N, head_dim]
        q = q.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention scores using batched matmul (GPU)
        scale = 1.0 / np.sqrt(head_dim)

        # Flatten batch and heads: [B*num_heads, N, head_dim]
        q_flat = q.reshape(B * self.num_heads, N, head_dim)
        k_flat = k.reshape(B * self.num_heads, N, head_dim)
        v_flat = v.reshape(B * self.num_heads, N, head_dim)

        # Q @ K^T using batched matmul: [B*H, N, D] @ [B*H, D, N] -> [B*H, N, N]
        q_gpu = from_numpy(q_flat.astype(np.float32))
        k_t_gpu = from_numpy(k_flat.transpose(0, 2, 1).astype(np.float32))
        scores_gpu = batched_matmul(q_gpu, k_t_gpu)
        scores = scores_gpu.to_numpy() * scale
        scores = scores.reshape(B, self.num_heads, N, N)

        # Add relative position bias
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.to_numpy()[:, None, None, :]
            scores = scores + (1.0 - mask) * (-1e9)

        # Softmax (CPU)
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

        # weights @ V using batched matmul: [B*H, N, N] @ [B*H, N, D] -> [B*H, N, D]
        attn_flat = attn_weights.reshape(B * self.num_heads, N, N)
        attn_gpu = from_numpy(attn_flat.astype(np.float32))
        v_gpu = from_numpy(v_flat.astype(np.float32))
        attn_out_gpu = batched_matmul(attn_gpu, v_gpu)
        attn_out = attn_out_gpu.to_numpy().reshape(B, self.num_heads, N, head_dim)

        # Reshape back: [B, N, inner_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B * N, inner_dim)

        # Output projection (GPU)
        attn_gpu = from_numpy(attn_out.astype(np.float32))
        o_wt = from_numpy(o_w.to_numpy().T.astype(np.float32))
        output = matmul(attn_gpu, o_wt).to_numpy().reshape(B, N, D)

        return from_numpy(output.astype(np.float32))

    def _feed_forward_gpu(self, x: GPUArray, layer_idx: int) -> GPUArray:
        """T5 gated FFN with GPU matmul."""
        from pygpukit.ops.matmul.generic import matmul

        x_np = x.to_numpy()
        B, N, D = x_np.shape
        prefix = f"encoder.block.{layer_idx}.layer.1.DenseReluDense"

        wi_0 = self.weights.get(f"{prefix}.wi_0.weight")
        wi_1 = self.weights.get(f"{prefix}.wi_1.weight")
        wo = self.weights.get(f"{prefix}.wo.weight")

        if wi_0 is None or wi_1 is None or wo is None:
            return from_numpy(x_np * 0.1)

        # Reshape for matmul: [B, N, D] -> [B*N, D]
        x_2d = from_numpy(x_np.reshape(B * N, D).astype(np.float32))

        # Transpose weights: [out_dim, in_dim] -> [in_dim, out_dim]
        wi_0t = from_numpy(wi_0.to_numpy().T.astype(np.float32))
        wi_1t = from_numpy(wi_1.to_numpy().T.astype(np.float32))
        wot = from_numpy(wo.to_numpy().T.astype(np.float32))

        # Gated FFN using GPU matmul
        gate = matmul(x_2d, wi_0t).to_numpy()
        gate = np.maximum(gate, 0)  # ReLU

        value = matmul(x_2d, wi_1t).to_numpy()

        hidden = gate * value
        hidden_gpu = from_numpy(hidden.astype(np.float32))
        output = matmul(hidden_gpu, wot).to_numpy().reshape(B, N, D)

        return from_numpy(output.astype(np.float32))

    def _rms_norm_gpu(
        self,
        x: GPUArray,
        gamma: GPUArray | None = None,
        eps: float = 1e-6,
    ) -> GPUArray:
        """RMS normalization (GPU-compatible)."""
        x_np = x.to_numpy()
        rms = np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + eps)
        x_norm = x_np / rms

        if gamma is not None:
            gamma_np = gamma.to_numpy()
            x_norm = x_norm * gamma_np

        return from_numpy(x_norm.astype(np.float32))


# T5-XXL configuration (used by SD3 and Flux)
class T5XXLEncoder(T5Encoder):
    """T5-XXL encoder (4096-dim, 24 layers)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_size", 4096)
        kwargs.setdefault("num_layers", 24)
        kwargs.setdefault("num_heads", 64)
        kwargs.setdefault("d_ff", 10240)
        kwargs.setdefault("max_length", 512)
        super().__init__(**kwargs)


__all__ = [
    "T5Encoder",
    "T5XXLEncoder",
]
