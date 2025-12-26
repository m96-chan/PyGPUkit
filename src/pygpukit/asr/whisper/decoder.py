"""Whisper decoder implementation.

The Whisper decoder generates text tokens from encoder hidden states:
1. Token embedding lookup
2. Sinusoidal positional embeddings
3. N transformer decoder layers:
   - Causal self-attention
   - Cross-attention to encoder outputs
   - FFN
4. Final layer normalization
5. Output projection to vocabulary

Architecture (Large-v3 / kotoba-whisper-v2.0):
- Input: token IDs [batch, seq_len]
- Encoder states: [batch, 1500, 1280]
- Transformer: 2-32 layers depending on distillation
- Output: logits [batch, seq_len, vocab_size]
"""

from __future__ import annotations

import math

import numpy as np

from ...core import GPUArray, from_numpy
from ...ops.matmul import matmul
from ...ops.nn import gelu, layernorm
from .config import WhisperConfig
from .loader import WhisperWeights


def _softmax_2d(x: GPUArray) -> GPUArray:
    """Softmax over last dimension for 2D tensor.

    Args:
        x: Input [batch, features]

    Returns:
        Softmax output [batch, features]
    """
    # Use GPU softmax kernel
    from ...ops.reduction import softmax

    return softmax(x)


def _softmax_4d(x: GPUArray) -> GPUArray:
    """Softmax over last dimension for 4D attention weights.

    Args:
        x: Input [batch, heads, seq_q, seq_k]

    Returns:
        Softmax output [batch, heads, seq_q, seq_k]
    """
    # Use GPU softmax kernel (supports 2D/3D/4D)
    from ...ops.reduction import softmax

    return softmax(x)


def _batched_matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    """Batched matrix multiplication for 4D tensors.

    Args:
        a: Input [batch, heads, M, K]
        b: Input [batch, heads, K, N]

    Returns:
        Output [batch, heads, M, N]
    """
    # Use GPU batched matmul kernel
    from ...ops.matmul import batched_matmul

    return batched_matmul(a, b)


def _create_causal_mask(seq_len: int, dtype: np.dtype) -> np.ndarray:
    """Create causal attention mask.

    Args:
        seq_len: Sequence length
        dtype: Output dtype

    Returns:
        Mask [1, 1, seq_len, seq_len] where upper triangle is -inf
    """
    mask = np.triu(np.ones((seq_len, seq_len), dtype=dtype) * float("-inf"), k=1)
    return mask.reshape(1, 1, seq_len, seq_len)


class WhisperDecoderLayer:
    """Single Whisper decoder transformer layer.

    Architecture:
        x = x + self_attention(layer_norm(x))
        x = x + cross_attention(layer_norm(x), encoder_hidden_states)
        x = x + ffn(layer_norm(x))
    """

    def __init__(
        self,
        config: WhisperConfig,
        layer_weights: dict,
    ):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.decoder_attention_heads
        self.head_dim = config.d_model // config.decoder_attention_heads

        # Load weights as GPUArrays
        self._load_weights(layer_weights)

    def _load_weights(self, weights: dict) -> None:
        """Load layer weights to GPU."""

        def _to_gpu(arr):
            """Convert numpy array to GPUArray, handling None."""
            return from_numpy(arr) if arr is not None else None

        # Self attention
        self.self_attn_q_weight = _to_gpu(weights["self_attn_q_weight"])
        self.self_attn_q_bias = _to_gpu(weights["self_attn_q_bias"])
        self.self_attn_k_weight = _to_gpu(weights["self_attn_k_weight"])
        self.self_attn_k_bias = _to_gpu(weights["self_attn_k_bias"])
        self.self_attn_v_weight = _to_gpu(weights["self_attn_v_weight"])
        self.self_attn_v_bias = _to_gpu(weights["self_attn_v_bias"])
        self.self_attn_out_weight = _to_gpu(weights["self_attn_out_weight"])
        self.self_attn_out_bias = _to_gpu(weights["self_attn_out_bias"])

        # Self attention layer norm
        self.self_attn_ln_weight = _to_gpu(weights["self_attn_layer_norm_weight"])
        self.self_attn_ln_bias = _to_gpu(weights["self_attn_layer_norm_bias"])

        # Cross attention
        self.cross_attn_q_weight = _to_gpu(weights["cross_attn_q_weight"])
        self.cross_attn_q_bias = _to_gpu(weights["cross_attn_q_bias"])
        self.cross_attn_k_weight = _to_gpu(weights["cross_attn_k_weight"])
        self.cross_attn_k_bias = _to_gpu(weights["cross_attn_k_bias"])
        self.cross_attn_v_weight = _to_gpu(weights["cross_attn_v_weight"])
        self.cross_attn_v_bias = _to_gpu(weights["cross_attn_v_bias"])
        self.cross_attn_out_weight = _to_gpu(weights["cross_attn_out_weight"])
        self.cross_attn_out_bias = _to_gpu(weights["cross_attn_out_bias"])

        # Cross attention layer norm
        self.cross_attn_ln_weight = _to_gpu(weights["cross_attn_layer_norm_weight"])
        self.cross_attn_ln_bias = _to_gpu(weights["cross_attn_layer_norm_bias"])

        # FFN
        self.fc1_weight = _to_gpu(weights["fc1_weight"])
        self.fc1_bias = _to_gpu(weights["fc1_bias"])
        self.fc2_weight = _to_gpu(weights["fc2_weight"])
        self.fc2_bias = _to_gpu(weights["fc2_bias"])

        # Final layer norm
        self.ffn_ln_weight = _to_gpu(weights["final_layer_norm_weight"])
        self.ffn_ln_bias = _to_gpu(weights["final_layer_norm_bias"])

    def __call__(
        self,
        x: GPUArray,
        encoder_hidden_states: GPUArray,
        causal_mask: GPUArray | None = None,
    ) -> GPUArray:
        """Forward pass through decoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            encoder_hidden_states: Encoder output [batch, enc_seq_len, d_model]
            causal_mask: Optional causal mask [1, 1, seq_len, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self attention block (with causal masking)
        residual = x
        x = self._layer_norm(x, self.self_attn_ln_weight, self.self_attn_ln_bias)
        x = self._self_attention(x, causal_mask)
        x = residual + x

        # Cross attention block
        residual = x
        x = self._layer_norm(x, self.cross_attn_ln_weight, self.cross_attn_ln_bias)
        x = self._cross_attention(x, encoder_hidden_states)
        x = residual + x

        # FFN block
        residual = x
        x = self._layer_norm(x, self.ffn_ln_weight, self.ffn_ln_bias)
        x = self._ffn(x)
        x = residual + x

        return x

    def _layer_norm(
        self, x: GPUArray, weight: GPUArray, bias: GPUArray, eps: float = 1e-5
    ) -> GPUArray:
        """Apply layer normalization."""
        return layernorm(x, weight, bias, eps=eps)

    def _self_attention(self, x: GPUArray, causal_mask: GPUArray | None = None) -> GPUArray:
        """Causal multi-head self attention.

        Args:
            x: Input [batch, seq_len, d_model]
            causal_mask: Causal mask [1, 1, seq_len, seq_len]

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project Q, K, V
        q = self._linear(x, self.self_attn_q_weight, self.self_attn_q_bias)
        k = self._linear(x, self.self_attn_k_weight, self.self_attn_k_bias)
        v = self._linear(x, self.self_attn_v_weight, self.self_attn_v_bias)

        # Reshape for multi-head attention: [batch, seq, n_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose to [batch, n_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = _batched_matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        # Softmax
        attn_weights = _softmax_4d(attn_weights)

        # Apply attention to values
        attn_output = _batched_matmul(attn_weights, v)

        # Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = self._linear(attn_output, self.self_attn_out_weight, self.self_attn_out_bias)

        return output

    def _cross_attention(self, x: GPUArray, encoder_hidden_states: GPUArray) -> GPUArray:
        """Cross attention to encoder outputs.

        Args:
            x: Decoder input [batch, dec_seq_len, d_model]
            encoder_hidden_states: Encoder output [batch, enc_seq_len, d_model]

        Returns:
            Attention output [batch, dec_seq_len, d_model]
        """
        batch_size = x.shape[0]
        dec_seq_len = x.shape[1]
        enc_seq_len = encoder_hidden_states.shape[1]

        # Q from decoder, K/V from encoder
        q = self._linear(x, self.cross_attn_q_weight, self.cross_attn_q_bias)
        k = self._linear(encoder_hidden_states, self.cross_attn_k_weight, self.cross_attn_k_bias)
        v = self._linear(encoder_hidden_states, self.cross_attn_v_weight, self.cross_attn_v_bias)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, dec_seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, enc_seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, enc_seq_len, self.n_heads, self.head_dim)

        # Transpose to [batch, n_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention (no causal mask for cross attention)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = _batched_matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Softmax
        attn_weights = _softmax_4d(attn_weights)

        # Apply attention to values
        attn_output = _batched_matmul(attn_weights, v)

        # Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, dec_seq_len, self.d_model)

        # Output projection
        output = self._linear(attn_output, self.cross_attn_out_weight, self.cross_attn_out_bias)

        return output

    def _ffn(self, x: GPUArray) -> GPUArray:
        """Feed-forward network with GELU activation.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            FFN output [batch, seq_len, d_model]
        """
        # fc1: d_model -> ffn_dim
        h = self._linear(x, self.fc1_weight, self.fc1_bias)

        # GELU activation
        h = gelu(h)

        # fc2: ffn_dim -> d_model
        output = self._linear(h, self.fc2_weight, self.fc2_bias)

        return output

    def _linear(self, x: GPUArray, weight: GPUArray, bias: GPUArray) -> GPUArray:
        """Linear projection: y = xW^T + b.

        Handles both 2D [batch, features] and 3D [batch, seq_len, features] input.
        """
        weight_t = weight.T
        out_features = weight.shape[0]

        if x.ndim == 3:
            batch, seq_len, in_features = x.shape
            x_2d = x.reshape(batch * seq_len, in_features)
            out_2d = matmul(x_2d, weight_t)
            # Add bias in 2D (broadcasting works naturally)
            if bias is not None:
                out_2d = out_2d + bias
            out = out_2d.reshape(batch, seq_len, out_features)
        else:
            out = matmul(x, weight_t)
            if bias is not None:
                out = out + bias
        return out


class WhisperDecoder:
    """Whisper text decoder.

    Generates text tokens from encoder hidden states using
    autoregressive decoding.
    """

    def __init__(self, config: WhisperConfig, weights: WhisperWeights):
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.decoder_layers
        self.vocab_size = config.vocab_size

        # Load weights
        self._load_weights(weights)

        # Create decoder layers
        self.layers = []
        for layer_weights in weights.decoder_layers:
            layer = WhisperDecoderLayer(config, layer_weights)
            self.layers.append(layer)

        # Cached causal mask
        self._cached_mask: GPUArray | None = None
        self._cached_mask_size: int = 0

    def _load_weights(self, weights: WhisperWeights) -> None:
        """Load decoder-specific weights."""

        def _to_gpu(arr):
            """Convert numpy array to GPUArray, handling None."""
            return from_numpy(arr) if arr is not None else None

        # Token embeddings
        self.embed_tokens = _to_gpu(weights.decoder_embed_tokens)

        # Positional embeddings
        self.embed_positions = _to_gpu(weights.decoder_embed_positions)

        # Final layer norm
        self.layer_norm_weight = _to_gpu(weights.decoder_layer_norm_weight)
        self.layer_norm_bias = _to_gpu(weights.decoder_layer_norm_bias)

        # Output projection
        self.proj_out = _to_gpu(weights.proj_out_weight)

    def __call__(
        self,
        input_ids: GPUArray,
        encoder_hidden_states: GPUArray,
        past_key_values: list | None = None,
    ) -> GPUArray:
        """Decode tokens given encoder outputs.

        Args:
            input_ids: Token IDs [batch, seq_len]
            encoder_hidden_states: Encoder output [batch, enc_seq_len, d_model]
            past_key_values: Optional cached key/values for incremental decoding

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]

        # Token embedding lookup
        x = self._embed_tokens(input_ids)

        # Add positional embeddings
        positions = self.embed_positions[:seq_len]
        # Add batch dimension for broadcasting: [seq_len, d_model] -> [1, seq_len, d_model]
        positions = positions.reshape(1, seq_len, -1)
        x = x + positions

        # Get causal mask
        causal_mask = self._get_causal_mask(seq_len, x.to_numpy().dtype)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, causal_mask)

        # Final layer norm
        x = layernorm(x, self.layer_norm_weight, self.layer_norm_bias)

        # Output projection to vocabulary
        # x is [batch, seq_len, d_model], proj_out is [vocab_size, d_model]
        batch, seq_len, d_model = x.shape
        x_2d = x.reshape(batch * seq_len, d_model)
        logits_2d = matmul(x_2d, self.proj_out.T)
        logits = logits_2d.reshape(batch, seq_len, -1)

        return logits

    def _embed_tokens(self, input_ids: GPUArray) -> GPUArray:
        """Lookup token embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        # CPU fallback implementation
        ids: np.ndarray = input_ids.to_numpy().astype(np.int64)
        embed = self.embed_tokens.to_numpy()

        batch_size, seq_len = ids.shape
        output = np.zeros((batch_size, seq_len, embed.shape[1]), dtype=embed.dtype)

        for b in range(batch_size):
            for s in range(seq_len):
                output[b, s] = embed[ids[b, s]]

        return from_numpy(output)

    def _get_causal_mask(self, seq_len: int, dtype: np.dtype) -> GPUArray:
        """Get or create causal attention mask.

        Args:
            seq_len: Sequence length
            dtype: Mask dtype

        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        if self._cached_mask is None or self._cached_mask_size < seq_len:
            mask = _create_causal_mask(seq_len, dtype)
            self._cached_mask = from_numpy(mask)
            self._cached_mask_size = seq_len
            return self._cached_mask

        # Slice cached mask if needed
        if self._cached_mask_size > seq_len:
            mask = self._cached_mask.to_numpy()[:, :, :seq_len, :seq_len]
            return from_numpy(mask)

        return self._cached_mask

    def generate(
        self,
        encoder_hidden_states: GPUArray,
        max_length: int = 448,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[int]:
        """Generate tokens autoregressively.

        Args:
            encoder_hidden_states: Encoder output [1, enc_seq_len, d_model]
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling

        Returns:
            List of generated token IDs
        """
        # Start with decoder start token
        tokens = [self.config.decoder_start_token_id]

        for _ in range(max_length - 1):
            # Create input tensor
            input_ids = from_numpy(np.array([tokens], dtype=np.int64))

            # Forward pass
            logits = self(input_ids, encoder_hidden_states)

            # Get logits for last token
            last_logits = logits.to_numpy()[0, -1, :]  # [vocab_size]

            # Apply temperature (skip for greedy decoding)
            if temperature > 0.0 and temperature != 1.0:
                last_logits = last_logits / temperature

            # Sample next token
            if top_k is not None:
                # Top-k sampling
                top_k_idx = np.argsort(last_logits)[-top_k:]
                top_k_logits = last_logits[top_k_idx]
                probs = np.exp(top_k_logits - np.max(top_k_logits))
                probs = probs / probs.sum()
                next_token = top_k_idx[np.random.choice(len(top_k_idx), p=probs)]
            else:
                # Greedy decoding
                next_token = int(np.argmax(last_logits))

            tokens.append(next_token)

            # Check for end of sequence
            if next_token == self.config.eos_token_id:
                break

        return tokens


def create_decoder(config: WhisperConfig, weights: WhisperWeights) -> WhisperDecoder:
    """Create Whisper decoder from config and weights.

    Args:
        config: Whisper model configuration
        weights: Loaded model weights

    Returns:
        Initialized WhisperDecoder

    Example:
        >>> config, weights = load_whisper_model("kotoba-tech/kotoba-whisper-v2.0")
        >>> decoder = create_decoder(config, weights)
        >>> logits = decoder(input_ids, encoder_hidden_states)
    """
    return WhisperDecoder(config, weights)


__all__ = [
    "WhisperDecoder",
    "WhisperDecoderLayer",
    "create_decoder",
]
