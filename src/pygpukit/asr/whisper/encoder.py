"""Whisper encoder implementation.

The Whisper encoder processes mel spectrograms through:
1. Conv1d stem (2 layers with GELU activation)
2. Sinusoidal positional embeddings
3. N transformer encoder layers (self-attention + FFN)
4. Final layer normalization

Architecture (Large-v3 / kotoba-whisper-v2.0):
- Input: [batch, n_mels, n_frames] = [batch, 128, 3000]
- Conv1d: 128 -> 1280 channels
- Transformer: 32 layers, 20 heads, 1280 dim
- Output: [batch, 1500, 1280]
"""

import math

import numpy as np

from ...core import GPUArray, from_numpy
from ...ops.matmul import matmul
from ...ops.nn import gelu, layernorm
from .config import WhisperConfig
from .loader import WhisperWeights


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


def _conv1d(
    x: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
    stride: int = 1,
    padding: int = 0,
) -> GPUArray:
    """1D convolution using im2col + matmul.

    Args:
        x: Input [batch, in_channels, length]
        weight: Kernel [out_channels, in_channels, kernel_size]
        bias: Bias [out_channels]
        stride: Stride
        padding: Padding

    Returns:
        Output [batch, out_channels, out_length]
    """
    # CPU fallback implementation using im2col
    # TODO: Implement native GPU conv1d kernel
    x_np = x.to_numpy()
    w_np = weight.to_numpy()
    b_np = bias.to_numpy() if bias is not None else None

    batch, in_channels, length = x_np.shape
    out_channels, _, kernel_size = w_np.shape

    # Apply padding
    if padding > 0:
        x_np = np.pad(x_np, ((0, 0), (0, 0), (padding, padding)), mode="constant")

    # Compute output length
    out_length = (x_np.shape[2] - kernel_size) // stride + 1

    # im2col: extract patches
    # Shape: [batch, in_channels * kernel_size, out_length]
    col = np.zeros((batch, in_channels * kernel_size, out_length), dtype=x_np.dtype)
    for i in range(out_length):
        start = i * stride
        end = start + kernel_size
        col[:, :, i] = x_np[:, :, start:end].reshape(batch, -1)

    # matmul: weight [out_channels, in_channels * kernel_size] @ col
    # Result: [batch, out_channels, out_length]
    w_flat = w_np.reshape(out_channels, -1)  # [out_channels, in_channels * kernel_size]
    out = np.zeros((batch, out_channels, out_length), dtype=x_np.dtype)
    for b in range(batch):
        out[b] = w_flat @ col[b]

    # Add bias
    if b_np is not None:
        out = out + b_np.reshape(1, -1, 1)

    return from_numpy(out)


class WhisperEncoderLayer:
    """Single Whisper encoder transformer layer.

    Architecture:
        x = x + self_attention(layer_norm(x))
        x = x + ffn(layer_norm(x))
    """

    def __init__(
        self,
        config: WhisperConfig,
        layer_weights: dict,
    ):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // config.encoder_attention_heads

        # Load weights as GPUArrays
        self._load_weights(layer_weights)

    def _load_weights(self, weights: dict) -> None:
        """Load layer weights to GPU."""

        def _to_gpu(arr):
            """Convert numpy array to GPUArray, handling None."""
            return from_numpy(arr) if arr is not None else None

        # Self attention
        self.q_weight = _to_gpu(weights["self_attn_q_weight"])
        self.q_bias = _to_gpu(weights["self_attn_q_bias"])
        self.k_weight = _to_gpu(weights["self_attn_k_weight"])
        self.k_bias = _to_gpu(weights["self_attn_k_bias"])
        self.v_weight = _to_gpu(weights["self_attn_v_weight"])
        self.v_bias = _to_gpu(weights["self_attn_v_bias"])
        self.out_weight = _to_gpu(weights["self_attn_out_weight"])
        self.out_bias = _to_gpu(weights["self_attn_out_bias"])

        # Self attention layer norm
        self.attn_ln_weight = _to_gpu(weights["self_attn_layer_norm_weight"])
        self.attn_ln_bias = _to_gpu(weights["self_attn_layer_norm_bias"])

        # FFN
        self.fc1_weight = _to_gpu(weights["fc1_weight"])
        self.fc1_bias = _to_gpu(weights["fc1_bias"])
        self.fc2_weight = _to_gpu(weights["fc2_weight"])
        self.fc2_bias = _to_gpu(weights["fc2_bias"])

        # Final layer norm
        self.ffn_ln_weight = _to_gpu(weights["final_layer_norm_weight"])
        self.ffn_ln_bias = _to_gpu(weights["final_layer_norm_bias"])

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self attention block
        residual = x
        x = self._layer_norm(x, self.attn_ln_weight, self.attn_ln_bias)
        x = self._self_attention(x)
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

    def _self_attention(self, x: GPUArray) -> GPUArray:
        """Multi-head self attention.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project Q, K, V
        q = self._linear(x, self.q_weight, self.q_bias)
        k = self._linear(x, self.k_weight, self.k_bias)
        v = self._linear(x, self.v_weight, self.v_bias)

        # Reshape for multi-head attention: [batch, seq, n_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose to [batch, n_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = _batched_matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Softmax over last dimension
        attn_weights = _softmax_4d(attn_weights)

        # Apply attention to values
        attn_output = _batched_matmul(attn_weights, v)

        # Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = self._linear(attn_output, self.out_weight, self.out_bias)

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
        # weight is [out_features, in_features], need to transpose
        weight_t = weight.T
        out_features = weight.shape[0]

        if x.ndim == 3:
            # Reshape [batch, seq_len, in_features] -> [batch * seq_len, in_features]
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


class WhisperEncoder:
    """Whisper audio encoder.

    Converts mel spectrograms to encoder hidden states.
    """

    def __init__(self, config: WhisperConfig, weights: WhisperWeights):
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.encoder_layers

        # Load weights
        self._load_weights(weights)

        # Create encoder layers
        self.layers = []
        for layer_weights in weights.encoder_layers:
            layer = WhisperEncoderLayer(config, layer_weights)
            self.layers.append(layer)

    def _load_weights(self, weights: WhisperWeights) -> None:
        """Load encoder-specific weights."""

        def _to_gpu(arr):
            """Convert numpy array to GPUArray, handling None."""
            return from_numpy(arr) if arr is not None else None

        # Conv1d stem
        self.conv1_weight = _to_gpu(weights.encoder_conv1_weight)
        self.conv1_bias = _to_gpu(weights.encoder_conv1_bias)
        self.conv2_weight = _to_gpu(weights.encoder_conv2_weight)
        self.conv2_bias = _to_gpu(weights.encoder_conv2_bias)

        # Positional embeddings
        self.embed_positions = _to_gpu(weights.encoder_embed_positions)

        # Final layer norm
        self.layer_norm_weight = _to_gpu(weights.encoder_layer_norm_weight)
        self.layer_norm_bias = _to_gpu(weights.encoder_layer_norm_bias)

    def __call__(self, mel: GPUArray) -> GPUArray:
        """Encode mel spectrogram to hidden states.

        Args:
            mel: Mel spectrogram [batch, n_mels, n_frames]
                 For kotoba-whisper: [batch, 128, 3000]

        Returns:
            Encoder hidden states [batch, seq_len, d_model]
            For kotoba-whisper: [batch, 1500, 1280]
        """
        # Conv1d stem: [batch, n_mels, n_frames] -> [batch, d_model, seq_len]
        x = self._conv_stem(mel)

        # Transpose to [batch, seq_len, d_model]
        x = x.transpose(0, 2, 1)

        # Add positional embeddings
        seq_len = x.shape[1]
        max_positions = self.embed_positions.shape[0]
        if seq_len > max_positions:
            # Clamp to available positions (should not happen with correct preprocessing)
            seq_len = max_positions
            x = x[:, :seq_len, :]
        positions = self.embed_positions[:seq_len]
        # Add batch dimension for broadcasting: [seq_len, d_model] -> [1, seq_len, d_model]
        positions = positions.reshape(1, seq_len, -1)
        x = x + positions

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = layernorm(x, self.layer_norm_weight, self.layer_norm_bias)

        return x

    def _conv_stem(self, mel: GPUArray) -> GPUArray:
        """Convolutional stem: 2 Conv1d layers with GELU.

        Conv1: n_mels -> d_model, kernel=3, padding=1
        Conv2: d_model -> d_model, kernel=3, stride=2, padding=1

        Args:
            mel: [batch, n_mels, n_frames]

        Returns:
            [batch, d_model, n_frames // 2]
        """
        # Conv1: [batch, n_mels, n_frames] -> [batch, d_model, n_frames]
        x = _conv1d(mel, self.conv1_weight, self.conv1_bias, padding=1)
        x = gelu(x)

        # Conv2: [batch, d_model, n_frames] -> [batch, d_model, n_frames // 2]
        x = _conv1d(x, self.conv2_weight, self.conv2_bias, stride=2, padding=1)
        x = gelu(x)

        return x


def create_encoder(config: WhisperConfig, weights: WhisperWeights) -> WhisperEncoder:
    """Create Whisper encoder from config and weights.

    Args:
        config: Whisper model configuration
        weights: Loaded model weights

    Returns:
        Initialized WhisperEncoder

    Example:
        >>> config, weights = load_whisper_model("kotoba-tech/kotoba-whisper-v2.0")
        >>> encoder = create_encoder(config, weights)
        >>> mel = preprocess_audio("audio.wav")  # [80, 3000]
        >>> hidden = encoder(mel.unsqueeze(0))  # [1, 1500, 1280]
    """
    return WhisperEncoder(config, weights)


__all__ = [
    "WhisperEncoder",
    "WhisperEncoderLayer",
    "create_encoder",
]
