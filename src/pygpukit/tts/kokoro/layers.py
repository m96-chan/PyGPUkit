"""Neural network layers for Kokoro TTS model.

Implements:
- Conv1d: 1D convolution layer
- PLBERTEncoder: Text encoder (BERT-style)
- StyleEncoder: Speaker style encoder
- Decoder: Mel spectrogram decoder
- ISTFTNet: Neural vocoder with ISTFT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.dtypes import bfloat16 as dt_bfloat16
from pygpukit.core.factory import from_numpy, zeros

if TYPE_CHECKING:
    from pygpukit.tts.kokoro.config import KokoroConfig


def _get_native():
    """Get the native module."""
    try:
        from pygpukit._native_loader import get_native_module

        return get_native_module()
    except ImportError:
        from pygpukit import _pygpukit_native

        return _pygpukit_native


# =============================================================================
# Basic Layers
# =============================================================================


class Linear:
    """Linear layer: y = xW^T + b

    Weights are stored as [out_features, in_features].
    """

    def __init__(self, weight: GPUArray, bias: GPUArray | None = None):
        self.weight = weight
        self.bias = bias
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass."""
        from pygpukit.ops.basic import bias_add_inplace, matmul, transpose

        weight_t = transpose(self.weight)
        y = matmul(x, weight_t)

        if self.bias is not None:
            bias_add_inplace(y, self.bias)

        return y


class LayerNorm:
    """Layer normalization."""

    def __init__(self, weight: GPUArray, bias: GPUArray | None = None, eps: float = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = weight.shape[0]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass."""
        from pygpukit.ops.basic import layernorm

        return layernorm(x, self.weight, self.bias, self.eps)


class Conv1d:
    """1D Convolution layer.

    Implements convolution using im2col + matmul for GPU efficiency.
    Input shape: [batch, in_channels, length]
    Output shape: [batch, out_channels, new_length]
    """

    def __init__(
        self,
        weight: GPUArray,  # [out_channels, in_channels, kernel_size]
        bias: GPUArray | None = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.out_channels = weight.shape[0]
        self.in_channels = weight.shape[1]
        self.kernel_size = weight.shape[2]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass using im2col + matmul.

        This is a simple CPU implementation for correctness.
        Can be optimized with a native CUDA kernel later.
        """
        # x: [batch, in_channels, length]
        batch_size = x.shape[0]
        length = x.shape[2]

        # Calculate output length
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        out_length = (length + 2 * self.padding - effective_kernel) // self.stride + 1

        # Convert to numpy for im2col (can be optimized later)
        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()

        # Pad input
        if self.padding > 0:
            x_np = np.pad(
                x_np, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant"
            )

        # im2col: extract patches
        col = np.zeros(
            (batch_size, self.in_channels, self.kernel_size, out_length), dtype=np.float32
        )

        for i in range(self.kernel_size):
            i_dilated = i * self.dilation
            for j in range(out_length):
                j_strided = j * self.stride
                col[:, :, i, j] = x_np[:, :, j_strided + i_dilated]

        # Reshape for matmul
        # col: [batch, in_channels * kernel_size, out_length]
        col = col.reshape(batch_size, -1, out_length)

        # weight: [out_channels, in_channels * kernel_size]
        w_reshaped = w_np.reshape(self.out_channels, -1)

        # Matmul: [batch, out_channels, out_length]
        out_np = np.einsum("bkl,ok->bol", col, w_reshaped)

        # Add bias
        if self.bias is not None:
            bias_np = self.bias.to_numpy()
            out_np = out_np + bias_np.reshape(1, -1, 1)

        return from_numpy(out_np.astype(np.float32))


class ConvTranspose1d:
    """1D Transposed Convolution (deconvolution) layer.

    Used for upsampling in the vocoder.
    """

    def __init__(
        self,
        weight: GPUArray,  # [in_channels, out_channels, kernel_size]
        bias: GPUArray | None = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.in_channels = weight.shape[0]
        self.out_channels = weight.shape[1]
        self.kernel_size = weight.shape[2]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass."""
        # x: [batch, in_channels, length]
        batch_size = x.shape[0]
        length = x.shape[2]

        # Calculate output length
        out_length = (
            (length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        )

        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()

        # Initialize output
        out_np = np.zeros((batch_size, self.out_channels, out_length), dtype=np.float32)

        # Scatter-add operation
        for i in range(length):
            for k in range(self.kernel_size):
                out_pos = i * self.stride - self.padding + k
                if 0 <= out_pos < out_length:
                    # out[:, :, out_pos] += x[:, :, i] @ w[:, :, k]
                    out_np[:, :, out_pos] += np.einsum("bi,io->bo", x_np[:, :, i], w_np[:, :, k])

        # Add bias
        if self.bias is not None:
            bias_np = self.bias.to_numpy()
            out_np = out_np + bias_np.reshape(1, -1, 1)

        return from_numpy(out_np)


# =============================================================================
# Activation Functions
# =============================================================================


def leaky_relu(x: GPUArray, negative_slope: float = 0.1) -> GPUArray:
    """Leaky ReLU activation."""
    x_np = x.to_numpy()
    out_np = np.where(x_np > 0, x_np, negative_slope * x_np)
    return from_numpy(out_np.astype(np.float32))


def tanh(x: GPUArray) -> GPUArray:
    """Tanh activation."""
    from pygpukit.ops.basic import tanh as gpu_tanh

    return gpu_tanh(x)


# =============================================================================
# PLBERT Text Encoder
# =============================================================================


class BertSelfAttention:
    """BERT self-attention layer."""

    def __init__(
        self,
        query: Linear,
        key: Linear,
        value: Linear,
        num_attention_heads: int,
        attention_head_size: int,
    ):
        self.query = query
        self.key = key
        self.value = value
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = num_attention_heads * attention_head_size

    def transpose_for_scores(self, x: GPUArray) -> GPUArray:
        """Reshape for multi-head attention."""
        # x: [batch, seq_len, all_head_size]
        # output: [batch, num_heads, seq_len, head_size]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x_np = x.to_numpy()
        x_reshaped = x_np.reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        )
        x_transposed = x_reshaped.transpose(0, 2, 1, 3)
        return from_numpy(x_transposed.astype(np.float32))

    def __call__(self, hidden_states: GPUArray, attention_mask: GPUArray | None = None) -> GPUArray:
        """Forward pass."""
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        q_np = query_layer.to_numpy()
        k_np = key_layer.to_numpy()
        v_np = value_layer.to_numpy()

        # Scaled dot-product attention
        attention_scores = np.matmul(q_np, k_np.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)

        if attention_mask is not None:
            mask_np = attention_mask.to_numpy()
            attention_scores = attention_scores + mask_np

        attention_probs = np.exp(attention_scores - attention_scores.max(axis=-1, keepdims=True))
        attention_probs = attention_probs / attention_probs.sum(axis=-1, keepdims=True)

        context = np.matmul(attention_probs, v_np)

        # Reshape back
        batch_size = context.shape[0]
        seq_len = context.shape[2]
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.all_head_size)

        return from_numpy(context.astype(np.float32))


class BertLayer:
    """Single BERT encoder layer."""

    def __init__(
        self,
        attention: BertSelfAttention,
        attention_output: Linear,
        attention_norm: LayerNorm,
        intermediate: Linear,
        output_dense: Linear,
        output_norm: LayerNorm,
    ):
        self.attention = attention
        self.attention_output = attention_output
        self.attention_norm = attention_norm
        self.intermediate = intermediate
        self.output_dense = output_dense
        self.output_norm = output_norm

    def __call__(self, hidden_states: GPUArray, attention_mask: GPUArray | None = None) -> GPUArray:
        """Forward pass."""
        from pygpukit.ops.basic import add, gelu

        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        hidden_states = self.attention_norm(add(attention_output, hidden_states))

        # Feed-forward
        intermediate_output = gelu(self.intermediate(hidden_states))
        layer_output = self.output_dense(intermediate_output)
        hidden_states = self.output_norm(add(layer_output, hidden_states))

        return hidden_states


class PLBERTEncoder:
    """PLBERT text encoder for Kokoro TTS.

    BERT-style transformer encoder that converts phoneme tokens to
    contextualized embeddings.
    """

    def __init__(
        self,
        config: KokoroConfig,
        embeddings: GPUArray,  # [vocab_size, hidden_size]
        position_embeddings: GPUArray,  # [max_position, hidden_size]
        layers: list[BertLayer],
        final_norm: LayerNorm | None = None,
    ):
        self.config = config
        self.embeddings = embeddings
        self.position_embeddings = position_embeddings
        self.layers = layers
        self.final_norm = final_norm

    def __call__(
        self,
        input_ids: GPUArray,  # [batch, seq_len]
        attention_mask: GPUArray | None = None,
    ) -> GPUArray:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (optional)

        Returns:
            Hidden states [batch, seq_len, hidden_size]
        """
        from pygpukit.ops.basic import add

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Token embeddings (numpy-based for simplicity)
        input_ids_np: np.ndarray = input_ids.to_numpy().astype(np.int32)
        embeddings_np = self.embeddings.to_numpy()
        token_embeds_np = embeddings_np[input_ids_np.flatten()].reshape(
            batch_size, seq_len, -1
        )
        token_embeds = from_numpy(token_embeds_np.astype(np.float32))

        # Position embeddings
        positions = np.arange(seq_len, dtype=np.int32)
        pos_embeds_np = self.position_embeddings.to_numpy()
        pos_embeds_np = pos_embeds_np[positions].reshape(1, seq_len, -1)
        pos_embeds = from_numpy(pos_embeds_np.astype(np.float32))

        # Combine embeddings
        hidden_states = add(token_embeds, pos_embeds)

        # Create attention mask if needed
        if attention_mask is not None:
            # Convert [batch, seq_len] to [batch, 1, 1, seq_len]
            mask_np = attention_mask.to_numpy()
            extended_mask = mask_np[:, np.newaxis, np.newaxis, :]
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_mask = from_numpy(extended_mask.astype(np.float32))

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


# =============================================================================
# Style Encoder
# =============================================================================


class StyleEncoder:
    """Style encoder for speaker conditioning.

    Converts text features and speaker embedding to style vector.
    """

    def __init__(
        self,
        convs: list[Conv1d],
        norm: LayerNorm | None = None,
        output_dim: int = 128,
    ):
        self.convs = convs
        self.norm = norm
        self.output_dim = output_dim

    def __call__(
        self,
        text_features: GPUArray,  # [batch, seq_len, hidden_dim]
        speaker_embedding: GPUArray | None = None,  # [batch, style_dim]
    ) -> GPUArray:
        """Forward pass.

        Args:
            text_features: Text encoder output [batch, seq_len, hidden_dim]
            speaker_embedding: Optional speaker style [batch, style_dim]

        Returns:
            Style conditioning [batch, style_dim]
        """
        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = text_features.to_numpy().transpose(0, 2, 1)
        x = from_numpy(x.astype(np.float32))

        # Apply convolutions
        for conv in self.convs:
            x = leaky_relu(conv(x))

        # Global average pooling: [batch, channels]
        x_np = x.to_numpy()
        x_pooled = x_np.mean(axis=-1)

        result = from_numpy(x_pooled.astype(np.float32))

        # Combine with speaker embedding if provided
        if speaker_embedding is not None:
            from pygpukit.ops.basic import add

            result = add(result, speaker_embedding)

        return result


# =============================================================================
# Decoder
# =============================================================================


class ResBlock1d:
    """1D Residual block with dilated convolutions."""

    def __init__(
        self,
        convs: list[Conv1d],
    ):
        self.convs = convs

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass with residual connection."""
        from pygpukit.ops.basic import add

        residual = x
        for i, conv in enumerate(self.convs):
            x = leaky_relu(x) if i > 0 else x
            x = conv(x)
        return add(x, residual)


class Decoder:
    """Mel spectrogram decoder.

    Converts text features + style to mel spectrogram.
    """

    def __init__(
        self,
        input_proj: Linear,
        layers: list[ResBlock1d | Conv1d],
        output_proj: Linear,
        n_mels: int = 80,
    ):
        self.input_proj = input_proj
        self.layers = layers
        self.output_proj = output_proj
        self.n_mels = n_mels

    def __call__(
        self,
        text_features: GPUArray,  # [batch, seq_len, hidden_dim]
        style: GPUArray,  # [batch, style_dim]
        durations: GPUArray | None = None,  # [batch, seq_len]
    ) -> GPUArray:
        """Forward pass.

        Args:
            text_features: Text encoder output
            style: Style conditioning
            durations: Duration per phoneme (optional, for alignment)

        Returns:
            Mel spectrogram [batch, n_mels, mel_len]
        """
        # Project input
        x = self.input_proj(text_features)

        # Add style conditioning (broadcast over sequence)
        style_expanded = style.to_numpy()[:, :, np.newaxis]
        x_np = x.to_numpy()

        # Simple duration expansion (repeat each frame by duration)
        if durations is not None:
            dur_np: np.ndarray = durations.to_numpy().astype(np.int32)
            expanded: list[np.ndarray] = []
            for b in range(x_np.shape[0]):
                frames: list[np.ndarray] = []
                for t in range(x_np.shape[1]):
                    dur = max(1, int(dur_np[b, t]))
                    frames.extend([x_np[b, t]] * dur)
                expanded.append(np.stack(frames))
            x_np = np.stack(expanded)

        x = from_numpy(x_np.astype(np.float32))

        # Transpose for conv: [batch, hidden, seq_len]
        x = from_numpy(x.to_numpy().transpose(0, 2, 1).astype(np.float32))

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x)

        # Transpose back and project to mel
        x = from_numpy(x.to_numpy().transpose(0, 2, 1).astype(np.float32))
        mel = self.output_proj(x)

        # Transpose to [batch, n_mels, mel_len]
        mel = from_numpy(mel.to_numpy().transpose(0, 2, 1).astype(np.float32))

        return mel


# =============================================================================
# ISTFTNet Vocoder
# =============================================================================


class ISTFTNet:
    """ISTFTNet vocoder for waveform synthesis.

    Converts mel spectrogram to audio waveform using upsampling
    and inverse STFT.
    """

    def __init__(
        self,
        config: KokoroConfig,
        ups: list[ConvTranspose1d],
        resblocks: list[list[ResBlock1d]],
        output_conv: Conv1d,
    ):
        self.config = config
        self.ups = ups
        self.resblocks = resblocks
        self.output_conv = output_conv

        # ISTFT parameters
        self.n_fft = config.gen_istft_n_fft
        self.hop_size = config.gen_istft_hop_size

    def __call__(self, mel: GPUArray) -> GPUArray:
        """Forward pass.

        Args:
            mel: Mel spectrogram [batch, n_mels, mel_len]

        Returns:
            Audio waveform [batch, audio_len]
        """
        x = mel

        # Upsampling stages
        for i, (up, resblock_group) in enumerate(zip(self.ups, self.resblocks)):
            x = leaky_relu(x)
            x = up(x)

            # Apply residual blocks and sum
            if resblock_group:
                xs = None
                for resblock in resblock_group:
                    if xs is None:
                        xs = resblock(x)
                    else:
                        xs_np = xs.to_numpy() + resblock(x).to_numpy()
                        xs = from_numpy(xs_np.astype(np.float32))
                x = from_numpy((xs.to_numpy() / len(resblock_group)).astype(np.float32))

        x = leaky_relu(x)
        x = self.output_conv(x)
        x = tanh(x)

        # ISTFT to convert to waveform
        # Output conv produces [batch, n_fft, frames]
        # We need to apply ISTFT
        x_np = x.to_numpy()

        # Simple overlap-add reconstruction
        batch_size = x_np.shape[0]
        frames = x_np.shape[2]
        audio_len = frames * self.hop_size + self.n_fft - self.hop_size

        audio = np.zeros((batch_size, audio_len), dtype=np.float32)
        window = np.hanning(self.n_fft).astype(np.float32)

        for i in range(frames):
            start = i * self.hop_size
            audio[:, start : start + self.n_fft] += x_np[:, :, i] * window

        # Normalize by window sum
        window_sum = np.zeros(audio_len, dtype=np.float32)
        for i in range(frames):
            start = i * self.hop_size
            window_sum[start : start + self.n_fft] += window**2
        window_sum = np.maximum(window_sum, 1e-8)
        audio = audio / window_sum

        return from_numpy(audio)


# =============================================================================
# Layer Building Utilities
# =============================================================================


def build_plbert_from_weights(
    config: KokoroConfig,
    weights: dict[str, GPUArray],
    prefix: str = "bert",
) -> PLBERTEncoder:
    """Build PLBERT encoder from weight dictionary.

    Args:
        config: Model configuration
        weights: Dictionary of weight tensors
        prefix: Weight name prefix

    Returns:
        PLBERTEncoder instance
    """
    # Build embeddings
    embeddings = weights.get(f"{prefix}.embeddings.word_embeddings.weight")
    position_embeddings = weights.get(f"{prefix}.embeddings.position_embeddings.weight")

    if embeddings is None or position_embeddings is None:
        raise ValueError(f"Missing embedding weights with prefix '{prefix}'")

    # Build transformer layers
    layers = []
    for i in range(config.plbert_num_hidden_layers):
        layer_prefix = f"{prefix}.encoder.layer.{i}"

        # Check if layer exists
        q_weight = weights.get(f"{layer_prefix}.attention.self.query.weight")
        if q_weight is None:
            break

        # Self-attention
        attention = BertSelfAttention(
            query=Linear(
                weights[f"{layer_prefix}.attention.self.query.weight"],
                weights.get(f"{layer_prefix}.attention.self.query.bias"),
            ),
            key=Linear(
                weights[f"{layer_prefix}.attention.self.key.weight"],
                weights.get(f"{layer_prefix}.attention.self.key.bias"),
            ),
            value=Linear(
                weights[f"{layer_prefix}.attention.self.value.weight"],
                weights.get(f"{layer_prefix}.attention.self.value.bias"),
            ),
            num_attention_heads=config.plbert_num_attention_heads,
            attention_head_size=config.plbert_hidden_size // config.plbert_num_attention_heads,
        )

        layer = BertLayer(
            attention=attention,
            attention_output=Linear(
                weights[f"{layer_prefix}.attention.output.dense.weight"],
                weights.get(f"{layer_prefix}.attention.output.dense.bias"),
            ),
            attention_norm=LayerNorm(
                weights[f"{layer_prefix}.attention.output.LayerNorm.weight"],
                weights.get(f"{layer_prefix}.attention.output.LayerNorm.bias"),
            ),
            intermediate=Linear(
                weights[f"{layer_prefix}.intermediate.dense.weight"],
                weights.get(f"{layer_prefix}.intermediate.dense.bias"),
            ),
            output_dense=Linear(
                weights[f"{layer_prefix}.output.dense.weight"],
                weights.get(f"{layer_prefix}.output.dense.bias"),
            ),
            output_norm=LayerNorm(
                weights[f"{layer_prefix}.output.LayerNorm.weight"],
                weights.get(f"{layer_prefix}.output.LayerNorm.bias"),
            ),
        )
        layers.append(layer)

    return PLBERTEncoder(
        config=config,
        embeddings=embeddings,
        position_embeddings=position_embeddings,
        layers=layers,
    )


__all__ = [
    # Basic layers
    "Linear",
    "LayerNorm",
    "Conv1d",
    "ConvTranspose1d",
    "ResBlock1d",
    # Activations
    "leaky_relu",
    "tanh",
    # Components
    "BertSelfAttention",
    "BertLayer",
    "PLBERTEncoder",
    "StyleEncoder",
    "Decoder",
    "ISTFTNet",
    # Utilities
    "build_plbert_from_weights",
]
