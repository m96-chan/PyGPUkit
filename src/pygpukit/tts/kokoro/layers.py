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
from pygpukit.core.factory import from_numpy

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
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")

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


class LSTM:
    """LSTM layer using native CUDA kernel.

    Implements unidirectional or bidirectional LSTM with PyTorch-compatible weights.

    Args:
        W_ih: Input-to-hidden weights [4*hidden_size, input_size]
        W_hh: Hidden-to-hidden weights [4*hidden_size, hidden_size]
        b_ih: Input bias [4*hidden_size]
        b_hh: Hidden bias [4*hidden_size]
        bidirectional: If True, runs bidirectional LSTM
        W_ih_reverse: Backward direction weights (only if bidirectional)
        W_hh_reverse: Backward direction weights (only if bidirectional)
        b_ih_reverse: Backward direction bias (only if bidirectional)
        b_hh_reverse: Backward direction bias (only if bidirectional)
    """

    def __init__(
        self,
        W_ih: GPUArray,
        W_hh: GPUArray,
        b_ih: GPUArray,
        b_hh: GPUArray,
        bidirectional: bool = False,
        W_ih_reverse: GPUArray | None = None,
        W_hh_reverse: GPUArray | None = None,
        b_ih_reverse: GPUArray | None = None,
        b_hh_reverse: GPUArray | None = None,
    ):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh
        self.bidirectional = bidirectional
        self.W_ih_reverse = W_ih_reverse
        self.W_hh_reverse = W_hh_reverse
        self.b_ih_reverse = b_ih_reverse
        self.b_hh_reverse = b_hh_reverse

        # Infer dimensions from weights
        self.hidden_size = W_hh.shape[1]
        self.input_size = W_ih.shape[1]

    def __call__(
        self,
        x: GPUArray,
        h0: GPUArray | None = None,
        c0: GPUArray | None = None,
    ) -> tuple[GPUArray, tuple[GPUArray, GPUArray]]:
        """Forward pass.

        Args:
            x: Input sequence [batch, seq_len, input_size]
            h0: Initial hidden state [num_layers * num_directions, batch, hidden_size]
            c0: Initial cell state [num_layers * num_directions, batch, hidden_size]

        Returns:
            Tuple of (output, (h_n, c_n)):
                output: Hidden states [batch, seq_len, hidden_size * num_directions]
                h_n: Final hidden state
                c_n: Final cell state
        """
        from pygpukit.ops.nn import lstm_bidirectional, lstm_forward

        if self.bidirectional:
            if self.W_ih_reverse is None:
                raise ValueError("Bidirectional LSTM requires reverse weights")

            output, h_n, c_n = lstm_bidirectional(
                x,
                self.W_ih,
                self.W_hh,
                self.b_ih,
                self.b_hh,
                self.W_ih_reverse,
                self.W_hh_reverse,
                self.b_ih_reverse,
                self.b_hh_reverse,
            )
        else:
            # Extract h0, c0 for single layer if provided
            h0_layer = h0
            c0_layer = c0

            output, h_n, c_n = lstm_forward(
                x, self.W_ih, self.W_hh, self.b_ih, self.b_hh, h0_layer, c0_layer
            )

        return output, (h_n, c_n)


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
        token_embeds_np = embeddings_np[input_ids_np.flatten()].reshape(batch_size, seq_len, -1)
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
        # Note: style conditioning is applied after duration expansion
        _ = style  # Reserved for future use
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
        for _i, (up, resblock_group) in enumerate(zip(self.ups, self.resblocks)):
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


# =============================================================================
# Weight Normalization and Instance Normalization
# =============================================================================


class WeightNormConv1d:
    """1D Convolution with weight normalization.

    Weight normalization decomposes weight W = g * (v / ||v||)
    where g is a scalar magnitude and v is the direction.
    """

    def __init__(
        self,
        weight_g: GPUArray,  # [out_channels, 1, 1] - magnitude
        weight_v: GPUArray,  # [out_channels, in_channels, kernel_size] - direction
        bias: GPUArray | None = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        self.weight_g = weight_g
        self.weight_v = weight_v
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.out_channels = weight_v.shape[0]
        self.in_channels = weight_v.shape[1]
        self.kernel_size = weight_v.shape[2]

    def _compute_weight(self) -> np.ndarray:
        """Compute normalized weight: W = g * (v / ||v||)."""
        g = self.weight_g.to_numpy()  # [out_channels, 1, 1]
        v = self.weight_v.to_numpy()  # [out_channels, in_channels, kernel_size]

        # Compute L2 norm of v along in_channels and kernel dimensions
        v_norm = np.sqrt((v**2).sum(axis=(1, 2), keepdims=True) + 1e-12)
        weight = g * (v / v_norm)
        return weight.astype(np.float32)

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass."""
        batch_size = x.shape[0]
        length = x.shape[2]

        # Compute normalized weight
        weight = self._compute_weight()

        # Calculate output length
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        out_length = (length + 2 * self.padding - effective_kernel) // self.stride + 1

        x_np = x.to_numpy()

        # Pad input
        if self.padding > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")

        # im2col
        col = np.zeros(
            (batch_size, self.in_channels, self.kernel_size, out_length), dtype=np.float32
        )
        for i in range(self.kernel_size):
            i_dilated = i * self.dilation
            for j in range(out_length):
                j_strided = j * self.stride
                col[:, :, i, j] = x_np[:, :, j_strided + i_dilated]

        col = col.reshape(batch_size, -1, out_length)
        w_reshaped = weight.reshape(self.out_channels, -1)
        out_np = np.einsum("bkl,ok->bol", col, w_reshaped)

        if self.bias is not None:
            bias_np = self.bias.to_numpy()
            out_np = out_np + bias_np.reshape(1, -1, 1)

        return from_numpy(out_np.astype(np.float32))


class InstanceNorm1d:
    """1D Instance Normalization.

    Normalizes each channel independently for each sample.
    Uses gamma and beta for affine transform.
    """

    def __init__(
        self,
        gamma: GPUArray,  # [channels] - scale
        beta: GPUArray,  # [channels] - shift
        eps: float = 1e-5,
    ):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.num_features = gamma.shape[0]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: y = gamma * (x - mean) / sqrt(var + eps) + beta."""
        x_np = x.to_numpy()  # [batch, channels, length]

        # Compute mean and var along length dimension
        mean = x_np.mean(axis=2, keepdims=True)
        var = x_np.var(axis=2, keepdims=True)

        # Normalize
        x_norm = (x_np - mean) / np.sqrt(var + self.eps)

        # Apply affine transform
        gamma = self.gamma.to_numpy().reshape(1, -1, 1)
        beta = self.beta.to_numpy().reshape(1, -1, 1)
        out = gamma * x_norm + beta

        return from_numpy(out.astype(np.float32))


class AdaIN:
    """Adaptive Instance Normalization.

    Computes style-dependent scale and shift from a style vector.
    y = scale * (x - mean) / std + shift
    where scale and shift are computed from the style vector.
    """

    def __init__(
        self,
        fc_weight: GPUArray,  # [2 * channels, style_dim]
        fc_bias: GPUArray,  # [2 * channels]
    ):
        self.fc_weight = fc_weight
        self.fc_bias = fc_bias
        self.num_features = fc_weight.shape[0] // 2

    def __call__(self, x: GPUArray, style: GPUArray, eps: float = 1e-5) -> GPUArray:
        """Forward pass.

        Args:
            x: Input [batch, channels, length]
            style: Style vector [batch, style_dim]

        Returns:
            Normalized and styled output [batch, channels, length]
        """
        x_np = x.to_numpy()
        style_np = style.to_numpy()

        # Compute scale and shift from style
        fc_w = self.fc_weight.to_numpy()
        fc_b = self.fc_bias.to_numpy()
        params = style_np @ fc_w.T + fc_b  # [batch, 2 * channels]

        scale = params[:, : self.num_features].reshape(-1, self.num_features, 1)
        shift = params[:, self.num_features :].reshape(-1, self.num_features, 1)

        # Instance normalization
        mean = x_np.mean(axis=2, keepdims=True)
        std = np.sqrt(x_np.var(axis=2, keepdims=True) + eps)
        x_norm = (x_np - mean) / std

        # Apply adaptive style
        out = scale * x_norm + shift

        return from_numpy(out.astype(np.float32))


# =============================================================================
# ALBERT Encoder (used by Kokoro instead of BERT)
# =============================================================================


class ALBERTLayer:
    """Single ALBERT layer with shared weights across layers."""

    def __init__(
        self,
        query: Linear,
        key: Linear,
        value: Linear,
        attention_dense: Linear,
        attention_norm: LayerNorm,
        ffn: Linear,
        ffn_output: Linear,
        full_layer_norm: LayerNorm,
        num_attention_heads: int,
        hidden_size: int,
    ):
        self.query = query
        self.key = key
        self.value = value
        self.attention_dense = attention_dense
        self.attention_norm = attention_norm
        self.ffn = ffn
        self.ffn_output = ffn_output
        self.full_layer_norm = full_layer_norm
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

    def transpose_for_scores(self, x: GPUArray) -> GPUArray:
        """Reshape for multi-head attention."""
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
        from pygpukit.ops.basic import add, gelu

        # Self-attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

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
        hidden_size = self.num_attention_heads * self.attention_head_size
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        context = from_numpy(context.astype(np.float32))

        # Attention output
        attention_output = self.attention_dense(context)
        hidden_states = self.attention_norm(add(attention_output, hidden_states))

        # Feed-forward
        ffn_output = gelu(self.ffn(hidden_states))
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_norm(add(ffn_output, hidden_states))

        return hidden_states


class ALBERTEncoder:
    """ALBERT encoder for Kokoro TTS.

    ALBERT shares weights across layers, making it more parameter-efficient.
    """

    def __init__(
        self,
        word_embeddings: GPUArray,
        position_embeddings: GPUArray,
        token_type_embeddings: GPUArray,
        embeddings_norm: LayerNorm,
        embedding_mapping: Linear,  # Maps from embedding dim to hidden dim
        layer: ALBERTLayer,  # Shared layer
        num_hidden_layers: int = 12,
    ):
        self.word_embeddings = word_embeddings
        self.position_embeddings = position_embeddings
        self.token_type_embeddings = token_type_embeddings
        self.embeddings_norm = embeddings_norm
        self.embedding_mapping = embedding_mapping
        self.layer = layer
        self.num_hidden_layers = num_hidden_layers

    def __call__(
        self,
        input_ids: GPUArray,
        attention_mask: GPUArray | None = None,
    ) -> GPUArray:
        """Forward pass."""

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Token embeddings
        input_ids_np: np.ndarray = input_ids.to_numpy().astype(np.int32)
        word_embeds_np = self.word_embeddings.to_numpy()
        token_embeds = word_embeds_np[input_ids_np.flatten()].reshape(batch_size, seq_len, -1)

        # Position embeddings
        positions = np.arange(seq_len, dtype=np.int32)
        pos_embeds_np = self.position_embeddings.to_numpy()
        pos_embeds = pos_embeds_np[positions].reshape(1, seq_len, -1)

        # Token type embeddings (all zeros for single sequence)
        token_type_embeds_np = self.token_type_embeddings.to_numpy()
        token_type_embeds = token_type_embeds_np[0].reshape(1, 1, -1)

        # Combine embeddings
        embeddings = token_embeds + pos_embeds + token_type_embeds
        embeddings = from_numpy(embeddings.astype(np.float32))
        embeddings = self.embeddings_norm(embeddings)

        # Project to hidden size
        hidden_states = self.embedding_mapping(embeddings)

        # Create attention mask
        if attention_mask is not None:
            mask_np = attention_mask.to_numpy()
            extended_mask = mask_np[:, np.newaxis, np.newaxis, :]
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_mask = from_numpy(extended_mask.astype(np.float32))

        # Apply shared layer multiple times
        for _ in range(self.num_hidden_layers):
            hidden_states = self.layer(hidden_states, attention_mask)

        return hidden_states


# =============================================================================
# Kokoro Text Encoder (CNN + LSTM)
# =============================================================================


class KokoroTextEncoder:
    """Text encoder for Kokoro TTS.

    Architecture: Embedding -> CNN layers -> BiLSTM
    """

    def __init__(
        self,
        embedding: GPUArray,  # [vocab_size, embed_dim]
        cnn_layers: list[tuple[WeightNormConv1d, InstanceNorm1d]],
        lstm: LSTM,
    ):
        self.embedding = embedding
        self.cnn_layers = cnn_layers
        self.lstm = lstm

    def __call__(self, input_ids: GPUArray) -> GPUArray:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Encoded features [batch, seq_len, hidden_dim]
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Embedding lookup
        input_ids_np: np.ndarray = input_ids.to_numpy().astype(np.int32)
        embed_np = self.embedding.to_numpy()
        x = embed_np[input_ids_np.flatten()].reshape(batch_size, seq_len, -1)
        x = from_numpy(x.astype(np.float32))

        # Transpose for CNN: [batch, embed_dim, seq_len]
        x = from_numpy(x.to_numpy().transpose(0, 2, 1).astype(np.float32))

        # CNN layers with instance norm
        for conv, norm in self.cnn_layers:
            x = conv(x)
            x = norm(x)
            x = leaky_relu(x)

        # Transpose back for LSTM: [batch, seq_len, channels]
        x = from_numpy(x.to_numpy().transpose(0, 2, 1).astype(np.float32))

        # BiLSTM
        output, _ = self.lstm(x)

        return output


# =============================================================================
# Kokoro AdaIN ResBlock
# =============================================================================


class AdaINResBlock:
    """Residual block with AdaIN for style conditioning."""

    def __init__(
        self,
        conv1: WeightNormConv1d,
        conv2: WeightNormConv1d,
        norm1: AdaIN,
        norm2: AdaIN,
        conv1x1: WeightNormConv1d | None = None,  # For channel mismatch
    ):
        self.conv1 = conv1
        self.conv2 = conv2
        self.norm1 = norm1
        self.norm2 = norm2
        self.conv1x1 = conv1x1

    def __call__(self, x: GPUArray, style: GPUArray) -> GPUArray:
        """Forward pass with style conditioning."""
        residual = x

        # First conv + AdaIN
        out = self.norm1(x, style)
        out = leaky_relu(out)
        out = self.conv1(out)

        # Second conv + AdaIN
        out = self.norm2(out, style)
        out = leaky_relu(out)
        out = self.conv2(out)

        # Residual connection (with 1x1 conv if needed)
        if self.conv1x1 is not None:
            residual = self.conv1x1(residual)

        out_np = out.to_numpy() + residual.to_numpy()
        return from_numpy(out_np.astype(np.float32))


# =============================================================================
# Builder Functions
# =============================================================================


def build_albert_from_weights(
    weights: dict[str, GPUArray],
    prefix: str = "bert",
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    hidden_size: int = 768,
) -> ALBERTEncoder:
    """Build ALBERT encoder from weight dictionary."""
    # Embeddings
    word_embeddings = weights[f"{prefix}.module.embeddings.word_embeddings.weight"]
    position_embeddings = weights[f"{prefix}.module.embeddings.position_embeddings.weight"]
    token_type_embeddings = weights[f"{prefix}.module.embeddings.token_type_embeddings.weight"]

    embeddings_norm = LayerNorm(
        weights[f"{prefix}.module.embeddings.LayerNorm.weight"],
        weights.get(f"{prefix}.module.embeddings.LayerNorm.bias"),
    )

    embedding_mapping = Linear(
        weights[f"{prefix}.module.encoder.embedding_hidden_mapping_in.weight"],
        weights.get(f"{prefix}.module.encoder.embedding_hidden_mapping_in.bias"),
    )

    # Shared ALBERT layer
    layer_prefix = f"{prefix}.module.encoder.albert_layer_groups.0.albert_layers.0"

    layer = ALBERTLayer(
        query=Linear(
            weights[f"{layer_prefix}.attention.query.weight"],
            weights.get(f"{layer_prefix}.attention.query.bias"),
        ),
        key=Linear(
            weights[f"{layer_prefix}.attention.key.weight"],
            weights.get(f"{layer_prefix}.attention.key.bias"),
        ),
        value=Linear(
            weights[f"{layer_prefix}.attention.value.weight"],
            weights.get(f"{layer_prefix}.attention.value.bias"),
        ),
        attention_dense=Linear(
            weights[f"{layer_prefix}.attention.dense.weight"],
            weights.get(f"{layer_prefix}.attention.dense.bias"),
        ),
        attention_norm=LayerNorm(
            weights[f"{layer_prefix}.attention.LayerNorm.weight"],
            weights.get(f"{layer_prefix}.attention.LayerNorm.bias"),
        ),
        ffn=Linear(
            weights[f"{layer_prefix}.ffn.weight"],
            weights.get(f"{layer_prefix}.ffn.bias"),
        ),
        ffn_output=Linear(
            weights[f"{layer_prefix}.ffn_output.weight"],
            weights.get(f"{layer_prefix}.ffn_output.bias"),
        ),
        full_layer_norm=LayerNorm(
            weights[f"{layer_prefix}.full_layer_layer_norm.weight"],
            weights.get(f"{layer_prefix}.full_layer_layer_norm.bias"),
        ),
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
    )

    return ALBERTEncoder(
        word_embeddings=word_embeddings,
        position_embeddings=position_embeddings,
        token_type_embeddings=token_type_embeddings,
        embeddings_norm=embeddings_norm,
        embedding_mapping=embedding_mapping,
        layer=layer,
        num_hidden_layers=num_hidden_layers,
    )


def build_text_encoder_from_weights(
    weights: dict[str, GPUArray],
    prefix: str = "text_encoder",
) -> KokoroTextEncoder:
    """Build Kokoro text encoder from weight dictionary."""
    # Embedding
    embedding = weights[f"{prefix}.module.embedding.weight"]

    # CNN layers (3 layers)
    cnn_layers = []
    for i in range(3):
        conv = WeightNormConv1d(
            weight_g=weights[f"{prefix}.module.cnn.{i}.0.weight_g"],
            weight_v=weights[f"{prefix}.module.cnn.{i}.0.weight_v"],
            bias=weights.get(f"{prefix}.module.cnn.{i}.0.bias"),
            padding=2,  # kernel_size=5, padding=2 for same output length
        )
        norm = InstanceNorm1d(
            gamma=weights[f"{prefix}.module.cnn.{i}.1.gamma"],
            beta=weights[f"{prefix}.module.cnn.{i}.1.beta"],
        )
        cnn_layers.append((conv, norm))

    # BiLSTM
    lstm = LSTM(
        W_ih=weights[f"{prefix}.module.lstm.weight_ih_l0"],
        W_hh=weights[f"{prefix}.module.lstm.weight_hh_l0"],
        b_ih=weights[f"{prefix}.module.lstm.bias_ih_l0"],
        b_hh=weights[f"{prefix}.module.lstm.bias_hh_l0"],
        bidirectional=True,
        W_ih_reverse=weights[f"{prefix}.module.lstm.weight_ih_l0_reverse"],
        W_hh_reverse=weights[f"{prefix}.module.lstm.weight_hh_l0_reverse"],
        b_ih_reverse=weights[f"{prefix}.module.lstm.bias_ih_l0_reverse"],
        b_hh_reverse=weights[f"{prefix}.module.lstm.bias_hh_l0_reverse"],
    )

    return KokoroTextEncoder(
        embedding=embedding,
        cnn_layers=cnn_layers,
        lstm=lstm,
    )


__all__ = [
    # Basic layers
    "Linear",
    "LayerNorm",
    "Conv1d",
    "ConvTranspose1d",
    "ResBlock1d",
    "WeightNormConv1d",
    "InstanceNorm1d",
    "AdaIN",
    "AdaINResBlock",
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
    "LSTM",
    "ALBERTLayer",
    "ALBERTEncoder",
    "KokoroTextEncoder",
    # Utilities
    "build_plbert_from_weights",
    "build_albert_from_weights",
    "build_text_encoder_from_weights",
]
