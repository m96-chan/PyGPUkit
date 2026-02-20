"""Speech Tokenizer Encoder for Qwen3-TTS.

Converts audio waveform to codec tokens for ICL voice cloning.

Architecture:
    Audio [samples]
        -> Conv Encoder (downsample + residual blocks)
        -> Transformer (8 layers)
        -> RVQ (semantic + acoustic quantizers)
    Output: codec codes [num_quantizers, seq_len]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpeechEncoderConfig:
    """Configuration for Speech Encoder."""

    sample_rate: int = 24000
    frame_rate: int = 12  # 12 Hz output
    hidden_dim: int = 512
    transformer_layers: int = 8
    num_heads: int = 8
    semantic_codebooks: int = 1
    acoustic_codebooks: int = 31
    codebook_size: int = 2048
    codebook_dim: int = 256


# =============================================================================
# Basic Layers
# =============================================================================


class Conv1d:
    """1D Convolution with optional dilation."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32) if bias is not None else None
        self.out_channels, self.in_channels, self.kernel_size = weight.shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        batch, in_ch, length = x.shape

        # Effective kernel size with dilation
        eff_kernel = self.dilation * (self.kernel_size - 1) + 1

        # Pad input
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")

        out_length = (x.shape[2] - eff_kernel) // self.stride + 1

        # im2col with dilation
        col = np.zeros((batch, self.in_channels * self.kernel_size, out_length), dtype=np.float32)
        for i in range(self.kernel_size):
            idx = i * self.dilation
            for j in range(out_length):
                j_strided = j * self.stride
                col[:, i * self.in_channels : (i + 1) * self.in_channels, j] = x[
                    :, :, j_strided + idx
                ]

        # matmul
        w_flat = self.weight.reshape(self.out_channels, -1)
        out = np.einsum("oi,bil->bol", w_flat, col)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class LayerNorm:
    """Layer Normalization."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward: normalize over last dim."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class LayerScale:
    """Layer Scale (learnable per-channel scaling)."""

    def __init__(self, scale: np.ndarray):
        self.scale = scale.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along axis."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# =============================================================================
# Encoder Components
# =============================================================================


class ResidualBlock:
    """Residual block with bottleneck structure."""

    def __init__(self, conv1: Conv1d, conv2: Conv1d):
        self.conv1 = conv1
        self.conv2 = conv2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        residual = x
        out = gelu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class SelfAttention:
    """Multi-head self-attention."""

    def __init__(
        self,
        q_proj: np.ndarray,
        k_proj: np.ndarray,
        v_proj: np.ndarray,
        o_proj: np.ndarray,
        num_heads: int = 8,
    ):
        self.q_proj = q_proj.astype(np.float32)
        self.k_proj = k_proj.astype(np.float32)
        self.v_proj = v_proj.astype(np.float32)
        self.o_proj = o_proj.astype(np.float32)
        self.num_heads = num_heads
        self.head_dim = q_proj.shape[0] // num_heads

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward: x [batch, seq_len, hidden_dim]."""
        batch, seq_len, hidden = x.shape

        # Project
        q = x @ self.q_proj.T  # [B, S, H]
        k = x @ self.k_proj.T
        v = x @ self.v_proj.T

        # Reshape for multi-head
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        scale = 1.0 / np.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis=-1)

        # Output
        out = attn @ v  # [B, heads, S, head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)
        out = out @ self.o_proj.T

        return out


class MLP:
    """Feed-forward MLP."""

    def __init__(self, fc1: np.ndarray, fc2: np.ndarray):
        self.fc1 = fc1.astype(np.float32)
        self.fc2 = fc2.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.fc1.T
        out = gelu(out)
        out = out @ self.fc2.T
        return out


class TransformerBlock:
    """Transformer block with layer scale."""

    def __init__(
        self,
        input_norm: LayerNorm,
        attn: SelfAttention,
        attn_scale: LayerScale,
        post_norm: LayerNorm,
        mlp: MLP,
        mlp_scale: LayerScale,
    ):
        self.input_norm = input_norm
        self.attn = attn
        self.attn_scale = attn_scale
        self.post_norm = post_norm
        self.mlp = mlp
        self.mlp_scale = mlp_scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual
        residual = x
        x = self.input_norm(x)
        x = self.attn(x)
        x = self.attn_scale(x)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        x = self.mlp_scale(x)
        x = x + residual

        return x


class VectorQuantizer:
    """Single codebook vector quantizer."""

    def __init__(self, embed_sum: np.ndarray, cluster_usage: np.ndarray):
        # Compute embeddings from EMA statistics
        usage = cluster_usage[:, None] + 1e-6
        self.embeddings = (embed_sum / usage).astype(np.float32)
        self.codebook_size, self.dim = self.embeddings.shape

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Quantize to nearest codebook entry.

        Args:
            x: [batch, seq_len, dim]

        Returns:
            codes: [batch, seq_len]
        """
        batch, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # L2 distance: ||x - e||^2 = ||x||^2 - 2*x*e + ||e||^2
        x_sq = (x_flat**2).sum(axis=1, keepdims=True)
        e_sq = (self.embeddings**2).sum(axis=1)
        dist = x_sq - 2 * x_flat @ self.embeddings.T + e_sq

        codes = np.argmin(dist, axis=1)
        return codes.reshape(batch, seq_len)


class ResidualVectorQuantizer:
    """RVQ with multiple codebook layers."""

    def __init__(
        self,
        input_proj: np.ndarray,
        output_proj: np.ndarray,
        quantizers: list[VectorQuantizer],
    ):
        self.input_proj = input_proj.astype(np.float32)
        self.output_proj = output_proj.astype(np.float32)
        self.quantizers = quantizers
        self.num_quantizers = len(quantizers)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode with residual quantization.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            codes: [batch, num_quantizers, seq_len]
        """
        batch, seq_len, _ = x.shape

        # Input projection (1x1 conv)
        x_proj = np.einsum("bsh,oh->bso", x, self.input_proj[:, :, 0])

        codes_list = []
        residual = x_proj.copy()

        for quantizer in self.quantizers:
            codes = quantizer.encode(residual)
            codes_list.append(codes)

            # Subtract quantized from residual
            quantized = quantizer.embeddings[codes]  # [B, S, dim]
            residual = residual - quantized

        return np.stack(codes_list, axis=1)  # [B, num_q, S]


# =============================================================================
# Main Speech Encoder
# =============================================================================


class SpeechEncoder:
    """Speech Tokenizer Encoder.

    Converts audio waveform to codec codes for ICL voice cloning.
    """

    def __init__(
        self,
        config: SpeechEncoderConfig,
        initial_conv: Conv1d,
        residual_blocks: list[ResidualBlock],
        downsample_convs: list[Conv1d],
        final_conv: Conv1d,
        downsample_final: Conv1d,
        downsample_extra: Conv1d,
        transformer_blocks: list[TransformerBlock],
        semantic_rvq: ResidualVectorQuantizer,
        acoustic_rvq: ResidualVectorQuantizer,
    ):
        self.config = config
        self.initial_conv = initial_conv
        self.residual_blocks = residual_blocks
        self.downsample_convs = downsample_convs
        self.final_conv = final_conv
        self.downsample_final = downsample_final
        self.downsample_extra = downsample_extra
        self.transformer_blocks = transformer_blocks
        self.semantic_rvq = semantic_rvq
        self.acoustic_rvq = acoustic_rvq

    def encode(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """Encode audio to codec codes.

        Args:
            audio: Audio waveform [samples] or [batch, samples]
            sample_rate: Audio sample rate

        Returns:
            Codec codes [num_quantizers, seq_len] or [batch, num_quantizers, seq_len]
        """
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[np.newaxis, :]

        # Add channel dim: [B, 1, samples]
        x = audio[:, np.newaxis, :].astype(np.float32)

        # Convolutional encoder
        x = self.initial_conv(x)  # [B, 64, T]

        # Residual blocks and downsampling
        # Structure: res_block -> downsample (for each downsample_conv)
        res_idx = 0
        for ds_conv in self.downsample_convs:
            x = self.residual_blocks[res_idx](x)
            res_idx += 1
            x = gelu(ds_conv(x))

        # Final residual block
        x = self.residual_blocks[res_idx](x)

        # Final conv (stride=4)
        x = gelu(self.final_conv(x))

        # Final layer to get 512 channels
        x = self.downsample_final(x)

        # Extra downsample (compress=2)
        x = self.downsample_extra(x)

        # Transpose for transformer: [B, C, T] -> [B, T, C]
        x = x.transpose(0, 2, 1)

        # Transformer
        for block in self.transformer_blocks:
            x = block(x)

        # Quantize
        semantic_codes = self.semantic_rvq.encode(x)  # [B, 1, T]
        acoustic_codes = self.acoustic_rvq.encode(x)  # [B, 31, T]

        # Concatenate
        codes = np.concatenate([semantic_codes, acoustic_codes], axis=1)  # [B, 32, T]

        if squeeze:
            codes = codes[0]

        return codes


# =============================================================================
# Model Loading
# =============================================================================


def load_speech_encoder(
    model_path: str | Path,
    config: SpeechEncoderConfig | None = None,
) -> SpeechEncoder:
    """Load speech encoder from Qwen3-TTS model.

    Args:
        model_path: Path to speech_tokenizer directory or model directory
        config: Optional config override

    Returns:
        SpeechEncoder instance
    """
    import torch
    from safetensors import safe_open

    if config is None:
        config = SpeechEncoderConfig()

    path = Path(model_path)
    if (path / "speech_tokenizer").exists():
        path = path / "speech_tokenizer"

    weights_path = path / "model.safetensors"

    weights: dict[str, np.ndarray] = {}
    with safe_open(str(weights_path), framework="pt") as f:
        for key in f.keys():
            if key.startswith("encoder."):
                tensor = f.get_tensor(key)
                name = key[len("encoder.") :]
                if tensor.dtype == torch.bfloat16:
                    weights[name] = tensor.float().cpu().numpy()
                else:
                    weights[name] = tensor.cpu().numpy()

    if not weights:
        raise ValueError(f"No encoder weights found in {weights_path}")

    def get_weight(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(f"Weight '{name}' not found in encoder")
        return weights[name]

    def get_bias(name: str) -> np.ndarray | None:
        return weights.get(name)

    def build_conv(prefix: str, stride: int = 1, padding: int = 0) -> Conv1d:
        return Conv1d(
            weight=get_weight(f"{prefix}.weight"),
            bias=get_bias(f"{prefix}.bias"),
            stride=stride,
            padding=padding,
        )

    def build_layer_norm(prefix: str) -> LayerNorm:
        return LayerNorm(
            weight=get_weight(f"{prefix}.weight"),
            bias=get_weight(f"{prefix}.bias"),
        )

    def build_layer_scale(prefix: str) -> LayerScale:
        return LayerScale(scale=get_weight(f"{prefix}.scale"))

    def build_attention(prefix: str) -> SelfAttention:
        return SelfAttention(
            q_proj=get_weight(f"{prefix}.q_proj.weight"),
            k_proj=get_weight(f"{prefix}.k_proj.weight"),
            v_proj=get_weight(f"{prefix}.v_proj.weight"),
            o_proj=get_weight(f"{prefix}.o_proj.weight"),
            num_heads=config.num_heads,
        )

    def build_mlp(prefix: str) -> MLP:
        return MLP(
            fc1=get_weight(f"{prefix}.fc1.weight"),
            fc2=get_weight(f"{prefix}.fc2.weight"),
        )

    def build_transformer_block(prefix: str) -> TransformerBlock:
        return TransformerBlock(
            input_norm=build_layer_norm(f"{prefix}.input_layernorm"),
            attn=build_attention(f"{prefix}.self_attn"),
            attn_scale=build_layer_scale(f"{prefix}.self_attn_layer_scale"),
            post_norm=build_layer_norm(f"{prefix}.post_attention_layernorm"),
            mlp=build_mlp(f"{prefix}.mlp"),
            mlp_scale=build_layer_scale(f"{prefix}.mlp_layer_scale"),
        )

    def build_quantizer(prefix: str, num_layers: int) -> ResidualVectorQuantizer:
        quantizers = []
        for i in range(num_layers):
            vq = VectorQuantizer(
                embed_sum=get_weight(f"{prefix}.layers.{i}.codebook.embed_sum"),
                cluster_usage=get_weight(f"{prefix}.layers.{i}.codebook.cluster_usage"),
            )
            quantizers.append(vq)

        return ResidualVectorQuantizer(
            input_proj=get_weight(f"{prefix}.input_proj.weight"),
            output_proj=get_weight(f"{prefix}.output_proj.weight"),
            quantizers=quantizers,
        )

    # Build model
    # Upsampling ratios for encoder (actually downsample): [8, 6, 5, 4] + compress=2
    # Total: 8 * 6 * 5 * 4 * 2 = 1920x downsample

    # Initial conv: 1 -> 64, kernel=7, padding=3
    initial_conv = build_conv("encoder.layers.0.conv", padding=3)

    # Residual blocks and downsampling
    # Structure: res_block -> downsample -> res_block -> downsample -> ...
    residual_blocks = [
        ResidualBlock(
            conv1=build_conv("encoder.layers.1.block.1.conv", padding=1),
            conv2=build_conv("encoder.layers.1.block.3.conv"),
        ),
        ResidualBlock(
            conv1=build_conv("encoder.layers.4.block.1.conv", padding=1),
            conv2=build_conv("encoder.layers.4.block.3.conv"),
        ),
        ResidualBlock(
            conv1=build_conv("encoder.layers.7.block.1.conv", padding=1),
            conv2=build_conv("encoder.layers.7.block.3.conv"),
        ),
        ResidualBlock(
            conv1=build_conv("encoder.layers.10.block.1.conv", padding=1),
            conv2=build_conv("encoder.layers.10.block.3.conv"),
        ),
    ]

    # Downsample convolutions with correct strides: [8, 6, 5, 4]
    downsample_convs = [
        build_conv("encoder.layers.3.conv", stride=8, padding=0),   # 64 -> 128, 8x
        build_conv("encoder.layers.6.conv", stride=6, padding=0),   # 128 -> 256, 6x
        build_conv("encoder.layers.9.conv", stride=5, padding=0),   # 256 -> 512, 5x
    ]

    # Final convolutions: stride=4
    final_conv = build_conv("encoder.layers.12.conv", stride=4, padding=0)  # 512 -> 1024, 4x
    downsample_final = build_conv("encoder.layers.14.conv", padding=1)  # 1024 -> 512

    # Additional downsample for compress=2
    downsample_extra = build_conv("downsample.conv", stride=2, padding=0)  # 2x

    # Transformer blocks
    transformer_blocks = [
        build_transformer_block(f"encoder_transformer.layers.{i}")
        for i in range(config.transformer_layers)
    ]

    # RVQ
    semantic_rvq = build_quantizer(
        "quantizer.semantic_residual_vector_quantizer",
        config.semantic_codebooks,
    )
    acoustic_rvq = build_quantizer(
        "quantizer.acoustic_residual_vector_quantizer",
        config.acoustic_codebooks,
    )

    return SpeechEncoder(
        config=config,
        initial_conv=initial_conv,
        residual_blocks=residual_blocks,
        downsample_convs=downsample_convs,
        final_conv=final_conv,
        downsample_final=downsample_final,
        downsample_extra=downsample_extra,
        transformer_blocks=transformer_blocks,
        semantic_rvq=semantic_rvq,
        acoustic_rvq=acoustic_rvq,
    )


__all__ = [
    "SpeechEncoder",
    "SpeechEncoderConfig",
    "load_speech_encoder",
]
