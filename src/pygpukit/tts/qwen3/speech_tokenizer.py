"""Speech Tokenizer (Codec) for Qwen3-TTS.

Implements a neural audio codec that converts audio waveforms to/from
discrete tokens using Residual Vector Quantization (RVQ).

Architecture:
    Encoder: Audio -> RVQ codes (discrete tokens)
    Decoder: RVQ codes -> Transformer -> Upsampler -> Vocoder -> Audio

Reference:
    Based on Qwen3-TTS tokenizer which uses Mimi-style audio codec
    with transformer-based decoder and SnakeBeta vocoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy


@dataclass
class SpeechTokenizerConfig:
    """Configuration for Speech Tokenizer."""

    # Audio settings
    sample_rate: int = 24000
    frame_rate: int = 12  # 12Hz or 25Hz

    # Encoder settings
    encoder_dim: int = 512
    encoder_layers: int = 8

    # Quantization
    num_quantizers: int = 8
    codebook_size: int = 2048
    codebook_dim: int = 256

    # Decoder settings
    latent_dim: int = 1024
    decoder_dim: int = 512
    decoder_layers: int = 8
    decoder_heads: int = 8
    decoder_head_dim: int = 64

    # Upsampling (to go from frame_rate to audio sample rate)
    upsampling_ratios: Sequence[int] = field(default_factory=lambda: [8, 5, 5, 5])
    upsample_rates: Sequence[int] = field(default_factory=lambda: [8, 5, 4, 2])

    # Attention
    sliding_window: int = 512
    rope_theta: float = 10000.0


class SpeechTokenizerOutput(NamedTuple):
    """Output from speech tokenizer encode."""

    audio_codes: list[np.ndarray]  # List of [num_quantizers, seq_len] per batch


class SpeechTokenizerDecodeOutput(NamedTuple):
    """Output from speech tokenizer decode."""

    audio: list[np.ndarray]  # List of audio waveforms [samples]
    sample_rate: int


# =============================================================================
# Basic Layers
# =============================================================================


class RMSNorm:
    """RMS Normalization."""

    def __init__(self, weight: GPUArray, eps: float = 1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        x_np = x.to_numpy()
        rms = np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + self.eps)
        x_norm = x_np / rms
        weight = self.weight.to_numpy()
        return from_numpy((x_norm * weight).astype(np.float32))


class CausalConv1d:
    """Causal 1D convolution with left padding."""

    def __init__(
        self,
        weight: GPUArray,  # [out_channels, in_channels, kernel_size]
        bias: GPUArray | None = None,
        stride: int = 1,
        dilation: int = 1,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = weight.shape[2]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward with causal (left) padding."""
        # Causal padding: pad only on the left
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        padding = effective_kernel - 1

        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()

        batch, in_ch, length = x_np.shape
        out_ch = w_np.shape[0]

        # Left padding only
        x_padded = np.pad(x_np, ((0, 0), (0, 0), (padding, 0)), mode="constant")

        out_length = (length + padding - effective_kernel) // self.stride + 1

        # Convolution
        col = np.zeros((batch, in_ch, self.kernel_size, out_length), dtype=np.float32)
        for i in range(self.kernel_size):
            i_dilated = i * self.dilation
            for j in range(out_length):
                j_strided = j * self.stride
                col[:, :, i, j] = x_padded[:, :, j_strided + i_dilated]

        col = col.reshape(batch, -1, out_length)
        w_reshaped = w_np.reshape(out_ch, -1)
        out_np = np.einsum("bkl,ok->bol", col, w_reshaped)

        if self.bias is not None:
            out_np = out_np + self.bias.to_numpy().reshape(1, -1, 1)

        return from_numpy(out_np.astype(np.float32))


class CausalTransConv1d:
    """Causal Transposed 1D convolution for upsampling."""

    def __init__(
        self,
        weight: GPUArray,  # [in_channels, out_channels, kernel_size]
        bias: GPUArray | None = None,
        stride: int = 1,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.kernel_size = weight.shape[2]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward with causal output."""
        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()

        batch, in_ch, length = x_np.shape
        out_ch = w_np.shape[1]

        # Output length
        out_length = (length - 1) * self.stride + self.kernel_size

        out_np = np.zeros((batch, out_ch, out_length), dtype=np.float32)

        for i in range(length):
            for k in range(self.kernel_size):
                out_pos = i * self.stride + k
                out_np[:, :, out_pos] += np.einsum("bi,io->bo", x_np[:, :, i], w_np[:, :, k])

        # Trim to maintain causality
        trim = self.kernel_size - self.stride
        if trim > 0:
            out_np = out_np[:, :, :-trim]

        if self.bias is not None:
            out_np = out_np + self.bias.to_numpy().reshape(1, -1, 1)

        return from_numpy(out_np.astype(np.float32))


class SnakeBeta:
    """SnakeBeta activation with learnable frequency and magnitude.

    snake(x) = x + (1/b) * sin^2(a * x)
    """

    def __init__(self, alpha: GPUArray, beta: GPUArray):
        self.alpha = alpha  # [channels]
        self.beta = beta  # [channels]

    def __call__(self, x: GPUArray) -> GPUArray:
        x_np = x.to_numpy()
        alpha = self.alpha.to_numpy().reshape(1, -1, 1)
        beta = self.beta.to_numpy().reshape(1, -1, 1)

        # snake(x) = x + (1/beta) * sin^2(alpha * x)
        out = x_np + (1.0 / (beta + 1e-8)) * np.sin(alpha * x_np) ** 2

        return from_numpy(out.astype(np.float32))


# =============================================================================
# Residual Vector Quantization
# =============================================================================


class VectorQuantizer:
    """Single-codebook vector quantizer."""

    def __init__(
        self,
        codebook: GPUArray,  # [codebook_size, codebook_dim]
    ):
        self.codebook = codebook
        self.codebook_size = codebook.shape[0]
        self.codebook_dim = codebook.shape[1]

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Quantize continuous vectors to discrete codes.

        Args:
            x: Input [batch, seq_len, dim]

        Returns:
            Codes [batch, seq_len] (int indices)
        """
        codebook = self.codebook.to_numpy()

        batch, seq_len, dim = x.shape

        # Compute distances to all codebook entries
        x_flat = x.reshape(-1, dim)
        distances = (
            np.sum(x_flat**2, axis=1, keepdims=True)
            - 2 * x_flat @ codebook.T
            + np.sum(codebook**2, axis=1)
        )

        # Find nearest codebook entry
        codes = np.argmin(distances, axis=1)
        return codes.reshape(batch, seq_len)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode discrete codes to continuous vectors.

        Args:
            codes: Indices [batch, seq_len]

        Returns:
            Vectors [batch, seq_len, dim]
        """
        codebook = self.codebook.to_numpy()
        batch, seq_len = codes.shape
        return codebook[codes.flatten()].reshape(batch, seq_len, -1)


class ResidualVectorQuantizer:
    """Residual Vector Quantization with multiple codebooks.

    Encodes residuals sequentially through multiple quantizers.
    """

    def __init__(
        self,
        quantizers: list[VectorQuantizer],
    ):
        self.quantizers = quantizers
        self.num_quantizers = len(quantizers)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode with residual quantization.

        Args:
            x: Input [batch, seq_len, dim]

        Returns:
            Codes [batch, num_quantizers, seq_len]
        """
        batch, seq_len, dim = x.shape
        codes_list = []
        residual = x.copy()

        for quantizer in self.quantizers:
            codes = quantizer.encode(residual)
            codes_list.append(codes)
            quantized = quantizer.decode(codes)
            residual = residual - quantized

        return np.stack(codes_list, axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode by summing all quantized values.

        Args:
            codes: Indices [batch, num_quantizers, seq_len]

        Returns:
            Vectors [batch, seq_len, dim]
        """
        batch, num_q, seq_len = codes.shape
        result: np.ndarray | None = None

        for i, quantizer in enumerate(self.quantizers[:num_q]):
            quantized = quantizer.decode(codes[:, i, :])
            if result is None:
                result = quantized
            else:
                result = result + quantized

        # Should never be None if num_q > 0
        assert result is not None, "No quantizers to decode"
        return result


# =============================================================================
# Decoder Components
# =============================================================================


class DecoderResidualUnit:
    """Residual unit for vocoder decoder."""

    def __init__(
        self,
        conv1: CausalConv1d,
        conv2: CausalConv1d,
        snake1: SnakeBeta,
        snake2: SnakeBeta,
    ):
        self.conv1 = conv1
        self.conv2 = conv2
        self.snake1 = snake1
        self.snake2 = snake2

    def __call__(self, x: GPUArray) -> GPUArray:
        residual = x
        out = self.snake1(x)
        out = self.conv1(out)
        out = self.snake2(out)
        out = self.conv2(out)
        return from_numpy((out.to_numpy() + residual.to_numpy()).astype(np.float32))


class DecoderBlock:
    """Upsampling block for vocoder decoder."""

    def __init__(
        self,
        snake: SnakeBeta,
        upsample: CausalTransConv1d,
        residual_units: list[DecoderResidualUnit],
    ):
        self.snake = snake
        self.upsample = upsample
        self.residual_units = residual_units

    def __call__(self, x: GPUArray) -> GPUArray:
        out = self.snake(x)
        out = self.upsample(out)
        for unit in self.residual_units:
            out = unit(out)
        return out


class ConvNeXtBlock:
    """ConvNeXt-style residual block."""

    def __init__(
        self,
        dwconv: CausalConv1d,  # Depthwise conv
        norm: RMSNorm,
        pwconv1: GPUArray,  # Pointwise weight [dim, 4*dim]
        pwconv2: GPUArray,  # Pointwise weight [4*dim, dim]
        gamma: GPUArray | None = None,  # Layer scale
    ):
        self.dwconv = dwconv
        self.norm = norm
        self.pwconv1 = pwconv1
        self.pwconv2 = pwconv2
        self.gamma = gamma

    def __call__(self, x: GPUArray) -> GPUArray:
        residual = x

        # Depthwise conv
        out = self.dwconv(x)

        # Transpose for norm and MLP: [B, C, T] -> [B, T, C]
        out_np = out.to_numpy().transpose(0, 2, 1)
        out = from_numpy(out_np.astype(np.float32))

        # Norm
        out = self.norm(out)

        # Pointwise MLPs (GELU activation)
        out_np = out.to_numpy()
        pw1 = self.pwconv1.to_numpy()
        pw2 = self.pwconv2.to_numpy()

        hidden = out_np @ pw1
        hidden = (
            hidden * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
        )
        out_np = hidden @ pw2

        # Layer scale
        if self.gamma is not None:
            out_np = out_np * self.gamma.to_numpy()

        # Transpose back: [B, T, C] -> [B, C, T]
        out_np = out_np.transpose(0, 2, 1)

        # Residual
        out_np = out_np + residual.to_numpy()

        return from_numpy(out_np.astype(np.float32))


# =============================================================================
# Main Speech Tokenizer
# =============================================================================


class SpeechTokenizer:
    """Neural audio codec for Qwen3-TTS.

    Converts audio waveforms to/from discrete tokens using:
    - Encoder: Convolutional encoder with RVQ
    - Decoder: Transformer + Upsampler + Vocoder
    """

    def __init__(
        self,
        config: SpeechTokenizerConfig,
        rvq: ResidualVectorQuantizer,
        pre_conv: CausalConv1d,
        decoder_blocks: list[DecoderBlock],
        post_conv: CausalConv1d,
        # Encoder components (for encode)
        encoder_conv: CausalConv1d | None = None,
        encoder_downsample: list[CausalConv1d] | None = None,
    ):
        self.config = config
        self.rvq = rvq
        self.pre_conv = pre_conv
        self.decoder_blocks = decoder_blocks
        self.post_conv = post_conv
        self.encoder_conv = encoder_conv
        self.encoder_downsample = encoder_downsample or []

    def encode(
        self,
        audio: np.ndarray | GPUArray,
        sr: int | None = None,
    ) -> SpeechTokenizerOutput:
        """Encode audio waveform to discrete codes.

        Args:
            audio: Audio waveform [batch, samples] or [samples]
            sr: Sample rate (resamples if different from config)

        Returns:
            SpeechTokenizerOutput with audio_codes list
        """
        if isinstance(audio, GPUArray):
            audio = audio.to_numpy()

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Resample if needed
        if sr is not None and sr != self.config.sample_rate:
            audio = self._resample(audio, sr, self.config.sample_rate)

        batch_size = audio.shape[0]

        # Encode each sample
        codes_list = []
        for b in range(batch_size):
            # Simple encoder: downsample and quantize
            x = audio[b : b + 1, np.newaxis, :]  # [1, 1, samples]
            x = from_numpy(x.astype(np.float32))

            # Encoder convolutions
            if self.encoder_conv is not None:
                x = self.encoder_conv(x)

            for ds_conv in self.encoder_downsample:
                x = ds_conv(x)

            # Transpose for RVQ: [1, channels, time] -> [1, time, channels]
            x_np = x.to_numpy().transpose(0, 2, 1)

            # Quantize
            codes = self.rvq.encode(x_np)  # [1, num_q, time]
            codes_list.append(codes[0])  # [num_q, time]

        return SpeechTokenizerOutput(audio_codes=codes_list)

    def decode(
        self,
        audio_codes: list[dict[str, np.ndarray]] | list[np.ndarray],
    ) -> SpeechTokenizerDecodeOutput:
        """Decode discrete codes to audio waveform.

        Args:
            audio_codes: List of code arrays or dicts with 'audio_codes' key
                Each array: [num_quantizers, seq_len]

        Returns:
            SpeechTokenizerDecodeOutput with audio list
        """
        audio_list = []

        for codes in audio_codes:
            if isinstance(codes, dict):
                codes = codes["audio_codes"]

            if isinstance(codes, GPUArray):
                codes = codes.to_numpy()

            # Ensure shape [1, num_q, seq_len]
            if codes.ndim == 2:
                codes = codes[np.newaxis, :, :]

            # Decode RVQ
            hidden = self.rvq.decode(codes)  # [1, seq_len, dim]

            # Transpose: [1, seq_len, dim] -> [1, dim, seq_len]
            hidden = hidden.transpose(0, 2, 1)
            hidden = from_numpy(hidden.astype(np.float32))

            # Pre-conv
            hidden = self.pre_conv(hidden)

            # Decoder blocks (upsampling + vocoder)
            for block in self.decoder_blocks:
                hidden = block(hidden)

            # Post-conv (final output)
            audio = self.post_conv(hidden)

            # Clamp to [-1, 1]
            audio_np = np.clip(audio.to_numpy(), -1.0, 1.0)
            audio_list.append(audio_np[0, 0])  # Remove batch and channel dims

        return SpeechTokenizerDecodeOutput(
            audio=audio_list,
            sample_rate=self.config.sample_rate,
        )

    def _resample(self, audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if src_sr == tgt_sr:
            return audio

        ratio = tgt_sr / src_sr
        batch, samples = audio.shape
        new_samples = int(samples * ratio)

        # Linear interpolation
        old_idx = np.linspace(0, samples - 1, new_samples)
        old_idx_floor = np.floor(old_idx).astype(int)
        old_idx_ceil = np.minimum(old_idx_floor + 1, samples - 1)
        frac = old_idx - old_idx_floor

        resampled = audio[:, old_idx_floor] * (1 - frac) + audio[:, old_idx_ceil] * frac

        return resampled.astype(np.float32)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: SpeechTokenizerConfig | None = None,
    ) -> SpeechTokenizer:
        """Load speech tokenizer from pretrained weights.

        Args:
            path: Path to model directory
            config: Optional configuration override

        Returns:
            SpeechTokenizer instance
        """
        # TODO: Implement weight loading from safetensors
        raise NotImplementedError("from_pretrained not yet implemented")

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, GPUArray],
        config: SpeechTokenizerConfig | None = None,
        prefix: str = "speech_tokenizer",
    ) -> SpeechTokenizer:
        """Build speech tokenizer from weight dictionary.

        Args:
            weights: Dictionary mapping weight names to GPUArrays
            config: Tokenizer configuration
            prefix: Weight name prefix

        Returns:
            SpeechTokenizer instance
        """
        if config is None:
            config = SpeechTokenizerConfig()

        def get_weight(name: str) -> GPUArray:
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name not in weights:
                raise KeyError(f"Weight '{full_name}' not found")
            return weights[full_name]

        def get_weight_optional(name: str) -> GPUArray | None:
            full_name = f"{prefix}.{name}" if prefix else name
            return weights.get(full_name)

        # Build RVQ
        quantizers = []
        for i in range(config.num_quantizers):
            cb_name = f"quantizer.codebooks.{i}"
            codebook = get_weight(cb_name)
            quantizers.append(VectorQuantizer(codebook))
        rvq = ResidualVectorQuantizer(quantizers)

        # Build pre_conv
        pre_conv = CausalConv1d(
            weight=get_weight("pre_conv.weight"),
            bias=get_weight_optional("pre_conv.bias"),
        )

        # Build decoder blocks
        decoder_blocks = []
        for i, rate in enumerate(config.upsample_rates):
            snake = SnakeBeta(
                alpha=get_weight(f"decoder.{i}.snake.alpha"),
                beta=get_weight(f"decoder.{i}.snake.beta"),
            )
            upsample = CausalTransConv1d(
                weight=get_weight(f"decoder.{i}.upsample.weight"),
                bias=get_weight_optional(f"decoder.{i}.upsample.bias"),
                stride=rate,
            )

            # Residual units
            residual_units = []
            for j in range(3):  # 3 residual units per block
                dilation = [1, 3, 9][j]
                unit = DecoderResidualUnit(
                    conv1=CausalConv1d(
                        weight=get_weight(f"decoder.{i}.residuals.{j}.conv1.weight"),
                        bias=get_weight_optional(f"decoder.{i}.residuals.{j}.conv1.bias"),
                        dilation=dilation,
                    ),
                    conv2=CausalConv1d(
                        weight=get_weight(f"decoder.{i}.residuals.{j}.conv2.weight"),
                        bias=get_weight_optional(f"decoder.{i}.residuals.{j}.conv2.bias"),
                    ),
                    snake1=SnakeBeta(
                        alpha=get_weight(f"decoder.{i}.residuals.{j}.snake1.alpha"),
                        beta=get_weight(f"decoder.{i}.residuals.{j}.snake1.beta"),
                    ),
                    snake2=SnakeBeta(
                        alpha=get_weight(f"decoder.{i}.residuals.{j}.snake2.alpha"),
                        beta=get_weight(f"decoder.{i}.residuals.{j}.snake2.beta"),
                    ),
                )
                residual_units.append(unit)

            decoder_blocks.append(DecoderBlock(snake, upsample, residual_units))

        # Build post_conv
        post_conv = CausalConv1d(
            weight=get_weight("post_conv.weight"),
            bias=get_weight_optional("post_conv.bias"),
        )

        return cls(
            config=config,
            rvq=rvq,
            pre_conv=pre_conv,
            decoder_blocks=decoder_blocks,
            post_conv=post_conv,
        )


__all__ = [
    "SpeechTokenizer",
    "SpeechTokenizerConfig",
    "SpeechTokenizerOutput",
    "SpeechTokenizerDecodeOutput",
    # Quantization
    "VectorQuantizer",
    "ResidualVectorQuantizer",
    # Layers
    "CausalConv1d",
    "CausalTransConv1d",
    "SnakeBeta",
    "RMSNorm",
    "ConvNeXtBlock",
    "DecoderBlock",
    "DecoderResidualUnit",
]
