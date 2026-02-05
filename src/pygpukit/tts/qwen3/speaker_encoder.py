"""ECAPA-TDNN Speaker Encoder for Qwen3-TTS.

Implements the Emphasized Channel Attention, Propagation and Aggregation
Time Delay Neural Network for speaker embedding extraction.

Architecture:
    Input (mel spectrogram) [batch, num_mels, time]
        -> TDNN layer
        -> SE-Res2Net blocks (with skip connections)
        -> Multi-layer Feature Aggregation
        -> Attentive Statistics Pooling
        -> Final FC projection
    Output (speaker embedding) [batch, embed_dim]

Reference:
    ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
    in TDNN Based Speaker Verification (https://arxiv.org/abs/2005.07143)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy


@dataclass
class SpeakerEncoderConfig:
    """Configuration for ECAPA-TDNN Speaker Encoder."""

    # Input
    num_mels: int = 128
    sample_rate: int = 24000

    # TDNN channels
    channels: Sequence[int] = (512, 512, 512, 512, 1536)

    # SE-Res2Net
    res2net_scale: int = 8
    se_reduction: int = 128

    # Output
    embed_dim: int = 192

    # Kernel sizes
    kernel_sizes: Sequence[int] = (5, 3, 3, 3, 1)
    dilations: Sequence[int] = (1, 2, 3, 4, 1)


# =============================================================================
# Basic Layers
# =============================================================================


class BatchNorm1d:
    """1D Batch Normalization."""

    def __init__(
        self,
        weight: GPUArray,  # [channels]
        bias: GPUArray,  # [channels]
        running_mean: GPUArray,  # [channels]
        running_var: GPUArray,  # [channels]
        eps: float = 1e-5,
    ):
        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass (inference mode - uses running stats)."""
        x_np = x.to_numpy()  # [batch, channels, length]

        mean = self.running_mean.to_numpy().reshape(1, -1, 1)
        var = self.running_var.to_numpy().reshape(1, -1, 1)
        gamma = self.weight.to_numpy().reshape(1, -1, 1)
        beta = self.bias.to_numpy().reshape(1, -1, 1)

        x_norm = (x_np - mean) / np.sqrt(var + self.eps)
        out = gamma * x_norm + beta

        return from_numpy(out.astype(np.float32))


class Conv1d:
    """1D Convolution layer."""

    def __init__(
        self,
        weight: GPUArray,  # [out_channels, in_channels, kernel_size]
        bias: GPUArray | None = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.out_channels = weight.shape[0]
        self.in_channels = weight.shape[1] * groups
        self.kernel_size = weight.shape[2]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass."""
        batch_size = x.shape[0]
        length = x.shape[2]

        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        out_length = (length + 2 * self.padding - effective_kernel) // self.stride + 1

        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()

        # Pad input
        if self.padding > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")

        if self.groups == 1:
            # Standard convolution
            col = np.zeros(
                (batch_size, self.in_channels, self.kernel_size, out_length), dtype=np.float32
            )
            for i in range(self.kernel_size):
                i_dilated = i * self.dilation
                for j in range(out_length):
                    j_strided = j * self.stride
                    col[:, :, i, j] = x_np[:, :, j_strided + i_dilated]

            col = col.reshape(batch_size, -1, out_length)
            w_reshaped = w_np.reshape(self.out_channels, -1)
            out_np = np.einsum("bkl,ok->bol", col, w_reshaped)
        else:
            # Grouped convolution
            in_channels_per_group = self.in_channels // self.groups
            out_channels_per_group = self.out_channels // self.groups

            out_np = np.zeros((batch_size, self.out_channels, out_length), dtype=np.float32)

            for g in range(self.groups):
                in_start = g * in_channels_per_group
                in_end = in_start + in_channels_per_group
                out_start = g * out_channels_per_group
                out_end = out_start + out_channels_per_group

                x_group = x_np[:, in_start:in_end, :]
                w_group = w_np[out_start:out_end, :, :]

                col = np.zeros(
                    (batch_size, in_channels_per_group, self.kernel_size, out_length),
                    dtype=np.float32,
                )
                for i in range(self.kernel_size):
                    i_dilated = i * self.dilation
                    for j in range(out_length):
                        j_strided = j * self.stride
                        col[:, :, i, j] = x_group[:, :, j_strided + i_dilated]

                col = col.reshape(batch_size, -1, out_length)
                w_reshaped = w_group.reshape(out_channels_per_group, -1)
                out_np[:, out_start:out_end, :] = np.einsum("bkl,ok->bol", col, w_reshaped)

        if self.bias is not None:
            bias_np = self.bias.to_numpy()
            out_np = out_np + bias_np.reshape(1, -1, 1)

        return from_numpy(out_np.astype(np.float32))


class Linear:
    """Linear layer."""

    def __init__(self, weight: GPUArray, bias: GPUArray | None = None):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: y = xW^T + b."""
        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()

        out = x_np @ w_np.T

        if self.bias is not None:
            out = out + self.bias.to_numpy()

        return from_numpy(out.astype(np.float32))


def relu(x: GPUArray) -> GPUArray:
    """ReLU activation."""
    x_np = x.to_numpy()
    return from_numpy(np.maximum(x_np, 0).astype(np.float32))


def sigmoid(x: GPUArray) -> GPUArray:
    """Sigmoid activation."""
    x_np = x.to_numpy()
    return from_numpy((1.0 / (1.0 + np.exp(-x_np))).astype(np.float32))


# =============================================================================
# ECAPA-TDNN Components
# =============================================================================


class TDNNBlock:
    """Time Delay Neural Network block.

    Conv1d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        conv: Conv1d,
        bn: BatchNorm1d,
    ):
        self.conv = conv
        self.bn = bn

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = relu(x)
        return x


class SEBlock:
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(
        self,
        fc1: Linear,  # [reduction, channels]
        fc2: Linear,  # [channels, reduction]
    ):
        self.fc1 = fc1
        self.fc2 = fc2

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: channel-wise attention."""
        # x: [batch, channels, length]
        x_np = x.to_numpy()

        # Global average pooling
        s = x_np.mean(axis=2)  # [batch, channels]
        s = from_numpy(s.astype(np.float32))

        # FC -> ReLU -> FC -> Sigmoid
        s = relu(self.fc1(s))
        s = sigmoid(self.fc2(s))

        # Scale channels
        s_np = s.to_numpy()[:, :, np.newaxis]
        out = x_np * s_np

        return from_numpy(out.astype(np.float32))


class Res2NetBlock:
    """Res2Net-style multi-scale feature extraction.

    Splits input channels into `scale` groups, processes each group
    with a 3x3 conv, and adds outputs hierarchically.
    """

    def __init__(
        self,
        convs: list[Conv1d],  # [scale-1] convolutions
        scale: int = 8,
    ):
        self.convs = convs
        self.scale = scale

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass with hierarchical residual connections."""
        x_np = x.to_numpy()
        batch, channels, length = x_np.shape
        # width = channels // self.scale (implicit in split)

        # Split into groups
        spx = np.split(x_np, self.scale, axis=1)

        outputs = []
        sp = None

        for i in range(self.scale):
            if i == 0:
                # First group: pass through
                sp = spx[i]
            elif i == 1:
                # Second group: conv only
                sp = self.convs[i - 1](from_numpy(spx[i].astype(np.float32))).to_numpy()
            else:
                # Other groups: add previous output, then conv
                sp = spx[i] + sp
                sp = self.convs[i - 1](from_numpy(sp.astype(np.float32))).to_numpy()

            outputs.append(sp)

        # Concatenate all groups
        out = np.concatenate(outputs, axis=1)
        return from_numpy(out.astype(np.float32))


class SERes2NetBlock:
    """SE-Res2Net block for ECAPA-TDNN.

    TDNNBlock -> Res2Net -> Conv1d -> SE -> Residual
    """

    def __init__(
        self,
        tdnn: TDNNBlock,
        res2net: Res2NetBlock,
        conv: Conv1d,
        bn: BatchNorm1d,
        se: SEBlock,
        shortcut: Conv1d | None = None,  # For channel mismatch
    ):
        self.tdnn = tdnn
        self.res2net = res2net
        self.conv = conv
        self.bn = bn
        self.se = se
        self.shortcut = shortcut

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass with residual connection."""
        residual = x

        # TDNN
        out = self.tdnn(x)

        # Res2Net
        out = self.res2net(out)

        # Conv + BN
        out = self.conv(out)
        out = self.bn(out)

        # SE attention
        out = self.se(out)

        # Residual
        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out_np = out.to_numpy() + residual.to_numpy()
        out = relu(from_numpy(out_np.astype(np.float32)))

        return out


class AttentiveStatisticsPooling:
    """Attentive Statistics Pooling for speaker embedding.

    Computes attention-weighted mean and standard deviation
    across the time dimension.
    """

    def __init__(
        self,
        attention_conv: Conv1d,
        attention_bn: BatchNorm1d,
        attention_fc: Linear,
    ):
        self.attention_conv = attention_conv
        self.attention_bn = attention_bn
        self.attention_fc = attention_fc

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: compute attention-weighted statistics.

        Args:
            x: Input [batch, channels, length]

        Returns:
            Concatenated mean and std [batch, 2 * channels]
        """
        x_np = x.to_numpy()
        batch, channels, length = x_np.shape

        # Compute attention weights
        # Conv -> Tanh -> Conv
        attn = self.attention_conv(x)
        attn = self.attention_bn(attn)
        attn_np = np.tanh(attn.to_numpy())

        # Global pooling for FC input
        attn_pooled = attn_np.mean(axis=2)
        attn_pooled = from_numpy(attn_pooled.astype(np.float32))
        attn_weights = self.attention_fc(attn_pooled)

        # Softmax over time
        attn_weights_np = attn_weights.to_numpy()
        attn_weights_np = attn_weights_np[:, :, np.newaxis]
        attn_weights_np = np.broadcast_to(attn_weights_np, (batch, channels, length))

        # Normalize
        attn_exp = np.exp(attn_weights_np - attn_weights_np.max(axis=2, keepdims=True))
        attn_softmax = attn_exp / attn_exp.sum(axis=2, keepdims=True)

        # Weighted mean
        mean = (x_np * attn_softmax).sum(axis=2)

        # Weighted std
        var = ((x_np - mean[:, :, np.newaxis]) ** 2 * attn_softmax).sum(axis=2)
        std = np.sqrt(var + 1e-8)

        # Concatenate mean and std
        out = np.concatenate([mean, std], axis=1)

        return from_numpy(out.astype(np.float32))


# =============================================================================
# Main Speaker Encoder
# =============================================================================


class SpeakerEncoder:
    """ECAPA-TDNN Speaker Encoder.

    Extracts speaker embeddings from mel spectrograms.

    Input: mel spectrogram [batch, num_mels, time]
    Output: speaker embedding [batch, embed_dim]
    """

    def __init__(
        self,
        config: SpeakerEncoderConfig,
        initial_tdnn: TDNNBlock,
        se_res2net_blocks: list[SERes2NetBlock],
        mfa_conv: Conv1d,
        asp: AttentiveStatisticsPooling,
        final_bn: BatchNorm1d,
        final_fc: Linear,
    ):
        self.config = config
        self.initial_tdnn = initial_tdnn
        self.se_res2net_blocks = se_res2net_blocks
        self.mfa_conv = mfa_conv
        self.asp = asp
        self.final_bn = final_bn
        self.final_fc = final_fc

    def __call__(self, x: GPUArray) -> GPUArray:
        """Extract speaker embedding.

        Args:
            x: Mel spectrogram [batch, num_mels, time]

        Returns:
            Speaker embedding [batch, embed_dim]
        """
        # Initial TDNN
        out = self.initial_tdnn(x)

        # SE-Res2Net blocks with skip connections
        outputs = [out]
        for block in self.se_res2net_blocks:
            out = block(out)
            outputs.append(out)

        # Multi-layer Feature Aggregation
        # Concatenate all outputs along channel dimension
        mfa_input = np.concatenate([o.to_numpy() for o in outputs], axis=1)
        mfa_input = from_numpy(mfa_input.astype(np.float32))
        out = self.mfa_conv(mfa_input)

        # Attentive Statistics Pooling
        out = self.asp(out)

        # Final BN + FC
        out = self.final_bn(from_numpy(out.to_numpy().reshape(out.shape[0], -1, 1)))
        out_np = out.to_numpy().squeeze(-1)
        out = self.final_fc(from_numpy(out_np.astype(np.float32)))

        return out

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, GPUArray],
        config: SpeakerEncoderConfig | None = None,
        prefix: str = "speaker_encoder",
    ) -> SpeakerEncoder:
        """Build speaker encoder from weight dictionary.

        Args:
            weights: Dictionary mapping weight names to GPUArrays
            config: Encoder configuration (uses default if None)
            prefix: Weight name prefix

        Returns:
            SpeakerEncoder instance
        """
        if config is None:
            config = SpeakerEncoderConfig()

        def get_weight(name: str) -> GPUArray:
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name not in weights:
                raise KeyError(f"Weight '{full_name}' not found")
            return weights[full_name]

        def get_weight_optional(name: str) -> GPUArray | None:
            full_name = f"{prefix}.{name}" if prefix else name
            return weights.get(full_name)

        def build_conv(name: str, **kwargs) -> Conv1d:
            return Conv1d(
                weight=get_weight(f"{name}.weight"),
                bias=get_weight_optional(f"{name}.bias"),
                **kwargs,
            )

        def build_bn(name: str) -> BatchNorm1d:
            return BatchNorm1d(
                weight=get_weight(f"{name}.weight"),
                bias=get_weight(f"{name}.bias"),
                running_mean=get_weight(f"{name}.running_mean"),
                running_var=get_weight(f"{name}.running_var"),
            )

        def build_linear(name: str) -> Linear:
            return Linear(
                weight=get_weight(f"{name}.weight"),
                bias=get_weight_optional(f"{name}.bias"),
            )

        def build_tdnn(name: str, **kwargs) -> TDNNBlock:
            return TDNNBlock(
                conv=build_conv(f"{name}.conv", **kwargs),
                bn=build_bn(f"{name}.bn"),
            )

        def build_se(name: str) -> SEBlock:
            return SEBlock(
                fc1=build_linear(f"{name}.fc1"),
                fc2=build_linear(f"{name}.fc2"),
            )

        def build_res2net(name: str, scale: int) -> Res2NetBlock:
            convs = []
            for i in range(1, scale):
                conv_name = f"{name}.convs.{i - 1}"
                if f"{prefix}.{conv_name}.weight" in weights:
                    convs.append(build_conv(conv_name, padding=1))
            return Res2NetBlock(convs=convs, scale=scale)

        def build_se_res2net(name: str, scale: int) -> SERes2NetBlock:
            shortcut = None
            shortcut_name = f"{name}.shortcut"
            if f"{prefix}.{shortcut_name}.weight" in weights:
                shortcut = build_conv(shortcut_name)

            return SERes2NetBlock(
                tdnn=build_tdnn(f"{name}.tdnn", padding=1),
                res2net=build_res2net(f"{name}.res2net", scale),
                conv=build_conv(f"{name}.conv"),
                bn=build_bn(f"{name}.bn"),
                se=build_se(f"{name}.se"),
                shortcut=shortcut,
            )

        # Build model
        initial_tdnn = build_tdnn("blocks.0", padding=2, dilation=1)

        se_res2net_blocks = []
        for i in range(1, len(config.channels) - 1):
            block = build_se_res2net(f"blocks.{i}", config.res2net_scale)
            se_res2net_blocks.append(block)

        mfa_conv = build_conv("mfa")

        asp = AttentiveStatisticsPooling(
            attention_conv=build_conv("asp.attention_conv", padding=0),
            attention_bn=build_bn("asp.attention_bn"),
            attention_fc=build_linear("asp.attention_fc"),
        )

        final_bn = build_bn("fc_bn")
        final_fc = build_linear("fc")

        return cls(
            config=config,
            initial_tdnn=initial_tdnn,
            se_res2net_blocks=se_res2net_blocks,
            mfa_conv=mfa_conv,
            asp=asp,
            final_bn=final_bn,
            final_fc=final_fc,
        )


# =============================================================================
# Mel Spectrogram Computation
# =============================================================================


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 12000.0,
) -> np.ndarray:
    """Compute mel spectrogram from audio waveform.

    Args:
        audio: Audio waveform [samples] or [batch, samples]
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window size
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel spectrogram [batch, n_mels, time] or [n_mels, time]
    """
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    batch_size, audio_len = audio.shape
    num_frames = (audio_len - n_fft) // hop_length + 1

    # Create mel filterbank
    mel_basis = _create_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)

    # Hann window
    window = np.hanning(win_length).astype(np.float32)

    mels = []
    for b in range(batch_size):
        # STFT
        frames = []
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[b, start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frame = frame * window
            spectrum = np.fft.rfft(frame)
            frames.append(np.abs(spectrum) ** 2)

        power_spec = np.stack(frames, axis=1)  # [n_fft//2+1, time]

        # Apply mel filterbank
        mel_spec = mel_basis @ power_spec  # [n_mels, time]

        # Log scale
        mel_spec = np.log(np.maximum(mel_spec, 1e-10))

        mels.append(mel_spec)

    result = np.stack(mels, axis=0)  # [batch, n_mels, time]

    if squeeze:
        result = result[0]

    return result.astype(np.float32)


def _create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Create mel filterbank matrix."""

    # Convert Hz to mel
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq), dtype=np.float32)

    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        # Left slope
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)

        # Right slope
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank


__all__ = [
    "SpeakerEncoder",
    "SpeakerEncoderConfig",
    "compute_mel_spectrogram",
    # Layers (for potential reuse)
    "TDNNBlock",
    "SEBlock",
    "Res2NetBlock",
    "SERes2NetBlock",
    "AttentiveStatisticsPooling",
    "BatchNorm1d",
    "Conv1d",
    "Linear",
]
