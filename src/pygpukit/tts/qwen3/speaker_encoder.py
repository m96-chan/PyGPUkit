"""ECAPA-TDNN Speaker Encoder for Qwen3-TTS.

Extracts speaker embeddings from mel spectrograms for voice cloning.

Architecture (Qwen3-TTS-Base):
    Input (mel spectrogram) [batch, num_mels, time]
        -> Initial TDNN (128 -> 512)
        -> 3x SE-Res2Net blocks (512 channels)
        -> Multi-layer Feature Aggregation (1536 -> 1536)
        -> Attentive Statistics Pooling (-> 3072)
        -> Final FC projection (3072 -> 2048)
    Output (speaker embedding) [batch, 2048]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pygpukit.core.array import GPUArray


@dataclass
class SpeakerEncoderConfig:
    """Configuration for ECAPA-TDNN Speaker Encoder."""

    num_mels: int = 128
    sample_rate: int = 24000
    channels: int = 512
    res2net_scale: int = 8  # 8 groups, 7 convs
    se_reduction: int = 128
    embed_dim: int = 2048


# =============================================================================
# Basic Layers (No BatchNorm in Qwen3-TTS-Base)
# =============================================================================


class Conv1d:
    """1D Convolution layer."""

    def __init__(
        self,
        weight: np.ndarray,  # [out_channels, in_channels, kernel_size]
        bias: np.ndarray | None = None,
        padding: int = 0,
    ):
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32) if bias is not None else None
        self.out_channels, self.in_channels, self.kernel_size = weight.shape
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        batch, in_ch, length = x.shape

        # Pad input
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")

        out_length = x.shape[2] - self.kernel_size + 1

        # im2col convolution
        col = np.zeros((batch, self.in_channels * self.kernel_size, out_length), dtype=np.float32)
        for i in range(self.kernel_size):
            col[:, i * self.in_channels : (i + 1) * self.in_channels, :] = x[:, :, i : i + out_length]

        # matmul
        w_flat = self.weight.reshape(self.out_channels, -1)
        out = np.einsum("oi,bil->bol", w_flat, col)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(x, 0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# =============================================================================
# ECAPA-TDNN Components
# =============================================================================


class Res2NetBlock:
    """Res2Net multi-scale feature extraction.

    Splits into `scale` groups, processes with convs hierarchically.
    """

    def __init__(self, convs: list[Conv1d], scale: int = 8):
        self.convs = convs  # 7 convs for scale=8
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward with hierarchical residual."""
        batch, channels, length = x.shape
        width = channels // self.scale

        # Split into groups
        splits = [x[:, i * width : (i + 1) * width, :] for i in range(self.scale)]

        outputs = []
        sp = None

        for i in range(self.scale):
            if i == 0:
                sp = splits[i]
            elif i == 1:
                sp = relu(self.convs[i - 1](splits[i]))
            else:
                sp = relu(self.convs[i - 1](splits[i] + sp))
            outputs.append(sp)

        return np.concatenate(outputs, axis=1)


class SEBlock:
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, conv1: Conv1d, conv2: Conv1d):
        self.conv1 = conv1
        self.conv2 = conv2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward: global pool -> FC -> ReLU -> FC -> Sigmoid -> scale."""
        # Global average pooling
        s = x.mean(axis=2, keepdims=True)  # [B, C, 1]

        # FC layers (as 1x1 conv)
        s = relu(self.conv1(s))
        s = sigmoid(self.conv2(s))

        return x * s


class SERes2NetBlock:
    """SE-Res2Net block for ECAPA-TDNN.

    Structure: TDNN1 -> Res2Net -> TDNN2 -> SE -> Residual
    """

    def __init__(
        self,
        tdnn1: Conv1d,
        res2net: Res2NetBlock,
        tdnn2: Conv1d,
        se: SEBlock,
    ):
        self.tdnn1 = tdnn1
        self.res2net = res2net
        self.tdnn2 = tdnn2
        self.se = se

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward with residual."""
        residual = x

        out = relu(self.tdnn1(x))
        out = self.res2net(out)
        out = relu(self.tdnn2(out))
        out = self.se(out)

        return relu(out + residual)


class AttentiveStatisticsPooling:
    """Attentive Statistics Pooling.

    Computes attention-weighted mean and std across time.
    """

    def __init__(self, tdnn: Conv1d, conv: Conv1d):
        self.tdnn = tdnn  # [4608, 128]
        self.conv = conv  # [128, 1536]

    def __call__(
        self,
        attn_input: np.ndarray,
        stats_input: np.ndarray,
    ) -> np.ndarray:
        """Forward: attention weights from attn_input, stats from stats_input.

        Args:
            attn_input: Features for attention computation [batch, 4608, time]
            stats_input: Features to pool [batch, 1536, time]

        Returns:
            [batch, 3072] (mean + std concatenated)
        """
        # Compute attention weights from concatenated features
        attn = relu(self.tdnn(attn_input))  # [B, 128, T]
        attn = self.conv(attn)  # [B, 1536, T]

        # Softmax over time
        attn_max = attn.max(axis=2, keepdims=True)
        attn_exp = np.exp(attn - attn_max)
        attn_weights = attn_exp / (attn_exp.sum(axis=2, keepdims=True) + 1e-8)

        # Weighted mean of stats_input
        mean = (stats_input * attn_weights).sum(axis=2)  # [B, 1536]

        # Weighted std
        var = ((stats_input - mean[:, :, np.newaxis]) ** 2 * attn_weights).sum(axis=2)
        std = np.sqrt(var + 1e-8)

        # Concatenate
        return np.concatenate([mean, std], axis=1)  # [B, 3072]


# =============================================================================
# Main Speaker Encoder
# =============================================================================


class SpeakerEncoder:
    """ECAPA-TDNN Speaker Encoder for voice cloning.

    Extracts speaker embeddings from mel spectrograms.
    """

    def __init__(
        self,
        config: SpeakerEncoderConfig,
        initial_conv: Conv1d,
        se_res2net_blocks: list[SERes2NetBlock],
        mfa_conv: Conv1d,
        asp: AttentiveStatisticsPooling,
        fc: Conv1d,
    ):
        self.config = config
        self.initial_conv = initial_conv
        self.se_res2net_blocks = se_res2net_blocks
        self.mfa_conv = mfa_conv
        self.asp = asp
        self.fc = fc

    def __call__(self, mel: np.ndarray | GPUArray) -> np.ndarray:
        """Extract speaker embedding from mel spectrogram.

        Args:
            mel: Mel spectrogram [batch, num_mels, time] or [num_mels, time]

        Returns:
            Speaker embedding [batch, embed_dim] or [embed_dim]
        """
        if isinstance(mel, GPUArray):
            mel = mel.to_numpy()

        squeeze = mel.ndim == 2
        if squeeze:
            mel = mel[np.newaxis, :, :]

        x = mel.astype(np.float32)

        # Initial TDNN
        x = relu(self.initial_conv(x))  # [B, 512, T]

        # SE-Res2Net blocks with skip connections
        outputs = [x]
        for block in self.se_res2net_blocks:
            x = block(x)
            outputs.append(x)

        # Multi-layer Feature Aggregation (concatenate SE-Res2Net block outputs)
        mfa_in = np.concatenate(outputs[1:], axis=1)  # [B, 1536, T] (512 * 3)
        mfa_out = relu(self.mfa_conv(mfa_in))  # [B, 1536, T]

        # Attentive Statistics Pooling
        # Attention computed from concat(mfa_in, mfa_out, mfa_out) = 4608 channels
        # Statistics computed over mfa_out = 1536 channels
        asp_attn_in = np.concatenate([mfa_in, mfa_out, mfa_out], axis=1)  # [B, 4608, T]
        x = self.asp(asp_attn_in, mfa_out)  # [B, 3072]

        # Final FC (as 1x1 conv, need to add time dim)
        x = x[:, :, np.newaxis]  # [B, 3072, 1]
        x = self.fc(x)[:, :, 0]  # [B, 2048]

        if squeeze:
            x = x[0]

        return x

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Extract speaker embedding from audio waveform.

        Args:
            audio: Audio waveform [samples] or [batch, samples]
            sample_rate: Audio sample rate

        Returns:
            Speaker embedding [embed_dim] or [batch, embed_dim]
        """
        mel = compute_mel_spectrogram(
            audio,
            sample_rate=sample_rate,
            n_mels=self.config.num_mels,
        )
        return self(mel)


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
    fmax: float | None = None,
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
        fmax: Maximum frequency (default: sample_rate / 2)

    Returns:
        Mel spectrogram [batch, n_mels, time] or [n_mels, time]
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    squeeze = audio.ndim == 1
    if squeeze:
        audio = audio[np.newaxis, :]

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

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq), dtype=np.float32)

    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)

        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank


# =============================================================================
# Model Loading
# =============================================================================


def load_speaker_encoder(
    model_path: str | Path,
    config: SpeakerEncoderConfig | None = None,
) -> SpeakerEncoder:
    """Load speaker encoder from Qwen3-TTS-Base model.

    Args:
        model_path: Path to Qwen3-TTS-Base model directory
        config: Optional config override

    Returns:
        SpeakerEncoder instance
    """
    import torch
    from safetensors import safe_open

    if config is None:
        config = SpeakerEncoderConfig()

    path = Path(model_path)
    weights_path = path / "model.safetensors"

    weights: dict[str, np.ndarray] = {}
    with safe_open(str(weights_path), framework="pt") as f:
        for key in f.keys():
            if key.startswith("speaker_encoder."):
                tensor = f.get_tensor(key)
                # Remove prefix
                name = key[len("speaker_encoder.") :]
                if tensor.dtype == torch.bfloat16:
                    weights[name] = tensor.float().cpu().numpy()
                else:
                    weights[name] = tensor.cpu().numpy()

    if not weights:
        raise ValueError(f"No speaker_encoder weights found in {weights_path}")

    def get_weight(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(f"Weight '{name}' not found")
        return weights[name]

    def get_bias(name: str) -> np.ndarray | None:
        return weights.get(name)

    def build_conv(prefix: str, padding: int = 0) -> Conv1d:
        return Conv1d(
            weight=get_weight(f"{prefix}.weight"),
            bias=get_bias(f"{prefix}.bias"),
            padding=padding,
        )

    def build_res2net(prefix: str) -> Res2NetBlock:
        convs = []
        for i in range(7):  # 7 convs for scale=8
            conv = build_conv(f"{prefix}.blocks.{i}.conv", padding=1)
            convs.append(conv)
        return Res2NetBlock(convs, scale=8)

    def build_se(prefix: str) -> SEBlock:
        return SEBlock(
            conv1=build_conv(f"{prefix}.conv1"),
            conv2=build_conv(f"{prefix}.conv2"),
        )

    def build_se_res2net_block(prefix: str) -> SERes2NetBlock:
        return SERes2NetBlock(
            tdnn1=build_conv(f"{prefix}.tdnn1.conv"),
            res2net=build_res2net(f"{prefix}.res2net_block"),
            tdnn2=build_conv(f"{prefix}.tdnn2.conv"),
            se=build_se(f"{prefix}.se_block"),
        )

    # Build model
    initial_conv = build_conv("blocks.0.conv", padding=2)  # kernel=5, pad=2

    se_res2net_blocks = [
        build_se_res2net_block("blocks.1"),
        build_se_res2net_block("blocks.2"),
        build_se_res2net_block("blocks.3"),
    ]

    mfa_conv = build_conv("mfa.conv")

    asp = AttentiveStatisticsPooling(
        tdnn=build_conv("asp.tdnn.conv"),
        conv=build_conv("asp.conv"),
    )

    fc = build_conv("fc")

    return SpeakerEncoder(
        config=config,
        initial_conv=initial_conv,
        se_res2net_blocks=se_res2net_blocks,
        mfa_conv=mfa_conv,
        asp=asp,
        fc=fc,
    )


__all__ = [
    "SpeakerEncoder",
    "SpeakerEncoderConfig",
    "load_speaker_encoder",
    "compute_mel_spectrogram",
]
