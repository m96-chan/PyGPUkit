"""Speech Tokenizer Decoder for Qwen3-TTS.

Converts codec tokens to audio waveform using PyGPUkit native ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.conv import conv1d, conv_transpose1d


@dataclass
class SpeechDecoderConfig:
    """Configuration for speech decoder."""

    codebook_size: int = 2048
    codebook_dim: int = 256
    latent_dim: int = 1024
    decoder_dim: int = 1536
    num_quantizers: int = 16
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3)  # Total: 480x
    sample_rate: int = 24000
    frame_rate: float = 12.5  # 24000 / 480 / 4 = 12.5 Hz


class Codebook:
    """Single codebook for vector quantization."""

    def __init__(self, embeddings: np.ndarray):
        """Initialize codebook.

        Args:
            embeddings: Codebook embeddings [codebook_size, dim]
        """
        self.embeddings = embeddings
        self.codebook_size, self.dim = embeddings.shape

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Lookup embeddings by indices.

        Args:
            indices: Token indices [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, dim]
        """
        return self.embeddings[indices]


class RVQDecoder:
    """Residual Vector Quantizer decoder."""

    def __init__(
        self,
        codebooks: list[Codebook],
        input_proj: np.ndarray | None = None,
        output_proj: np.ndarray | None = None,
    ):
        self.codebooks = codebooks
        self.input_proj = input_proj
        self.output_proj = output_proj
        self.num_quantizers = len(codebooks)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes to embeddings.

        Args:
            codes: Indices [batch, num_quantizers, seq_len]

        Returns:
            Summed embeddings [batch, seq_len, dim]
        """
        batch, num_q, seq_len = codes.shape
        result = None

        for i in range(min(num_q, self.num_quantizers)):
            quantized = self.codebooks[i].decode(codes[:, i, :])
            if result is None:
                result = quantized
            else:
                result = result + quantized

        assert result is not None

        # Apply output projection if available
        if self.output_proj is not None:
            # output_proj shape: [out_dim, in_dim, 1] (1D conv)
            proj = self.output_proj[:, :, 0]  # [out_dim, in_dim]
            result = np.einsum("bsd,od->bso", result, proj)

        return result


def snake_beta(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """SnakeBeta activation function.

    snake(x) = x + (1/beta) * sin^2(alpha * x)
    """
    return x + (1.0 / (beta + 1e-6)) * np.sin(alpha * x) ** 2


class ConvBlock:
    """1D Convolution block using PyGPUkit native ops."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None):
        """Initialize conv block.

        Args:
            weight: Conv weight [out_channels, in_channels, kernel_size]
            bias: Conv bias [out_channels]
        """
        self.out_channels, self.in_channels, self.kernel_size = weight.shape
        self.weight_gpu = from_numpy(weight.astype(np.float32))
        self.bias_gpu = from_numpy(bias.astype(np.float32)) if bias is not None else None
        self.padding = self.kernel_size // 2

    def __call__(self, x: np.ndarray | GPUArray) -> np.ndarray:
        """Forward pass using PyGPUkit conv1d.

        Args:
            x: Input [batch, channels, seq_len]

        Returns:
            Output [batch, out_channels, seq_len]
        """
        if isinstance(x, np.ndarray):
            x_gpu = from_numpy(x.astype(np.float32))
        else:
            x_gpu = x

        out = conv1d(x_gpu, self.weight_gpu, self.bias_gpu, stride=1, padding=self.padding)
        return out.to_numpy()


class ConvTransposeBlock:
    """1D Transposed Convolution for upsampling using PyGPUkit."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        stride: int = 1,
    ):
        """Initialize conv transpose block.

        Args:
            weight: Conv weight [in_channels, out_channels, kernel_size]
            bias: Conv bias [out_channels]
            stride: Upsampling stride
        """
        self.stride = stride
        self.in_channels, self.out_channels, self.kernel_size = weight.shape
        self.weight_gpu = from_numpy(weight.astype(np.float32))
        self.bias_gpu = from_numpy(bias.astype(np.float32)) if bias is not None else None
        # Padding to maintain output_length = input_length * stride
        self.padding = (self.kernel_size - stride) // 2
        self.output_padding = (self.kernel_size - stride) % 2

    def __call__(self, x: np.ndarray | GPUArray) -> np.ndarray:
        """Forward pass with upsampling using PyGPUkit conv_transpose1d.

        Args:
            x: Input [batch, in_channels, seq_len]

        Returns:
            Output [batch, out_channels, seq_len * stride]
        """
        if isinstance(x, np.ndarray):
            x_gpu = from_numpy(x.astype(np.float32))
        else:
            x_gpu = x

        out = conv_transpose1d(
            x_gpu,
            self.weight_gpu,
            self.bias_gpu,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        return out.to_numpy()


class ResBlock:
    """Residual block with SnakeBeta activation."""

    def __init__(
        self,
        conv1_weight: np.ndarray,
        conv1_bias: np.ndarray | None,
        conv2_weight: np.ndarray,
        conv2_bias: np.ndarray | None,
        act1_alpha: np.ndarray,
        act1_beta: np.ndarray,
        act2_alpha: np.ndarray,
        act2_beta: np.ndarray,
    ):
        self.conv1 = ConvBlock(conv1_weight, conv1_bias)
        self.conv2 = ConvBlock(conv2_weight, conv2_bias)
        self.act1_alpha = act1_alpha
        self.act1_beta = act1_beta
        self.act2_alpha = act2_alpha
        self.act2_beta = act2_beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        residual = x
        x = snake_beta(x, self.act1_alpha[:, None], self.act1_beta[:, None])
        x = self.conv1(x)
        x = snake_beta(x, self.act2_alpha[:, None], self.act2_beta[:, None])
        x = self.conv2(x)
        return x + residual


class DecoderBlock:
    """Decoder block with upsample + residual blocks."""

    def __init__(
        self,
        pre_act_alpha: np.ndarray,
        pre_act_beta: np.ndarray,
        upsample: ConvTransposeBlock,
        res_blocks: list[ResBlock],
    ):
        self.pre_act_alpha = pre_act_alpha
        self.pre_act_beta = pre_act_beta
        self.upsample = upsample
        self.res_blocks = res_blocks

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = snake_beta(x, self.pre_act_alpha[:, None], self.pre_act_beta[:, None])
        x = self.upsample(x)
        for res in self.res_blocks:
            x = res(x)
        return x


class SpeechDecoder:
    """Speech tokenizer decoder (vocoder).

    Converts codec tokens to audio waveform.
    """

    def __init__(
        self,
        config: SpeechDecoderConfig,
        rvq_first: RVQDecoder,
        rvq_rest: RVQDecoder,
        pre_transformer_proj: np.ndarray,
        pre_transformer_bias: np.ndarray | None,
        initial_conv: ConvBlock,
        decoder_blocks: list[DecoderBlock],
        final_act_alpha: np.ndarray,
        final_act_beta: np.ndarray,
        final_conv: ConvBlock,
    ):
        self.config = config
        self.rvq_first = rvq_first
        self.rvq_rest = rvq_rest
        self.pre_transformer_proj = pre_transformer_proj
        self.pre_transformer_bias = pre_transformer_bias
        self.initial_conv = initial_conv
        self.decoder_blocks = decoder_blocks
        self.final_act_alpha = final_act_alpha
        self.final_act_beta = final_act_beta
        self.final_conv = final_conv

    def decode_codes_to_latent(self, codes: np.ndarray) -> np.ndarray:
        """Decode codec tokens to latent representation.

        Args:
            codes: Codec tokens [batch, num_quantizers, seq_len] or [num_quantizers, seq_len]

        Returns:
            Latent [batch, latent_dim, seq_len]
        """
        if codes.ndim == 2:
            codes = codes[np.newaxis, :, :]

        batch, num_q, seq_len = codes.shape

        # Decode first quantizer (semantic)
        first_emb = self.rvq_first.decode(codes[:, :1, :])  # [batch, seq_len, dim]

        # Decode rest quantizers (acoustic)
        if num_q > 1:
            rest_emb = self.rvq_rest.decode(codes[:, 1:, :])  # [batch, seq_len, dim]
            latent = first_emb + rest_emb
        else:
            latent = first_emb

        # Project to latent_dim
        latent = latent @ self.pre_transformer_proj.T  # [batch, seq_len, latent_dim]
        if self.pre_transformer_bias is not None:
            latent = latent + self.pre_transformer_bias

        # Transpose to [batch, latent_dim, seq_len]
        latent = latent.transpose(0, 2, 1)

        return latent

    def decode_latent_to_audio(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent representation to audio.

        Args:
            latent: Latent [batch, latent_dim, seq_len]

        Returns:
            Audio [batch, samples]
        """
        x = self.initial_conv(latent)

        for block in self.decoder_blocks:
            x = block(x)

        x = snake_beta(x, self.final_act_alpha[:, None], self.final_act_beta[:, None])
        x = self.final_conv(x)

        # Remove channel dim and return
        return x[:, 0, :]

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Full decode: codes → audio.

        Args:
            codes: Codec tokens [batch, num_quantizers, seq_len] or [num_quantizers, seq_len]

        Returns:
            Audio waveform [batch, samples] or [samples]
        """
        squeeze = codes.ndim == 2
        if squeeze:
            codes = codes[np.newaxis, :, :]

        latent = self.decode_codes_to_latent(codes)
        audio = self.decode_latent_to_audio(latent)

        if squeeze:
            audio = audio[0]

        return audio


def load_speech_decoder(
    model_path: str | Path,
    config: SpeechDecoderConfig | None = None,
) -> SpeechDecoder:
    """Load speech decoder from safetensors.

    Args:
        model_path: Path to speech_tokenizer directory
        config: Optional config override

    Returns:
        SpeechDecoder instance
    """
    import torch
    from safetensors import safe_open

    if config is None:
        config = SpeechDecoderConfig()

    path = Path(model_path)
    weights_path = path / "model.safetensors"

    weights = {}
    with safe_open(str(weights_path), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                weights[key] = tensor.float().cpu().numpy()
            else:
                weights[key] = tensor.cpu().numpy()

    def get_weight(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(f"Weight '{name}' not found")
        return weights[name]

    def get_weight_optional(name: str) -> np.ndarray | None:
        return weights.get(name)

    # Build codebooks from embedding_sum / cluster_usage
    def build_codebook(prefix: str) -> Codebook:
        embed_sum = get_weight(f"{prefix}._codebook.embedding_sum")
        cluster_usage = get_weight(f"{prefix}._codebook.cluster_usage")
        # Normalize: embeddings = embed_sum / cluster_usage
        usage = cluster_usage[:, None] + 1e-6
        embeddings = embed_sum / usage
        return Codebook(embeddings.astype(np.float32))

    # Build RVQ first (semantic, 1 quantizer)
    rvq_first_cb = [build_codebook("decoder.quantizer.rvq_first.vq.layers.0")]
    rvq_first_out = get_weight_optional("decoder.quantizer.rvq_first.output_proj.weight")
    rvq_first = RVQDecoder(rvq_first_cb, output_proj=rvq_first_out)

    # Build RVQ rest (acoustic, 15 quantizers)
    rvq_rest_cbs = []
    for i in range(15):
        cb = build_codebook(f"decoder.quantizer.rvq_rest.vq.layers.{i}")
        rvq_rest_cbs.append(cb)
    rvq_rest_out = get_weight_optional("decoder.quantizer.rvq_rest.output_proj.weight")
    rvq_rest = RVQDecoder(rvq_rest_cbs, output_proj=rvq_rest_out)

    # Pre-transformer projection
    pre_proj = get_weight("decoder.pre_transformer.output_proj.weight")
    pre_bias = get_weight_optional("decoder.pre_transformer.output_proj.bias")

    # Initial conv
    initial_conv = ConvBlock(
        get_weight("decoder.decoder.0.conv.weight"),
        get_weight_optional("decoder.decoder.0.conv.bias"),
    )

    # Decoder blocks (1-4)
    decoder_blocks = []
    for block_idx in range(1, 5):
        prefix = f"decoder.decoder.{block_idx}"

        # Pre-activation
        pre_alpha = get_weight(f"{prefix}.block.0.alpha")
        pre_beta = get_weight(f"{prefix}.block.0.beta")

        # Upsample (ConvTranspose)
        up_weight = get_weight(f"{prefix}.block.1.conv.weight")
        up_bias = get_weight_optional(f"{prefix}.block.1.conv.bias")
        stride = config.upsample_rates[block_idx - 1]
        upsample = ConvTransposeBlock(up_weight, up_bias, stride=stride)

        # Residual blocks (2, 3, 4)
        res_blocks = []
        for res_idx in range(2, 5):
            res_prefix = f"{prefix}.block.{res_idx}"
            res = ResBlock(
                conv1_weight=get_weight(f"{res_prefix}.conv1.conv.weight"),
                conv1_bias=get_weight_optional(f"{res_prefix}.conv1.conv.bias"),
                conv2_weight=get_weight(f"{res_prefix}.conv2.conv.weight"),
                conv2_bias=get_weight_optional(f"{res_prefix}.conv2.conv.bias"),
                act1_alpha=get_weight(f"{res_prefix}.act1.alpha"),
                act1_beta=get_weight(f"{res_prefix}.act1.beta"),
                act2_alpha=get_weight(f"{res_prefix}.act2.alpha"),
                act2_beta=get_weight(f"{res_prefix}.act2.beta"),
            )
            res_blocks.append(res)

        block = DecoderBlock(pre_alpha, pre_beta, upsample, res_blocks)
        decoder_blocks.append(block)

    # Final activation and conv
    final_alpha = get_weight("decoder.decoder.5.alpha")
    final_beta = get_weight("decoder.decoder.5.beta")
    final_conv = ConvBlock(
        get_weight("decoder.decoder.6.conv.weight"),
        get_weight_optional("decoder.decoder.6.conv.bias"),
    )

    return SpeechDecoder(
        config=config,
        rvq_first=rvq_first,
        rvq_rest=rvq_rest,
        pre_transformer_proj=pre_proj,
        pre_transformer_bias=pre_bias,
        initial_conv=initial_conv,
        decoder_blocks=decoder_blocks,
        final_act_alpha=final_alpha,
        final_act_beta=final_beta,
        final_conv=final_conv,
    )


__all__ = [
    "SpeechDecoder",
    "SpeechDecoderConfig",
    "load_speech_decoder",
]
