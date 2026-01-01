"""GPU-native FLUX Transformer model.

Main transformer implementation for FLUX.1 text-to-image generation.
Uses GPU-native operations to minimize H2D/D2H transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.models.flux.attention import layer_norm
from pygpukit.diffusion.models.flux.blocks import joint_block, single_block
from pygpukit.diffusion.models.flux.embeddings import (
    get_rope_frequencies,
    prepare_image_ids,
    prepare_text_ids,
    timestep_embedding,
)
from pygpukit.diffusion.models.flux.ops import gpu_linear, gpu_silu


@dataclass
class FluxConfig:
    """FLUX transformer configuration."""

    in_channels: int = 64
    out_channels: int | None = None
    hidden_size: int = 3072
    num_layers: int = 19  # Joint blocks
    num_single_layers: int = 38  # Single blocks
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    joint_attention_dim: int = 4096  # T5 encoder dim
    pooled_projection_dim: int = 768  # CLIP pooled dim
    guidance_embeds: bool = False  # True for dev, False for schnell
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)

    @property
    def head_dim(self) -> int:
        return self.attention_head_dim


class FluxTransformer:
    """GPU-native FLUX transformer for text-to-image generation.

    Implements the FLUX.1 architecture with:
    - 19 joint transformer blocks (image + text cross-attention)
    - 38 single transformer blocks (self-attention)
    - RoPE position embeddings
    - AdaLN-Zero modulation

    Uses GPU-native operations to minimize H2D/D2H transfers during forward pass.
    """

    def __init__(
        self,
        config: FluxConfig,
        weights: dict[str, GPUArray],
    ):
        """Initialize FLUX transformer.

        Args:
            config: Model configuration.
            weights: Dictionary of model weights (already on GPU as GPUArray).
        """
        self.config = config
        self.weights = weights

        # Pre-computed RoPE frequencies (will be set during first forward pass)
        self._rope_cos: GPUArray | None = None
        self._rope_sin: GPUArray | None = None
        self._last_img_seq_len: int = 0
        self._last_txt_seq_len: int = 0

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        dtype: str = "float32",
    ) -> FluxTransformer:
        """Load FLUX transformer from safetensors.

        Args:
            path: Path to model directory or safetensors file.
            dtype: Weight dtype ("float32" or "float16").

        Returns:
            Loaded FluxTransformer instance.
        """
        path = Path(path)

        # Find safetensors file
        if path.is_dir():
            # Check for HuggingFace cache structure
            cache_path = path / "models--black-forest-labs--FLUX.1-schnell"
            if cache_path.exists():
                # Find the latest snapshot
                snapshots = list((cache_path / "snapshots").iterdir())
                if snapshots:
                    transformer_path = snapshots[0] / "transformer"
                    if transformer_path.exists():
                        path = transformer_path

            # Look for safetensors files
            st_files = list(path.glob("*.safetensors"))
            if not st_files:
                st_files = list(path.glob("**/*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

            # Use the first file or concatenate if sharded
            if len(st_files) == 1:
                st_path = st_files[0]
            else:
                # Multiple files - load all
                st_path = st_files
        else:
            st_path = path

        # Load weights using torch for bfloat16 support
        import torch

        weights: dict[str, GPUArray] = {}
        torch_dtype = torch.float32 if dtype == "float32" else torch.float16

        if isinstance(st_path, list):
            # Multiple safetensors files
            for sf in st_path:
                with safe_open(str(sf), framework="pt") as f:
                    for name in f.keys():
                        tensor = f.get_tensor(name).to(torch_dtype).numpy()
                        weights[name] = from_numpy(tensor)
        else:
            with safe_open(str(st_path), framework="pt") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name).to(torch_dtype).numpy()
                    weights[name] = from_numpy(tensor)

        # Detect configuration from weights
        config = cls._detect_config(weights)

        return cls(config, weights)

    @staticmethod
    def _detect_config(weights: dict[str, GPUArray]) -> FluxConfig:
        """Detect model configuration from weights."""
        # Count transformer blocks
        num_layers = 0
        num_single_layers = 0

        for name in weights.keys():
            if name.startswith("transformer_blocks."):
                idx = int(name.split(".")[1])
                num_layers = max(num_layers, idx + 1)
            elif name.startswith("single_transformer_blocks."):
                idx = int(name.split(".")[1])
                num_single_layers = max(num_single_layers, idx + 1)

        # Get hidden size from x_embedder
        if "x_embedder.weight" in weights:
            hidden_size = weights["x_embedder.weight"].shape[0]
        else:
            hidden_size = 3072

        # Check for guidance embeddings
        guidance_embeds = "guidance_in.in_layer.weight" in weights

        return FluxConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            guidance_embeds=guidance_embeds,
        )

    def _get_rope_frequencies(
        self,
        img_ids: np.ndarray,
        txt_ids: np.ndarray,
    ) -> tuple[GPUArray, GPUArray]:
        """Get or compute RoPE frequencies.

        Caches RoPE frequencies to avoid recomputation when sequence lengths match.
        """
        img_seq_len = img_ids.shape[0]
        txt_seq_len = txt_ids.shape[0]

        # Check if we can reuse cached frequencies
        if (
            self._rope_cos is not None
            and self._rope_sin is not None
            and self._last_img_seq_len == img_seq_len
            and self._last_txt_seq_len == txt_seq_len
        ):
            return self._rope_cos, self._rope_sin

        # Compute new frequencies
        rope_cos, rope_sin = get_rope_frequencies(
            img_ids,
            txt_ids,
            axes_dim=self.config.axes_dims_rope,
        )

        # Cache as GPUArray
        self._rope_cos = from_numpy(rope_cos)
        self._rope_sin = from_numpy(rope_sin)
        self._last_img_seq_len = img_seq_len
        self._last_txt_seq_len = txt_seq_len

        return self._rope_cos, self._rope_sin

    def forward(
        self,
        hidden_states: GPUArray,
        encoder_hidden_states: GPUArray,
        pooled_projections: GPUArray,
        timestep: np.ndarray,
        img_ids: np.ndarray | None = None,
        txt_ids: np.ndarray | None = None,
        guidance: np.ndarray | None = None,
    ) -> GPUArray:
        """GPU-native forward pass of FLUX transformer.

        Keeps data on GPU throughout computation to minimize transfers.

        Args:
            hidden_states: Latent image [B, img_seq_len, in_channels].
            encoder_hidden_states: T5 text embeddings [B, txt_seq_len, 4096].
            pooled_projections: CLIP pooled embedding [B, 768].
            timestep: Diffusion timestep [B].
            img_ids: Image position IDs [B, img_seq_len, 3].
            txt_ids: Text position IDs [B, txt_seq_len, 3].
            guidance: Guidance scale (only for dev variant) [B].

        Returns:
            Predicted noise/velocity [B, img_seq_len, in_channels].
        """
        B = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]

        # Prepare position IDs if not provided
        if img_ids is None:
            # Assume square image
            h = w = int(np.sqrt(img_seq_len))
            img_ids = prepare_image_ids(B, h, w)[0]  # [img_seq_len, 3]
        else:
            img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids

        if txt_ids is None:
            txt_ids = prepare_text_ids(B, txt_seq_len)[0]  # [txt_seq_len, 3]
        else:
            txt_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids

        # Get RoPE frequencies (cached on GPU)
        rope_cos, rope_sin = self._get_rope_frequencies(img_ids, txt_ids)

        # Embed image latents using GPU-native linear
        # [B, img_seq_len, in_channels] -> [B, img_seq_len, hidden_size]
        x_2d = hidden_states.reshape(B * img_seq_len, self.config.in_channels)
        x = gpu_linear(x_2d, self.weights["x_embedder.weight"], self.weights.get("x_embedder.bias"))
        x = x.reshape(B, img_seq_len, self.config.hidden_size)

        # Embed text using GPU-native linear
        # [B, txt_seq_len, 4096] -> [B, txt_seq_len, hidden_size]
        txt_2d = encoder_hidden_states.reshape(B * txt_seq_len, self.config.joint_attention_dim)
        txt = gpu_linear(
            txt_2d,
            self.weights["context_embedder.weight"],
            self.weights.get("context_embedder.bias"),
        )
        txt = txt.reshape(B, txt_seq_len, self.config.hidden_size)

        # Time + text embedding (GPU-native)
        temb = self._compute_time_text_embedding(timestep, pooled_projections, guidance)

        # Joint transformer blocks
        for i in range(self.config.num_layers):
            x, txt = joint_block(
                x,
                txt,
                temb,
                self.weights,
                prefix=f"transformer_blocks.{i}",
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.head_dim,
            )

        # Single transformer blocks (keep img/txt separate like diffusers)
        for i in range(self.config.num_single_layers):
            txt, x = single_block(
                x,  # hidden_states (img)
                txt,  # encoder_hidden_states (txt)
                temb,
                self.weights,
                prefix=f"single_transformer_blocks.{i}",
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.head_dim,
            )

        # Final layer: AdaLN + projection (GPU-native)
        x_final = self._final_layer(x, temb)

        return x_final

    def _compute_time_text_embedding(
        self,
        timestep: np.ndarray,
        pooled_text: GPUArray,
        guidance: np.ndarray | None = None,
    ) -> GPUArray:
        """Compute combined time + text embedding using GPU-native ops.

        Args:
            timestep: Timestep values [B].
            pooled_text: CLIP pooled embedding [B, 768] as GPUArray.
            guidance: Guidance scale (only for dev) [B].

        Returns:
            Combined embedding [B, hidden_size] as GPUArray.
        """
        # Timestep embedding: sinusoidal -> MLP
        # FLUX uses timestep directly in [0, 1] range (no scaling)
        t_emb = timestep_embedding(timestep, dim=256)  # [B, 256]
        t_emb_gpu = from_numpy(t_emb)

        # Time projection: Linear -> SiLU -> Linear (GPU-native)
        t_proj = gpu_linear(
            t_emb_gpu,
            self.weights["time_text_embed.timestep_embedder.linear_1.weight"],
            self.weights.get("time_text_embed.timestep_embedder.linear_1.bias"),
        )
        t_proj = gpu_silu(t_proj)
        temb = gpu_linear(
            t_proj,
            self.weights["time_text_embed.timestep_embedder.linear_2.weight"],
            self.weights.get("time_text_embed.timestep_embedder.linear_2.bias"),
        )

        # Text projection: Linear -> SiLU -> Linear (GPU-native)
        text_proj = gpu_linear(
            pooled_text,
            self.weights["time_text_embed.text_embedder.linear_1.weight"],
            self.weights.get("time_text_embed.text_embedder.linear_1.bias"),
        )
        text_proj = gpu_silu(text_proj)
        text_emb = gpu_linear(
            text_proj,
            self.weights["time_text_embed.text_embedder.linear_2.weight"],
            self.weights.get("time_text_embed.text_embedder.linear_2.bias"),
        )

        # Combine - need numpy for now (can add GPU add later)
        temb_np = temb.to_numpy()
        text_emb_np = text_emb.to_numpy()
        combined = temb_np + text_emb_np

        # Guidance embedding (only for dev variant)
        if self.config.guidance_embeds and guidance is not None:
            g_emb = timestep_embedding(guidance * 1000, dim=256)
            g_emb_gpu = from_numpy(g_emb)

            g_proj = gpu_linear(
                g_emb_gpu,
                self.weights["time_text_embed.guidance_embedder.linear_1.weight"],
                self.weights.get("time_text_embed.guidance_embedder.linear_1.bias"),
            )
            g_proj = gpu_silu(g_proj)
            g_emb_out = gpu_linear(
                g_proj,
                self.weights["time_text_embed.guidance_embedder.linear_2.weight"],
                self.weights.get("time_text_embed.guidance_embedder.linear_2.bias"),
            )

            combined = combined + g_emb_out.to_numpy()

        return from_numpy(combined.astype(np.float32))

    def _final_layer(
        self,
        x: GPUArray,
        temb: GPUArray,
    ) -> GPUArray:
        """GPU-native final normalization and projection.

        Args:
            x: Hidden states [B, img_seq_len, D].
            temb: Time embedding [B, D].

        Returns:
            Output [B, img_seq_len, out_channels].
        """
        B = x.shape[0]
        seq_len = x.shape[1]
        D = x.shape[2]

        # AdaLN Continuous: emb -> SiLU -> Linear -> (scale, shift)
        norm_linear_w = self.weights["norm_out.linear.weight"]
        norm_linear_b = self.weights.get("norm_out.linear.bias")

        # SiLU on temb (GPU-native)
        temb_silu = gpu_silu(temb)

        # Project to scale/shift
        mod = gpu_linear(temb_silu, norm_linear_w, norm_linear_b)
        mod_np = mod.to_numpy()

        # Split into scale and shift (diffusers order)
        scale, shift = np.split(mod_np, 2, axis=-1)

        # Apply normalization
        x_norm = layer_norm(x)
        x_norm_np = x_norm.to_numpy() if isinstance(x_norm, GPUArray) else x_norm
        x_mod = x_norm_np * (1.0 + scale[:, None, :]) + shift[:, None, :]

        # Output projection (GPU-native)
        proj_out_w = self.weights["proj_out.weight"]
        proj_out_b = self.weights.get("proj_out.bias")

        x_2d = x_mod.reshape(B * seq_len, D).astype(np.float32)
        output = gpu_linear(from_numpy(x_2d), proj_out_w, proj_out_b)
        output_np = output.to_numpy()

        out_channels = self.config.out_channels or self.config.in_channels
        output_np = output_np.reshape(B, seq_len, out_channels)

        return from_numpy(output_np.astype(np.float32))


__all__ = ["FluxConfig", "FluxTransformer"]
