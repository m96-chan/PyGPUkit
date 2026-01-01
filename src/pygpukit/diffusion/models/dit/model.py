"""PixArt Transformer model.

Implements the PixArt-Sigma DiT architecture with proper attention and FFN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.config import PixArtSpec

from .adaln import layer_norm
from .attention import cross_attention, self_attention
from .embeddings import (
    caption_projection,
    get_2d_sincos_pos_embed,
    patch_embed,
    timestep_embedding,
    unpatchify,
)
from .ffn import geglu_ffn


class PixArtTransformer:
    """PixArt-Sigma Transformer model.

    Architecture:
        - Patch embedding with 2x2 patches
        - 28 transformer blocks with AdaLN-Zero
        - Self-attention + Cross-attention + GEGLU FFN
        - Output projection
    """

    def __init__(
        self,
        spec: PixArtSpec,
        weights: dict[str, GPUArray],
    ):
        """Initialize PixArt transformer.

        Args:
            spec: Model specification.
            weights: Pre-loaded weights.
        """
        self.spec = spec
        self.weights = weights
        self.hidden_size = spec.hidden_size
        self.num_layers = spec.num_layers
        self.num_heads = spec.num_heads
        self.head_dim = spec.hidden_size // spec.num_heads
        self.patch_size = spec.patch_size

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        dtype: str = "float32",
    ) -> PixArtTransformer:
        """Load PixArt transformer from SafeTensors.

        Args:
            path: Path to model directory or file.
            dtype: Weight dtype.

        Returns:
            Loaded model.
        """
        from safetensors import safe_open

        path = Path(path)

        # Find safetensors file
        if path.is_dir():
            model_path = path / "diffusion_pytorch_model.safetensors"
            if not model_path.exists():
                st_files = list(path.glob("*.safetensors"))
                if st_files:
                    model_path = st_files[0]
                else:
                    raise FileNotFoundError(f"No safetensors found in {path}")
        else:
            model_path = path

        # Load weights
        weights = {}
        with safe_open(str(model_path), framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if dtype == "float16":
                    tensor = tensor.astype(np.float16)
                else:
                    tensor = tensor.astype(np.float32)
                weights[name] = from_numpy(tensor)

        # Detect spec from weights
        hidden_size = weights["pos_embed.proj.bias"].shape[0]
        num_blocks = sum(
            1
            for k in weights
            if k.startswith("transformer_blocks.") and k.endswith(".attn1.to_q.weight")
        )

        spec = PixArtSpec(
            name="pixart_sigma",
            hidden_size=hidden_size,
            num_layers=num_blocks,
            num_heads=hidden_size // 72,  # PixArt uses 72 head_dim
            conditioning_type="cross_attn",
            text_encoder_dim=4096,
            pos_embed_type="sinusoidal",
            in_channels=4,
            out_channels=8,
            cross_attention_dim=4096,
        )

        return cls(spec, weights)

    def forward(
        self,
        latent: GPUArray,
        timestep: float,
        encoder_hidden_states: GPUArray,
        pooled_projections: GPUArray | None = None,
        guidance: float | None = None,
    ) -> GPUArray:
        """Forward pass through PixArt transformer.

        Args:
            latent: Noisy latent [B, C, H, W].
            timestep: Timestep value (0-1000).
            encoder_hidden_states: Text embeddings [B, seq_len, 4096].
            pooled_projections: Unused for PixArt.
            guidance: Unused for PixArt.

        Returns:
            Predicted noise/velocity [B, out_C, H, W].
        """
        B, C, H, W = latent.shape

        # 1. Patch embedding
        x = self._patch_embed(latent)  # [B, num_patches, D]

        # 2. Timestep embedding
        t_emb = self._timestep_embed(timestep, B)  # [B, D]

        # 3. Caption projection
        text = self._caption_projection(encoder_hidden_states)  # [B, seq_len, D]

        # 4. Compute AdaLN conditioning
        adaln_cond = self._compute_adaln_conditioning(t_emb)  # [B, 6, D]

        # 5. Transformer blocks
        for i in range(self.num_layers):
            x = self._transformer_block(x, text, adaln_cond, i)

        # 6. Final norm and output projection (pass t_emb for final modulation)
        x = self._final_layer(x, t_emb, H, W)

        return x

    def _patch_embed(self, x: GPUArray) -> GPUArray:
        """Patch embedding with 2D sinusoidal positional embedding."""
        proj_w = self.weights.get("pos_embed.proj.weight")
        proj_b = self.weights.get("pos_embed.proj.bias")

        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        if proj_w is not None:
            x_proj = patch_embed(x, proj_w, proj_b, self.patch_size)
        else:
            # Fallback: manual patch extraction
            x_np = x.to_numpy()
            x_np = x_np.reshape(B, C, h_patches, self.patch_size, w_patches, self.patch_size)
            x_np = x_np.transpose(0, 2, 4, 1, 3, 5).reshape(B, h_patches * w_patches, -1)
            x_proj = from_numpy(x_np.astype(np.float32))

        # Add 2D sinusoidal positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, (h_patches, w_patches))
        x_proj_np = x_proj.to_numpy()
        x_proj_np = (
            x_proj_np + pos_embed[None, :, :]
        )  # [1, num_patches, D] broadcast to [B, num_patches, D]

        return from_numpy(x_proj_np.astype(np.float32))

    def _timestep_embed(self, timestep: float, batch_size: int) -> GPUArray:
        """Timestep embedding."""
        prefix = "adaln_single.emb.timestep_embedder"
        linear1_w = self.weights.get(f"{prefix}.linear_1.weight")
        linear1_b = self.weights.get(f"{prefix}.linear_1.bias")
        linear2_w = self.weights.get(f"{prefix}.linear_2.weight")
        linear2_b = self.weights.get(f"{prefix}.linear_2.bias")

        return timestep_embedding(
            timestep,
            self.hidden_size,
            linear1_w,
            linear1_b,
            linear2_w,
            linear2_b,
            batch_size,
        )

    def _caption_projection(self, text: GPUArray) -> GPUArray:
        """Project text embeddings."""
        prefix = "caption_projection"
        linear1_w = self.weights.get(f"{prefix}.linear_1.weight")
        linear1_b = self.weights.get(f"{prefix}.linear_1.bias")
        linear2_w = self.weights.get(f"{prefix}.linear_2.weight")
        linear2_b = self.weights.get(f"{prefix}.linear_2.bias")

        if linear1_w is not None:
            return caption_projection(text, linear1_w, linear1_b, linear2_w, linear2_b)

        return text

    def _compute_adaln_conditioning(self, t_emb: GPUArray) -> GPUArray:
        """Compute global AdaLN conditioning."""
        linear_w = self.weights.get("adaln_single.linear.weight")
        linear_b = self.weights.get("adaln_single.linear.bias")

        if linear_w is None:
            # Return zeros if not available
            B = t_emb.shape[0]
            return from_numpy(np.zeros((B, 6, self.hidden_size), dtype=np.float32))

        from pygpukit.ops.matmul.generic import matmul

        t_np = t_emb.to_numpy()

        # Apply SiLU before the final linear (silu = x * sigmoid(x))
        t_silu = t_np * (1.0 / (1.0 + np.exp(-t_np)))

        w = linear_w.to_numpy().T.astype(np.float32)
        cond = matmul(from_numpy(t_silu.astype(np.float32)), from_numpy(w)).to_numpy()

        if linear_b is not None:
            cond = cond + linear_b.to_numpy()

        # Reshape to [B, 6, D]
        B = t_np.shape[0]
        cond = cond.reshape(B, 6, -1)

        return from_numpy(cond.astype(np.float32))

    def _transformer_block(
        self,
        x: GPUArray,
        text: GPUArray,
        adaln_cond: GPUArray,
        layer_idx: int,
    ) -> GPUArray:
        """Single transformer block with AdaLN-Zero."""
        prefix = f"transformer_blocks.{layer_idx}"

        # Get scale_shift_table for this block
        scale_shift = self.weights.get(f"{prefix}.scale_shift_table")
        if scale_shift is None:
            # Skip if no weights
            return x

        scale_shift_np = scale_shift.to_numpy()  # [6, D]
        adaln_cond_np = adaln_cond.to_numpy()  # [B, 6, D]

        # Combine global conditioning with per-block table
        # modulation = scale_shift_table + adaln_single output
        modulation = scale_shift_np[None, :, :] + adaln_cond_np  # [B, 6, D]

        # Extract modulation parameters
        shift_msa = modulation[:, 0, :]  # [B, D]
        scale_msa = modulation[:, 1, :]  # [B, D]
        gate_msa = modulation[:, 2, :]  # [B, D]
        shift_mlp = modulation[:, 3, :]  # [B, D]
        scale_mlp = modulation[:, 4, :]  # [B, D]
        gate_mlp = modulation[:, 5, :]  # [B, D]

        x_np = x.to_numpy()

        # === Self-Attention ===
        # Norm + modulate
        x_norm = layer_norm(x_np)
        x_mod = x_norm * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        # Self-attention
        attn_out = self._self_attention(from_numpy(x_mod.astype(np.float32)), layer_idx)

        # Gate and residual
        gate_msa_expanded = gate_msa[:, None, :]  # [B, 1, D]
        x_np = x_np + attn_out.to_numpy() * gate_msa_expanded

        # === Cross-Attention ===
        # Cross-attention with text (no modulation for cross-attn in PixArt)
        cross_out = self._cross_attention(from_numpy(x_np.astype(np.float32)), text, layer_idx)
        x_np = x_np + cross_out.to_numpy()

        # === FFN ===
        # Norm + modulate
        x_norm = layer_norm(x_np)
        x_mod = x_norm * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]

        # GEGLU FFN
        ffn_out = self._ffn(from_numpy(x_mod.astype(np.float32)), layer_idx)

        # Gate and residual
        gate_mlp_expanded = gate_mlp[:, None, :]
        x_np = x_np + ffn_out.to_numpy() * gate_mlp_expanded

        return from_numpy(x_np.astype(np.float32))

    def _self_attention(self, x: GPUArray, layer_idx: int) -> GPUArray:
        """Self-attention for a transformer block."""
        prefix = f"transformer_blocks.{layer_idx}.attn1"

        q_w = self.weights.get(f"{prefix}.to_q.weight")
        k_w = self.weights.get(f"{prefix}.to_k.weight")
        v_w = self.weights.get(f"{prefix}.to_v.weight")
        out_w = self.weights.get(f"{prefix}.to_out.0.weight")

        q_b = self.weights.get(f"{prefix}.to_q.bias")
        k_b = self.weights.get(f"{prefix}.to_k.bias")
        v_b = self.weights.get(f"{prefix}.to_v.bias")
        out_b = self.weights.get(f"{prefix}.to_out.0.bias")

        if q_w is None:
            return x

        return self_attention(
            x,
            q_w,
            k_w,
            v_w,
            out_w,
            q_b,
            k_b,
            v_b,
            out_b,
            num_heads=self.num_heads,
        )

    def _cross_attention(self, x: GPUArray, context: GPUArray, layer_idx: int) -> GPUArray:
        """Cross-attention with text embeddings."""
        prefix = f"transformer_blocks.{layer_idx}.attn2"

        q_w = self.weights.get(f"{prefix}.to_q.weight")
        k_w = self.weights.get(f"{prefix}.to_k.weight")
        v_w = self.weights.get(f"{prefix}.to_v.weight")
        out_w = self.weights.get(f"{prefix}.to_out.0.weight")

        q_b = self.weights.get(f"{prefix}.to_q.bias")
        k_b = self.weights.get(f"{prefix}.to_k.bias")
        v_b = self.weights.get(f"{prefix}.to_v.bias")
        out_b = self.weights.get(f"{prefix}.to_out.0.bias")

        if q_w is None:
            return from_numpy(np.zeros_like(x.to_numpy()))

        return cross_attention(
            x,
            context,
            q_w,
            k_w,
            v_w,
            out_w,
            q_b,
            k_b,
            v_b,
            out_b,
            num_heads=self.num_heads,
        )

    def _ffn(self, x: GPUArray, layer_idx: int) -> GPUArray:
        """GEGLU Feed-Forward Network."""
        prefix = f"transformer_blocks.{layer_idx}.ff.net"

        gate_w = self.weights.get(f"{prefix}.0.proj.weight")
        gate_b = self.weights.get(f"{prefix}.0.proj.bias")
        down_w = self.weights.get(f"{prefix}.2.weight")
        down_b = self.weights.get(f"{prefix}.2.bias")

        if gate_w is None:
            return x

        return geglu_ffn(x, gate_w, gate_b, down_w, down_b)

    def _final_layer(self, x: GPUArray, t_emb: GPUArray, H: int, W: int) -> GPUArray:
        """Final normalization and output projection."""
        x_np = x.to_numpy()
        t_emb_np = t_emb.to_numpy()  # [B, D] - timestep embedding for final modulation

        # Get global scale_shift_table
        scale_shift = self.weights.get("scale_shift_table")
        if scale_shift is not None:
            ss_np = scale_shift.to_numpy()  # [2, D]
            shift = ss_np[0]  # [D]
            scale = ss_np[1]  # [D]

            # Add timestep embedding to shift (timestep-dependent modulation)
            # shift_final = scale_shift_table[0] + t_emb
            shift_final = shift + t_emb_np  # [B, D] broadcast

            # Apply final norm + modulation
            x_norm = layer_norm(x_np)
            # Expand shift to [B, N, D] for broadcasting
            x_np = x_norm * (1.0 + scale) + shift_final[:, None, :]
        else:
            x_np = layer_norm(x_np)

        x = from_numpy(x_np.astype(np.float32))

        # Output projection
        proj_w = self.weights.get("proj_out.weight")
        proj_b = self.weights.get("proj_out.bias")

        if proj_w is not None:
            return unpatchify(
                x,
                H,
                W,
                out_channels=self.spec.out_channels,
                patch_size=self.patch_size,
                proj_weight=proj_w,
                proj_bias=proj_b,
            )

        # Fallback unpatchify
        B, num_patches, D = x_np.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        out_dim = self.spec.out_channels * self.patch_size * self.patch_size

        # Simple projection
        if D != out_dim:
            np.random.seed(99)
            proj = np.random.randn(D, out_dim).astype(np.float32) / np.sqrt(D)
            x_np = np.dot(x_np, proj)

        x_np = x_np.reshape(B, h_p, w_p, self.spec.out_channels, self.patch_size, self.patch_size)
        x_np = x_np.transpose(0, 3, 1, 4, 2, 5).reshape(B, self.spec.out_channels, H, W)

        return from_numpy(x_np.astype(np.float32))


__all__ = ["PixArtTransformer"]
