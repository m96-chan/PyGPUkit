"""GPU-native transformer blocks for FLUX.

Provides JointBlock (double) and SingleBlock implementations.
Most operations stay on GPU to minimize H2D/D2H transfers.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.models.flux.attention import (
    joint_attention,
    layer_norm,
    single_attention,
)
from pygpukit.diffusion.models.flux.ops import (
    gpu_gelu,
    gpu_linear,
    gpu_silu,
)


def adaln_zero(
    x: GPUArray,
    emb: GPUArray,
    linear_weight: GPUArray,
    linear_bias: GPUArray | None,
    num_outputs: int = 6,
    eps: float = 1e-6,
) -> tuple[GPUArray, ...]:
    """GPU-native Adaptive Layer Normalization Zero.

    Args:
        x: Input tensor [B, seq_len, D].
        emb: Conditioning embedding [B, D].
        linear_weight: Modulation projection [num_outputs * D, D].
        linear_bias: Modulation bias [num_outputs * D].
        num_outputs: Number of modulation outputs (6 for joint, 3 for single).
        eps: LayerNorm epsilon.

    Returns:
        Tuple of (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp) for 6 outputs
        or (normalized_x, gate) for 3 outputs.
    """
    B, seq_len, D = x.shape

    # SiLU activation on embedding
    emb_silu = gpu_silu(emb)

    # Project to modulation parameters using GPU-native linear
    # emb_silu: [B, D], linear_weight: [num_outputs * D, D]
    mod = gpu_linear(emb_silu, linear_weight, linear_bias)  # [B, num_outputs * D]

    # Split into components - need numpy for split operation
    mod_np = mod.to_numpy()
    mod_split = np.split(mod_np, num_outputs, axis=-1)  # List of [B, D] arrays

    # Layer norm (stays partially on GPU)
    x_norm = layer_norm(x, eps)
    x_norm_np = x_norm.to_numpy() if isinstance(x_norm, GPUArray) else x_norm

    if num_outputs == 6:
        # Joint block: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_split

        # Apply shift and scale to normalized x
        x_mod = x_norm_np * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        return (
            from_numpy(x_mod.astype(np.float32)),
            from_numpy(gate_msa.astype(np.float32)),
            from_numpy(shift_mlp.astype(np.float32)),
            from_numpy(scale_mlp.astype(np.float32)),
            from_numpy(gate_mlp.astype(np.float32)),
        )

    elif num_outputs == 3:
        # Single block: shift, scale, gate
        shift, scale, gate = mod_split

        # Apply shift and scale
        x_mod = x_norm_np * (1.0 + scale[:, None, :]) + shift[:, None, :]

        return (
            from_numpy(x_mod.astype(np.float32)),
            from_numpy(gate.astype(np.float32)),
        )

    else:
        raise ValueError(f"num_outputs must be 3 or 6, got {num_outputs}")


def gelu(x: GPUArray) -> GPUArray:
    """GPU-native GELU activation."""
    return gpu_gelu(x)


def feedforward(
    x: GPUArray,
    up_proj_weight: GPUArray,
    up_proj_bias: GPUArray | None,
    down_proj_weight: GPUArray,
    down_proj_bias: GPUArray | None,
) -> GPUArray:
    """GPU-native Feed-forward network with GELU activation.

    FLUX uses standard GELU: Linear(hidden_dim) -> GELU -> Linear(D)

    Args:
        x: Input [B, seq_len, D].
        up_proj_weight: Up projection [hidden_dim, D].
        down_proj_weight: Down projection [D, hidden_dim].

    Returns:
        Output [B, seq_len, D].
    """
    B, seq_len, D = x.shape

    # Reshape to 2D for linear operations
    x_2d = x.reshape(B * seq_len, D)

    # Up projection using GPU-native linear
    hidden = gpu_linear(x_2d, up_proj_weight, up_proj_bias)

    # GELU activation (GPU-native)
    hidden = gpu_gelu(hidden)

    # Down projection
    output = gpu_linear(hidden, down_proj_weight, down_proj_bias)

    return output.reshape(B, seq_len, D)


def joint_block(
    hidden_states: GPUArray,
    encoder_hidden_states: GPUArray,
    temb: GPUArray,
    weights: dict[str, GPUArray],
    prefix: str,
    rope_cos: np.ndarray | GPUArray,
    rope_sin: np.ndarray | GPUArray,
    num_heads: int = 24,
    head_dim: int = 128,
) -> tuple[GPUArray, GPUArray]:
    """GPU-native Joint transformer block for FLUX.

    Processes image and text streams in parallel with joint attention.

    Args:
        hidden_states: Image hidden states [B, img_len, D].
        encoder_hidden_states: Text hidden states [B, txt_len, D].
        temb: Time embedding [B, D].
        weights: Weight dictionary.
        prefix: Weight prefix (e.g., "transformer_blocks.0").
        rope_cos, rope_sin: RoPE frequencies.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Tuple of (image_output, text_output).
    """
    # Get weights helper
    def get_weight(name: str) -> GPUArray | None:
        return weights.get(f"{prefix}.{name}")

    # AdaLN for image stream
    norm1_linear_w = get_weight("norm1.linear.weight")
    norm1_linear_b = get_weight("norm1.linear.bias")
    img_mod, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = adaln_zero(
        hidden_states, temb, norm1_linear_w, norm1_linear_b, num_outputs=6
    )

    # AdaLN for text stream
    norm1_ctx_linear_w = get_weight("norm1_context.linear.weight")
    norm1_ctx_linear_b = get_weight("norm1_context.linear.bias")
    txt_mod, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = adaln_zero(
        encoder_hidden_states, temb, norm1_ctx_linear_w, norm1_ctx_linear_b, num_outputs=6
    )

    # Joint attention (GPU-native)
    attn_img, attn_txt = joint_attention(
        img_mod,
        txt_mod,
        q_weight=weights[f"{prefix}.attn.to_q.weight"],
        k_weight=weights[f"{prefix}.attn.to_k.weight"],
        v_weight=weights[f"{prefix}.attn.to_v.weight"],
        q_bias=weights.get(f"{prefix}.attn.to_q.bias"),
        k_bias=weights.get(f"{prefix}.attn.to_k.bias"),
        v_bias=weights.get(f"{prefix}.attn.to_v.bias"),
        add_q_weight=weights[f"{prefix}.attn.add_q_proj.weight"],
        add_k_weight=weights[f"{prefix}.attn.add_k_proj.weight"],
        add_v_weight=weights[f"{prefix}.attn.add_v_proj.weight"],
        add_q_bias=weights.get(f"{prefix}.attn.add_q_proj.bias"),
        add_k_bias=weights.get(f"{prefix}.attn.add_k_proj.bias"),
        add_v_bias=weights.get(f"{prefix}.attn.add_v_proj.bias"),
        out_weight=weights[f"{prefix}.attn.to_out.0.weight"],
        out_bias=weights.get(f"{prefix}.attn.to_out.0.bias"),
        add_out_weight=weights[f"{prefix}.attn.to_add_out.weight"],
        add_out_bias=weights.get(f"{prefix}.attn.to_add_out.bias"),
        norm_q_weight=weights[f"{prefix}.attn.norm_q.weight"],
        norm_k_weight=weights[f"{prefix}.attn.norm_k.weight"],
        norm_added_q_weight=weights[f"{prefix}.attn.norm_added_q.weight"],
        norm_added_k_weight=weights[f"{prefix}.attn.norm_added_k.weight"],
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    # Residual with gating for image
    # img = img + gate * attn_img
    img_np = hidden_states.to_numpy()
    attn_img_np = attn_img.to_numpy()
    gate_img_np = img_gate_msa.to_numpy()
    img_np = img_np + gate_img_np[:, None, :] * attn_img_np

    # Residual with gating for text
    txt_np = encoder_hidden_states.to_numpy()
    attn_txt_np = attn_txt.to_numpy()
    gate_txt_np = txt_gate_msa.to_numpy()
    txt_np = txt_np + gate_txt_np[:, None, :] * attn_txt_np

    # FFN for image
    img_norm2 = layer_norm(from_numpy(img_np.astype(np.float32)))
    img_norm2_np = img_norm2.to_numpy() if isinstance(img_norm2, GPUArray) else img_norm2
    img_scale_mlp_np = img_scale_mlp.to_numpy()
    img_shift_mlp_np = img_shift_mlp.to_numpy()
    img_ffn_in = img_norm2_np * (1.0 + img_scale_mlp_np[:, None, :]) + img_shift_mlp_np[:, None, :]

    ff_gate_w = get_weight("ff.net.0.proj.weight")
    ff_gate_b = get_weight("ff.net.0.proj.bias")
    ff_down_w = get_weight("ff.net.2.weight")
    ff_down_b = get_weight("ff.net.2.bias")

    img_ffn_out = feedforward(
        from_numpy(img_ffn_in.astype(np.float32)), ff_gate_w, ff_gate_b, ff_down_w, ff_down_b
    )
    img_ffn_out_np = img_ffn_out.to_numpy()
    img_gate_mlp_np = img_gate_mlp.to_numpy()
    img_np = img_np + img_gate_mlp_np[:, None, :] * img_ffn_out_np

    # FFN for text
    txt_norm2 = layer_norm(from_numpy(txt_np.astype(np.float32)))
    txt_norm2_np = txt_norm2.to_numpy() if isinstance(txt_norm2, GPUArray) else txt_norm2
    txt_scale_mlp_np = txt_scale_mlp.to_numpy()
    txt_shift_mlp_np = txt_shift_mlp.to_numpy()
    txt_ffn_in = txt_norm2_np * (1.0 + txt_scale_mlp_np[:, None, :]) + txt_shift_mlp_np[:, None, :]

    ff_ctx_gate_w = get_weight("ff_context.net.0.proj.weight")
    ff_ctx_gate_b = get_weight("ff_context.net.0.proj.bias")
    ff_ctx_down_w = get_weight("ff_context.net.2.weight")
    ff_ctx_down_b = get_weight("ff_context.net.2.bias")

    txt_ffn_out = feedforward(
        from_numpy(txt_ffn_in.astype(np.float32)),
        ff_ctx_gate_w,
        ff_ctx_gate_b,
        ff_ctx_down_w,
        ff_ctx_down_b,
    )
    txt_ffn_out_np = txt_ffn_out.to_numpy()
    txt_gate_mlp_np = txt_gate_mlp.to_numpy()
    txt_np = txt_np + txt_gate_mlp_np[:, None, :] * txt_ffn_out_np

    return from_numpy(img_np.astype(np.float32)), from_numpy(txt_np.astype(np.float32))


def single_block(
    hidden_states: GPUArray,
    encoder_hidden_states: GPUArray,
    temb: GPUArray,
    weights: dict[str, GPUArray],
    prefix: str,
    rope_cos: np.ndarray | GPUArray,
    rope_sin: np.ndarray | GPUArray,
    num_heads: int = 24,
    head_dim: int = 128,
) -> tuple[GPUArray, GPUArray]:
    """GPU-native Single transformer block for FLUX.

    Self-attention on concatenated [text, image] sequence with parallel MLP.
    Matches diffusers behavior: takes separate img/txt, returns separate img/txt.

    Args:
        hidden_states: Image hidden states [B, img_len, D].
        encoder_hidden_states: Text hidden states [B, txt_len, D].
        temb: Time embedding [B, D].
        weights: Weight dictionary.
        prefix: Weight prefix (e.g., "single_transformer_blocks.0").
        rope_cos, rope_sin: RoPE frequencies.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Tuple of (encoder_hidden_states, hidden_states) matching diffusers output.
    """
    img_np = hidden_states.to_numpy()
    txt_np = encoder_hidden_states.to_numpy()

    B, img_len, D = img_np.shape
    _, txt_len, _ = txt_np.shape

    # Concatenate for processing: [txt, img]
    x_np = np.concatenate([txt_np, img_np], axis=1)  # [B, txt_len + img_len, D]
    seq_len = txt_len + img_len
    residual = x_np.copy()

    # Get weights helper
    def get_weight(name: str) -> GPUArray | None:
        return weights.get(f"{prefix}.{name}")

    # AdaLN (3 outputs for single block)
    norm_linear_w = get_weight("norm.linear.weight")
    norm_linear_b = get_weight("norm.linear.bias")
    x_mod, gate = adaln_zero(
        from_numpy(x_np.astype(np.float32)), temb, norm_linear_w, norm_linear_b, num_outputs=3
    )

    # Self-attention (GPU-native, no output projection in single blocks)
    attn_out = single_attention(
        x_mod,
        q_weight=weights[f"{prefix}.attn.to_q.weight"],
        k_weight=weights[f"{prefix}.attn.to_k.weight"],
        v_weight=weights[f"{prefix}.attn.to_v.weight"],
        q_bias=weights.get(f"{prefix}.attn.to_q.bias"),
        k_bias=weights.get(f"{prefix}.attn.to_k.bias"),
        v_bias=weights.get(f"{prefix}.attn.to_v.bias"),
        norm_q_weight=weights[f"{prefix}.attn.norm_q.weight"],
        norm_k_weight=weights[f"{prefix}.attn.norm_k.weight"],
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    attn_out_np = attn_out.to_numpy()

    # Parallel MLP
    proj_mlp_w = get_weight("proj_mlp.weight")
    proj_mlp_b = get_weight("proj_mlp.bias")

    x_mod_np = x_mod.to_numpy()
    x_mod_2d = x_mod_np.reshape(B * seq_len, D)
    mlp_hidden = gpu_linear(from_numpy(x_mod_2d.astype(np.float32)), proj_mlp_w, proj_mlp_b)
    mlp_hidden = gpu_gelu(mlp_hidden)
    mlp_hidden_np = mlp_hidden.to_numpy().reshape(B, seq_len, -1)

    # Concatenate attention and MLP outputs
    combined = np.concatenate([attn_out_np, mlp_hidden_np], axis=-1)

    # Output projection with gating
    proj_out_w = get_weight("proj_out.weight")
    proj_out_b = get_weight("proj_out.bias")

    combined_2d = combined.reshape(B * seq_len, -1)
    output = gpu_linear(from_numpy(combined_2d.astype(np.float32)), proj_out_w, proj_out_b)
    output_np = output.to_numpy().reshape(B, seq_len, D)

    # Apply gating and residual
    gate_np = gate.to_numpy()
    output_np = gate_np[:, None, :] * output_np
    output_np = residual + output_np

    # Split back to txt and img
    txt_out = output_np[:, :txt_len, :]
    img_out = output_np[:, txt_len:, :]

    # Return tuple matching diffusers: (encoder_hidden_states, hidden_states)
    return from_numpy(txt_out.astype(np.float32)), from_numpy(img_out.astype(np.float32))


__all__ = [
    "adaln_zero",
    "gelu",
    "feedforward",
    "joint_block",
    "single_block",
]
