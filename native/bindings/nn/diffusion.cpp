/**
 * Diffusion model operations: GroupNorm, AdaLN, Cross-Attention, Conv2D
 */
#include "../bindings_common.hpp"

void init_nn_diffusion(py::module_& m) {
    // GroupNorm
    m.def("group_norm", &ops::group_norm,
          py::arg("input"), py::arg("gamma"), py::arg("beta"),
          py::arg("num_groups"), py::arg("eps") = 1e-5f,
          "Group normalization for diffusion models (VAE, UNet)\n"
          "input: [N, C, H, W], gamma/beta: [C]\n"
          "Normalizes over (C/num_groups, H, W) for each group");

    // AdaLN
    m.def("adaln", &ops::adaln,
          py::arg("input"), py::arg("scale"), py::arg("shift"),
          py::arg("eps") = 1e-5f,
          "Adaptive Layer Normalization for DiT models\n"
          "y = (x - mean) / sqrt(var + eps) * (1 + scale) + shift\n"
          "input: [B, N, D], scale/shift: [B, D]");

    // AdaLN-Zero
    m.def("adaln_zero", &ops::adaln_zero,
          py::arg("input"), py::arg("scale"), py::arg("shift"),
          py::arg("gate"), py::arg("residual"), py::arg("eps") = 1e-5f,
          "AdaLN-Zero for DiT with gated residual\n"
          "y = residual + gate * (normalized * (1 + scale) + shift)\n"
          "input: [B, N, D], scale/shift/gate: [B, D], residual: [B, N, D]");

    // Cross-Attention
    m.def("cross_attention", &ops::cross_attention,
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale") = 0.0f,
          "Cross-attention for text-to-image conditioning (no causal mask)\n"
          "Q: [n_heads, q_len, head_dim] (from image latents)\n"
          "K: [n_heads, kv_len, head_dim] (from text embeddings)\n"
          "V: [n_heads, kv_len, head_dim]\n"
          "scale: 1/sqrt(head_dim), computed automatically if <= 0");

    // Conv2D 1x1
    m.def("conv2d_1x1", &ops::conv2d_1x1,
          py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
          "1x1 pointwise convolution\n"
          "input: [N, C_in, H, W], weight: [C_out, C_in]\n"
          "bias: [C_out] or None");

    // Conv2D 3x3
    m.def("conv2d_3x3", &ops::conv2d_3x3,
          py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
          py::arg("pad_h") = 1, py::arg("pad_w") = 1,
          py::arg("stride_h") = 1, py::arg("stride_w") = 1,
          "3x3 direct convolution (optimized)\n"
          "input: [N, C_in, H, W], weight: [C_out, C_in, 3, 3]");

    // im2col
    m.def("im2col", &ops::im2col,
          py::arg("input"),
          py::arg("K_h"), py::arg("K_w"),
          py::arg("pad_h"), py::arg("pad_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("dil_h") = 1, py::arg("dil_w") = 1,
          "im2col for general convolution\n"
          "input: [N, C, H, W] -> output: [N, C*K_h*K_w, H_out*W_out]\n"
          "Use with GEMM for Conv2D");

    // col2im
    m.def("col2im", &ops::col2im,
          py::arg("input"),
          py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("K_h"), py::arg("K_w"),
          py::arg("pad_h"), py::arg("pad_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("dil_h") = 1, py::arg("dil_w") = 1,
          "col2im for transposed convolution\n"
          "input: [N, C*K_h*K_w, H_in*W_in] -> output: [N, C, H, W]");

    // ========================================================================
    // FLUX-specific operations
    // ========================================================================

    // LayerNorm simple (no learnable params)
    m.def("layer_norm_simple", &ops::layer_norm_simple,
          py::arg("input"), py::arg("eps") = 1e-6f,
          "Layer normalization without learnable parameters\n"
          "input: [B, N, D] or [rows, D]");

    // Modulate
    m.def("modulate", &ops::modulate,
          py::arg("input"), py::arg("scale"), py::arg("shift"),
          "Modulate: y = x * (1 + scale) + shift\n"
          "input: [B, N, D], scale/shift: [B, D]");

    // Gated residual
    m.def("gated_residual", &ops::gated_residual,
          py::arg("residual"), py::arg("gate"), py::arg("value"),
          "Gated residual: y = residual + gate * value\n"
          "residual/value: [B, N, D], gate: [B, D]");

    // Scale tensor
    m.def("scale_tensor", &ops::scale_tensor,
          py::arg("input"), py::arg("scale"),
          "Scale tensor: y = x * scale");

    // Concat axis 1 (3D and 4D)
    m.def("concat_axis1", &ops::concat_axis1,
          py::arg("a"), py::arg("b"),
          "Concatenate along axis 1\n"
          "3D: [B, N1, D] + [B, N2, D] -> [B, N1+N2, D]\n"
          "4D: [B, N1, H, D] + [B, N2, H, D] -> [B, N1+N2, H, D]");

    // Split axis 1 (3D and 4D)
    m.def("split_axis1", &ops::split_axis1,
          py::arg("input"), py::arg("split_size"),
          "Split along axis 1\n"
          "3D: [B, N, D] -> ([B, split_size, D], [B, N-split_size, D])\n"
          "4D: [B, N, H, D] -> ([B, split_size, H, D], [B, N-split_size, H, D])");

    // Apply RoPE
    m.def("apply_rope", &ops::apply_rope,
          py::arg("x"), py::arg("cos_freq"), py::arg("sin_freq"),
          "Apply Rotary Position Embedding\n"
          "x: [B, N, H, D], cos/sin: [N, D]");
}
