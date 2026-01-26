"""Llama 4 model implementation for PyGPUkit.

Llama 4 architecture differences from Llama 3:
- QK L2 normalization (no gamma, parameterless)
- iRoPE temperature scaling instead of RoPE
- All layers use NoPE (no_rope_layers=[1]*48)

Reference: HuggingFace Transformers Llama4
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.basic import (
    matmul,
    rmsnorm,
    silu,
)
from pygpukit.ops.nn import l2norm, sdpa_irope


@dataclass
class Llama4Config:
    """Llama 4 text model configuration."""

    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    attn_scale: float = 0.1
    floor_scale: float = 8192.0
    use_qk_norm: bool = True
    max_position_embeddings: int = 10485760
    no_rope_layers: list[int] | None = None  # 1 = NoPE (no RoPE), 0 = RoPE

    @classmethod
    def from_json(cls, path: str | Path) -> Llama4Config:
        """Load config from HuggingFace config.json."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        text_config = data.get("text_config", data)
        return cls(
            vocab_size=text_config.get("vocab_size", 202048),
            hidden_size=text_config.get("hidden_size", 5120),
            intermediate_size=text_config.get("intermediate_size", 8192),
            num_hidden_layers=text_config.get("num_hidden_layers", 48),
            num_attention_heads=text_config.get("num_attention_heads", 40),
            num_key_value_heads=text_config.get("num_key_value_heads", 8),
            head_dim=text_config.get("head_dim", 128),
            rms_norm_eps=text_config.get("rms_norm_eps", 1e-5),
            attn_scale=text_config.get("attn_scale", 0.1),
            floor_scale=text_config.get("floor_scale", 8192.0),
            use_qk_norm=text_config.get("use_qk_norm", True),
            max_position_embeddings=text_config.get("max_position_embeddings", 10485760),
            no_rope_layers=text_config.get("no_rope_layers"),
        )


class Llama4Attention:
    """Llama 4 attention with QK L2 norm and iRoPE."""

    def __init__(
        self,
        q_proj: GPUArray,
        k_proj: GPUArray,
        v_proj: GPUArray,
        o_proj: GPUArray,
        config: Llama4Config,
        use_rope: bool = True,
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.config = config
        self.use_rope = use_rope

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

    def forward(self, hidden: GPUArray, positions: GPUArray) -> GPUArray:
        """Forward pass with QK norm and iRoPE SDPA.

        Args:
            hidden: [seq_len, hidden_size]
            positions: [seq_len] int64 position indices

        Returns:
            Output tensor [seq_len, hidden_size]
        """
        seq_len = hidden.shape[0]

        # Project Q, K, V
        # hidden: [seq_len, hidden_size]
        # q_proj.T: [hidden_size, num_heads * head_dim]
        q = matmul(hidden, self.q_proj)  # [seq_len, num_heads * head_dim]
        k = matmul(hidden, self.k_proj)  # [seq_len, num_kv_heads * head_dim]
        v = matmul(hidden, self.v_proj)  # [seq_len, num_kv_heads * head_dim]

        # Reshape to [seq_len, num_heads, head_dim]
        q = q.reshape((seq_len, self.num_heads, self.head_dim))
        k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))

        # Apply QK L2 normalization
        # Note: Per HuggingFace, QK norm is only applied when use_rope=True,
        # but empirically it improves stability for Llama Guard 4 NoPE layers too
        if self.config.use_qk_norm:
            # L2 norm: x * rsqrt(mean(x^2) + eps)
            # Reshape to [seq_len * num_heads, head_dim] for l2norm
            q_flat = q.reshape((seq_len * self.num_heads, self.head_dim))
            k_flat = k.reshape((seq_len * self.num_kv_heads, self.head_dim))
            # Use config's rms_norm_eps for L2 norm (default 1e-5)
            q_flat = l2norm(q_flat, eps=self.config.rms_norm_eps)
            k_flat = l2norm(k_flat, eps=self.config.rms_norm_eps)
            q = q_flat.reshape((seq_len, self.num_heads, self.head_dim))
            k = k_flat.reshape((seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to [num_heads, seq_len, head_dim] for SDPA
        q_t = q.transpose((1, 0, 2))  # [num_heads, seq_len, head_dim]
        k_t = k.transpose((1, 0, 2))  # [num_kv_heads, seq_len, head_dim]
        v_t = v.transpose((1, 0, 2))  # [num_kv_heads, seq_len, head_dim]

        # SDPA with iRoPE temperature scaling (Llama 4 specific)
        attn_out = sdpa_irope(
            q_t,
            k_t,
            v_t,
            positions,
            attn_scale=self.config.attn_scale,
            floor_scale=self.config.floor_scale,
            causal_offset=0,
        )  # [num_heads, seq_len, head_dim]

        # Transpose back to [seq_len, num_heads, head_dim]
        attn_out = attn_out.transpose((1, 0, 2))

        # Reshape to [seq_len, num_heads * head_dim]
        attn_out = attn_out.reshape((seq_len, self.num_heads * self.head_dim))

        # Output projection
        output = matmul(attn_out, self.o_proj)  # [seq_len, hidden_size]
        return output


class Llama4MLP:
    """Llama 4 MLP with SiLU activation."""

    def __init__(
        self,
        gate_proj: GPUArray,
        up_proj: GPUArray,
        down_proj: GPUArray,
    ):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, hidden: GPUArray) -> GPUArray:
        """Forward pass: down_proj(silu(gate_proj(x)) * up_proj(x))."""
        from pygpukit.ops.elementwise import mul

        gate = matmul(hidden, self.gate_proj)
        up = matmul(hidden, self.up_proj)
        gate = silu(gate)
        # Element-wise multiplication using native CUDA kernel
        gate_up = mul(gate, up)
        output = matmul(gate_up, self.down_proj)
        return output


class Llama4Block:
    """Single Llama 4 transformer block."""

    def __init__(
        self,
        attn: Llama4Attention,
        mlp: Llama4MLP,
        input_norm_weight: GPUArray,
        post_attn_norm_weight: GPUArray,
        rms_norm_eps: float,
    ):
        self.attn = attn
        self.mlp = mlp
        self.input_norm_weight = input_norm_weight
        self.post_attn_norm_weight = post_attn_norm_weight
        self.rms_norm_eps = rms_norm_eps

    def forward(self, hidden: GPUArray, positions: GPUArray) -> GPUArray:
        """Forward pass with residual connections."""
        from pygpukit.ops.basic import add

        # Self-attention with residual
        normed = rmsnorm(hidden, self.input_norm_weight, self.rms_norm_eps)
        attn_out = self.attn.forward(normed, positions)
        hidden = add(hidden, attn_out)

        # MLP with residual
        normed = rmsnorm(hidden, self.post_attn_norm_weight, self.rms_norm_eps)
        mlp_out = self.mlp.forward(normed)
        hidden = add(hidden, mlp_out)

        return hidden


class Llama4Model:
    """Llama 4 text model for inference."""

    def __init__(
        self,
        config: Llama4Config,
        embed_tokens: GPUArray,
        blocks: list[Llama4Block],
        final_norm_weight: GPUArray,
        lm_head: GPUArray,
    ):
        self.config = config
        self.embed_tokens = embed_tokens
        self.blocks = blocks
        self.final_norm_weight = final_norm_weight
        self.lm_head = lm_head

    def forward(self, input_ids: np.ndarray) -> GPUArray:
        """Forward pass.

        Args:
            input_ids: [seq_len] int64 token IDs

        Returns:
            Logits tensor [seq_len, vocab_size]
        """
        seq_len = len(input_ids)

        # Token embedding lookup
        embed_np = self.embed_tokens.to_numpy()
        hidden_np = embed_np[input_ids]
        hidden = from_numpy(hidden_np)

        # Position indices
        positions = from_numpy(np.arange(seq_len, dtype=np.int64))

        # Transformer blocks
        for block in self.blocks:
            hidden = block.forward(hidden, positions)

        # Final norm
        hidden = rmsnorm(hidden, self.final_norm_weight, self.config.rms_norm_eps)

        # LM head projection
        logits = matmul(hidden, self.lm_head)

        return logits

    @classmethod
    def from_safetensors(cls, model_path: str | Path) -> Llama4Model:
        """Load Llama 4 model from safetensors files.

        Args:
            model_path: Path to model directory containing config.json and safetensors

        Returns:
            Loaded Llama4Model instance
        """
        from pygpukit.llm.safetensors import Dtype, load_safetensors

        model_path = Path(model_path)

        # Load config
        config = Llama4Config.from_json(model_path / "config.json")
        print(
            f"Llama 4 config: {config.num_hidden_layers} layers, "
            f"{config.num_attention_heads} heads, {config.hidden_size} hidden"
        )

        # Load using PyGPUkit's safetensors loader
        index_path = model_path / "model.safetensors.index.json"
        st = load_safetensors(str(index_path))

        # Helper to get weight as GPUArray
        def get_weight(name: str) -> GPUArray:
            """Load tensor and convert to GPUArray."""
            info = st.tensor_info(name)
            data = st.tensor_bytes(name)

            # BF16 is stored as uint16, keep it that way for now
            if info.dtype == Dtype.BFloat16:
                arr = np.frombuffer(data, dtype=np.uint16).reshape(info.shape)
            elif info.dtype == Dtype.Float16:
                arr = np.frombuffer(data, dtype=np.float16).reshape(info.shape)
            elif info.dtype == Dtype.Float32:
                arr = np.frombuffer(data, dtype=np.float32).reshape(info.shape)
            else:
                raise ValueError(f"Unsupported dtype: {info.dtype_name}")

            return from_numpy(arr.copy())

        # Load embeddings
        embed_tokens = get_weight("language_model.model.embed_tokens.weight")

        # Load blocks
        blocks = []
        for i in range(config.num_hidden_layers):
            prefix = f"language_model.model.layers.{i}"

            # Attention weights (need to transpose for our matmul convention)
            q_proj = get_weight(f"{prefix}.self_attn.q_proj.weight")
            k_proj = get_weight(f"{prefix}.self_attn.k_proj.weight")
            v_proj = get_weight(f"{prefix}.self_attn.v_proj.weight")
            o_proj = get_weight(f"{prefix}.self_attn.o_proj.weight")

            # Transpose: HF stores [out, in], we need [in, out] for matmul(x, W)
            q_proj = q_proj.transpose((1, 0))
            k_proj = k_proj.transpose((1, 0))
            v_proj = v_proj.transpose((1, 0))
            o_proj = o_proj.transpose((1, 0))

            # Check if this layer uses RoPE (no_rope_layers[i] == 0) or NoPE (no_rope_layers[i] == 1)
            use_rope = True
            if config.no_rope_layers is not None and i < len(config.no_rope_layers):
                use_rope = config.no_rope_layers[i] == 0

            attn = Llama4Attention(q_proj, k_proj, v_proj, o_proj, config, use_rope=use_rope)

            # MLP weights
            gate_proj = get_weight(f"{prefix}.feed_forward.gate_proj.weight").transpose((1, 0))
            up_proj = get_weight(f"{prefix}.feed_forward.up_proj.weight").transpose((1, 0))
            down_proj = get_weight(f"{prefix}.feed_forward.down_proj.weight").transpose((1, 0))

            mlp = Llama4MLP(gate_proj, up_proj, down_proj)

            # Norm weights
            input_norm = get_weight(f"{prefix}.input_layernorm.weight")
            post_attn_norm = get_weight(f"{prefix}.post_attention_layernorm.weight")

            block = Llama4Block(attn, mlp, input_norm, post_attn_norm, config.rms_norm_eps)
            blocks.append(block)

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{config.num_hidden_layers} blocks")

        # Final norm
        final_norm = get_weight("language_model.model.norm.weight")

        # LM head (transpose for matmul)
        lm_head = get_weight("language_model.lm_head.weight").transpose((1, 0))

        print(f"Model loaded: {len(blocks)} blocks, {embed_tokens.shape[0]} vocab")

        return cls(config, embed_tokens, blocks, final_norm, lm_head)


def generate(
    model: Llama4Model,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    eos_token_id: int | list[int] = 200001,
) -> np.ndarray:
    """Simple greedy generation.

    Args:
        model: Llama4Model instance
        input_ids: Initial token IDs [seq_len]
        max_new_tokens: Maximum tokens to generate
        eos_token_id: EOS token ID(s) to stop generation

    Returns:
        Generated token IDs including input
    """
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = set(eos_token_id)

    current_ids = list(input_ids)

    for _ in range(max_new_tokens):
        # Forward pass
        logits = model.forward(np.array(current_ids, dtype=np.int64))

        # Get last token logits
        last_logits = logits.to_numpy()[-1]

        # Convert BF16 to float32 if needed
        if last_logits.dtype == np.uint16:
            last_logits = (last_logits.astype(np.uint32) << 16).view(np.float32)

        # Greedy: argmax
        next_token = int(np.argmax(last_logits))
        current_ids.append(next_token)

        if next_token in eos_token_ids:
            break

    return np.array(current_ids, dtype=np.int64)
