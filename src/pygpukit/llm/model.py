"""LLM model components for PyGPUkit.

Provides transformer building blocks for GPT-2 style models:
- MLP block (fc1 -> gelu -> fc2)
- TransformerBlock (ln -> mlp -> residual)
- GPT2Model (embedding -> blocks -> lm_head)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.basic import add, bias_add_inplace, gelu, layernorm, matmul, transpose

if TYPE_CHECKING:
    pass


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model.

    GPT-2 Small defaults:
        vocab_size=50257, n_embd=768, n_layer=12, n_head=12
    """

    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_positions: int = 1024
    layer_norm_eps: float = 1e-5

    @property
    def n_inner(self) -> int:
        """Inner dimension of MLP (4 * n_embd)."""
        return 4 * self.n_embd


class Linear:
    """Linear layer: y = xW^T + b

    Weights are stored as [out_features, in_features] (PyTorch convention).
    Forward pass uses GPU transpose and bias_add_inplace for efficiency.
    """

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray | None = None,
    ):
        """Initialize Linear layer.

        Args:
            weight: Weight matrix [out_features, in_features]
            bias: Optional bias vector [out_features]
        """
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.ndim}D")
        self.weight = weight  # [out_features, in_features]
        self.bias = bias
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]
        # Pre-transpose weight for efficient forward pass
        self._weight_t: GPUArray | None = None

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: y = xW^T + b

        Args:
            x: Input tensor [batch, in_features]

        Returns:
            Output tensor [batch, out_features]
        """
        if x.ndim != 2:
            raise ValueError(f"input must be 2D [batch, in_features], got {x.ndim}D")
        if x.shape[1] != self.in_features:
            raise ValueError(f"input features {x.shape[1]} doesn't match weight {self.in_features}")

        # Lazy transpose - compute once and cache
        # weight: [out_features, in_features] -> weight_t: [in_features, out_features]
        if self._weight_t is None:
            self._weight_t = transpose(self.weight)

        # y = x @ weight_t: [batch, in_features] @ [in_features, out_features] = [batch, out_features]
        y = matmul(x, self._weight_t)

        # Add bias in-place on GPU if present
        if self.bias is not None:
            bias_add_inplace(y, self.bias)

        return y


class MLP:
    """MLP block for GPT-2.

    Structure: fc1 -> gelu -> fc2
    fc1: [n_embd] -> [n_inner]
    fc2: [n_inner] -> [n_embd]
    """

    def __init__(
        self,
        c_fc_weight: GPUArray,
        c_fc_bias: GPUArray | None,
        c_proj_weight: GPUArray,
        c_proj_bias: GPUArray | None,
    ):
        """Initialize MLP block.

        Args:
            c_fc_weight: First linear weight [n_inner, n_embd]
            c_fc_bias: First linear bias [n_inner]
            c_proj_weight: Second linear weight [n_embd, n_inner]
            c_proj_bias: Second linear bias [n_embd]
        """
        self.c_fc = Linear(c_fc_weight, c_fc_bias)
        self.c_proj = Linear(c_proj_weight, c_proj_bias)

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: fc1 -> gelu -> fc2

        Args:
            x: Input tensor [batch, n_embd]

        Returns:
            Output tensor [batch, n_embd]
        """
        h = self.c_fc(x)
        h = gelu(h)
        h = self.c_proj(h)
        return h


class LayerNorm:
    """Layer normalization with learnable parameters."""

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray,
        eps: float = 1e-5,
    ):
        """Initialize LayerNorm.

        Args:
            weight: Scale parameter (gamma) [features]
            bias: Shift parameter (beta) [features]
            eps: Epsilon for numerical stability
        """
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass.

        Args:
            x: Input tensor [batch, features]

        Returns:
            Normalized tensor [batch, features]
        """
        return layernorm(x, self.weight, self.bias, self.eps)


class CausalSelfAttention:
    """Causal self-attention for GPT-2.

    Structure:
    - c_attn: [n_embd] -> [3*n_embd] (Q, K, V projection)
    - Split into n_head heads
    - Q @ K^T / sqrt(d_k) with causal mask
    - Softmax
    - Attention @ V
    - c_proj: [n_embd] -> [n_embd]
    """

    def __init__(
        self,
        c_attn_weight: GPUArray,
        c_attn_bias: GPUArray | None,
        c_proj_weight: GPUArray,
        c_proj_bias: GPUArray | None,
        n_head: int,
        n_embd: int,
    ):
        """Initialize CausalSelfAttention.

        Args:
            c_attn_weight: QKV projection weight [3*n_embd, n_embd]
            c_attn_bias: QKV projection bias [3*n_embd]
            c_proj_weight: Output projection weight [n_embd, n_embd]
            c_proj_bias: Output projection bias [n_embd]
            n_head: Number of attention heads
            n_embd: Embedding dimension
        """
        self.c_attn = Linear(c_attn_weight, c_attn_bias)
        self.c_proj = Linear(c_proj_weight, c_proj_bias)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass with causal self-attention.

        Args:
            x: Input tensor [seq_len, n_embd]

        Returns:
            Output tensor [seq_len, n_embd]
        """
        import numpy as np

        seq_len = x.shape[0]

        # QKV projection: [seq_len, n_embd] -> [seq_len, 3*n_embd]
        qkv = self.c_attn(x)
        qkv_np = qkv.to_numpy()

        # Split into Q, K, V: each [seq_len, n_embd]
        q_np = qkv_np[:, :self.n_embd]
        k_np = qkv_np[:, self.n_embd:2*self.n_embd]
        v_np = qkv_np[:, 2*self.n_embd:]

        # Reshape for multi-head: [seq_len, n_head, head_dim]
        q_np = q_np.reshape(seq_len, self.n_head, self.head_dim)
        k_np = k_np.reshape(seq_len, self.n_head, self.head_dim)
        v_np = v_np.reshape(seq_len, self.n_head, self.head_dim)

        # Transpose to [n_head, seq_len, head_dim] for batched attention
        q_np = q_np.transpose(1, 0, 2)
        k_np = k_np.transpose(1, 0, 2)
        v_np = v_np.transpose(1, 0, 2)

        # Compute attention scores: [n_head, seq_len, seq_len]
        scale = 1.0 / np.sqrt(self.head_dim)
        # Q @ K^T: [n_head, seq_len, head_dim] @ [n_head, head_dim, seq_len]
        attn_scores = np.matmul(q_np, k_np.transpose(0, 2, 1)) * scale

        # Apply causal mask (lower triangular)
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        attn_scores[:, causal_mask] = -1e9

        # Softmax over last dimension
        attn_scores_max = attn_scores.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn_scores - attn_scores_max)
        attn_weights = attn_exp / attn_exp.sum(axis=-1, keepdims=True)

        # Attention @ V: [n_head, seq_len, head_dim]
        attn_output = np.matmul(attn_weights, v_np)

        # Transpose back: [seq_len, n_head, head_dim]
        attn_output = attn_output.transpose(1, 0, 2)

        # Reshape to [seq_len, n_embd]
        attn_output = attn_output.reshape(seq_len, self.n_embd)

        # Output projection
        out = from_numpy(attn_output.astype(np.float32))
        out = self.c_proj(out)

        return out


class TransformerBlock:
    """Full transformer block with attention and MLP.

    Structure: ln_1 -> attention -> residual -> ln_2 -> mlp -> residual
    """

    def __init__(
        self,
        ln_1_weight: GPUArray,
        ln_1_bias: GPUArray,
        attn: CausalSelfAttention | None,
        ln_2_weight: GPUArray,
        ln_2_bias: GPUArray,
        mlp: MLP,
        eps: float = 1e-5,
    ):
        """Initialize TransformerBlock.

        Args:
            ln_1_weight: First LayerNorm weight [n_embd]
            ln_1_bias: First LayerNorm bias [n_embd]
            attn: CausalSelfAttention module (None for MLP-only mode)
            ln_2_weight: Second LayerNorm weight [n_embd]
            ln_2_bias: Second LayerNorm bias [n_embd]
            mlp: MLP block
            eps: LayerNorm epsilon
        """
        self.ln_1 = LayerNorm(ln_1_weight, ln_1_bias, eps)
        self.attn = attn
        self.ln_2 = LayerNorm(ln_2_weight, ln_2_bias, eps)
        self.mlp = mlp

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: ln_1 -> attn -> residual -> ln_2 -> mlp -> residual

        Args:
            x: Input tensor [seq_len, n_embd]

        Returns:
            Output tensor [seq_len, n_embd]
        """
        # Attention block (if available)
        if self.attn is not None:
            h = self.ln_1(x)
            h = self.attn(h)
            x = add(x, h)

        # MLP block
        h = self.ln_2(x)
        h = self.mlp(h)
        return add(x, h)


class GPT2Model:
    """GPT-2 model with full transformer blocks.

    Structure:
    - Token embedding
    - Position embedding
    - Transformer blocks (attention + MLP)
    - Final LayerNorm
    - LM head (tied to embedding)
    """

    def __init__(
        self,
        config: GPT2Config,
        wte: GPUArray,  # Token embedding [vocab_size, n_embd]
        wpe: GPUArray,  # Position embedding [n_positions, n_embd]
        blocks: list[TransformerBlock],
        ln_f_weight: GPUArray,
        ln_f_bias: GPUArray,
    ):
        """Initialize GPT-2 model.

        Args:
            config: Model configuration
            wte: Token embedding weights [vocab_size, n_embd]
            wpe: Position embedding weights [n_positions, n_embd]
            blocks: List of transformer blocks
            ln_f_weight: Final LayerNorm weight
            ln_f_bias: Final LayerNorm bias
        """
        self.config = config
        self.wte = wte
        self.wpe = wpe
        self.blocks = blocks
        self.ln_f = LayerNorm(ln_f_weight, ln_f_bias, config.layer_norm_eps)

    def __call__(self, input_ids: list[int], position_ids: list[int] | None = None) -> GPUArray:
        """Forward pass.

        Args:
            input_ids: Token IDs [seq_len]
            position_ids: Optional position IDs [seq_len]

        Returns:
            Hidden states [seq_len, n_embd]
        """
        import numpy as np

        seq_len = len(input_ids)

        if position_ids is None:
            position_ids = list(range(seq_len))

        # Get embeddings by indexing (CPU for MVP)
        wte_np = self.wte.to_numpy()
        wpe_np = self.wpe.to_numpy()

        # Token embeddings: select rows from wte
        token_embeds = wte_np[input_ids]  # [seq_len, n_embd]

        # Position embeddings: select rows from wpe
        pos_embeds = wpe_np[position_ids]  # [seq_len, n_embd]

        # Combine embeddings
        hidden = from_numpy((token_embeds + pos_embeds).astype(np.float32))

        # Apply transformer blocks
        for block in self.blocks:
            hidden = block(hidden)

        # Final LayerNorm
        hidden = self.ln_f(hidden)

        return hidden

    def lm_head(self, hidden: GPUArray) -> GPUArray:
        """Compute logits from hidden states.

        Args:
            hidden: Hidden states [seq_len, n_embd]

        Returns:
            Logits [seq_len, vocab_size]
        """
        # LM head is tied to embedding weights
        # logits = hidden @ wte.T
        wte_np = self.wte.to_numpy()
        hidden_np = hidden.to_numpy()
        logits = hidden_np @ wte_np.T
        return from_numpy(logits.astype(hidden_np.dtype))

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = greedy argmax)

        Returns:
            List of all token IDs (input + generated)
        """
        import numpy as np

        tokens = list(input_ids)

        for _ in range(max_new_tokens):
            # Truncate to max context length
            context = tokens[-self.config.n_positions :]

            # Forward pass
            hidden = self(context)

            # Get logits for last position
            logits = self.lm_head(hidden)
            logits_np = logits.to_numpy()
            last_logits = logits_np[-1]  # [vocab_size]

            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature

            # Greedy decoding (argmax)
            next_token = int(np.argmax(last_logits))
            tokens.append(next_token)

            # Stop at EOS (50256 for GPT-2)
            if next_token == 50256:
                break

        return tokens


def load_gpt2_from_safetensors(
    model_path: str,
    config: GPT2Config | None = None,
    load_attention: bool = True,
) -> GPT2Model:
    """Load GPT-2 model from safetensors file.

    Args:
        model_path: Path to model.safetensors file
        config: Model configuration (defaults to GPT-2 small)
        load_attention: Whether to load attention weights (default: True)

    Returns:
        GPT2Model instance
    """
    from pygpukit.llm import SafeTensorsFile

    if config is None:
        config = GPT2Config()

    st = SafeTensorsFile(model_path)

    # Helper to load tensor
    def load_tensor(name: str) -> GPUArray:
        data = st.tensor_bytes(name)
        info = st.tensor_info(name)

        import numpy as np

        # Determine numpy dtype
        dtype_map = {
            0: np.float32,  # Float32
            1: np.float16,  # Float16
            2: np.float32,  # BFloat16 -> convert to float32 for now
            3: np.float64,  # Float64
        }
        np_dtype = dtype_map.get(info.dtype, np.float32)

        # Create numpy array from bytes
        arr = np.frombuffer(data, dtype=np_dtype).reshape(info.shape)
        return from_numpy(arr.copy())

    def try_load_tensor(name: str) -> GPUArray | None:
        """Try to load tensor, return None if not found."""
        if name in st.tensor_names:
            return load_tensor(name)
        return None

    # Load embeddings
    wte = load_tensor("wte.weight")
    wpe = load_tensor("wpe.weight")

    # Load blocks
    blocks = []
    for i in range(config.n_layer):
        prefix = f"h.{i}."

        # Check if MLP weights exist
        mlp_c_fc_w_name = f"{prefix}mlp.c_fc.weight"
        if mlp_c_fc_w_name not in st.tensor_names:
            # Skip blocks without MLP (shouldn't happen for GPT-2)
            continue

        # LayerNorm 1 (before attention)
        ln_1_w = load_tensor(f"{prefix}ln_1.weight")
        ln_1_b = load_tensor(f"{prefix}ln_1.bias")

        # Attention (optional)
        attn = None
        if load_attention:
            attn_c_attn_w_name = f"{prefix}attn.c_attn.weight"
            if attn_c_attn_w_name in st.tensor_names:
                attn_c_attn_w = load_tensor(attn_c_attn_w_name)
                attn_c_attn_b = try_load_tensor(f"{prefix}attn.c_attn.bias")
                attn_c_proj_w = load_tensor(f"{prefix}attn.c_proj.weight")
                attn_c_proj_b = try_load_tensor(f"{prefix}attn.c_proj.bias")

                attn = CausalSelfAttention(
                    attn_c_attn_w, attn_c_attn_b,
                    attn_c_proj_w, attn_c_proj_b,
                    config.n_head, config.n_embd
                )

        # LayerNorm 2 (before MLP)
        ln_2_w = load_tensor(f"{prefix}ln_2.weight")
        ln_2_b = load_tensor(f"{prefix}ln_2.bias")

        # MLP
        mlp_c_fc_w = load_tensor(f"{prefix}mlp.c_fc.weight")
        mlp_c_fc_b = try_load_tensor(f"{prefix}mlp.c_fc.bias")
        mlp_c_proj_w = load_tensor(f"{prefix}mlp.c_proj.weight")
        mlp_c_proj_b = try_load_tensor(f"{prefix}mlp.c_proj.bias")

        mlp = MLP(mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b)
        block = TransformerBlock(
            ln_1_w, ln_1_b, attn,
            ln_2_w, ln_2_b, mlp,
            config.layer_norm_eps
        )
        blocks.append(block)

    # Final LayerNorm
    ln_f_w = load_tensor("ln_f.weight")
    ln_f_b = load_tensor("ln_f.bias")

    return GPT2Model(config, wte, wpe, blocks, ln_f_w, ln_f_b)
