# LLM Support Guide

PyGPUkit provides native support for loading and running LLM models with efficient GPU acceleration.

## Quick Start

```python
from pygpukit.llm import load_model_from_safetensors, detect_model_spec, load_safetensors

# Auto-detect and load any supported model
st = load_safetensors("model.safetensors")
spec = detect_model_spec(st.tensor_names)
model = load_model_from_safetensors("model.safetensors", dtype="float16", spec=spec)

# Generate text (use HuggingFace tokenizers for production)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
input_ids = tokenizer.encode("Hello, world!").ids

output_ids = model.generate(
    input_ids,
    max_new_tokens=32,
    temperature=0.7,
    use_cache=True,
)
print(tokenizer.decode(output_ids))
```

---

## Supported Models

| Architecture | Models | Features |
|--------------|--------|----------|
| **GPT-2** | GPT-2 (all sizes) | LayerNorm, GELU, Position Embedding |
| **LLaMA** | LLaMA 2/3, TinyLlama, Mistral | RMSNorm, SiLU, RoPE, GQA |
| **Qwen3** | Qwen3 (all sizes) | RMSNorm, SiLU, RoPE, GQA, QK-Norm |

---

## Tokenizer Policy

> **Important:** PyGPUkit's core responsibility is **GPU execution**, not tokenization.

- The model API expects **token IDs as input**, not raw text
- For production use, we recommend [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)
- The built-in `Tokenizer` class is **experimental** and intended for demos only

```python
# Recommended: HuggingFace tokenizers
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
input_ids = tokenizer.encode("Hello").ids
output_text = tokenizer.decode(output_ids)

# Experimental: PyGPUkit built-in (demos only)
from pygpukit.llm import Tokenizer
tok = Tokenizer("tokenizer.json")  # May not work with all formats
```

---

## SafeTensors Loading

### Single File

```python
from pygpukit.llm import SafeTensorsFile, load_safetensors

# Load a safetensors file
st = load_safetensors("model.safetensors")

# File information
print(f"Number of tensors: {st.num_tensors}")
print(f"File size: {st.file_size / 1e9:.2f} GB")
print(f"Tensors: {st.tensor_names}")
```

### Sharded Models (Large Models)

```python
from pygpukit.llm import load_safetensors

# Automatically handles sharded models
st = load_safetensors("model.safetensors.index.json")
print(f"Shards: {len(st._shard_files)}")
print(f"Total tensors: {st.num_tensors}")

# Access tensors transparently (lazy loading)
info = st.tensor_info("model.embed_tokens.weight")
data = st.tensor_bytes("model.embed_tokens.weight")
```

### Tensor Metadata

```python
from pygpukit.llm import SafeTensorsFile

st = SafeTensorsFile("model.safetensors")

# Get tensor info without loading data
info = st.tensor_info("model.embed_tokens.weight")
print(f"Name: {info.name}")
print(f"Shape: {info.shape}")
print(f"Dtype: {info.dtype_name}")  # float16, bfloat16, float32
print(f"Size: {info.size_bytes / 1e6:.1f} MB")
```

---

## Model Loading

### Automatic Detection

```python
from pygpukit.llm import load_model_from_safetensors, detect_model_spec, load_safetensors

# Load safetensors and detect model type
st = load_safetensors("model.safetensors")
spec = detect_model_spec(st.tensor_names)
print(f"Detected: {spec.name}")  # "gpt2", "llama", or "qwen3"

# Load model with detected spec
model = load_model_from_safetensors(
    "model.safetensors",
    dtype="float16",  # or "float32"
    spec=spec,
)
```

### Architecture-Specific Loaders

```python
from pygpukit.llm import (
    load_gpt2_from_safetensors,
    load_llama_from_safetensors,
    load_qwen3_from_safetensors,
)

# GPT-2
model = load_gpt2_from_safetensors("gpt2.safetensors")

# LLaMA / Mistral
model = load_llama_from_safetensors("llama.safetensors", dtype="float16")

# Qwen3
model = load_qwen3_from_safetensors("qwen3.safetensors", dtype="float16")
```

### ModelSpec

```python
from pygpukit.llm import GPT2_SPEC, LLAMA_SPEC, QWEN3_SPEC, MODEL_SPECS

# Pre-defined specs
print(GPT2_SPEC.name)        # "gpt2"
print(GPT2_SPEC.norm_type)   # "layernorm"
print(GPT2_SPEC.activation)  # "gelu"
print(GPT2_SPEC.use_rope)    # False

print(LLAMA_SPEC.name)       # "llama"
print(LLAMA_SPEC.norm_type)  # "rmsnorm"
print(LLAMA_SPEC.activation) # "silu"
print(LLAMA_SPEC.use_rope)   # True

print(QWEN3_SPEC.name)       # "qwen3"
print(QWEN3_SPEC.use_qk_norm) # True (QK normalization)

# Registry
MODEL_SPECS["gpt2"]   # GPT2_SPEC
MODEL_SPECS["llama"]  # LLAMA_SPEC
MODEL_SPECS["qwen3"]  # QWEN3_SPEC
MODEL_SPECS["qwen2"]  # LLAMA_SPEC (uses LLaMA structure)
```

---

## Text Generation

### Basic Generation

```python
from pygpukit.llm import load_model_from_safetensors, detect_model_spec, load_safetensors
from tokenizers import Tokenizer

# Load model
st = load_safetensors("model.safetensors")
spec = detect_model_spec(st.tensor_names)
model = load_model_from_safetensors("model.safetensors", dtype="float16", spec=spec)

# Tokenize
tokenizer = Tokenizer.from_file("tokenizer.json")
input_ids = tokenizer.encode("The quick brown fox").ids

# Generate with KV-cache
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    eos_token_id=tokenizer.token_to_id("</s>"),
    use_cache=True,  # Enable KV-cache for faster generation
)

print(tokenizer.decode(output_ids))
```

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_ids` | `list[int]` | required | Input token IDs |
| `max_new_tokens` | `int` | 100 | Maximum tokens to generate |
| `temperature` | `float` | 1.0 | Sampling temperature (0 = greedy) |
| `top_k` | `int` | 50 | Top-k sampling |
| `top_p` | `float` | 1.0 | Nucleus sampling threshold |
| `eos_token_id` | `int` | None | Stop at this token |
| `use_cache` | `bool` | True | Enable KV-cache |

### Manual Forward Pass

```python
# Forward pass without generation
hidden, kv_cache = model(input_ids, use_cache=True)

# Get logits
logits = model.get_logits(hidden)
logits_np = logits.to_numpy()

# Get next token (greedy)
next_token = int(logits_np[-1].argmax())

# Continue with KV-cache
hidden, kv_cache = model([next_token], past_key_values=kv_cache, use_cache=True)
```

---

## Hybrid Attention

PyGPUkit uses hybrid CPU/GPU attention for optimal performance:

| Phase | Backend | Reason |
|-------|---------|--------|
| **Prefill** (seq_len > 1) | GPU SDPA | Parallelizable, high throughput |
| **Decode** (seq_len = 1) | CPU | Avoids kernel launch overhead |

This is automatic and requires no configuration.

---

## Model Components

### TransformerConfig

```python
from pygpukit.llm import TransformerConfig

config = TransformerConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,      # GQA: fewer KV heads than Q heads
    intermediate_size=14336,
    norm_type="rmsnorm", # "rmsnorm" or "layernorm"
    activation="silu",   # "silu" or "gelu"
    use_rope=True,
    max_position_embeddings=4096,
    norm_eps=1e-5,
    rope_theta=10000.0,
)

# Computed properties
print(config.head_dim)      # hidden_size // num_heads
print(config.num_kv_groups) # num_heads // num_kv_heads
```

### CausalTransformerModel

```python
from pygpukit.llm import CausalTransformerModel

# All model aliases point to CausalTransformerModel
from pygpukit.llm import GPT2Model, LlamaModel
assert GPT2Model is CausalTransformerModel
assert LlamaModel is CausalTransformerModel

# Model properties
model.config        # TransformerConfig
model.spec          # ModelSpec (GPT2_SPEC, LLAMA_SPEC, etc.)
model.embed_tokens  # Embedding weights
model.blocks        # List of TransformerBlock
model.final_norm    # Final layer norm
model.lm_head       # LM head weights (may be tied to embed_tokens)
```

### Building Blocks

```python
from pygpukit.llm import (
    Attention,      # Unified attention (hybrid CPU/GPU)
    MLP,            # Feed-forward network
    Norm,           # RMSNorm or LayerNorm
    TransformerBlock,
    Linear,
)

# Aliases for compatibility
from pygpukit.llm import (
    RMSNorm,            # = Norm
    LayerNorm,          # = Norm
    CausalSelfAttention, # = Attention
    LlamaAttention,     # = Attention
    LlamaMLP,           # = MLP
    LlamaBlock,         # = TransformerBlock
)
```

---

## Performance

### Tested Results (RTX 3090 Ti)

| Model | Size | Dtype | Throughput |
|-------|------|-------|------------|
| GPT-2 | 124M | FP32 | 8.7 tok/s |
| TinyLlama | 1.1B | FP16 | 1.8 tok/s |
| Qwen3 | 8B | FP16 | 0.2 tok/s |

> **Note:** Current implementation uses hybrid CPU/GPU attention. Full GPU attention will significantly improve decode performance.

### Memory Usage

| Model | FP32 | FP16 |
|-------|------|------|
| GPT-2 (124M) | ~500 MB | ~250 MB |
| LLaMA 7B | ~28 GB | ~14 GB |
| Qwen3 8B | ~32 GB | ~16 GB |

---

## API Reference

### SafeTensorsFile

| Method/Property | Description |
|-----------------|-------------|
| `SafeTensorsFile(path)` | Open safetensors file |
| `load_safetensors(path)` | Auto-detect single/sharded |
| `.tensor_names` | List of tensor names |
| `.num_tensors` | Number of tensors |
| `.file_size` | File size in bytes |
| `.tensor_info(name)` | Get TensorInfo |
| `.tensor_bytes(name)` | Get raw bytes |
| `.tensor_as_f32(name)` | Get as float32 numpy array |

### Model Loading

| Function | Description |
|----------|-------------|
| `load_model_from_safetensors(path, dtype, spec)` | Unified loader |
| `detect_model_spec(tensor_names)` | Auto-detect architecture |
| `load_gpt2_from_safetensors(path, dtype)` | Load GPT-2 |
| `load_llama_from_safetensors(path, dtype)` | Load LLaMA |
| `load_qwen3_from_safetensors(path, dtype)` | Load Qwen3 |

### CausalTransformerModel

| Method | Description |
|--------|-------------|
| `__call__(input_ids, position_ids, past_key_values, use_cache)` | Forward pass |
| `generate(input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id, use_cache)` | Text generation |
| `get_logits(hidden)` | Compute logits from hidden states |

### Tokenizer (Experimental)

| Method/Property | Description |
|-----------------|-------------|
| `Tokenizer(path)` | Load from tokenizer.json |
| `.vocab_size` | Vocabulary size |
| `.bos_token_id` | BOS token ID |
| `.eos_token_id` | EOS token ID |
| `.encode(text)` | Encode text to IDs |
| `.decode(ids)` | Decode IDs to text |

---

## Complete Example

```python
"""End-to-end LLM inference with PyGPUkit."""
from pygpukit.llm import load_model_from_safetensors, detect_model_spec, load_safetensors
from tokenizers import Tokenizer
import time

# Paths (adjust for your model)
MODEL_PATH = "model.safetensors"
TOKENIZER_PATH = "tokenizer.json"

# Load model
print("Loading model...")
st = load_safetensors(MODEL_PATH)
spec = detect_model_spec(st.tensor_names)
print(f"Detected architecture: {spec.name}")

model = load_model_from_safetensors(MODEL_PATH, dtype="float16", spec=spec)
print(f"Layers: {model.config.num_layers}, Hidden: {model.config.hidden_size}")

# Load tokenizer (HuggingFace)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Generate
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt).ids
print(f"Prompt: {prompt}")
print(f"Input tokens: {len(input_ids)}")

start = time.perf_counter()
output_ids = model.generate(
    input_ids,
    max_new_tokens=32,
    temperature=0.7,
    use_cache=True,
)
elapsed = time.perf_counter() - start

output_text = tokenizer.decode(output_ids)
new_tokens = len(output_ids) - len(input_ids)

print(f"Output: {output_text}")
print(f"Generated {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
```
