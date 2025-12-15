# LLM Support Guide

PyGPUkit provides native support for loading and running LLM models with efficient GPU acceleration.

## SafeTensors Loading

SafeTensors is a safe, fast format for storing tensors. PyGPUkit uses memory-mapped loading for zero-copy access.

### Basic Usage

```python
from pygpukit.llm import SafeTensorsFile, load_safetensors

# Load a safetensors file
st = load_safetensors("model.safetensors")

# File information
print(f"Number of tensors: {st.num_tensors}")
print(f"File size: {st.file_size / 1e9:.2f} GB")

# List all tensor names
print(f"Tensors: {st.tensor_names}")
```

### Tensor Metadata

```python
from pygpukit.llm import SafeTensorsFile

st = SafeTensorsFile("model.safetensors")

# Get tensor info without loading data
info = st.tensor_info("model.embed_tokens.weight")
print(f"Name: {info.name}")
print(f"Shape: {info.shape}")
print(f"Dtype: {info.dtype_name}")  # float16, bfloat16, float32, etc.
print(f"Size: {info.size_bytes / 1e6:.1f} MB")
print(f"Elements: {info.numel}")
```

### Loading Tensor Data

```python
from pygpukit.llm import SafeTensorsFile
import pygpukit as gpk
import numpy as np

st = SafeTensorsFile("model.safetensors")

# Get raw bytes
data = st.tensor_bytes("model.embed_tokens.weight")

# Load as float32 numpy array (if tensor is float32)
np_array = st.tensor_as_f32("model.embed_tokens.weight")

# Manual conversion for other dtypes
info = st.tensor_info("model.layers.0.self_attn.q_proj.weight")
data = st.tensor_bytes("model.layers.0.self_attn.q_proj.weight")

if info.dtype_name == "float16":
    np_arr = np.frombuffer(data, dtype=np.float16).reshape(info.shape)
elif info.dtype_name == "bfloat16":
    # BFloat16 needs special handling
    raw = np.frombuffer(data, dtype=np.uint16).reshape(info.shape)
    np_arr = raw.view(np.float32)  # Reinterpret as float32
```

### Iterating Over Tensors

```python
from pygpukit.llm import SafeTensorsFile

st = SafeTensorsFile("model.safetensors")

# Check if tensor exists
if "model.embed_tokens.weight" in st:
    print("Embedding found!")

# Iterate over all tensors
for name in st.tensor_names:
    info = st.tensor_info(name)
    print(f"{name}: {info.shape} ({info.dtype_name})")
```

---

## Tokenizer

PyGPUkit includes a BPE tokenizer compatible with HuggingFace's `tokenizer.json` format.

### Loading a Tokenizer

```python
from pygpukit.llm import Tokenizer

# Load from file
tok = Tokenizer("tokenizer.json")

# Or from JSON string
import json
with open("tokenizer.json") as f:
    json_str = f.read()
tok = Tokenizer.from_json(json_str)
```

### Encoding and Decoding

```python
from pygpukit.llm import Tokenizer

tok = Tokenizer("tokenizer.json")

# Encode text to token IDs
text = "Hello, world! How are you?"
token_ids = tok.encode(text)
print(f"Token IDs: {token_ids}")

# Decode back to text
decoded = tok.decode(token_ids)
print(f"Decoded: {decoded}")

# Single token operations
token_str = tok.id_to_token(123)  # Get token string for ID
token_id = tok.token_to_id("hello")  # Get ID for token string
```

### Special Tokens

```python
from pygpukit.llm import Tokenizer

tok = Tokenizer("tokenizer.json")

print(f"Vocabulary size: {tok.vocab_size}")
print(f"BOS token ID: {tok.bos_token_id}")  # Beginning of sequence
print(f"EOS token ID: {tok.eos_token_id}")  # End of sequence
print(f"PAD token ID: {tok.pad_token_id}")  # Padding
```

---

## Model Components

PyGPUkit provides building blocks for constructing neural network models.

### Linear Layer

```python
from pygpukit.llm import Linear
import pygpukit as gpk
import numpy as np

# Create weights [out_features, in_features]
weight = gpk.from_numpy(np.random.randn(3072, 768).astype(np.float32))
bias = gpk.from_numpy(np.random.randn(3072).astype(np.float32))

# Create linear layer
linear = Linear(weight, bias)

# Forward pass: y = xW^T + b
x = gpk.from_numpy(np.random.randn(32, 768).astype(np.float32))
y = linear(x)  # [32, 3072]

# Properties
print(f"In features: {linear.in_features}")
print(f"Out features: {linear.out_features}")
```

### LayerNorm

```python
from pygpukit.llm import LayerNorm
import pygpukit as gpk

features = 768
weight = gpk.ones(features)  # gamma
bias = gpk.zeros(features)   # beta

ln = LayerNorm(weight, bias, eps=1e-5)

x = gpk.from_numpy(np.random.randn(32, 768).astype(np.float32))
y = ln(x)  # Normalized output [32, 768]
```

### MLP Block

```python
from pygpukit.llm import MLP
import pygpukit as gpk
import numpy as np

n_embd = 768
n_inner = 3072  # 4 * n_embd

# Create weights
c_fc_w = gpk.from_numpy(np.random.randn(n_inner, n_embd).astype(np.float32))
c_fc_b = gpk.from_numpy(np.random.randn(n_inner).astype(np.float32))
c_proj_w = gpk.from_numpy(np.random.randn(n_embd, n_inner).astype(np.float32))
c_proj_b = gpk.from_numpy(np.random.randn(n_embd).astype(np.float32))

mlp = MLP(c_fc_w, c_fc_b, c_proj_w, c_proj_b)

# Forward: fc1 -> gelu -> fc2
x = gpk.from_numpy(np.random.randn(32, n_embd).astype(np.float32))
y = mlp(x)  # [32, 768]
```

### TransformerBlock

```python
from pygpukit.llm import TransformerBlock, MLP, LayerNorm
import pygpukit as gpk

n_embd = 768

# LayerNorm weights
ln_w = gpk.ones(n_embd)
ln_b = gpk.zeros(n_embd)

# MLP weights (as above)
mlp = MLP(c_fc_w, c_fc_b, c_proj_w, c_proj_b)

# Create transformer block: ln -> mlp -> residual
block = TransformerBlock(ln_w, ln_b, mlp, eps=1e-5)

x = gpk.from_numpy(np.random.randn(32, n_embd).astype(np.float32))
y = block(x)  # [32, 768] with residual connection
```

---

## GPT-2 Model

PyGPUkit includes a GPT-2 model implementation (MLP-only for MVP).

### Loading from SafeTensors

```python
from pygpukit.llm import GPT2Config, load_gpt2_from_safetensors

# Default GPT-2 Small config
config = GPT2Config(
    vocab_size=50257,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_positions=1024,
)

# Load model
model = load_gpt2_from_safetensors("gpt2.safetensors", config)
```

### Forward Pass

```python
from pygpukit.llm import load_gpt2_from_safetensors, Tokenizer

model = load_gpt2_from_safetensors("gpt2.safetensors")
tok = Tokenizer("tokenizer.json")

# Tokenize input
text = "The quick brown fox"
input_ids = tok.encode(text)

# Forward pass
hidden = model(input_ids)  # [seq_len, n_embd]

# Get logits
logits = model.lm_head(hidden)  # [seq_len, vocab_size]

# Get next token prediction
import numpy as np
next_token_logits = logits.to_numpy()[-1]
next_token_id = int(np.argmax(next_token_logits))
print(f"Next token: {tok.decode([next_token_id])}")
```

### Text Generation

```python
from pygpukit.llm import load_gpt2_from_safetensors, Tokenizer

model = load_gpt2_from_safetensors("gpt2.safetensors")
tok = Tokenizer("tokenizer.json")

# Generate text
prompt = "Once upon a time"
input_ids = tok.encode(prompt)

# Generate with greedy decoding
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=1.0,  # 1.0 = greedy argmax
)

generated_text = tok.decode(output_ids)
print(generated_text)
```

> **Note:** The current implementation is MLP-only (no attention mechanism).
> It's meant as a demonstration of the loading/inference pipeline.
> Full attention will be added in future versions.

---

## Complete Example

```python
"""Load and inspect a model from HuggingFace."""
from pygpukit.llm import SafeTensorsFile, Tokenizer
import pygpukit as gpk

# Download model files first:
# huggingface-cli download gpt2 --local-dir ./gpt2

# Load safetensors
st = SafeTensorsFile("gpt2/model.safetensors")

print("=" * 50)
print(f"Model: GPT-2")
print(f"Tensors: {st.num_tensors}")
print(f"Size: {st.file_size / 1e6:.1f} MB")
print("=" * 50)

# Print all tensor shapes
for name in sorted(st.tensor_names):
    info = st.tensor_info(name)
    print(f"  {name}: {info.shape} ({info.dtype_name})")

# Load tokenizer
tok = Tokenizer("gpt2/tokenizer.json")
print(f"\nVocabulary: {tok.vocab_size} tokens")

# Test tokenization
text = "Hello, world!"
ids = tok.encode(text)
print(f"\n'{text}' -> {ids}")
print(f"{ids} -> '{tok.decode(ids)}'")
```

---

## API Reference

### SafeTensorsFile

| Method/Property | Description |
|-----------------|-------------|
| `SafeTensorsFile(path)` | Open safetensors file |
| `.tensor_names` | List of tensor names |
| `.num_tensors` | Number of tensors |
| `.file_size` | File size in bytes |
| `.tensor_info(name)` | Get TensorInfo for tensor |
| `.tensor_bytes(name)` | Get raw bytes |
| `.tensor_as_f32(name)` | Get as float32 numpy array |

### TensorInfo

| Property | Description |
|----------|-------------|
| `.name` | Tensor name |
| `.dtype` | Dtype as integer |
| `.dtype_name` | Dtype as string |
| `.shape` | Tensor shape |
| `.offset` | Byte offset in file |
| `.size_bytes` | Size in bytes |
| `.numel` | Number of elements |

### Tokenizer

| Method/Property | Description |
|-----------------|-------------|
| `Tokenizer(path)` | Load from tokenizer.json |
| `Tokenizer.from_json(str)` | Load from JSON string |
| `.vocab_size` | Vocabulary size |
| `.bos_token_id` | BOS token ID |
| `.eos_token_id` | EOS token ID |
| `.pad_token_id` | PAD token ID |
| `.encode(text)` | Encode text to IDs |
| `.decode(ids)` | Decode IDs to text |
| `.id_to_token(id)` | Get token for ID |
| `.token_to_id(token)` | Get ID for token |

### GPT2Config

| Property | Default | Description |
|----------|---------|-------------|
| `vocab_size` | 50257 | Vocabulary size |
| `n_embd` | 768 | Embedding dimension |
| `n_layer` | 12 | Number of layers |
| `n_head` | 12 | Number of attention heads |
| `n_positions` | 1024 | Max sequence length |
| `layer_norm_eps` | 1e-5 | LayerNorm epsilon |
