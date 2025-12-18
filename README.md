
# PyGPUkit — Lightweight GPU Runtime for Python
*A minimal, modular GPU runtime with Rust-powered scheduler, NVRTC JIT compilation, and a clean NumPy-like API.*

[![PyPI version](https://badge.fury.io/py/PyGPUkit.svg)](https://badge.fury.io/py/PyGPUkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, quick start, basic usage |
| [API Reference](docs/api.md) | Complete API documentation with examples |
| [LLM Guide](docs/llm.md) | SafeTensors, GPT-2/LLaMA/Qwen3 inference |
| [Performance Tuning](docs/performance.md) | TF32, FP16, CUTLASS optimization |
| [Scheduler Guide](docs/scheduler.md) | Multi-LLM concurrent execution |

---

## Overview
**PyGPUkit** is a lightweight GPU runtime for Python that provides:
- **Single-binary distribution** — works with just GPU drivers, no CUDA Toolkit needed
- **Rust-powered scheduler** with admission control, QoS, and resource partitioning
- **NVRTC JIT** (optional) for custom kernel compilation
- A NumPy-like `GPUArray` type
- Kubernetes-inspired GPU scheduling (bandwidth + memory guarantees)

PyGPUkit aims to be the "micro-runtime for GPU computing": small, fast, and ideal for research, inference tooling, DSP, and real-time systems.

> **Note:** PyGPUkit is NOT a PyTorch/CuPy replacement—it's a lightweight runtime for custom GPU workloads where full ML frameworks are overkill.

---

## What's New in v0.2.10

### Dynamic cuBLASLt Loading
cuBLASLt is now loaded dynamically at runtime, enabling true **driver-only deployment**. No CUDA Toolkit installation required on target machines.

| Feature | Description |
|---------|-------------|
| **Dynamic Loading** | `LoadLibrary`/`dlopen` for cuBLASLt DLL |
| **Descriptor Caching** | GEMM descriptors cached per (M, N, K, dtype) |
| **2.67x Faster** | 224 matmuls: 395ms → 148ms |

```python
# Works with just GPU drivers - no CUDA Toolkit needed
import pygpukit as gk
C = A @ B  # Uses dynamically-loaded cuBLASLt for small batch sizes
```

### CUDA Graph Optimizations
- Eliminated GPU allocations in position/random buffer updates
- Direct `copy_from_numpy` for H2D transfers during graph replay

### Performance (Qwen3-8B, RTX 3090 Ti)
| Mode | Throughput |
|------|------------|
| Standard decode | 1.85 tok/s |
| CUDA Graph | 2.12 tok/s |

---

## What's New in v0.2.9

### Unified LLM Interface
A single `CausalTransformerModel` now supports multiple architectures through the `ModelSpec` abstraction.

| Architecture | Features | Status |
|--------------|----------|--------|
| **GPT-2** | LayerNorm, GELU, Position Embedding | ✅ Tested |
| **LLaMA 2/3** | RMSNorm, SiLU, RoPE, GQA | ✅ Tested |
| **Qwen3** | RMSNorm, SiLU, RoPE, GQA, QK-Norm | ✅ Tested |

```python
from pygpukit.llm import load_model_from_safetensors, detect_model_spec, load_safetensors

# Auto-detect and load any supported model
st = load_safetensors("model.safetensors")
spec = detect_model_spec(st.tensor_names)  # Returns GPT2_SPEC, LLAMA_SPEC, or QWEN3_SPEC
model = load_model_from_safetensors("model.safetensors", dtype="float16", spec=spec)

# Generate with KV-cache
output_ids = model.generate(
    input_ids,
    max_new_tokens=64,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    use_cache=True,  # KV-cache for efficient generation
)
```

### Hybrid Attention Execution
Automatic CPU/GPU switching for optimal performance:

| Phase | Backend | Reason |
|-------|---------|--------|
| **Prefill** (seq_len > 1) | GPU SDPA | Parallelizable |
| **Decode** (seq_len = 1) | CPU | Avoids kernel launch overhead |

### New LLM Operations
| Operation | Description |
|-----------|-------------|
| `gpk.sdpa_causal(q, k, v)` | Scaled Dot-Product Attention with causal mask |
| `gpk.rope_inplace(x, freqs)` | Rotary Position Embedding (in-place) |
| `gpk.silu(x)` | SiLU/Swish activation |
| `gpk.rmsnorm(x, weight, eps)` | RMS Layer Normalization |

### Sharded Model Support
Load large models split across multiple safetensors files:

```python
from pygpukit.llm import load_safetensors

# Automatically handles sharded models
st = load_safetensors("model.safetensors.index.json")  # Returns ShardedSafeTensorsFile
print(f"Shards: {len(st._shard_files)}, Tensors: {st.num_tensors}")
```

---

## What's New in v0.2.7

### CUTLASS Epilogue Fusion
Fused Linear + Bias + GELU operations using CUTLASS epilogue fusion for improved performance in transformer workloads.

```python
import pygpukit as gpk
import numpy as np

# Create tensors
batch, in_feat, out_feat = 512, 768, 3072
input = gpk.from_numpy(np.random.randn(batch, in_feat).astype(np.float32))
weight = gpk.from_numpy(np.random.randn(out_feat, in_feat).astype(np.float32))
bias = gpk.from_numpy(np.random.randn(out_feat).astype(np.float32))

# Fused linear + bias + GELU (single kernel, no intermediate memory)
output = gpk.linear_bias_gelu(input, weight, bias)
```

### Multi-SM CUTLASS Kernels
Runtime SM detection with architecture-optimized kernel variants:

| Architecture | GPU Examples | Pipeline | Features |
|-------------|--------------|----------|----------|
| **SM80** | A100 | 4-stage | 48KB shared memory |
| **SM86** | RTX 3090, RTX 3080 | 5-stage | 100KB shared memory |
| **SM89** | RTX 4090, RTX 4080 | 6-stage | Ada Lovelace optimizations |
| **SM90** | H100 | CUTLASS 3.x | WGMMA/TMA instructions |
| **SM100/120** | Blackwell (B100, B200) | CUTLASS 3.x | Next-gen TensorCore |

> **Note:** SM100+ (Blackwell) requires CUDA 13.x. Windows wheels include SM100/120 support.

### New Operations
| Operation | Description |
|-----------|-------------|
| `gpk.transpose(a)` | GPU-native matrix transpose |
| `gpk.bias_add_inplace(out, bias)` | In-place bias addition |
| `gpk.linear_bias_gelu(x, w, b)` | Fused linear + bias + GELU |

### API Improvements
- Complete public API exports (all operations accessible via `gpk.*`)
- Consistent snake_case naming convention
- Full docstrings for all public functions

---

## LLM Support

PyGPUkit includes built-in support for loading and running LLM models.
See the [LLM Guide](docs/llm.md) for detailed documentation.

**Important:** PyGPUkit's core responsibility is **GPU execution**, not tokenization.
- The model API expects **token IDs as input**, not raw text
- For production tokenization, use [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)
- The built-in `Tokenizer` class is **experimental** and intended for demos only

```python
from pygpukit.llm import SafeTensorsFile, load_model_from_safetensors, detect_model_spec

# Load safetensors (memory-mapped, zero-copy)
st = SafeTensorsFile("model.safetensors")
print(f"Tensors: {st.num_tensors}, Size: {st.file_size / 1e9:.2f} GB")

# Load model with automatic architecture detection
spec = detect_model_spec(st.tensor_names)
model = load_model_from_safetensors("model.safetensors", dtype="float16", spec=spec)

# Generate with token IDs (use HuggingFace tokenizers for production)
input_ids = [1, 2, 3, 4]  # Your tokenizer's output
output_ids = model.generate(input_ids, max_new_tokens=32)
```

| Component | Description |
|-----------|-------------|
| `SafeTensorsFile` | Memory-mapped .safetensors loading |
| `CausalTransformerModel` | Unified model for GPT-2, LLaMA, Qwen3 |
| `load_model_from_safetensors` | Load model with auto-detection |
| `detect_model_spec` | Auto-detect model architecture |
| `Tokenizer` | **Experimental** BPE tokenizer (demos only) |

---

## What's New in v0.2.6

### CUTLASS Backend (Default)
NVIDIA CUTLASS v4.3.0 is now the default GEMM backend, delivering optimized TensorCore performance out of the box.

| Feature | Description |
|---------|-------------|
| **TF32 TensorCore** | 31+ TFLOPS for FP32 inputs (automatic) |
| **FP16 TensorCore** | 63 TFLOPS |
| **BF16 TensorCore** | 63 TFLOPS |
| **Zero Config** | No environment variables needed |

```python
import pygpukit as gpk
import numpy as np

# CUTLASS TF32 is automatic for FP32 (31+ TFLOPS)
a = gpk.from_numpy(np.random.randn(8192, 8192).astype(np.float32))
b = gpk.from_numpy(np.random.randn(8192, 8192).astype(np.float32))
c = a @ b  # Uses CUTLASS TF32 TensorCore

# For full FP32 precision (no TF32), set:
# PYGPUKIT_NO_TF32=1
```

### Multi-LLM Concurrent Execution
Run multiple AI models (LLM, TTS, Vision) concurrently on a single GPU with independent CUDA streams and VRAM budgets.

| Feature | Description |
|---------|-------------|
| **Execution Control** | User controls execution order |
| **Stream Isolation** | No implicit sync between streams |
| **VRAM Budgeting** | Safe memory sharing per model |
| **Concurrent Safety** | "Running simultaneously doesn't break" |
| **asyncio Integration** | Native Python async/await support |

> **Note:** On a single GPU, Multi-LLM scheduling enables **concurrent execution, not faster execution**, for compute-bound workloads. Speedup benefits apply to I/O-bound workloads or multi-GPU setups.

```python
import asyncio
from pygpukit.scheduler import (
    create_context, context_session, GB, initialize
)

# Create execution contexts with VRAM budgets
initialize(device_id=0)
llm_ctx = create_context("llm", max_vram=4 * GB)
tts_ctx = create_context("tts", max_vram=2 * GB)

async def run_parallel():
    async with context_session(llm_ctx), context_session(tts_ctx):
        # Run models concurrently with asyncio.gather
        llm_task = asyncio.create_task(run_llm_inference())
        tts_task = asyncio.create_task(run_tts_synthesis())

        text, audio = await asyncio.gather(llm_task, tts_task)
        return text, audio

result = asyncio.run(run_parallel())
```

### FP16/BF16 TensorCore (via CUTLASS)
| Feature | Description |
|---------|-------------|
| **FP16 TensorCore** | 63 TFLOPS (automatic via CUTLASS) |
| **BF16 TensorCore** | 63 TFLOPS (automatic via CUTLASS) |
| **FP32 Accumulation** | Numerical stability maintained |

```python
import pygpukit as gpk
import numpy as np

# FP16 TensorCore matmul (63 TFLOPS on RTX 3090 Ti)
# No environment variable needed - CUTLASS is automatic
a = gpk.from_numpy(np.random.randn(8192, 8192).astype(np.float16))
b = gpk.from_numpy(np.random.randn(8192, 8192).astype(np.float16))
c = a @ b  # Uses CUTLASS TensorCore
```

> **Note:** CUTLASS requires matrix dimensions divisible by 16.

---

## What's New in v0.2.5

### FP16 / BF16 Support
| Feature | Description |
|---------|-------------|
| **FP16 (float16)** | Half-precision floating point |
| **BF16 (bfloat16)** | Brain floating point (better dynamic range) |
| **FP32 Accumulation** | Numerical stability via FP32 intermediate |
| **Type Conversion** | `astype()` for seamless dtype conversion |

```python
import pygpukit as gpk
import numpy as np

# FP16 operations
a = gpk.from_numpy(np.random.randn(1024, 1024).astype(np.float16))
b = gpk.from_numpy(np.random.randn(1024, 1024).astype(np.float16))
c = a @ b  # FP16 matmul

# BF16 operations
arr = np.random.randn(1024, 1024).astype(np.float32)
a_bf16 = gpk.from_numpy(arr).astype(gpk.bfloat16)
b_bf16 = gpk.from_numpy(arr).astype(gpk.bfloat16)
c_bf16 = a_bf16 @ b_bf16  # BF16 matmul
result = c_bf16.astype(gpk.float32)  # Convert back to FP32
```

### Reduction Operations
| Operation | Description |
|-----------|-------------|
| `gpk.sum(a)` | Sum of all elements |
| `gpk.mean(a)` | Mean of all elements |
| `gpk.max(a)` | Maximum element |

### Operator Overloads
```python
c = a + b   # Element-wise add
c = a - b   # Element-wise subtract
c = a * b   # Element-wise multiply
c = a / b   # Element-wise divide
c = a @ b   # Matrix multiplication
```

---

## What's New in v0.2.4

### Single-Binary Distribution
| Feature | Description |
|---------|-------------|
| **Driver-only mode** | Only `nvcuda.dll` (GPU driver) required |
| **Dynamic NVRTC** | JIT loaded at runtime, optional |
| **No cudart dependency** | Eliminated CUDA Runtime dependency |
| **Smaller wheel** | No bundled DLLs |

```python
import pygpukit as gp

# Works with just GPU drivers!
print(f"CUDA: {gp.is_cuda_available()}")      # True (if GPU driver installed)
print(f"NVRTC: {gp.is_nvrtc_available()}")    # True (if CUDA Toolkit installed)
print(f"NVRTC Path: {gp.get_nvrtc_path()}")   # Path to NVRTC DLL (if available)
```

### TF32 TensorCore GEMM
| Feature | Description |
|---------|-------------|
| **PTX mma.sync** | Direct TensorCore access via inline PTX assembly |
| **cp.async Pipeline** | Double-buffered async memory transfers |
| **TF32 Precision** | 19-bit mantissa (vs FP32's 23-bit), ~0.1% per-op error |
| **SM 80+ Required** | Ampere architecture (RTX 30XX+) required |

---

## Performance

### Benchmark Comparison (RTX 3090 Ti, 8192×8192)

| Library | FP32 | TF32 | FP16 | BF16 | Requirements |
|---------|------|------|------|------|--------------|
| **NumPy** (OpenBLAS) | ~0.8 TFLOPS | — | — | — | CPU only |
| **cuBLAS** | ~21 TFLOPS | ~59 TFLOPS | ~75 TFLOPS | ~83 TFLOPS | CUDA Toolkit |
| **PyGPUkit** (CUTLASS) | 18 TFLOPS | **31 TFLOPS** | **63 TFLOPS** | **63 TFLOPS** | GPU drivers only |

> Built-in matmul kernels are pre-compiled. Driver-Only and Full (JIT) modes have identical matmul performance. JIT is only needed for custom kernels.

### PyGPUkit Performance by Matrix Size

| Matrix Size | FP32 (NO_TF32) | TF32 (CUTLASS) | FP16 (CUTLASS) | BF16 (CUTLASS) |
|-------------|----------------|----------------|----------------|----------------|
| 2048×2048 | 9.6 TFLOPS | 13 TFLOPS | 15 TFLOPS | 21 TFLOPS |
| 4096×4096 | 14.7 TFLOPS | 22 TFLOPS | 44 TFLOPS | 44 TFLOPS |
| 8192×8192 | 18 TFLOPS | **31 TFLOPS** | **63 TFLOPS** | **63 TFLOPS** |

> **Note:** CUTLASS is automatic for compatible sizes (16-aligned). Use `PYGPUKIT_NO_TF32=1` for full FP32 precision.

---

## Installation

```bash
pip install pygpukit
```

From source:
```bash
git clone https://github.com/m96-chan/PyGPUkit
cd PyGPUkit
pip install -e .
```

### Requirements
- Python 3.10+
- NVIDIA GPU with drivers installed
- **Optional:** CUDA Toolkit (for JIT compilation of custom kernels)

> **Note:** NVRTC (NVIDIA Runtime Compiler) is included in CUDA Toolkit.
> Pre-compiled GPU operations (matmul, add, mul, etc.) work with just GPU drivers.

### Supported GPUs

| Generation | Architecture | Examples | Status |
|------------|-------------|----------|--------|
| **Ampere** | SM80-86 | A100, RTX 3090, RTX 3080 | Fully supported |
| **Ada Lovelace** | SM89 | RTX 4090, RTX 4080 | Fully supported |
| **Hopper** | SM90 | H100, H200 | Fully supported |
| **Blackwell** | SM100-120 | B100, B200 | Supported (CUDA 13.x) |
| Turing/Older | SM < 80 | RTX 20XX, GTX 10XX | **NOT supported** |

### Runtime Modes
| Mode | Requirements | Features |
|------|-------------|----------|
| **Full JIT** | GPU drivers + CUDA Toolkit | All features including custom kernels |
| **Pre-compiled** | GPU drivers only | Built-in ops (matmul, add, mul) |
| **CPU simulation** | None | Testing/development without GPU |

---

## Quick Start

### Basic Operations
```python
import pygpukit as gp

# Allocate arrays
x = gp.zeros((1024, 1024), dtype="float32")
y = gp.ones((1024, 1024), dtype="float32")

# Operations
z = gp.add(x, y)
w = gp.matmul(x, y)

# CPU <-> GPU transfer
arr = z.to_numpy()
garr = gp.from_numpy(arr)
```

### Custom JIT Kernel (requires CUDA Toolkit)
```python
src = '''
extern "C" __global__
void scale(float* x, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= factor;
}
'''

if gp.is_nvrtc_available():
    kernel = gp.jit(src, func="scale")
    kernel(x, factor=0.5, n=x.size)
else:
    print("JIT not available. Using pre-compiled ops.")
```

### Rust Scheduler
```python
import _pygpukit_rust as rust

# Memory Pool with LRU eviction
pool = rust.MemoryPool(quota=100 * 1024 * 1024, enable_eviction=True)
block = pool.allocate(4096)

# QoS-aware task scheduling
evaluator = rust.QosPolicyEvaluator(total_memory=8*1024**3, total_bandwidth=1.0)
task = rust.QosTaskMeta.guaranteed("task-1", "Critical Task", 256*1024*1024)
result = evaluator.evaluate(task)

# GPU Partitioning
manager = rust.PartitionManager(rust.PartitionConfig(total_memory=8*1024**3))
manager.create_partition("inference", "Inference",
    rust.PartitionLimits().memory(4*1024**3).compute(0.5))
```

---

## Features

### Core Infrastructure (Rust)
| Feature | Description |
|---------|-------------|
| **Memory Pool** | LRU eviction, size-class free lists |
| **Scheduler** | Priority queue, memory reservation |
| **Transfer Engine** | Separate H2D/D2H streams, priority |
| **Kernel Dispatch** | Per-stream limits, lifecycle tracking |

### Advanced Scheduler
| Feature | Description |
|---------|-------------|
| **Admission Control** | Deterministic admission, quota enforcement |
| **QoS Policy** | Guaranteed/Burstable/BestEffort tiers |
| **Kernel Pacing** | Bandwidth-based throttling per stream |
| **GPU Partitioning** | Resource isolation, multi-tenant support |
| **Multi-LLM Execution** | Concurrent AI model execution with stream isolation |
| **asyncio Integration** | Native Python async/await for concurrent inference |

---

## Project Goals
1. Provide the smallest usable GPU runtime for Python
2. Expose GPU scheduling (bandwidth, memory, partitioning)
3. Make writing custom GPU kernels easy
4. Serve as a building block for inference engines, DSP systems, and real-time workloads

---

## Project Structure
```
PyGPUkit/
  src/pygpukit/    # Python API (NumPy-compatible)
  native/          # C++ backend (CUDA Driver API, NVRTC)
  rust/            # Rust backend (memory pool, scheduler)
    pygpukit-core/   # Pure Rust core logic
    pygpukit-python/ # PyO3 bindings
  docs/            # Documentation guides
  examples/        # Demo scripts
  scripts/         # Build scripts, benchmarks
  tests/           # Test suite
```

---

## Roadmap

### Released

| Version | Highlights |
|---------|------------|
| **v0.1** | GPUArray, NVRTC JIT, add/mul/matmul, wheels |
| **v0.2.0** | Rust scheduler (QoS, partitioning), memory pool (LRU), 106 tests |
| **v0.2.1** | API stabilization, error propagation |
| **v0.2.2** | Ampere SGEMM (cp.async, float4), 18 TFLOPS FP32 |
| **v0.2.3** | TF32 TensorCore (PTX mma.sync), 28 TFLOPS |
| **v0.2.4** | **Single-binary distribution**, dynamic NVRTC, driver-only mode |
| **v0.2.5** | **FP16/BF16 support**, reduction ops, operator overloads, TF32 v2 (~30 TFLOPS) |
| **v0.2.6** | **CUTLASS backend** (31 TFLOPS TF32, 63 TFLOPS FP16/BF16), Multi-LLM concurrent execution |
| **v0.2.7** | **Epilogue fusion** (linear+bias+gelu), Multi-SM kernels, API review |
| **v0.2.8** | CUTLASS v4.3.3 update, auto-update workflow |
| **v0.2.9** | **Unified LLM interface** (CausalTransformerModel), ModelSpec abstraction, GPT-2/LLaMA/Qwen3 support |

### Planned

| Version | Goals |
|---------|-------|
| **v0.2.9** | **General LLM Execution** — Attention layer, GPT-2 E2E inference, GPT-2/GPT-Neo/LLaMA architecture support |
| **v0.3** | Triton backend, advanced ops (softmax), MPS/MIG |

---

## API Stability & Backward Compatibility

### Version Policy
- **v0.2.x**: Backward compatible within minor versions. New features may be added, but existing APIs remain stable.
- **v0.3+**: May introduce breaking changes with deprecation warnings in prior version.

### Stable Public API (v0.2.x)
All functions exported via `pygpukit.*` are part of the stable public API:

| Category | Functions |
|----------|-----------|
| **Factory** | `zeros`, `ones`, `empty`, `from_numpy` |
| **Elementwise** | `add`, `sub`, `mul`, `div` |
| **Math** | `exp`, `log`, `relu`, `gelu` |
| **Matrix** | `matmul`, `transpose` |
| **Reductions** | `sum`, `mean`, `max` |
| **Neural** | `layernorm`, `bias_add_inplace`, `linear_bias_gelu` |
| **Types** | `GPUArray`, `DataType`, `float32`, `float64`, `float16`, `bfloat16` |
| **LLM** | `llm.SafeTensorsFile`, `llm.CausalTransformerModel`, `llm.load_model_from_safetensors` |
| **LLM (Experimental)** | `llm.Tokenizer` (use HuggingFace tokenizers for production) |

### Deprecation Policy
APIs to be removed will emit `DeprecationWarning` for at least one minor version before removal.

---

## Contributing
Contributions and discussions are welcome!
Please open Issues for feature requests, bugs, or design proposals.

---

## License
MIT License

---

## Acknowledgements
Inspired by: CUDA Runtime, NVRTC, PyCUDA, CuPy, Triton

PyGPUkit aims to fill the gap for a tiny, embeddable GPU runtime for Python.
