
# PyGPUkit — Lightweight GPU Runtime for Python
*A minimal, modular GPU runtime with Rust-powered scheduler, NVRTC JIT compilation, and a clean NumPy-like API.*

[![PyPI version](https://badge.fury.io/py/PyGPUkit.svg)](https://badge.fury.io/py/PyGPUkit)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub stars](https://img.shields.io/github/stars/m96-chan/PyGPUkit?style=social)](https://github.com/m96-chan/PyGPUkit)


[![Python](https://img.shields.io/pypi/pyversions/PyGPUkit.svg)](https://pypi.org/project/PyGPUkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SM](https://img.shields.io/badge/SM-80%20%7C%2086%20%7C%2089%20%7C%2090%20%7C%20100%20%7C%20120a-blue.svg)](#supported-gpus)
[![Downloads](https://img.shields.io/pypi/dm/PyGPUkit.svg)](https://pypi.org/project/PyGPUkit/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

### When GPU optimizations change your results, something is wrong.

*A minimal, deterministic GPU runtime for Python.*  
Built for people who care about **correctness**, **reproducibility**, and **real performance**.

- CUDA Graph that doesn't lie
- cuBLASLt without hidden state
- FP8 / NVF4 / w8a16 done explicitly
- Rust-powered scheduler for real GPU concurrency

This is not a framework.
This is a GPU runtime.
---

## Why PyGPUkit Exists

Modern GPU stacks optimize aggressively.  
Sometimes, they optimize **correctness away**.

PyGPUkit exists because:

- CUDA Graph replay can change numerical results
- cuBLASLt may depend on hidden workspace state
- Stream-0 synchronization hides performance bugs
- “It’s faster” often means “it’s nondeterministic”

PyGPUkit chooses:

- **Explicit** over implicit
- **Determinism** over magic
- **Measurable behavior** over benchmark-only claims

---

## What PyGPUkit Is NOT

- ❌ Not a PyTorch replacement
- ❌ Not a training framework
- ❌ Not a convenience-first library
- ❌ Not safe if you ignore GPU semantics
- ❌ Not designed for "just works" expectations

PyGPUkit is for people who want to *see* and *control*
what their GPU is actually doing.

---

## Core Capabilities (TL;DR)

- 🚀 Driver-only deployment (no CUDA Toolkit required)
- 🧠 Deterministic CUDA Graph execution
- ⚙️ Explicit stream & memory control
- 🧮 FP8 / NVF4 / BF16 / TF32 done right
- 🎛️ Rust-based GPU scheduler with QoS & partitioning
- 🔊 GPU-native audio & DSP (no cuFFT dependency)

---

## Real-World GPU Pathologies (Observed)

- Same input, different output with CUDA Graph replay
- FP8 GEMM producing correct averages but wrong tokens
- cuBLASLt performance variance across runs
- H2D stalls masked by stream-0 synchronization

All of these are **reproducible**.  
All of them are **documented**.  
All of them are **why PyGPUkit exists**.

These are not theoretical.
They were all observed in production or real benchmarks.

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

## What's New in v0.2.16

### MoE (Mixture of Experts) Support
Full support for Mixtral-style MoE models with custom CUDA kernels:

| Component | Description |
|-----------|-------------|
| **MoE Kernels** | TopK routing, softmax, token permutation, gather/scatter |
| **Grouped GEMM** | Batched expert dispatch with per-row expert IDs |
| **MoELayer** | Python layer with router + expert FFN dispatch |
| **MIXTRAL_SPEC** | Auto-detection for Mixtral 8x7B models |

```python
from pygpukit.llm import load_model_from_safetensors, detect_model_spec

# Auto-detect MoE model
spec = detect_model_spec(tensor_names)  # Returns MIXTRAL_SPEC for MoE
model = load_model_from_safetensors("mixtral.safetensors", spec=spec)
```

### Thinking Model Support
Qwen3 Thinking model support with `<think>...</think>` block parsing:

```python
# examples/chat_cli_thinking.py
python examples/chat_cli_thinking.py --model F:/LLM/Qwen3-4B-Thinking
```

- Streaming output with thinking/answer separation
- `/think` command to toggle thinking display
- CUDA Graph support for faster decode

### New GEMV Kernels (SM120)

| Kernel | A dtype | B dtype | Speedup vs BF16 |
|--------|---------|---------|-----------------|
| **FP8/FP8 (W8A8)** | FP8 E4M3 | FP8 E4M3 | **6-22x** |
| **NVF4/NVF4 (W4A4)** | NVF4 | NVF4 | Memory priority |
| **Int4 GEMV** | BF16 | Int4 | Large K dimensions |

### New GEMM Kernels (SM120)

| Kernel | Description |
|--------|-------------|
| **W8A16 GEMM** | FP8 weight + BF16 activation (CUTLASS) |
| **Int8 Native** | Exact int8 via dp4a (CUDA cores) |
| **Int4 via Int8** | 4-bit approximation via TensorCore |
| **Grouped GEMM v2** | Per-row expert IDs for MoE |

### Kernel Directory Restructure
Organized matmul kernels by `{gemm|gemv}/{input}/{output}/{arch}/`:

```
native/ops/matmul/
├── gemm/fp8/bf16/sm120/w8a16_gemm.cu
├── gemm/fp8/fp8/sm120/fp8_cutlass.cu
├── gemv/fp8/fp8/sm120/fp8_gemv.cu
├── gemv/nvf4/nvf4/sm120/nvf4_gemv.cu
└── gemv/int4/int4/sm120/int4_gemv.cu
```

### Development Tooling
- **Claude Code Skills**: Build, benchmark, lint, test automation
- **Subagents**: kernel-reviewer, perf-analyzer, api-designer
- **CONTRIBUTING.md**: Contribution guidelines
- **MCP Integration**: Serena, Context7, Memory servers

### Kernel Cleanup
Removed redundant slow kernels:

| Removed | Kept | Reason |
|---------|------|--------|
| FP8 GEMV basic | FP8 GEMV opt | [N,K] layout 3-9x faster |
| Int8 via FP8 | Int8 native dp4a | Exact results |

---

## What's New in v0.2.15


### Whisper ASR Module
Full GPU-accelerated Whisper speech recognition:

| Component | Description |
|-----------|-------------|
| **WhisperEncoder** | Conv1d stem + transformer with GPU attention |
| **WhisperDecoder** | Autoregressive decoder with cross-attention |
| **WhisperModel** | High-level API: `from_pretrained()`, `transcribe()` |
| **Preprocessing** | GPU mel spectrogram (30s pad/trim, normalization) |
| **Streaming** | `transcribe_streaming()` for long audio |

### GEMV Kernels (SM120)
Optimized GEMV for LLM decode (M=1):

| Kernel | Feature | Speedup |
|--------|---------|---------|
| **BF16 GEMV** | BF16x2 vectorized loads | 25-40% vs scalar |
| **NVF4 GEMV** | Pre-scaled LUT | 73% less bandwidth |
| **Linear layer** | Auto GEMV for M=1 | 1.3-2.4x vs matmul |

### FP8 I/O GEMM (SM120)
Pure FP8 input/output GEMM for FP8 model inference (Llama 3.1 FP8, Qwen FP8, etc.):

| Function | Description |
|----------|-------------|
| `matmul_fp8_fp8_sm120` | FP8 E4M3 input -> FP8 E4M3 output (unity scaling) |
| `matmul_fp8_fp8_blockwise_sm120` | FP8 with block-wise scale_A / scale_B |
| `fp8_fp8_get_scale_sizes` | Get required scale factor sizes for (M, N, K) |
| `fp8_fp8_sm120_available` | Check SM120 FP8 I/O availability |

```python
import pygpukit as gpk
import numpy as np

# Check availability
if gpk.fp8_fp8_sm120_available():
    # Get scale sizes for blockwise scaling
    sfa_size, sfb_size = gpk.fp8_fp8_get_scale_sizes(M, N, K)

    # Blockwise scaled FP8 GEMM (for real FP8 models)
    scale_a = gpk.from_numpy(np.ones(sfa_size, dtype=np.float32))
    scale_b = gpk.from_numpy(np.ones(sfb_size, dtype=np.float32))
    C = gpk.matmul_fp8_fp8_blockwise_sm120(A_fp8, B_fp8, scale_a, scale_b)
```

### Pure NVF4 GEMM (446 TFLOPS)
GPU-side BF16->NVF4 quantization with 3-stage pipeline for maximum throughput:

| Matrix Size | TFLOPS | Notes |
|-------------|--------|-------|
| 8192x8192 | 261 | Branchless vectorized loads |
| 12288x12288 | 383 | 3-stage async pipeline |
| 16384x16384 | **446** | Direct write to user buffer |

### New Math Operations
Extended math operations for GPU computing:

| Category | Operations |
|----------|------------|
| **Trigonometric** | `sin`, `cos` |
| **Power/Root** | `sqrt`, `rsqrt` |
| **Sign** | `abs`, `neg` |
| **Comparison** | `clamp`, `where` |
| **Activation** | `sigmoid`, `tanh` |
| **Reduction** | `argmax`, `min`, `sum_axis` |

```python
import pygpukit as gpk

# Trigonometric
y = gpk.sin(x)
y = gpk.cos(x)

# Power operations
y = gpk.sqrt(x)
y = gpk.rsqrt(x)  # 1/sqrt(x)

# Element-wise comparison
y = gpk.clamp(x, min_val=-1.0, max_val=1.0)
y = gpk.where(cond, x, y)  # cond ? x : y

# New activations
y = gpk.sigmoid(x)
y = gpk.tanh(x)

# New reductions
idx = gpk.argmax(x)     # Index of maximum
val = gpk.min(x)        # Minimum value
y = gpk.sum_axis(x, 1)  # Sum along axis
```

### uint8/int8 NumPy Support
`from_numpy` now supports uint8 and int8 arrays for FP8 data handling:

```python
# FP8 data stored as uint8
fp8_data = np.array([...], dtype=np.uint8)
gpu_fp8 = gpk.from_numpy(fp8_data)
```

---


> **Previous versions (v0.2.4 - v0.2.14):** See [CHANGELOG.md](CHANGELOG.md) for complete release history.


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
## Performance

### RTX 5090 Benchmark (SM120a, CUDA 13.1)

#### Standard Precision (8192x8192)

| Precision | TFLOPS | Notes |
|-----------|--------|-------|
| **FP32** | 80 | CUDA cores |
| **TF32** | 87 | TensorCore |
| **FP16** | 170 | TensorCore |
| **BF16** | **173** | TensorCore |

#### Quantized GEMM (M=8192, K=4096, N=14336)

| Format | TFLOPS | Error | Notes |
|--------|--------|-------|-------|
| **FP8xFP8** | **217** | ~0.1% | CUTLASS SM120 blockwise |
| **W8A16** | 50 | ~0.1% | FP8 weight, BF16 activation |
| **Int8 (via FP8)** | 142 | ~3.5% | TensorCore approximation |
| **Int8 (dp4a)** | 44 | **0%** | Exact, CUDA cores |
| **Int4 (via Int8)** | 121 | ~0.1% | TensorCore approximation |

#### NVF4 (4-bit NormalFloat) GEMM

| Matrix Size | TFLOPS | Notes |
|-------------|--------|-------|
| 8192x8192 | 261 | Pre-quantized |
| 12288x12288 | 383 | 3-stage pipeline |
| 16384x16384 | **446** | Peak performance |

> **Note:** NVF4xNVF4 achieves 4x memory bandwidth reduction vs BF16 with minimal accuracy loss.

### RTX 3090 Ti Benchmark (SM86)

| Matrix Size | FP32 | TF32 | FP16 | BF16 |
|-------------|------|------|------|------|
| 2048×2048 | 9.6 TFLOPS | 13 TFLOPS | 15 TFLOPS | 21 TFLOPS |
| 4096×4096 | 14.7 TFLOPS | 22 TFLOPS | 44 TFLOPS | 44 TFLOPS |
| 8192×8192 | 18 TFLOPS | **31 TFLOPS** | **63 TFLOPS** | **63 TFLOPS** |

> **Note:** CUTLASS is automatic for compatible sizes (16-aligned). Use `PYGPUKIT_NO_TF32=1` for full FP32 precision.

### GEMV Performance (RTX 5090, SM120a)

For LLM decode (M=1), custom GEMV kernels for different quantization formats:

| Layer | K | N | BF16 | W8A8 | W4A16 | Int4 |
|-------|------|-------|------|------|-------|------|
| Qwen-7B hidden | 4096 | 4096 | 65 us | **10 us** | 140 us | 31 us |
| Qwen-7B MLP up | 4096 | 14336 | 125 us | **17 us** | 141 us | 47 us |
| Qwen-7B MLP down | 14336 | 4096 | 399 us | **22 us** | 404 us | 58 us |
| Qwen-72B hidden | 8192 | 8192 | 232 us | **21 us** | 252 us | 51 us |
| Qwen-72B MLP up | 8192 | 29568 | 324 us | 146 us | 436 us | **112 us** |
| Qwen-72B MLP down | 29568 | 8192 | 839 us | 170 us | 1393 us | **129 us** |

| Kernel | Format | Memory vs BF16 | Best For |
|--------|--------|----------------|----------|
| **BF16 GEMV** | A:BF16, B:BF16 | 100% | Baseline |
| **W8A8 GEMV** | A:FP8, B:FP8 | 50% | Speed priority (6-18x faster) |
| **W4A16 GEMV** | A:BF16, B:NVF4 | 25% | Memory priority |
| **Int4 GEMV** | A:BF16, B:Int4 | 25% | Large K dimensions |

> **Note:** W8A8 (FP8/FP8) GEMV is 6-18x faster than BF16 for typical sizes on SM120. Int4 GEMV excels at very large K (29568+) where it matches or beats W8A8.

### GEMV Quantization Trade-offs (Explicit)

Why is W4A16 faster than NVF4/NVF4 despite both using 4-bit weights?

| Kernel | A (Activation) | B (Weight) | Dequant Work | Speed |
|--------|---------------|------------|--------------|-------|
| **W4A16** | BF16 (native) | NVF4 (4-bit) | 1x (B only) | **104 us** |
| **NVF4/NVF4** | NVF4 (4-bit) | NVF4 (4-bit) | 2x (A + B) | 219 us |

**Per Scale Block (32 elements):**
| Operation | W4A16 | NVF4/NVF4 |
|-----------|-------|-----------|
| Scale load | 1 (B) | 2 (A + B) |
| Scale decode (LUT) | 1 | 2 |
| Pre-scaled LUT build | 16 mul | 16 mul |

**Per Element:**
| Operation | W4A16 | NVF4/NVF4 |
|-----------|-------|-----------|
| A conversion | BF16->float (free) | LUT lookup |
| B conversion | LUT lookup | LUT lookup |

**Conclusion:** NVF4/NVF4 trades speed for memory. Use when:
- Memory-constrained (A is 4x smaller)
- Batch inference with large A tensors

For single-token decode (M=1), **W4A16 or FP8 is recommended**.

### Comprehensive GEMV Benchmark (RTX 5090, SM120a)

All GEMV kernels compared on Qwen2.5-7B gate_proj (K=3584, N=18944):

| Kernel | A dtype | B dtype | Weight Size | Time (us) | vs BF16 |
|--------|---------|---------|-------------|-----------|---------|
| BF16 | BF16 | BF16 | 129.5 MB | 119 | 1.00x |
| FP8/BF16 (W8A16) | BF16 | FP8 | 64.8 MB | 272 | 0.44x |
| **FP8/FP8 (W8A8)** | FP8 | FP8 | 64.8 MB | **19** | **6.2x** |
| NVF4/BF16 (W4A16) | BF16 | NVF4 | 32.4 MB | 106 | 1.12x |
| NVF4/NVF4 (W4A4) | NVF4 | NVF4 | 32.4 MB | 217 | 0.55x |

**Performance by Layer Type:**

| Layer | K | N | Best Kernel | Speedup |
|-------|---|---|-------------|---------|
| gate_proj | 3584 | 18944 | FP8/FP8 | 6.2x |
| down_proj | 18944 | 3584 | FP8/FP8 | 22.7x |
| o_proj | 3584 | 3584 | FP8/FP8 | 6.8x |
| qkv_proj | 3584 | 512 | FP8/FP8 | 9.1x |

> **Recommendation:** FP8/FP8 is optimal for SM120 (Blackwell). NVF4/BF16 (W4A16) provides the best balance when FP8 compute is unavailable.

### NVF4-BF16 GEMM Performance (RTX 5090, SM120a)

4-bit NVF4 GEMM with BF16 I/O using CUTLASS block-scaled tensor operations:

| Matrix Size | NVF4xBF16 | NVF4xNVF4 | Notes |
|-------------|-----------|-----------|-------|
| 4096×4096 | 64 TFLOPS | 87 TFLOPS | GPU-side quantization |
| 8192×8192 | 168 TFLOPS | 261 TFLOPS | 3-stage async pipeline |
| 16384×16384 | — | **446 TFLOPS** | Peak performance |

> **Note:** GPU-side BF16->NVF4 quantization with unit scaling. No host-device copies. Ideal for memory-bound LLM inference with 4x bandwidth reduction vs BF16.

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
- **CUDA 13.0+** (required for SM120/Blackwell features)
- **Optional:** CUDA Toolkit (for JIT compilation of custom kernels)

#### Minimum Driver Versions (CUDA 13.x)
| Platform | Minimum Driver |
|----------|---------------|
| Linux | **590.44.01** or later |
| Windows | **572.16** or later (Game Ready/Studio) |

> **Note:** NVRTC (NVIDIA Runtime Compiler) is included in CUDA Toolkit.
> Pre-compiled GPU operations (matmul, add, mul, etc.) work with just GPU drivers.

### Supported GPUs

| Generation | Architecture | Examples | Status |
|------------|-------------|----------|--------|
| **Ampere** | SM80-86 | A100, RTX 3090, RTX 3080 | Fully supported |
| **Ada Lovelace** | SM89 | RTX 4090, RTX 4080 | Fully supported |
| **Hopper** | SM90 | H100, H200 | Fully supported |
| **Blackwell** | SM100-120 | B100, B200, RTX 5090 | **CUDA 13.0+ required** |
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
  .claude/         # Claude Code configuration
    skills/          # Development workflow skills
    agents/          # Specialized subagents
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
| **v0.2.10** | **Dynamic cuBLASLt loading**, CUDA Graph optimizations, descriptor caching |
| **v0.2.11** | **Batch decode** (6.8x speedup), Decode Strategy framework, Driver API async, Dual CUDA builds, RTX 5090 (SM120) |
| **v0.2.12** | **Advanced audio processing** (ISTFT, Griffin-Lim, HPSS, CQT, pitch detection, time stretch) |
| **v0.2.15** | **FP8 I/O GEMM** (blockwise scaling), Pure NVF4 (446 TFLOPS), New math ops (sin, cos, sqrt, rsqrt, abs, neg, clamp, where, sigmoid, tanh, argmax, min, sum_axis) |
| **v0.2.16** | **MoE support** (Mixtral), Thinking models (Qwen3), W8A8/W4A4 GEMV, W8A16/Int8/Int4 GEMM, Kernel restructure |

### Planned

| Version | Goals |
|---------|-------|
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
| **Elementwise** | `add`, `sub`, `mul`, `div`, `neg`, `abs`, `clamp`, `where` |
| **Math** | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `softmax` |
| **Matrix** | `matmul`, `transpose` |
| **Reductions** | `sum`, `sum_axis`, `mean`, `max`, `min`, `argmax` |
| **Neural** | `layernorm`, `rmsnorm`, `silu`, `sdpa_causal`, `rope_inplace`, `bias_add_inplace`, `linear_bias_gelu` |
| **Types** | `GPUArray`, `DataType`, `float32`, `float64`, `float16`, `bfloat16`, `int32`, `int64`, `int8`, `uint8` |
| **LLM** | `llm.SafeTensorsFile`, `llm.CausalTransformerModel`, `llm.load_model_from_safetensors` |
| **LLM (Experimental)** | `llm.Tokenizer` (use HuggingFace tokenizers for production) |

### Deprecation Policy
APIs to be removed will emit `DeprecationWarning` for at least one minor version before removal.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork and clone
2. Create feature branch
3. Build: `./build.sh 86` (Git Bash)
4. Run checks: `ruff check`, `mypy`, `pytest`
5. Submit PR

**We Accept:** Performance improvements, bug fixes, new GPU ops, documentation
**We Reject:** cuda-python dependencies, training features, SM < 80 support

---

## License
MIT License

---

## Acknowledgements

Inspired by and built upon:
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - Runtime, Driver API, NVRTC
- [CUTLASS](https://github.com/NVIDIA/cutlass) - TensorCore GEMM optimization techniques
- [Codon](https://github.com/exaloop/codon) - High-performance Python compiler with GPU support
- [CuPy](https://github.com/cupy/cupy)
- [Triton](https://github.com/triton-lang/triton)

PyGPUkit aims to fill the gap for a tiny, embeddable GPU runtime for Python.

---

If this project saved you from a silent GPU bug,
or helped you trust your results again,
consider giving it a ⭐.

Correctness deserves visibility.

---
