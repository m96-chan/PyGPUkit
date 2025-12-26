
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

## What's New in v0.2.15

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
| 8192x8192 | 320 | Branchless vectorized loads |
| 12288x12288 | 400 | 3-stage async pipeline |
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

## What's New in v0.2.14

### Packaging Fixes
v0.2.13 and v0.2.14 fix wheel RECORD file issues that caused PyPI deprecation warnings.

| Version | Issue | Fix |
|---------|-------|-----|
| v0.2.14 | Windows wheel missing `licenses/LICENSE` in RECORD | Added `-Recurse` to scan dist-info subdirectories |
| v0.2.13 | Hardcoded version in release workflow | Dynamic dist-info folder detection |

**Recommended:** Use v0.2.15 or later.

```bash
pip install pygpukit>=0.2.15
```

---

## What's New in v0.2.12

### GPU Audio Processing (Driver-Only)
Comprehensive audio processing operations with custom Radix-2 FFT - no cuFFT dependency.

| Category | Operations |
|----------|------------|
| **Time-Frequency** | `stft`, `istft`, `griffin_lim` |
| **Spectral Features** | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast` |
| **Pitch Detection** | `detect_pitch_yin`, `detect_pitch_yin_frames`, `autocorrelation` |
| **Music Analysis** | `cqt`, `chroma_stft`, `chroma_cqt`, `zero_crossing_rate` |
| **Source Separation** | `hpss`, `harmonic`, `percussive` |
| **Time/Pitch** | `time_stretch`, `pitch_shift` |

```python
from pygpukit.ops import audio
import numpy as np

# Load audio
samples = np.random.randn(16000).astype(np.float32)  # 1 sec @ 16kHz
buf = audio.from_pcm(samples, sample_rate=16000)

# STFT -> Magnitude -> ISTFT roundtrip
stft_out = audio.stft(buf, n_fft=512, hop_length=160)
mag = audio.magnitude_spectrum(stft_out)
reconstructed = audio.griffin_lim(mag, n_iter=32)

# Spectral features
centroid = audio.spectral_centroid(mag, sample_rate=16000)
flatness = audio.spectral_flatness(mag)

# HPSS (Harmonic-Percussive Separation)
harmonic, percussive = audio.hpss(mag, kernel_size=17)

# Time stretch (slow down to half speed)
slow = audio.time_stretch(buf, rate=0.5)

# Pitch shift (+12 semitones = 1 octave up)
higher = audio.pitch_shift(buf, sample_rate=16000, n_steps=12)
```

### Previous Audio Features (v0.2.11)
| Feature | Description |
|---------|-------------|
| **STFT** | Custom Radix-2 FFT (no cuFFT) |
| **Mel Filterbank** | Whisper-compatible preprocessing |
| **MFCC** | DCT-II based extraction |
| **VAD** | Voice Activity Detection |
| **Streaming** | Ring buffer, windowing |

---

## What's New in v0.2.11

### Batch Decode Support
Batch decoding enables processing multiple tokens in parallel, achieving near-linear speedup with TensorCore utilization.

| Batch Size | Per Token (us) | Throughput | Speedup |
|------------|---------------|------------|---------|
| 1 | 381,303 | 2.6 tok/s | 1.00x |
| 2 | 205,030 | 4.9 tok/s | 1.86x |
| 4 | 108,521 | 9.2 tok/s | 3.51x |
| 8 | 55,845 | 17.9 tok/s | **6.83x** |

### Decode Strategy Framework
Modular decode strategies for different use cases:

```python
from pygpukit.llm import DecodeM1, DecodeM1Graph, DecodeBatch, DecodeJacobi

# Standard single-token decode
m1 = DecodeM1()
m1.bind(model)

# CUDA Graph accelerated decode
m1_graph = DecodeM1Graph()
m1_graph.bind(model)
m1_graph.init_graph(max_seq_len=512)

# Batch decode for high throughput
batch = DecodeBatch(batch_size=8)
batch.bind(model)
```

| Strategy | Throughput | Use Case |
|----------|-----------|----------|
| DecodeM1 | 3.2 tok/s | Simple, low memory |
| DecodeM1Graph | 2.2 tok/s | Reduced kernel launch overhead |
| DecodeBatch (batch=8) | **19.6 tok/s** | High throughput |

### CUDA Graph Improvements
- Volatile reads for proper graph replay (attention, embedding, KV cache kernels)
- Separate `DecodeM1Graph` strategy for cleaner architecture
- Fixed stream handling for RoPE and SDPA operations

### Driver API Async Memory Operations
New async memory transfer functions using CUDA Driver API:

```python
from pygpukit.core import memcpy_host_to_device_async, pinned_malloc, pinned_free

# Pinned memory for faster transfers
pinned_ptr = pinned_malloc(size_bytes)
memcpy_host_to_device_async(device_ptr, pinned_ptr, size_bytes, stream)
```

### CUDA 13.x Required
Starting from v0.2.15, PyGPUkit requires **CUDA 13.0+** for SM120 (Blackwell) support:

| Module | CUDA Version | SM Support |
|--------|-------------|------------|
| `_pygpukit_native_cu131` | CUDA 13.1 | SM 80-120 (Blackwell) |

> **Note:** CUDA 12.x builds have been discontinued. SM120 features (FP8 I/O GEMM, NVF4 GEMM) require CUDA 13.0+.

### RTX 5090 Support
Full support for NVIDIA Blackwell consumer GPUs (SM120) via CUDA 13.x build.

### Qwen2 Architecture Support
Added `QWEN2_SPEC` for Qwen2/Qwen2.5 model family:

```python
from pygpukit.llm import detect_model_spec, QWEN2_SPEC

spec = detect_model_spec(tensor_names)  # Auto-detects Qwen2
# Or explicitly: spec = QWEN2_SPEC
```

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
| **Qwen2/2.5** | RMSNorm, SiLU, RoPE, GQA | ✅ Tested |
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

### GEMV Performance (RTX 5090, SM120a)

For LLM decode (M=1), custom GEMV kernels significantly outperform cuBLASLt:

| Model Layer | K | N | cuBLASLt | BF16 GEMV | NVF4 GEMV | Memory |
|-------------|------|-------|----------|-----------|-----------|--------|
| Qwen-7B hidden | 4096 | 4096 | 413us | **97us** | 152us | 73% less |
| Qwen-7B MLP | 4096 | 11008 | 418us | **96us** | 153us | 73% less |
| Qwen-72B hidden | 8192 | 8192 | 799us | 266us | **265us** | 73% less |
| Qwen-72B MLP | 8192 | 29568 | 1603us | **375us** | 454us | 73% less |

| Kernel | Description | Use Case |
|--------|-------------|----------|
| **BF16 GEMV** | Custom BF16 kernel optimized for M=1 | Speed priority |
| **NVF4 GEMV** | 4-bit NVF4 weights with block scaling | Memory priority (73% reduction) |

> **Note:** For large K (8192+), NVF4 matches BF16 speed while using 73% less memory. Ideal for memory-constrained LLM inference.

### NVF4-BF16 GEMM Performance (RTX 5090, SM120a)

4-bit NVF4 GEMM with BF16 I/O using CUTLASS block-scaled tensor operations:

| Matrix Size | TFLOPS (median) | TFLOPS (max) | Time (ms) |
|-------------|-----------------|--------------|-----------|
| 4096×4096 | 53 | 55 | 2.6 |
| 8192×8192 | 141 | 143 | 7.8 |
| 12288×12288 | 201 | 216 | 18.5 |
| 16384×16384 | **246** | **252** | 35.8 |

> **Note:** GPU-side BF16→NVF4 quantization with unit scaling. No host-device copies. Ideal for memory-bound LLM inference with 4x bandwidth reduction vs BF16.

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
