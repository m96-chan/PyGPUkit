
# PyGPUkit — Lightweight GPU Runtime for Python
*A minimal, modular GPU runtime with Rust-powered scheduler, NVRTC JIT compilation, and a clean NumPy-like API.*

[![PyPI version](https://badge.fury.io/py/PyGPUkit.svg)](https://badge.fury.io/py/PyGPUkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- RTX 30XX series (Ampere, SM 80+) and above
- Older GPUs (RTX 20XX, GTX 10XX, etc.) are **NOT supported** (SM < 80)

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
  examples/        # Demo scripts
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

### Planned

| Version | Goals |
|---------|-------|
| **v0.2.7** | Full API review, documentation, backward compatibility |
| **v0.3** | Triton backend, advanced ops (softmax, layernorm), MPS/MIG |

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
