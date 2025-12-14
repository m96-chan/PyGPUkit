
# PyGPUkit — Lightweight GPU Runtime for Python
*A minimal, modular GPU runtime with Rust-powered scheduler, NVRTC JIT compilation, and a clean NumPy-like API.*

[![PyPI version](https://badge.fury.io/py/PyGPUkit.svg)](https://badge.fury.io/py/PyGPUkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview
**PyGPUkit** is a lightweight GPU runtime for Python that provides:
- **Rust-powered scheduler** with admission control, QoS, and resource partitioning
- NVRTC-based JIT kernel compilation
- A NumPy-like `GPUArray` type
- Kubernetes-inspired GPU scheduling (bandwidth + memory guarantees)
- Extensible operator set (add/mul/matmul, custom kernels)
- Minimal dependencies and embeddable runtime

PyGPUkit aims to be the "micro-runtime for GPU computing": small, fast, and ideal for research, inference tooling, DSP, and real-time systems.

---

## Opening Paragraph (Goal Statement)

PyGPUkit aims to simplify GPU development by reducing dependency on complex CUDA Toolkit installations and fragile GPU environments.
Its goal is to make GPU programming feel like using a standard Python library: installable via pip with minimal setup. PyGPUkit provides high-performance GPU kernels, memory management, and scheduling through a NumPy-like API and a Kubernetes-inspired resource model, allowing developers to use GPUs explicitly, predictably, and productively.

> **Note:** PyGPUkit requires NVIDIA GPU drivers. NVRTC (JIT compilation) is **optional** — pre-compiled kernels work without CUDA Toolkit. It is NOT a PyTorch/CuPy replacement—it's a lightweight runtime for custom GPU workloads, research, and real-time systems where full ML frameworks are overkill.

---

## v0.2.3 Features (NEW)

### TF32 TensorCore GEMM
| Feature | Description |
|---------|-------------|
| **PTX mma.sync** | Direct TensorCore access via inline PTX assembly |
| **cp.async Pipeline** | Double-buffered async memory transfers |
| **TF32 Precision** | 19-bit mantissa (vs FP32's 23-bit), ~0.1% per-op error |
| **SM 80+ Required** | Ampere architecture (RTX 30XX+) required |

### Benchmark Comparison (RTX 3090 Ti, 8192×8192×8192)

| Library | FP32 | TF32 | Requires | Notes |
|---------|------|------|----------|-------|
| **NumPy** (OpenBLAS) | ~0.8 TFLOPS | — | CPU only | CPU baseline |
| **cuBLAS** | ~21 TFLOPS | ~59 TFLOPS | CUDA Toolkit | [NVIDIA benchmark](https://forums.developer.nvidia.com/t/a40-and-3090-gemm-performance-test-data/249424) |
| **PyGPUkit** (Driver-Only) | 17.7 TFLOPS | 28.2 TFLOPS | **GPU drivers only** | No CUDA Toolkit needed! |
| **PyGPUkit** (CUDA Toolkit) | 17.7 TFLOPS | 30.3 TFLOPS | CUDA Toolkit | +JIT compilation |

> **v0.2.4+**: PyGPUkit is now a **single-binary distribution** — pre-compiled GPU operations work with just NVIDIA drivers installed. CUDA Toolkit is only needed for JIT compilation of custom kernels. Performance is virtually identical between modes.

### PyGPUkit Performance by Size (Driver-Only)
| Matrix Size | FP32 | TF32 |
|-------------|------|------|
| 2048×2048 | 8.7 TFLOPS | 12.2 TFLOPS |
| 4096×4096 | 14.2 TFLOPS | 22.0 TFLOPS |
| 8192×8192 | 17.7 TFLOPS | **28.2 TFLOPS** |

### Core Infrastructure (Rust)
| Feature | Description |
|---------|-------------|
| **Memory Pool** | LRU eviction, size-class free lists |
| **Scheduler** | Priority queue, memory reservation |
| **Transfer Engine** | Separate H2D/D2H streams, priority |
| **Kernel Dispatch** | Per-stream limits, lifecycle tracking |

### Advanced Features (Rust)
| Feature | Description |
|---------|-------------|
| **Admission Control** | Deterministic admission, quota enforcement |
| **QoS Policy** | Guaranteed/Burstable/BestEffort tiers |
| **Kernel Pacing** | Bandwidth-based throttling per stream |
| **Micro-Slicing** | Kernel splitting, round-robin fairness |
| **Pinned Memory** | Page-locked host memory with pooling |
| **Kernel Cache** | PTX caching, LRU eviction, TTL |
| **GPU Partitioning** | Resource isolation, multi-tenant support |

---

## Features
- **Lightweight** — smaller footprint than PyTorch/CuPy (not a replacement)
- **Modular** — runtime / memory / scheduler / JIT / ops
- **Rust Backend** — memory pool, scheduler, dispatch in Rust
- **GPUArray** with NumPy interop
- **NVRTC JIT** for CUDA kernels
- **Advanced Scheduler** with memory & bandwidth guarantees
- **106 Rust tests** for core components

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

Requirements:
- Python 3.10+
- NVIDIA GPU with drivers installed
- **Optional:** CUDA Toolkit (for JIT compilation of custom kernels)

> **Note:** NVRTC (NVIDIA Runtime Compiler) is included in CUDA Toolkit.
> Pre-compiled GPU operations (matmul, add, mul, etc.) work with just GPU drivers.
> CUDA Toolkit is only needed if you want to write and compile custom CUDA kernels at runtime.

**Supported GPUs:**
- RTX 30XX series (Ampere, SM 80+) and above
- Performance tuning is optimized for GPUs with large L2 cache (6MB+)
- Older GPUs (RTX 20XX, GTX 10XX, etc.) are **NOT supported** (SM < 80)

**Runtime Modes:**
| Mode | Requirements | Features |
|------|-------------|----------|
| **Full JIT** | GPU drivers + CUDA Toolkit | All features including custom kernels |
| **Pre-compiled only** | GPU drivers only | Built-in ops (matmul, add, etc.) |
| **CPU simulation** | None | Testing/development without GPU |

Check NVRTC availability:
```python
import pygpukit as gp
print(f"CUDA: {gp.is_cuda_available()}")
print(f"NVRTC: {gp.is_nvrtc_available()}")
```

---

## Project Goals
1. Provide the smallest usable GPU runtime for Python
2. Expose GPU scheduling (bandwidth, memory, partitioning)
3. Make writing custom GPU kernels easy
4. Serve as a building block for inference engines, DSP systems, and real-time workloads

---

## Usage Examples

### Allocate Arrays
```python
import pygpukit as gp

x = gp.zeros((1024, 1024), dtype="float32")
y = gp.ones((1024, 1024), dtype="float32")
```

### Basic Operations
```python
z = gp.add(x, y)
w = gp.matmul(x, y)
```

### CPU <-> GPU Transfer
```python
arr = z.to_numpy()
garr = gp.from_numpy(arr)
```

### Custom NVRTC Kernel (requires CUDA Toolkit)
```cuda
extern "C" __global__
void scale(float* x, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= factor;
}
```

```python
# Check if JIT is available before using custom kernels
if gp.is_nvrtc_available():
    kernel = gp.jit(src, func="scale")
    kernel(x, factor=0.5, n=x.size)
else:
    print("JIT requires CUDA Toolkit. Using pre-compiled ops instead.")
```

### Rust Scheduler (v0.2)
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

# Scheduler — Kubernetes-Inspired GPU Orchestration

PyGPUkit includes an experimental scheduler that treats a single GPU as a **multi-tenant compute node**, similar to how Kubernetes orchestrates CPU workloads. The goal is to provide **resource isolation, guarantees, and fair sharing** across multiple GPU tasks.

### **Core Capabilities**

---

## **1. GPU Memory Reservation**
Tasks may request a guaranteed block of GPU memory.

- Hard guarantees -> task is rejected if memory cannot be allocated
- Soft guarantees -> best-effort allocation
- Overcommit strategies (evict to host when pressure is high)
- Reclaim policies (LRU GPUArray eviction)

**Example:**
```python
task = scheduler.submit(
    fn,
    memory="512MB",
)
```

---

## **2. GPU Bandwidth Guarantees / Throttling**
Tasks may request a specific percentage of GPU compute bandwidth.

Bandwidth control is implemented via:
- Stream priority
- Kernel pacing (launch intervals)
- Micro-slicing large kernels
- Cooperative time-quantized scheduling
- Persistent dispatcher kernels (planned)

**Example:**
```python
task = scheduler.submit(
    fn,
    bandwidth=0.20,   # 20% GPU compute share
)
```

---

## **3. Logical GPU Partitioning**
PyGPUkit implements **software-defined GPU slicing**, similar in spirit to Kubernetes device plugin resource partitioning.

Slices may define:
- Memory quota
- Bandwidth share
- Stream priority band
- Isolation level

Useful for:
- Multi-tenant inference servers
- Real-time audio/DSP workloads
- Background/foreground GPU task separation

---

## **4. Scheduling Policies**
The scheduler supports multiple policies:

- **Guaranteed** — exclusive reservation, strict QoS
- **Burstable** — partial guarantees, opportunistic bandwidth
- **BestEffort** — uses leftover GPU cycles
- **Priority scheduling**
- **Deadline scheduling** (planned)
- **Weighted fair sharing**

**Example:**
```python
task = scheduler.submit(
    fn,
    policy="guaranteed",
    memory="1GB",
    bandwidth=0.10,
)
```

---

## **5. Admission Control**
Before executing a task, the scheduler performs:

- Resource validation
- Quota check
- QoS matching
- Scheduling feasibility

Results in:
- **admitted**
- **queued**
- **rejected**

---

## **6. Monitoring & Introspection**
PyGPUkit exposes live metrics:

- Memory usage per task
- SM occupancy and GPU utilization
- Throttling / pacing logs
- Queue position / execution state
- Reclaim/eviction count

**Example:**
```python
stats = scheduler.stats(task_id)
```

---

## **7. Soft Isolation Model**
While not OS-level isolation, each GPU task is provided:

- Dedicated stream groups
- Guaranteed memory pools
- Kernel pacing to enforce bandwidth
- Optional sandboxed GPUArray region

This provides practical multi-tenant safety without MIG/MPS.

---

# Project Structure
```
PyGPUkit/
  src/pygpukit/    # Python API (NumPy-compatible)
  native/          # C++ backend (CUDA Driver/Runtime/NVRTC)
  rust/            # Rust backend (memory pool, scheduler, dispatch)
    pygpukit-core/   # Pure Rust core logic
    pygpukit-python/ # PyO3 bindings
  examples/        # Demo scripts
  tests/           # Test suite
```

---

## Roadmap

### **v0.1 — v0.2.3 (Released)**

| Version | Highlights |
|---------|------------|
| **v0.1** | GPUArray, NVRTC JIT, add/mul/matmul, wheels |
| **v0.2.0** | Rust scheduler (QoS, admission control, partitioning), memory pool (LRU), kernel cache, 106 Rust tests |
| **v0.2.1** | API stabilization, error propagation |
| **v0.2.2** | Ampere SGEMM (cp.async, float4), 18 TFLOPS FP32 |
| **v0.2.3** | TF32 TensorCore (PTX mma.sync), 27.5 TFLOPS |

### **v0.2.4 — Single-Binary Distribution (Current)**
- [x] **Single-binary wheel** — no CUDA Toolkit required for pre-compiled ops
- [x] **Dynamic NVRTC loading** — JIT available when Toolkit installed
- [x] **Driver-only mode** — only `nvcuda.dll` required (from GPU drivers)
- [x] `is_nvrtc_available()` / `get_nvrtc_version()` / `get_nvrtc_path()` API
- [x] Graceful fallback when NVRTC unavailable
- [x] Performance tests made informational (always PASS with TFLOPS summary)
- [ ] Actual PyTorch/NumPy comparison benchmarks
- [ ] Large GPU memory test (16GB continuous alloc/free)

### **v0.2.5 — Distributed Phase**
- [ ] Multi-GPU Detection
- [ ] NCCL / peer-to-peer preliminary support
- [ ] Scheduler multi-device support

### **v0.2.6 — Pre-v0.3 Finalization**
- [ ] Full API review
- [ ] Backward compatibility policy
- [ ] JIT build options, safety measures, env vars cleanup
- [ ] Documentation

### **v0.3 (Planned)**
- [ ] Triton optional backend
- [ ] Advanced ops (softmax, layernorm)
- [ ] Inference-oriented plugin system
- [ ] MPS/MIG integration

---

## Contributing
Contributions and discussions are welcome!
Please open Issues for feature requests, bugs, or design proposals.

---

## License
MIT License

---

## Acknowledgements
Inspired by:
- CUDA Runtime
- NVRTC
- PyCUDA
- CuPy
- Triton

PyGPUkit aims to fill the gap for a tiny, embeddable GPU runtime for Python.
