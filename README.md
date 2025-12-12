
# PyGPUkit — Lightweight GPU Runtime for Python
*A minimal, modular GPU runtime with NVRTC JIT compilation, GPU scheduling, and a clean NumPy-like API.*

---

## 🚀 Overview
**PyGPUkit** is a lightweight GPU runtime for Python that provides:
- NVRTC-based JIT kernel compilation
- A NumPy-like `GPUArray` type
- Kubernetes-inspired GPU scheduler (bandwidth + memory guarantees)
- Extensible operator set (add/mul/matmul, custom kernels)
- Minimal dependencies and embeddable runtime

PyGPUkit aims to be the “micro-runtime for GPU computing”: small, fast, and ideal for research, inference tooling, DSP, and real-time systems.

---

## ✨ Features
- ⚡ **Lightweight** — no PyTorch/CuPy overhead
- 🧩 **Modular** — runtime / memory / scheduler / JIT / ops
- 📦 **GPUArray** with NumPy interop
- 🛠 **NVRTC JIT** for CUDA kernels
- 🎼 **Advanced Scheduler** with memory & bandwidth guarantees
- 🔌 Optional Triton backend (planned)
- 🧪 Test-friendly runtime

---

## 🔧 Installation
(Available after first PyPI release)

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
- Python 3.9+
- CUDA 11+
- NVRTC available
- NVIDIA GPU

**Supported GPUs:**
- RTX 30XX series (Ampere) and above
- Performance tuning is optimized for GPUs with large L2 cache (6MB+)
- Older GPUs (RTX 20XX, GTX 10XX, etc.) are NOT tuned and may have suboptimal performance

---

## 🧭 Project Goals
1. Provide the smallest usable GPU runtime for Python
2. Expose GPU scheduling (bandwidth, memory, partitioning)
3. Make writing custom GPU kernels easy
4. Serve as a building block for inference engines, DSP systems, and real-time workloads

---

## 📚 Usage Examples

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

### CPU ↔ GPU Transfer
```python
arr = z.to_numpy()
garr = gp.from_numpy(arr)
```

### Custom NVRTC Kernel
```cuda
extern "C" __global__
void scale(float* x, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= factor;
}
```

```python
kernel = gp.jit(src, func="scale")
kernel(x, factor=0.5, n=x.size)
```

---

# 🎼 Scheduler — Kubernetes‑Inspired GPU Orchestration

PyGPUkit includes an experimental scheduler that treats a single GPU as a **multi-tenant compute node**, similar to how Kubernetes orchestrates CPU workloads. The goal is to provide **resource isolation, guarantees, and fair sharing** across multiple GPU tasks.

### **Core Capabilities**

---

## **1. GPU Memory Reservation**
Tasks may request a guaranteed block of GPU memory.

- Hard guarantees → task is rejected if memory cannot be allocated
- Soft guarantees → best‑effort allocation
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
- Micro‑slicing large kernels
- Cooperative time‑quantized scheduling
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
PyGPUkit implements **software‑defined GPU slicing**, similar in spirit to Kubernetes device plugin resource partitioning.

Slices may define:
- Memory quota
- Bandwidth share
- Stream priority band
- Isolation level

Useful for:
- Multi‑tenant inference servers
- Real‑time audio/DSP workloads
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
While not OS‑level isolation, each GPU task is provided:

- Dedicated stream groups
- Guaranteed memory pools
- Kernel pacing to enforce bandwidth
- Optional sandboxed GPUArray region

This provides practical multi‑tenant safety without MIG/MPS.

---

# 🏗 Proposed Directory Structure
```
PyGPUkit/
  core/         # NVRTC wrapper, device info
  memory/       # GPUArray, allocators
  scheduler/    # orchestration, partitioning, throttling
  ops/          # built-in kernels
  jit/          # JIT compiler + cache
  python/       # high-level Python API
  examples/
  tests/
```

---

## 🧪 Roadmap

### **v0.1 (MVP)**
- GPUArray
- NVRTC JIT
- add/mul/matmul ops
- Basic stream manager
- Packaging + wheels

### **v0.2**
- Scheduler (memory + bandwidth guarantees)
- Kernel cache
- NumPy interop
- Benchmarks

### **v0.3**
- Triton optional backend
- Advanced ops (softmax, layernorm)
- Inference‑oriented plugin system

---

## 🤝 Contributing
Contributions and discussions are welcome!
Please open Issues for feature requests, bugs, or design proposals.

---

## 📄 License
MIT License

---

## ⭐ Acknowledgements
Inspired by:
- CUDA Runtime
- NVRTC
- PyCUDA
- CuPy
- Triton

PyGPUkit aims to fill the gap for a tiny, embeddable GPU runtime for Python.
