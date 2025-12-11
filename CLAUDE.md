# PyGPUkit - Claude Code Guidelines

## GPU Backend Model (CRITICAL)

### Mental Model for Code Generation

```
Python API →
  pybind11 →
    C++ backend →
       CUDA Driver API (cu*)
       CUDA Runtime API (cuda*)
       NVRTC for JIT

NO cuda-python
NO external Python CUDA dependencies
ALL GPU kernels compiled at runtime
PTX → SASS handled by NVIDIA driver
```

### 1. PyGPUkit does NOT depend on cuda-python

**Do NOT mention or require:**
- ❌ `cuda-python`
- ❌ `numba.cuda`
- ❌ `cupy.cuda`
- ❌ PyCUDA-style wrappers

PyGPUkit uses its own C++ backend with CUDA Driver API / Runtime API / NVRTC.

### 2. GPU Initialization (CUDA Driver & Runtime API only)

GPU availability is detected via these C++ calls:
- `cudaGetDeviceCount()`
- `cudaDriverGetVersion()`
- `cudaRuntimeGetVersion()`
- `nvrtcVersion()`

CPU fallback happens only if one of these fails.

**NEVER claim:** "install cuda-python to enable GPU mode."

### 3. NVRTC JIT for All Kernels

PyGPUkit **MUST NOT** ship precompiled GPU kernels.

All GPU kernels are:
1. Stored as CUDA C source code (string literals)
2. Compiled at runtime using NVRTC
3. Loaded using CUDA Driver API (`cuModuleLoadData`, etc.)

This design removes CUDA version dependency.

### 4. API Preference Order

1. **CUDA Driver API** (`cu*`) - Core of PyGPUkit
2. **NVRTC** - Runtime compiler
3. **CUDA Runtime API** (`cuda*`) - Convenience functions

Driver API is the primary interface.

### 5. CUDA Version Compatibility via PTX JIT

**Assume:**
- NVRTC generates PTX
- NVIDIA driver compiles PTX to GPU machine code (SASS)
- PyGPUkit itself is version-agnostic

**NEVER** tie PyGPUkit to CUDA version numbers in error messages.

### 6. GPU Code Generation Pipeline

```
source.cu (string)
    ↓ NVRTC
PTX
    ↓ CUDA Driver API
CUmodule
    ↓
CUfunction
```

- ❌ No precompiled cubin files
- ❌ No static compilation
- ❌ No nvcc step

### 7. CPU Fallback = Fully Supported Backend

When GPU is unavailable, PyGPUkit must:
- Run scheduler in CPU simulation mode
- Use NumPy as backend for GPUArray ops
- Disable NVRTC
- Still expose full API (no errors)

**Do NOT** require CUDA Toolkit on macOS.

### 8. Backend Loader Model

Python loads a shared library:
- Linux: `_pygpukit_native.cpython-3xx-x86_64-linux-gnu.so`
- Windows: `_pygpukit_native.cp3xx-win_amd64.pyd`
- macOS: CPU backend only

The GPU backend is NOT a Python package like cuda-python.

### 9. Correct Error Messages

**NEVER generate:**
- ❌ "Please install cuda-python"
- ❌ "GPU mode requires the cuda-python package"
- ❌ "CUDA is missing because no Python bindings are found"

**Instead use:**
- ✅ "CUDA driver not detected"
- ✅ "NVRTC JIT compiler not available"
- ✅ "No GPU devices found (cudaGetDeviceCount == 0)"
- ✅ "Falling back to CPU simulation backend"

---

## Tech Stack

PyGPUkit is a **Rust + C++ + Python** hybrid project.

```
PyGPUkit/
│
├── src/pygpukit/  → Python API (NumPy-compatible)
│
├── native/
│   ├── core/      → C++ (CUDA Runtime/Driver API)
│   ├── jit/       → C++ (NVRTC)
│   ├── ops/       → C++ (CUDA kernels)
│   └── bindings/  → pybind11
│
├── rust/ (v0.2+)
│   ├── memory/    → Rust (LRU, pool allocator)
│   └── scheduler/ → Rust (state management)
│
└── examples/      → Demo scripts
```

### Language Responsibilities

| Component | Language | Reason |
|-----------|----------|--------|
| Python API | Python | NumPy-compatible user interface |
| CUDA Driver/Runtime | C++ | Direct hardware access |
| NVRTC JIT | C++ | Kernel compilation |
| Memory Pool/LRU | Rust (v0.2) | Safe, fast memory management |
| Scheduler State | Rust (v0.2) | Thread-safe state machine |
| Kernel Launch | C++ | CUDA kernel dispatch |
| Bindings | pybind11 | C++ to Python |

---

## Critical Rules

### DO NOT

1. **Do NOT** use or mention `cuda-python` - it is NOT a dependency
2. **Do NOT** call CUDA APIs from Python directly
3. **Do NOT** implement memory management in pure Python (use Rust in v0.2)
4. **Do NOT** ship precompiled CUDA kernels
5. **Do NOT** require specific CUDA toolkit versions at runtime

### DO

1. **DO** use C++ for all CUDA Driver/Runtime API calls
2. **DO** compile all kernels at runtime with NVRTC
3. **DO** use pybind11 for C++ to Python bindings
4. **DO** keep Python layer thin - only API surface and NumPy interop
5. **DO** support CPU fallback when GPU unavailable

---

## Build System

- **C++/CUDA**: CMake with CUDA toolkit
- **Python**: scikit-build-core for CMake integration
- **Rust** (v0.2+): Cargo with PyO3
- **CI/CD**: cibuildwheel with CUDA

---

## Current State (v0.1)

- ✅ Native C++ backend with CUDA Runtime/Driver API
- ✅ NVRTC JIT compilation
- ✅ pybind11 bindings
- ✅ Zero-copy Python↔Native interop
- ✅ CPU simulation fallback
- ✅ 73 tests pass
- ✅ Verified on RTX 3090 Ti (2152 GFLOPS matmul)

## Next Steps (v0.2)

1. Implement Rust memory pool with LRU eviction
2. Implement Rust scheduler state machine
3. Add tiled matmul with shared memory
4. Add async memory transfers
