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

### 10. DLL Loading Model (Windows)

#### v0.1.x (Current):
- **Requires CUDA Toolkit installation**
- Loads DLLs from `CUDA_PATH/bin`
- Links against `cudart`, `cuda_driver`, `nvrtc`

#### v0.2 (Planned - Driver-Only Mode):
**PyGPUkit will NOT depend on CUDA Toolkit installation.**

1. **NVRTC DLL** (`nvrtc64_xx_0.dll`) will be **shipped inside the wheel**
   - Load via `os.add_dll_directory(<package_dir>)`
   - NOT from `CUDA_PATH/bin`

2. **CUDA Driver** (`nvcuda.dll`) is provided by **NVIDIA GPU drivers**
   - Always available if user has NVIDIA GPU
   - Do NOT search CUDA Toolkit paths

3. **cudart64_xx.dll will NOT be used**
   - PyGPUkit will use CUDA Driver API + NVRTC only

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

## Kernel Optimization Directives (CRITICAL)

**Target GPU architectures:** Ampere (SM 80–86), Ada (SM 89), Hopper (SM 90)
**Architectures below SM80 are officially unsupported.**

### 1. Kernel Design Philosophy

**DO NOT** use classic shared-memory tiling as the main optimization.
On Ampere, L2 is large and fast; naive or warp-level kernels outperform tiled kernels.

**Prefer:**
- L2-friendly memory access patterns
- Coalesced loads (`ld.global.cs`)
- Warp-level primitives (shuffle, reduce)
- Tensor-core paths when possible (`wmma`, `mma.sync`)
- Asynchronous copy (`cp.async`) for global→shared prefetch

**Avoid:**
- Unnecessary `__syncthreads()`
- Complex shared-memory patterns designed for Pascal/Turing
- Block sizes > 256 unless occupancy analysis explicitly shows benefit

### 2. Kernel Autoselection Rules

```cpp
int sm = device_sm_major * 10 + device_sm_minor;

if (sm >= 90) {
    // Hopper/Ada
    use_mma_sync_kernels();
} else if (sm >= 80) {
    // Ampere (A100, 3090, 3080)
    use_ampere_optimized_kernels();
} else {
    throw std::runtime_error("PyGPUkit requires SM >= 80 (Ampere)");
}
```

**No fallback kernels for older GPUs.**

### 3. MatMul Optimization Directives

For Ampere, implement two variants:
- **A. L2-optimized naive kernel** (fast for fp32)
- **B. Warp-level MMA kernel** (tensor core)

Block sizes:
```cpp
blockDim = (16, 16) or (32, 8)
grid = ceil((M,N)/block)
```

**Do NOT** increase blockDim to 32×32 unless profiler proves faster.

**Prefer:**
- `__ldg()` or modern `ld.global.cs` patterns
- Avoid shared-memory tiles except for mma kernels

**Enable Tensor Core fast paths for:**
- FP16
- BF16
- TF32 (Ampere only)

For mma kernels:
```
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

### 4. Memory Access Optimization Rules

- Align pointers to 128 bytes where possible
- Ensure loads are coalesced across warps
- Prefer `float4` / `half8` vectorized loads
- Avoid bank conflicts in shared memory (power of 2 strides)
- Use register blocking aggressively (Ampere has huge register file)

### 5. Remove Legacy Code

**DELETE or AVOID:**
- Pascal/Turing shared-memory kernels
- 32×32 tiled kernels
- Any kernel heavily relying on `__syncthreads()` inside inner loops
- SM60–75 fallback paths
- Shared-memory based matmul unless using mma

### 6. Benchmark Expectations (Target)

| GPU | FP32 naive-opt | FP32 MMA | Notes |
|-----|---------------|----------|-------|
| RTX 3090 | 2.1–2.3 TFLOPS | 9+ TFLOPS | TF32 or FP16 |
| A100 | 5.5+ TFLOPS | 156 TFLOPS | tensor cores |

If performance regresses from naive baseline, re-profile.

### 7. CMake Compilation Flags

```cmake
-arch=sm_80
--expt-relaxed-constexpr
--use_fast_math
```

For portability: allow runtime switch to sm_89, sm_90.

---

## Build System

- **C++/CUDA**: CMake with CUDA toolkit
- **Python**: scikit-build-core for CMake integration
- **Rust** (v0.2+): Cargo with PyO3
- **CI/CD**: cibuildwheel with CUDA

---

## Branch Strategy

| Change Type | Branch | Flow |
|-------------|--------|------|
| Hotfix (v0.1.x) | main | Direct push → tag |
| Minor/Major (v0.2+) | feature/* | Branch → PR → CI test → main → tag |

**Why feature branches for v0.2+:**
- CI runs tests on PR before merge
- Review changes before merging to main
- Avoid breaking main with incomplete features

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

1. ✅ Implement Python memory pool with LRU eviction
2. ✅ Implement Python scheduler state machine
3. Add L2-optimized naive matmul kernel (target: 2.1-2.3 TFLOPS)
4. Add SM >= 80 runtime check (reject older GPUs)
5. Add Tensor Core MMA kernel for FP16/TF32
6. Add async memory transfers
