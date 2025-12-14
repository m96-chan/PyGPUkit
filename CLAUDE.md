# PyGPUkit - Claude Code Guidelines

---

## Goal Statement

PyGPUkit aims to free developers from the complexity of CUDA Toolkit, Anaconda, and fragile GPU environments. Its goal is to make GPU programming and model execution feel like using a standard Python library: installable via pip, minimal setup, and no mandatory external SDKs. PyGPUkit provides high-performance GPU kernels, memory management, scheduling, and model execution (e.g. SafeTensors) through a NumPy-like API and a Kubernetes-inspired resource model, allowing developers to use GPUs explicitly, predictably, and productively without fighting their environment.

## Project Goals

1. Provide the smallest usable GPU runtime for Python
2. Expose GPU scheduling (bandwidth, memory, partitioning)
3. Make writing custom GPU kernels easy
4. Serve as a building block for inference engines, DSP systems, and real-time workloads

---

## Architecture

### Layer Model

```
Python (High-level orchestration only)
    ↓
Rust (Core scheduling, memory management, GPU coordination)
    ↓
C++ (CUDA Driver/Runtime API, NVRTC, kernel launch)
```

**Python is ONLY a high-level orchestration layer.**
The core scheduling, memory management, GPU coordination, and performance-critical components **MUST** remain implemented in Rust.

### Directory Structure

```
PyGPUkit/
├── src/pygpukit/           # Python API (NumPy-compatible)
├── native/
│   ├── core/               # C++ (CUDA Runtime/Driver API)
│   ├── jit/                # C++ (NVRTC)
│   ├── ops/                # C++ (CUDA kernels)
│   └── bindings/           # pybind11
├── rust/
│   ├── pygpukit-core/      # Pure Rust GPU runtime
│   │   └── src/
│   │       ├── memory/     # MemoryPool, LRU, size-class allocator
│   │       ├── scheduler/  # Task state machine, QoS policies
│   │       └── device.rs   # DeviceCapabilities, KernelType
│   └── pygpukit-python/    # PyO3 bindings
├── examples/
└── tests/
```

### Language Responsibilities

| Component | Language | Reason |
|-----------|----------|--------|
| Python API | Python | NumPy-compatible user interface |
| CUDA Driver/Runtime | C++ | Direct hardware access |
| NVRTC JIT | C++ | Kernel compilation |
| Memory Pool/LRU | Rust | Safe, fast memory management |
| Scheduler State | Rust | Thread-safe state machine |
| Kernel Launch | C++ | CUDA kernel dispatch |
| Bindings | pybind11, PyO3 | C++/Rust to Python |

### Required Rust Components (MUST NOT be removed)

1. **Rust memory pool** (with LRU eviction)
2. **Rust GPU scheduler state machine**
3. **Rust-side async GPU memory transfer engine**
4. **Rust-side kernel dispatch controller**

### Architecture Rules

1. **pygpukit-core is the authoritative runtime** - MemoryPool, Scheduler, Task, LRU, SizeClass MUST be implemented here
2. **All GPU memory management MUST live in** `rust/pygpukit-core/src/memory/`
3. **All scheduling logic MUST live in** `rust/pygpukit-core/src/scheduler/`
4. **Python bindings MUST be thin wrappers only** - no logic duplication
5. **When adding new features, always add them to Rust first**, then expose via PyO3

---

## GPU Backend Model

### Code Generation Pipeline

```
Python API → pybind11 → C++ backend → CUDA Driver API (cu*) / Runtime API (cuda*) / NVRTC

source.cu (string) → NVRTC → PTX → CUDA Driver API → CUmodule → CUfunction
```

- NO cuda-python
- NO external Python CUDA dependencies
- ALL GPU kernels compiled at runtime
- PTX → SASS handled by NVIDIA driver

### Dependencies

PyGPUkit uses its own C++ backend with CUDA Driver API / Runtime API / NVRTC.

**Do NOT mention or require:**
- ❌ `cuda-python`
- ❌ `numba.cuda`
- ❌ `cupy.cuda`
- ❌ PyCUDA-style wrappers

### GPU Initialization

GPU availability is detected via these C++ calls:
- `cudaGetDeviceCount()`
- `cudaDriverGetVersion()`
- `cudaRuntimeGetVersion()`
- `nvrtcVersion()`

CPU fallback happens only if one of these fails.

### CPU Fallback

When GPU is unavailable, PyGPUkit must:
- Run scheduler in CPU simulation mode
- Use NumPy as backend for GPUArray ops
- Disable NVRTC
- Still expose full API (no errors)

### Backend Loader Model

Python loads a shared library:
- Linux: `_pygpukit_native.cpython-3xx-x86_64-linux-gnu.so`
- Windows: `_pygpukit_native.cp3xx-win_amd64.pyd`
- macOS: CPU backend only

### DLL Loading Model (Windows)

**v0.1.x (Current):**
- Requires CUDA Toolkit installation
- Loads DLLs from `CUDA_PATH/bin`

**v0.2+ (Planned - Driver-Only Mode):**
- NVRTC DLL shipped inside the wheel
- CUDA Driver (`nvcuda.dll`) provided by NVIDIA GPU drivers
- No cudart dependency

### Error Messages

**NEVER generate:**
- ❌ "Please install cuda-python"
- ❌ "GPU mode requires the cuda-python package"

**Instead use:**
- ✅ "CUDA driver not detected"
- ✅ "NVRTC JIT compiler not available"
- ✅ "No GPU devices found (cudaGetDeviceCount == 0)"
- ✅ "Falling back to CPU simulation backend"

---

## Critical Rules

### DO NOT

1. Use or mention `cuda-python` - it is NOT a dependency
2. Call CUDA APIs from Python directly
3. Implement memory management in pure Python (use Rust)
4. Ship precompiled CUDA kernels
5. Require specific CUDA toolkit versions at runtime
6. Convert Rust features to Python, Cython, Numba, or pure CUDA kernels
7. Delete Rust tasks from roadmap
8. Simplify architecture by removing Rust layer

### DO

1. Use C++ for all CUDA Driver/Runtime API calls
2. Compile all kernels at runtime with NVRTC
3. Use pybind11 for C++ to Python bindings
4. Keep Python layer thin - only API surface and NumPy interop
5. Support CPU fallback when GPU unavailable
6. Add new features to Rust first, then expose via PyO3

---

## Kernel Optimization

### Target Architectures

- **Supported:** Ampere (SM 80–86), Ada (SM 89), Hopper (SM 90)
- **Unsupported:** Architectures below SM80

### Design Philosophy

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
- Block sizes > 256 unless occupancy analysis proves benefit

### Kernel Autoselection

```cpp
int sm = device_sm_major * 10 + device_sm_minor;

if (sm >= 90) {
    use_mma_sync_kernels();  // Hopper/Ada
} else if (sm >= 80) {
    use_ampere_optimized_kernels();  // Ampere
} else {
    throw std::runtime_error("PyGPUkit requires SM >= 80 (Ampere)");
}
```

### MatMul Variants

For Ampere, implement two variants:
- **L2-optimized naive kernel** (fast for FP32)
- **Warp-level MMA kernel** (TensorCore for TF32/FP16/BF16)

Block sizes: `(16, 16)` or `(32, 8)` - do NOT increase to 32×32 unless profiler proves faster.

### Memory Access Rules

- Align pointers to 128 bytes where possible
- Ensure loads are coalesced across warps
- Prefer `float4` / `half8` vectorized loads
- Avoid bank conflicts in shared memory
- Use register blocking aggressively

### Benchmark Targets

| GPU | FP32 | TF32 TensorCore |
|-----|------|-----------------|
| RTX 3090 Ti | 18 TFLOPS | 27+ TFLOPS |
| A100 | 5.5+ TFLOPS | 156 TFLOPS |

**Achieved (v0.2.3):** TF32 on RTX 3090 Ti: **27.38 TFLOPS** (8192×8192×8192)

### CMake Flags

```cmake
-arch=sm_80
--expt-relaxed-constexpr
--use_fast_math
```

---

## TF32 TensorCore Implementation

### PTX mma.sync Fragment Mapping

**CRITICAL**: PTX inline assembly `mma.sync` has DIFFERENT fragment layouts than WMMA API.
Verified empirically using `dump_c_fragment.cu`.

#### `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`

Each thread in a warp (lane 0-31) holds:
- **A fragment**: 4 registers (16×8 matrix, row-major)
- **B fragment**: 2 registers (8×8 matrix, col-major)
- **C fragment**: 4 registers (16×8 matrix)

```
A fragment (16×8):
  a[0] = A[lane/4][lane%4]           // rows 0-7,  cols 0-3
  a[1] = A[lane/4 + 8][lane%4]       // rows 8-15, cols 0-3
  a[2] = A[lane/4][lane%4 + 4]       // rows 0-7,  cols 4-7
  a[3] = A[lane/4 + 8][lane%4 + 4]   // rows 8-15, cols 4-7

B fragment (8×8):
  b[0] = B[lane%4][lane/4]           // rows 0-3, cols 0-7
  b[1] = B[lane%4 + 4][lane/4]       // rows 4-7, cols 0-7

C fragment (16×8) - KEY DIFFERENCE FROM WMMA:
  c[0] = C[lane/4][(lane%4)*2]       // rows 0-7,  cols 0,2,4,6
  c[1] = C[lane/4][(lane%4)*2 + 1]   // rows 0-7,  cols 1,3,5,7
  c[2] = C[lane/4 + 8][(lane%4)*2]   // rows 8-15, cols 0,2,4,6
  c[3] = C[lane/4 + 8][(lane%4)*2 + 1] // rows 8-15, cols 1,3,5,7
```

#### Common Mistakes

1. **C fragment column stride**: PTX uses `(lane%4)*2` (stride 2), NOT `lane%4` (stride 1)
2. **C fragment pairs**: c[0],c[1] are adjacent columns; c[2],c[3] are +8 rows

#### WMMA API vs PTX

| Aspect | WMMA API | PTX mma.sync |
|--------|----------|--------------|
| Fragment types | `wmma::fragment<>` | Raw registers |
| Layout | Opaque (compiler-managed) | Must match PTX spec exactly |
| Flexibility | Limited shapes | Full control |

#### Size Difference

| API | A | B | C |
|-----|---|---|---|
| WMMA 16×16×8 | 16×8 | 8×16 | 16×16 |
| PTX m16n8k8 | 16×8 | 8×8 | 16×8 |

PTX m16n8k8 uses only the left half (cols 0-7) of WMMA's B/C.

### cp.async Double-Buffering

**Common Bug**: Prefetching into the wrong stage.

```cpp
// WRONG - overwrites current buffer
for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr = kt & 1;
    if (kt + 2 < num_k_tiles) {
        load_async((kt+2) & 1, kt + 2);  // BUG!
    }
    process(curr);
}

// CORRECT - prefetch into OTHER stage
load_async(0, 0);
cp_async_wait_0();

for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr = kt & 1;
    int next = curr ^ 1;  // OTHER stage

    if (kt + 1 < num_k_tiles) {
        load_async(next, kt + 1);
    }
    process(curr);
    cp_async_wait_0();
}
```

**Key Insight**: Always prefetch into the stage you're NOT currently reading from.

### Verified WMMA Kernel

```cpp
// WMMA row_major × row_major (PASS)
fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
fragment<accumulator, 16, 16, 8, float> c_frag;

load_matrix_sync(a_frag, A + k, K);
load_matrix_sync(b_frag, B + k * N, N);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C, c_frag, N, mem_row_major);
```

**Note:** `row_major` A + `col_major` B combination fails due to different memory layout interpretation.

### File Locations

- `native/ops/matmul_f32_tf32.cuh` - TF32 kernel
- `native/ops/basic.cu` - Dispatch logic
- Environment variable `PYGPUKIT_ALLOW_TF32=1` to enable

---

## Development Workflow

### Kernel Development Cycle

```
Edit → Build → Validate → Benchmark → Commit
```

**Always commit after validation and benchmark, regardless of results.**

### Commit Rules

1. Commit after every validation/benchmark completion, regardless of outcome
2. Include benchmark results in commit message
3. Never proceed to next kernel edit until commit is complete
4. Never overwrite a working kernel without committing first

### Commit Message Format

```
wip(tf32): <summary of changes>

Benchmark results (RTX 3090 Ti):
- 2048x2048: XX.XX TFLOPS
- 4096x4096: XX.XX TFLOPS
- 8192x8192: XX.XX TFLOPS

Correctness: <PASS/FAIL>
```

### Commit Triggers (ABSOLUTE)

You MUST commit immediately when:

1. **Benchmark improves** in ANY matrix size (even +0.01 TFLOPS)
2. **Correctness achieved** (relative error < 1e-3 for all sizes)
3. **After EVERY benchmark execution** - even if no improvement, commit with `bench: results logged (no improvement)`

### Regression Handling

If performance or correctness degrades:
- MUST revert to the previous commit BEFORE continuing

**Rationale:**
- Prevent losing fast kernel versions
- Track performance changes over time
- Preserve trial-and-error history

---

## Design Principles

### 1. GPU Systems Toolkit, Not ML Framework

PyGPUkit is **not** a replacement for PyTorch, JAX, or TensorFlow.
Its purpose is to provide **low-level, explicit, and controllable GPU execution primitives**.

- Focus: memory, kernels, scheduling, bandwidth, latency
- Not focus: autograd graphs, optimizers, training loops

### 2. Performance Is a Prerequisite, Not the Goal

High performance is assumed. Optimization enables scheduling, concurrency, and predictability.

- Slower-than-cuBLAS requires justification
- Faster-than-cuBLAS is welcome, but not mandatory
- Performance regressions are unacceptable without explicit trade-offs

### 3. NumPy-like Semantics

User-facing APIs should resemble **NumPy-style array operations**.

- `C = A @ B` is preferred over opaque operator graphs
- Explicit is better than implicit
- Users should understand when and how GPU work is executed

### 4. GPU Scheduling Is First-Class

PyGPUkit treats the GPU as a **shared, schedulable resource** (Kubernetes-inspired).

- Admission control, QoS, memory reservation, kernel pacing
- Scheduling decisions are explicit and inspectable
- Kernels are workloads, not side effects

### 5. SafeTensors Are Immutable Resources

SafeTensors are treated as **immutable, read-only GPU resources**.

- No in-place mutation
- No hidden ownership or lifecycle coupling

### 6. Using cuBLAS / CUTLASS Is Not a Failure

Leveraging vendor or OSS-optimized kernels is acceptable and encouraged.

- Value lies in orchestration, scheduling, and integration
- Reusing proven kernels is preferable to reinventing them

### 7. Determinism and Correctness Are Explicit

- TF32 precision loss is acceptable when explicitly enabled
- FP32 correctness must remain available
- Non-determinism must be explainable and bounded

---

## Non-goals

1. **Full Training Framework** - No optimizers, training loops, dataset pipelines, autograd engines
2. **Abstracting Away GPU Reality** - Memory transfers, sync points, kernel costs, precision trade-offs are NOT hidden
3. **Supporting Legacy GPUs** - Only Ampere/Ada and newer; Turing and below are out of scope
4. **PyTorch API Compatibility** - Clarity over familiarity; APIs may diverge intentionally
5. **"Magic" Performance** - No undocumented heuristics; all optimizations must be explainable

---

## Build System

- **C++/CUDA**: CMake with CUDA toolkit
- **Python**: scikit-build-core for CMake integration
- **Rust**: Cargo with PyO3
- **CI/CD**: cibuildwheel with CUDA

---

## Branch Strategy

| Change Type | Branch | Flow |
|-------------|--------|------|
| Hotfix (v0.1.x) | main | Direct push → tag |
| Minor/Major (v0.2+) | feature/* | Branch → PR → CI test → main → tag |

---

## Current State

### v0.1 (Released)
- ✅ Native C++ backend with CUDA Runtime/Driver API
- ✅ NVRTC JIT compilation
- ✅ pybind11 bindings
- ✅ Zero-copy Python↔Native interop
- ✅ CPU simulation fallback

### v0.2.x (Released)
- ✅ Rust memory pool with LRU eviction
- ✅ Rust GPU scheduler state machine
- ✅ L2-optimized naive matmul (18 TFLOPS)
- ✅ TF32 TensorCore GEMM (27 TFLOPS)
- ✅ SM >= 80 runtime check
- ✅ 106 Rust tests

### Remaining Work
- Rust-side async memory transfer engine
- Rust-side kernel dispatch controller
- Python API wrappers for Rust scheduler/memory pool (thin wrappers only)
