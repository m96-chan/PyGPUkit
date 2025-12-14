# PyGPUkit - Claude Code Guidelines

---

## IMPORTANT IMPLEMENTATION DIRECTIVES FOR PyGPUkit (MANDATORY)

You **MUST** preserve and prioritize all Rust-based components.
You are **NOT** allowed to delete, simplify, or replace Rust modules with Python equivalents.

### Required Rust Components (MUST NOT be removed)

1. **Rust memory pool** (with LRU eviction)
2. **Rust GPU scheduler state machine**
3. **Rust-side async GPU memory transfer engine**
4. **Rust-side kernel dispatch controller**

### Architecture Principle

```
Python (High-level orchestration only)
    ↓
Rust (Core scheduling, memory management, GPU coordination)
    ↓
C++ (CUDA Driver/Runtime API, NVRTC, kernel launch)
```

**Python is ONLY a high-level orchestration layer.**
The core scheduling, memory management, GPU coordination, and performance-critical components **MUST** remain implemented in Rust.

### Prohibited Actions

- ❌ **DO NOT** convert Rust features to Python, Cython, Numba, or pure CUDA kernels
- ❌ **DO NOT** delete Rust tasks from roadmap
- ❌ **DO NOT** mark Rust tasks as "implemented in Python"
- ❌ **DO NOT** simplify architecture by removing Rust layer

### If Rust tasks were removed, RESTORE them:

- "Implement Rust memory pool with LRU eviction"
- "Implement Rust scheduler state machine"
- "Add async memory transfers (Rust-side)"
- "Add kernel dispatch controller (Rust-side)"

**Rust is the authoritative implementation for all performance-critical subsystems.**

---

## PyGPUkit Rust Architecture Specification (MANDATORY)

You **MUST** follow the directory structure below exactly.
Do **NOT** delete, merge, or simplify these modules.
Python is only the binding layer; the Rust crates contain the authoritative logic.

```
rust/
├── Cargo.toml                               # Workspace root
├── pygpukit-core/                           # Pure Rust GPU runtime
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── memory/
│       │   ├── mod.rs
│       │   ├── block.rs                     # GPU memory block representation
│       │   ├── pool.rs                      # MemoryPool with LRU eviction
│       │   └── size_class.rs                # Size-class allocator
│       └── scheduler/
│           ├── mod.rs
│           ├── task.rs                      # TaskState, QoS Policies, Metadata
│           └── core.rs                      # Scheduler state machine (Rust)
└── pygpukit-python/
    ├── Cargo.toml
    ├── pyproject.toml                       # Maturin configuration
    └── src/
        ├── lib.rs                           # Exposes _pygpukit_rust to Python
        ├── memory.rs                        # PyO3 bindings for MemoryPool
        └── scheduler.rs                     # PyO3 bindings for Scheduler
```

### Architecture Rules (Non-Negotiable)

1. **pygpukit-core is the authoritative runtime.**
   - MemoryPool, Scheduler, Task, LRU, SizeClass MUST be implemented here.
   - Python MUST NOT reimplement these concepts.

2. **All GPU memory management MUST live in:**
   `rust/pygpukit-core/src/memory/`

3. **All scheduling logic MUST live in:**
   `rust/pygpukit-core/src/scheduler/`

4. **Python bindings MUST be thin wrappers only.**
   - No logic duplication
   - No scheduler in Python
   - No memory pool in Python

5. **When adding new features, always add them to Rust first.**
   Then expose via PyO3.

6. **Never remove:**
   - `block.rs`
   - `pool.rs`
   - `size_class.rs`

---

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

| GPU | FP32 naive-opt | TF32 TensorCore | Notes |
|-----|---------------|-----------------|-------|
| RTX 3090 Ti | 18 TFLOPS | 27+ TFLOPS | Achieved with cp.async pipeline |
| A100 | 5.5+ TFLOPS | 156 TFLOPS | tensor cores |

**Achieved Results (v0.2.3)**:
- TF32 on RTX 3090 Ti: **27.38 TFLOPS** (8192×8192×8192)
- Correctness: ~3-5% relative error (expected for TF32 precision)

If performance regresses from naive baseline, re-profile.

### 7. CMake Compilation Flags

```cmake
-arch=sm_80
--expt-relaxed-constexpr
--use_fast_math
```

For portability: allow runtime switch to sm_89, sm_90.

### 8. PTX mma.sync Fragment Mapping (VERIFIED)

**CRITICAL**: PTX inline assembly `mma.sync` has DIFFERENT fragment layouts than WMMA API.
The following mappings were verified empirically using `dump_c_fragment.cu`.

#### PTX `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`

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

#### WMMA API vs PTX Inline ASM

| Aspect | WMMA API | PTX mma.sync |
|--------|----------|--------------|
| Fragment types | `wmma::fragment<>` | Raw registers |
| Layout | Opaque (compiler-managed) | Must match PTX spec exactly |
| Flexibility | Limited shapes | Full control |
| Performance | Good | Potentially better |

**Recommendation**: Use PTX for maximum performance, but VERIFY fragment mappings with test code.

### 9. cp.async Double-Buffering Pipeline (CRITICAL)

**Common Bug**: Prefetching into the wrong stage.

#### WRONG (causes correctness bug):
```cpp
// Prefetch kt+2 into stage (kt+2)&1 — WRONG!
// On kt=0, this prefetches into stage 0 while READING from stage 0
for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr = kt & 1;
    if (kt + 2 < num_k_tiles) {
        load_async((kt+2) & 1, kt + 2);  // BUG: overwrites current!
    }
    process(curr);
}
```

#### CORRECT (simple double-buffering):
```cpp
// Prefetch kt+1 into the OTHER stage
load_async(0, 0);
cp_async_wait_0();

for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr = kt & 1;
    int next = curr ^ 1;  // OTHER stage

    if (kt + 1 < num_k_tiles) {
        load_async(next, kt + 1);  // Prefetch into OTHER buffer
    }
    process(curr);  // Read from current buffer
    cp_async_wait_0();
}
```

**Key Insight**: Always prefetch into the stage you're NOT currently reading from.

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

### Rust Components (MANDATORY - DO NOT REPLACE WITH PYTHON)
1. ✅ Implement Rust memory pool with LRU eviction - DONE (27 tests pass)
2. ✅ Implement Rust GPU scheduler state machine - DONE (with memory reservation, dependencies)
3. Add Rust-side async memory transfer engine
4. Add Rust-side kernel dispatch controller

### CUDA/C++ Components
5. ✅ Add L2-optimized naive matmul kernel (target: 2.1-2.3 TFLOPS) - DONE: 2.2 TFLOPS
6. ✅ Add SM >= 80 runtime check (reject older GPUs)
7. Add Tensor Core MMA kernel for FP16/TF32

### Python Components (Orchestration Only)
8. Python API wrappers for Rust scheduler (thin wrappers only)
9. Python API wrappers for Rust memory pool (thin wrappers only)

---

## Kernel Development Workflow (MANDATORY)

カーネル開発時は以下のワークフローを**必ず**守ること：

### 1. 開発サイクル

```
Edit → Build → Validate → Benchmark → Commit
```

**どんな結果でもValidationとBenchmarkが完了したら必ずコミットする。**

### 2. コミットルール

- Validation/Benchmarkが終わったら**結果に関わらず**コミット
- コミットメッセージにベンチマーク結果を必ず記載

### 3. コミットメッセージ形式

```
wip(tf32): <変更内容の要約>

Benchmark results (RTX 3090 Ti):
- 2048x2048: XX.XX TFLOPS
- 4096x4096: XX.XX TFLOPS
- 8192x8192: XX.XX TFLOPS

Correctness: <PASS/FAIL>
```

### 4. 理由

- 高速だったバージョンに戻せなくなることを防ぐ
- パフォーマンスの変化を追跡可能にする
- 試行錯誤の履歴を保存する

---

## Commit Enforcement Rules (ABSOLUTE)

YOU MUST perform a git commit immediately under ANY of the following conditions:

### 1. Benchmark Improvement

If benchmark results improve in ANY matrix size:
- 2048, 4096, or 8192 shows higher TFLOPS than all previous runs
- Improvement = ANY positive increase (even +0.01 TFLOPS)

### 2. Correctness Achievement

If correctness becomes PASS for all tested sizes:
- relative error < 1e-3 for all matrices

### 3. After EVERY Benchmark Execution

- EVEN IF results are worse
- EVEN IF no improvement is observed
- You MUST create a commit with message: `bench: results logged (no improvement)`

### 4. Commit Before Proceeding

- You MUST NOT proceed to next kernel edit UNTIL the commit is complete

### 5. Never Overwrite Without Commit

- You MUST NEVER overwrite a working kernel without committing it first

### 6. Revert on Regression

If performance or correctness DEGRADES:
- You MUST revert to the previous commit BEFORE continuing

**These rules are absolute. No exceptions.**

了解。**そのまま `CLAUDE.md` に貼れる形**で書くね。
トーンは「ClaudeCodeが迷わない」「実装判断でブレない」ことを最優先にしてる。

---

## Non-goals / Design Principles

### Design Principles

#### 1. PyGPUkit is a GPU Systems Toolkit, not a ML Framework

PyGPUkit is **not** a replacement for PyTorch, JAX, or TensorFlow.
Its purpose is to provide **low-level, explicit, and controllable GPU execution primitives** on top of which higher-level systems may be built.

* Focus: memory, kernels, scheduling, bandwidth, latency
* Not focus: autograd graphs, optimizers, training loops

#### 2. Performance Is a Prerequisite, Not the Goal

High performance is assumed.
Optimization exists to **enable scheduling, concurrency, and predictability**, not as an end in itself.

* Slower-than-cuBLAS requires justification
* Faster-than-cuBLAS is welcome, but not mandatory
* Performance regressions are unacceptable without explicit trade-offs

#### 3. NumPy-like Semantics Over Framework-specific APIs

User-facing APIs should resemble **NumPy-style array operations**, not framework-specific abstractions.

* `C = A @ B` is preferred over opaque operator graphs
* Explicit is better than implicit
* Users should understand when and how GPU work is executed

#### 4. GPU Scheduling Is a First-Class Concept

PyGPUkit treats the GPU as a **shared, schedulable resource**, similar to Kubernetes concepts.

* Admission control, QoS, memory reservation, kernel pacing
* Scheduling decisions are explicit and inspectable
* Kernels are workloads, not side effects

#### 5. SafeTensors Are Immutable Resources

SafeTensors are treated as **immutable, read-only GPU resources**.

* No in-place mutation
* No hidden ownership or lifecycle coupling
* Comparable to ConfigMaps or mounted volumes, not model objects

#### 6. Using cuBLAS / CUTLASS Is Not a Failure

Leveraging vendor or OSS-optimized kernels is acceptable and encouraged.

* Value lies in orchestration, scheduling, and integration
* Reusing proven kernels is preferable to reinventing them
* Custom kernels exist where scheduling or constraints require them

#### 7. Determinism and Correctness Are Explicitly Defined

Numerical behavior must be **documented and intentional**.

* TF32 precision loss is acceptable when explicitly enabled
* FP32 correctness must remain available
* Non-determinism must be explainable and bounded

---

### Non-goals

#### 1. Full Training Framework

PyGPUkit does **not** aim to provide:

* Optimizers
* Training loops
* Dataset pipelines
* Autograd engines

These belong in higher-level frameworks.

#### 2. Abstracting Away GPU Reality

PyGPUkit will **not hide**:

* Memory transfers
* Synchronization points
* Kernel launch costs
* Precision trade-offs

Users are expected to understand GPU fundamentals.

#### 3. Supporting Legacy or Low-End GPUs

The project intentionally targets **modern GPUs (Ampere / Ada and newer)**.

* Older architectures (e.g., Turing and below) are out of scope
* Features may assume Tensor Cores, large shared memory, and modern instructions

#### 4. API Compatibility With PyTorch

API compatibility with PyTorch is **not a goal**.

* Familiarity is secondary to clarity
* PyGPUkit APIs may diverge intentionally for correctness or performance reasons

#### 5. “Magic” Performance

No undocumented heuristics or hidden behavior.

* All optimizations must be explainable
* Performance comes from design, not surprise


---

## TF32 TensorCore GEMM Development Notes

### WMMA vs PTX mma.sync

**重要な発見 (2024-12):**

1. **WMMA API** (`nvcuda::wmma`) は動作確認済み
   - `row_major` A + `row_major` B の組み合わせで正常動作
   - `row_major` A + `col_major` B は**メモリレイアウトの解釈が異なり失敗**

2. **PTX mma.sync** の正しいマッピングはまだ特定中
   - m16n8k8 のフラグメントレイアウトが複雑
   - WMMA の `debug_dump_fragments` で実際のマッピングを確認可能

### 動作確認済みカーネル

```cpp
// WMMA row_major × row_major (PASS)
fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
fragment<accumulator, 16, 16, 8, float> c_frag;

load_matrix_sync(a_frag, A + k, K);      // ldA = K
load_matrix_sync(b_frag, B + k * N, N);  // ldB = N (row-major storage)
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C, c_frag, N, mem_row_major);
```

### テスト結果 (WMMA row_row)

| M | N | K | max_err | rel_err | Status |
|---|---|---|---------|---------|--------|
| 16 | 16 | 8 | 0.0055 | 0.05% | PASS |
| 16 | 16 | 16 | 0.0089 | 0.07% | PASS |
| 16 | 16 | 32 | 0.0094 | 0.06% | PASS |
| 16 | 16 | 64 | 0.0205 | 0.10% | PASS |
| 16 | 16 | 128 | 0.0247 | 0.08% | PASS |
| 16 | 16 | 256 | 0.0373 | 0.08% | PASS |

### WMMA 16×16×8 フラグメントマッピング (実測値)

`dump_fragments.cu` による実測結果:

#### A fragment (16×8 matrix_a, row_major)
```cpp
// Thread t (0-31):
int a_row = t / 4;      // 0-7
int a_col = t % 4;      // 0-3

a[0] = A[a_row][a_col]           // rows 0-7,  cols 0-3
a[1] = A[a_row + 8][a_col]       // rows 8-15, cols 0-3
a[2] = A[a_row][a_col + 4]       // rows 0-7,  cols 4-7
a[3] = A[a_row + 8][a_col + 4]   // rows 8-15, cols 4-7
```

#### B fragment (8×16 matrix_b, row_major)
```cpp
// Thread t (0-31):
int b_row = t % 4;      // 0-3
int b_col = t / 4;      // 0-7

b[0] = B[b_row][b_col]           // rows 0-3, cols 0-7
b[1] = B[b_row + 4][b_col]       // rows 4-7, cols 0-7
b[2] = B[b_row][b_col + 8]       // rows 0-3, cols 8-15
b[3] = B[b_row + 4][b_col + 8]   // rows 4-7, cols 8-15
```

#### サイズの違い
| API | A | B | C |
|-----|---|---|---|
| WMMA 16×16×8 | 16×8 | 8×16 | 16×16 |
| PTX m16n8k8 | 16×8 | 8×8 | 16×8 |

PTX m16n8k8 は WMMA の **B/C の左半分** (cols 0-7) のみを使用。

#### C fragment マッピング (実測: dump_c_fragment.cu)
```cpp
int c_row = t / 4;           // 0-7
int c_col = (t % 4) * 2;     // 0, 2, 4, 6
c[0] = C[c_row][c_col]        // rows 0-7, cols even
c[1] = C[c_row][c_col + 1]    // rows 0-7, cols odd
c[2] = C[c_row + 8][c_col]    // rows 8-15, cols even
c[3] = C[c_row + 8][c_col + 1]// rows 8-15, cols odd
```

### 正確性テスト (C fragment 修正後) - 全 PASS
- 256³〜4096³: rel_err ≈ 8e-4 (0.08%)
- 決定性100回: PASS

### 次のステップ

1. ✅ WMMAの正しいフラグメントマッピングを `dump_fragments` で確認
2. ✅ C fragment マッピングを `dump_c_fragment` で確認・修正
3. ✅ 全正確性テスト PASS
4. パフォーマンス最適化 (現状 11-18 TFLOPS → 目標 22-35 TFLOPS)

### ファイル構成

- `native/ops/matmul_f32_tf32.cuh` - TF32カーネル
- `native/ops/basic.cu` - ディスパッチロジック (line 848-854)
- `dump_fragments.cu` - フラグメントマッピング確認用
- 環境変数 `PYGPUKIT_ALLOW_TF32=1` で有効化
