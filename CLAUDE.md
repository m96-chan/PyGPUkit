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
│   ├── core/               # GPUArray, backend abstraction
│   ├── ops/                # GPU operations (matmul, nn, audio, etc.)
│   ├── llm/                # LLM inference (Qwen, LLaMA)
│   │   ├── models/         # Model implementations
│   │   └── sampling/       # Token sampling strategies
│   └── asr/                # Speech recognition (Whisper)
│       ├── preprocessing.py    # Audio preprocessing (mel, normalize)
│       └── whisper/            # Whisper model implementation
│           ├── config.py       # WhisperConfig
│           ├── loader.py       # SafeTensors loader
│           ├── encoder.py      # Whisper encoder
│           ├── decoder.py      # Whisper decoder
│           └── model.py        # WhisperModel high-level API
├── native/
│   ├── core/               # C++ (CUDA Runtime/Driver API)
│   ├── jit/                # C++ (NVRTC)
│   ├── ops/                # C++ (CUDA kernels)
│   │   └── matmul/         # MatMul kernels (see below)
│   └── bindings/           # pybind11
├── rust/
│   ├── pygpukit-core/      # Pure Rust GPU runtime
│   │   └── src/
│   │       ├── memory/     # MemoryPool, LRU, size-class allocator
│   │       ├── scheduler/  # Task state machine, QoS policies
│   │       └── device.rs   # DeviceCapabilities, KernelType
│   └── pygpukit-python/    # PyO3 bindings
├── examples/
├── benchmarks/             # Performance benchmarks
└── tests/
```

### MatMul Kernel Structure

```
native/ops/matmul/
├── common/                          # Shared utilities
│   └── aligned_copy_sm120.cuh
├── gemm/                            # GEMM kernels (M > 1)
│   └── {input_dtype}/{output_dtype}/{arch}/{compute}_{suffix}.{cu,cuh}
├── gemv/                            # GEMV kernels (M = 1)
│   └── {input_dtype}/{output_dtype}/{arch}/{compute}_{suffix}.{cu,cuh}
├── cublaslt.cuh                     # cuBLASLt wrapper
├── matmul.cu                        # Main dispatcher
└── matmul_cutlass.cu                # CUTLASS dispatcher
```

**Path Convention:** `{gemm|gemv}/{input_dtype}/{output_dtype}/{arch}/{compute}_{suffix}.cu`

| Component | Values | Examples |
|-----------|--------|----------|
| `input_dtype` | `f32`, `bf16`, `fp8`, `nvf4` | Input tensor dtype |
| `output_dtype` | `f32`, `bf16`, `fp8` | Output tensor dtype |
| `arch` | `generic`, `sm80`, `sm90`, `sm100`, `sm120` | Target architecture |
| `compute` | `naive`, `wmma`, `mma`, `cutlass` | Compute method |
| `suffix` | `blockwise`, `kernels`, etc. | Variant identifier |

**Examples:**
```
gemm/bf16/bf16/sm120/bf16_cutlass.cuh    # BF16->BF16 GEMM, SM120, CUTLASS
gemm/fp8/f32/sm90/fp8_cutlass.cu         # FP8->F32 GEMM, SM90, CUTLASS
gemm/nvf4/bf16/sm120/nvf4_cutlass.cu     # NVF4->BF16 GEMM, SM120, CUTLASS
gemv/bf16/bf16/sm120/nvf4.cu             # NVF4->BF16 GEMV, SM120
gemm/f32/f32/generic/tf32_mma.cuh        # TF32 GEMM, generic (SM80+)
```

### Module Separation Policy

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `llm/` | Text generation | Text tokens | Text tokens |
| `asr/` | Speech recognition | Audio waveform | Text |
| `ops/` | Low-level GPU ops | GPUArray | GPUArray |

**Rationale**: Modules are separated by **modality** (audio vs text), not by architecture (transformer). This follows industry conventions (HuggingFace, OpenAI API) and enables clean future expansion (TTS, vision, etc.).

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

**v0.1.x:**
- Requires CUDA Toolkit installation
- Loads DLLs from `CUDA_PATH/bin`

**v0.2.x (Current):**
- cuBLASLt loaded dynamically at runtime
- Searches: `CUDA_PATH/bin/x64` → `CUDA_PATH/bin` → system PATH
- Descriptor caching for matmul performance
- Falls back gracefully if cuBLASLt unavailable

```cpp
// Dynamic loading sequence
cublasLt64_13.dll  // CUDA 13.x
cublasLt64_12.dll  // CUDA 12.x
cublasLt64_11.dll  // CUDA 11.x
```

**Future (Driver-Only Mode):**
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
9. Use emoji or non-ASCII characters in source code or comments (cp932/Shift-JIS compatibility)

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

- **Supported:** Ampere (SM 80-86), Ada (SM 89), Hopper (SM 90), Blackwell (SM 100, 120a)
- **Unsupported:** Architectures below SM80
- **Build default:** SM 80, 86, 89, 90, 100, 120a (CUDA 13.1+)

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

#### MatMul Performance

| GPU | FP32 | TF32 TensorCore |
|-----|------|-----------------|
| RTX 3090 Ti | 18 TFLOPS | 27+ TFLOPS |
| A100 | 5.5+ TFLOPS | 156 TFLOPS |

**Achieved (v0.2.3):** TF32 on RTX 3090 Ti: **27.38 TFLOPS** (8192×8192×8192)

#### LLM Inference (Qwen3-8B, RTX 3090 Ti, FP16)

**Single Token Decode (M=1):**

| Mode | Tokens/sec | ms/token |
|------|-----------|----------|
| Non-graph decode | 1.84 | 544 |
| CUDA Graph decode | 2.19 | 457 |
| Speedup | **1.19x** | - |

**Batch Decode (v0.2.11):**

| Batch Size | Per Token (us) | Throughput | Speedup |
|------------|---------------|------------|---------|
| 1 | 381,303 | 2.6 tok/s | 1.00x |
| 2 | 205,030 | 4.9 tok/s | 1.86x |
| 4 | 108,521 | 9.2 tok/s | 3.51x |
| 8 | 55,845 | 17.9 tok/s | **6.83x** |

**E2E Batch Verification (32 tokens):**

| Method | Time (ms) | tok/s | Speedup |
|--------|----------|-------|---------|
| Sequential | 14,541 | 2.13 | 1.00x |
| Batch Verify (batch=4) | 4,082 | 7.59 | 3.56x |
| Batch Verify (batch=8) | 2,147 | 14.44 | **6.77x** |

**Decode Strategy Benchmark (v0.2.11):**

Model: Qwen2.5-7B-Instruct (bfloat16), RTX 3090 Ti

| Strategy | tok/s | Speedup | Notes |
|----------|-------|---------|-------|
| DecodeM1 (baseline) | 3.2 | 1.00x | Single token per step |
| DecodeBatch (batch=8) | 19.6 | **6.06x** | TensorCore efficient |
| DecodeSpeculative | 1.4 | 0.42x | Self-speculative (early layers) |
| DecodeJacobi | 1.7 | 0.53x | Parallel iterative |

**Note:** Large models (8B+) are GPU compute-bound; CUDA Graph benefit is modest. Batch decode shows near-linear scaling with TensorCore utilization.

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

## TF32 Optimization Research (Issue #53)

### Current Performance Status

| Metric | Value |
|--------|-------|
| Current | **27.38 TFLOPS** (8192×8192) |
| RTX 3090 Ti TF32 Theoretical | ~40 TFLOPS |
| cuBLAS Reference | ~59 TFLOPS |
| Gap to cuBLAS | **47%** |

### Current Implementation Parameters

```
Block Tile: BM=128, BN=128, BK=16
Warp Tile: WARP_TILES_M=2, WARP_TILES_N=8 (32×64 per warp)
MMA Instruction: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
Pipeline: 2-stage double buffering
Thread Block: 256 threads (8 warps)
Shared Memory: ~37KB/block → occupancy ~16.7%
```

### CUTLASS Optimization Techniques

#### 1. Swizzled Shared Memory Layout (High Priority)

Current implementation uses simple padding (`A_PAD=4, B_PAD=4`) but bank conflicts are not fully eliminated.

**CUTLASS Approach:**
```cpp
// XOR-based swizzle pattern
int store_column = (lane_id % 8) ^ (lane_id / 8);
```

- Store and Load phases use transposed index relationship
- XOR operation applied per 8×8 block unit
- Combined with `ldmatrix` for fully bank conflict-free access

**Key Insight:**
> "the indexing in the 'Loading from Shared Memory to Registers' slide is transposed from the indexing in 'Load from Global/Store to Shared' slide."

#### 2. ldmatrix Instruction (High Priority)

Current implementation manually loads from shared memory to registers:
```cpp
// Current implementation
float a0 = smA[curr][tile_m + a_row_base][kk + a_col_base];
```

**CUTLASS Approach:**
- Uses `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
- Single instruction loads four 8×8 matrices (entire warp)

**TF32 Limitation:**
> "ldmatrix cannot transpose 32-bit data. CUTLASS uses 32-bit shared memory load to load data from shared memory to the registers to do the transpose right before calling tf32 tensor core."

#### 3. Multi-stage Pipeline (Medium-High Priority)

Current: 2-stage → CUTLASS default: **4-stage**

**Past Failed Attempt:**
> "3-stage pipeline: -28% (50% more smem reduced occupancy)"

**Considerations:**
- Trade-off between shared memory usage and occupancy
- RTX 3090 Ti: 100KB/SM available
- Current 37KB → 4-stage at ~74KB should fit

### Recommended Implementation Order

| Priority | Optimization | Expected Gain | Difficulty |
|----------|-------------|---------------|------------|
| 1 | Swizzled shared memory layout | +10-15% | Medium |
| 2 | 4-stage pipeline (proper smem sizing) | +5-10% | Medium |
| 3 | Warp tile tuning (BM/BN/BK re-tuning) | +5-10% | Low |
| 4 | Epilogue fusion (bias + activation) | Memory reduction | Medium |

### Path to 35 TFLOPS

- Current: 27.38 TFLOPS (68% of target)
- Swizzle + 4-stage: 32-34 TFLOPS expected
- Fine-tuning: 35+ TFLOPS

### Reference Materials

- [CUTLASS TF32 GEMM Example](https://github.com/NVIDIA/cutlass/blob/main/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu)
- [CUTLASS Efficient GEMM Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS Swizzled Layouts Discussion](https://github.com/NVIDIA/cutlass/discussions/1130)
- [Understanding CUTLASS Permuted Shared Memory](https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697)
- [Dissecting Tensor Cores (Academic Paper)](https://arxiv.org/pdf/2206.02874)

---

## Development Workflow

### Kernel Development Cycle

```
Edit → Build → Validate → Benchmark → Commit
```

**Always commit after validation and benchmark, regardless of results.**

### Build Instructions (IMPORTANT)

**Git Bashからビルド（推奨）：**

```bash
cd /d/Projects/m96-chan/PyGPUkit
./build.sh 86       # SM 86のみ (RTX 3090 Ti)
./build.sh 120a     # SM 120aのみ (RTX 5090)
./build.sh          # デフォルト: SM 120a
```

**注意事項：**
- RTX 5090 (SM 120a) はCUDA 13.1以降が必要
- サポートSM: 80, 86, 89, 90, 100, 120a

### Pre-Commit Checks (MANDATORY)

**Before EVERY commit, run these checks:**

```bash
# 1. Ruff lint check (auto-fix and format)
git ls-files "*.py" | xargs python -m ruff check --fix
git ls-files "*.py" | xargs python -m ruff format

# 2. Mypy type check
python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc
```

**NEVER commit without passing ALL checks.** CI will reject PRs with lint/type errors.

### PR Checklist (MANDATORY before `gh pr create`)

Before creating a PR, verify ALL of the following:

```bash
# 1. Lint passes
git ls-files "*.py" | xargs python -m ruff check

# 2. Mypy passes
python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc

# 3. Tests pass
python -m pytest tests/ -v

# 4. Benchmark runs (optional but recommended)
python benchmark.py --quick
```

**DO NOT create PR until all checks pass locally.**

### Commit Rules

1. **Run lint check before commit** (see above)
2. Commit after every validation/benchmark completion, regardless of outcome
3. Include benchmark results in commit message
4. Never proceed to next kernel edit until commit is complete
5. Never overwrite a working kernel without committing first

### Commit Message Format

```
wip(tf32): <summary of changes>

Benchmark results (RTX 5090):
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

### Benchmarking

**Always use `benchmark.py` for performance measurement.**

```bash
# Full benchmark (all dtypes, all sizes)
python benchmark.py

# Quick mode (fewer warmup/iterations)
python benchmark.py --quick

# Specific sizes
python benchmark.py --sizes 4096 8192

# TF32 kernel version selection
python benchmark.py --tf32-version v1   # WMMA API
python benchmark.py --tf32-version v2   # PTX mma.sync (default)
```

**Output includes:**
- Kernel-only timing (no D2H copy overhead)
- Correctness verification (relative error)
- README.md-ready table format

**Environment Variables:**
- `PYGPUKIT_ALLOW_TF32=1` - Enable TF32 TensorCore
- `PYGPUKIT_TF32_V2=1` - Use PTX mma.sync kernel (default when TF32 enabled)

---

## CUDA Graph Guidelines

M=1 decode separates CUDA Graph and Non-Graph versions.

### Graph Version Requirements

Use CUDA Graph ONLY when ALL conditions are met:

1. **Fixed shapes/dtypes/RoPE tables** - No dynamic changes during replay
2. **Identical kernel path** - warmup / capture / replay use the same code path
3. **No KV cache pollution** - Graph must not write to real KV cache during warmup/capture
4. **H2D copies on capture stream** - All host-to-device copies must be on the stream being captured

### Fallback Rules

If any condition is NOT met, fallback to Non-Graph version.

### Prohibited in Graph

- Conditional branches based on runtime values
- `copy_to` operations (use direct buffer writes instead)
- Any operation that reads from or writes to KV cache
- SDPA (Scaled Dot-Product Attention) - always run outside graph

### Implementation Pattern

```python
# Graph captures ONLY stateless operations:
# - Embedding lookup (via GPU pointer)
# - Linear projections (QKV, O, MLP)
# - RMSNorm
# - RoPE (via pre-computed GPU tables)

# These run OUTSIDE graph:
# - KV cache update
# - SDPA attention
# - Any operation that depends on context_len at runtime
```

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

## LLM Inference Architecture

### Overview

PyGPUkit includes a minimal LLM inference engine for SafeTensors models (Qwen, LLaMA, etc.).

```
SafeTensors → Model Loading → Prefill → Decode Loop → Token Output
                                ↓
                         CUDA Graph (optional)
```

### Decode Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Standard** | `model.forward()` with allocation | Simple usage |
| **Zero-Alloc** | `_decode_step_zero_alloc()` | Low-latency |
| **CUDA Graph** | `_decode_step_graph_replay()` | Reduced kernel launch overhead |
| **Jacobi** | Parallel iterative decode | Speculative execution |

### CUDA Graph Implementation

#### Capture Stream

All kernels must use `internal::get_capture_stream()` for CUDA Graph compatibility:

```cpp
cudaStream_t stream = internal::get_capture_stream();
my_kernel<<<grid, block, 0, stream>>>(...);
```

**Critical**: Kernels launched without stream parameter will NOT be captured in the graph.

#### Pointer-Based Kernels

For dynamic values during graph replay, use `_ptr` kernel variants:

```cpp
// Static value (captured at graph creation)
sdpa_causal_fixed_cache(..., context_len, ...);

// Pointer-based (read from GPU buffer at runtime)
sdpa_causal_fixed_cache_ptr(..., context_len_buf, max_kv_len, ...);
```

#### DecodeBuffers

Pre-allocated buffers for zero-allocation decode:

```python
@dataclass
class DecodeBuffers:
    hidden: GPUArray       # [1, hidden_size]
    q: GPUArray            # [1, num_heads, head_dim]
    k: GPUArray            # [1, num_kv_heads, head_dim]
    v: GPUArray            # [1, num_kv_heads, head_dim]
    attn_out: GPUArray     # [num_heads, 1, head_dim]
    # ... (layer-shared, reused across all layers)
```

#### Graph Capture Flow

```python
model.init_decode_graph(max_seq_len=512)  # Capture graph

# Replay loop
for i in range(num_tokens):
    logits = model._decode_step_graph_replay(token_id, position, context_len)
    next_token = sample(logits)
```

#### Performance Notes

| Scenario | CUDA Graph Speedup |
|----------|-------------------|
| Full decode loop (with D2H) | ~1.2x |
| Kernel-only (large model) | ~1.0x (GPU-bound) |
| Small model / many kernels | Higher benefit |

**Limitation**: Current implementation has 2 device syncs per replay (H2D visibility + completion wait), which reduces benefit for large models.

### KV Cache

Fixed-length KV cache with GQA support:

```python
# Initialize
for block in model.blocks:
    block.attn.init_fixed_cache(max_seq_len, dtype="float16")

# Prefill
hidden, past_kv = model(input_ids, use_cache=True)
for i, block in enumerate(model.blocks):
    kv_cache_prefill_gqa(past_kv[i][0], block.attn._k_cache, num_heads, start_pos=0)
    kv_cache_prefill_gqa(past_kv[i][1], block.attn._v_cache, num_heads, start_pos=0)

# Backup/Restore for benchmarking
kv_backup = model.snapshot_kv_cache()
model.restore_kv_cache(kv_backup)
```

### Jacobi Decoding

Parallel iterative generation for speculative execution:

```python
# Initialize Jacobi buffers
model.init_jacobi_decode(lookahead_k=4, max_seq_len=512)

# Parallel decode
accepted_tokens = model.jacobi_decode_step(draft_tokens, position)
```

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

### v0.2.10 (Released)
- ✅ CUDA Graph for single-token decode (M=1)
- ✅ cuBLASLt dynamic loading with descriptor caching
- ✅ Top-k sampling in graph capture
- ✅ Zero-allocation decode path (DecodeBuffers)

### v0.2.11 (Current)
- ✅ CUDA Graph stream fix (RoPE/SDPA now properly captured)
- ✅ Batch decode support (seq_len > 1)
- ✅ Jacobi decoding for parallel iterative generation
- ✅ Self-Speculative decoding framework
- ✅ GPU-side Lookahead KV Cache
- ✅ CUDA Events API

### Remaining Work
- Rust-side async memory transfer engine
- Rust-side kernel dispatch controller
- Python API wrappers for Rust scheduler/memory pool (thin wrappers only)

---

## Development Environment

### Build Instructions

**Git Bashからビルド（推奨）：**

```bash
cd /d/Projects/m96-chan/PyGPUkit
./build.sh 86       # SM 86のみ (RTX 3090 Ti)
./build.sh 120a     # SM 120aのみ (RTX 5090)
./build.sh          # デフォルト: SM 120a
```

**サポートSM:** 80, 86, 89, 90, 100, 120a

### Local Development Hardware

| Machine | GPU | SM | CUDA Toolkit | Notes |
|---------|-----|-----|--------------|-------|
| Primary | RTX 5090 | 120a | 13.1 | Blackwell GeForce, FP8 testing |
| Secondary | RTX 3090 Ti | 86 | 12.x | Ampere, TF32 benchmarks |

### Tokenizer

**PyGPUkit内蔵のTokenizerは使用しない。HuggingFace `tokenizers`ライブラリを使用する。**

```python
# 推奨: HuggingFace tokenizers
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("/path/to/tokenizer.json")

# 非推奨: 内蔵Tokenizer (互換性問題あり)
# from pygpukit.llm import Tokenizer
```

### LLM Models Directory

**Primary model storage:** `F:/LLM/`

All LLM models for inference testing are stored in `F:/LLM/`. Use this path when loading models.

```
F:/LLM/
├── Qwen2.5-7B-Instruct/           # Main test model
├── Qwen3-8B/                       # Qwen3 variant
├── TinyLlama-1.1B-Chat-v1.0/      # Small model for quick tests
└── ...
```

**Usage example:**
```python
from pygpukit.llm import QwenModel

model = QwenModel.from_safetensors("F:/LLM/Qwen2.5-7B-Instruct")
```

**Note:** HuggingFace cache (`~/.cache/huggingface/`) may also contain models but `F:/LLM/` is the canonical location.

---

## Claude Code Configuration

### Skills (.claude/skills/)

Development workflow automation:

| Skill | Description |
|-------|-------------|
| `build` | Build native module with SM selection |
| `benchmark` | Run matmul performance benchmarks |
| `lint` | Ruff lint + format |
| `typecheck` | Mypy type check |
| `test` | Run pytest |
| `precommit` | Pre-commit checks (lint + typecheck) |
| `check-all` | Full validation (lint + typecheck + test) |
| `chat-test` | LLM inference testing |
| `kernel-dev` | Kernel development workflow |

### Subagents (.claude/agents/)

Specialized agents for specific tasks:

| Agent | Model | Description |
|-------|-------|-------------|
| `kernel-reviewer` | opus | CUDA kernel code review |
| `perf-analyzer` | opus | Benchmark analysis and optimization |
| `api-designer` | sonnet | Python API design review |
| `commit-helper` | haiku | Commit message and PR generation |
| `doc-generator` | haiku | Documentation updates |

### Usage

Skills and agents are automatically invoked based on task context. Examples:

- "Build for RTX 5090" -> `build` skill
- "Review the kernel changes" -> `kernel-reviewer` agent
- "Analyze benchmark results" -> `perf-analyzer` agent
- "Commit these changes" -> `commit-helper` agent
