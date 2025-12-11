# PyGPUkit - Claude Code Guidelines

## Tech Stack (IMPORTANT)

PyGPUkit is a **Rust + C++ + Python** hybrid project.

```
PyGPUkit/
│
├── python/        → Python API (NumPy-compatible, pybind11 bindings)
│
├── core/
│   ├── C++ (CUDA Runtime API)
│   └── Rust backend (opt-in)
│
├── memory/
│   ├── Rust (LRU, pool allocator)
│   └── Python shim
│
├── scheduler/
│   ├── Rust (state management)
│   └── C++ (kernel launch wrappers)
│
└── jit/
    ├── C++ (NVRTC)
    └── Python wrappers
```

### Language Responsibilities

| Component | Language | Reason |
|-----------|----------|--------|
| Python API | Python | NumPy-compatible user interface |
| CUDA Runtime/Driver | C++ | Direct hardware access |
| NVRTC JIT | C++ | Kernel compilation |
| Memory Pool/LRU | Rust | Safe, fast memory management |
| Scheduler State | Rust | Thread-safe state machine |
| Kernel Launch | C++ | CUDA kernel dispatch |
| Bindings | pybind11 + PyO3 | C++/Rust to Python |

## Critical Rules

### DO NOT

1. **Do NOT** use or mention `cuda-python` - it is NOT a dependency
2. **Do NOT** call CUDA APIs from Python directly
3. **Do NOT** implement memory management in pure Python (use Rust)
4. **Do NOT** implement scheduler state in pure Python (use Rust)
5. **Do NOT** suggest pure-Python solutions for performance-critical code

### DO

1. **DO** use C++ for all CUDA Runtime/Driver API calls
2. **DO** use Rust for memory pool, LRU eviction, scheduler state
3. **DO** use pybind11 for C++ to Python bindings
4. **DO** use PyO3/maturin for Rust to Python bindings
5. **DO** keep Python layer thin - only API surface and NumPy interop

## GPU Detection

GPU availability is detected via native C++ calls ONLY:
- `cudaGetDeviceCount`
- `cudaDriverGetVersion`
- `cudaRuntimeGetVersion`
- `nvrtcVersion`

CPU fallback occurs ONLY when native backend fails to load or no GPU detected.

## Build System

- **C++**: CMake with CUDA toolkit
- **Rust**: Cargo with `cxx` or PyO3
- **Python**: setuptools/maturin hybrid build
- **Bindings**: pybind11 (C++) + PyO3 (Rust)

## Current State (v0.1)

- Python prototype with CPU simulation backend (for API design/testing)
- Native C++/Rust backend NOT yet implemented
- 73 tests pass using CPU simulation
- Real GPU execution requires native backend implementation

## Next Steps for Native Backend

1. Implement `core/` in C++ with CUDA Runtime API
2. Implement `memory/` in Rust with LRU allocator
3. Implement `jit/` in C++ with NVRTC
4. Create pybind11 bindings for C++ components
5. Create PyO3 bindings for Rust components
6. Hybrid build with maturin + CMake
