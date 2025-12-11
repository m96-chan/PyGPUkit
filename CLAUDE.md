# PyGPUkit - Claude Code Guidelines

## Architecture

PyGPUkit does **NOT** depend on cuda-python.

The GPU backend is implemented in **C++** using:
- CUDA Runtime API (`cuda*.h`)
- CUDA Driver API (`cu*.h`)
- NVRTC for JIT compilation

Python bindings are provided via **pybind11**.

## GPU Detection

GPU detection must ONLY use:
- `cudaGetDeviceCount`
- `cudaDriverGetVersion`
- `cudaRuntimeGetVersion`
- `nvrtcVersion`

CPU fallback occurs ONLY if these native calls fail.

## Implementation Rules

1. **Do NOT** generate code that imports or requires `cuda-python`
2. **Do NOT** generate diagnostics mentioning `cuda-python`
3. **Do NOT** suggest installing `cuda-python` as a solution
4. Python layer must NOT call CUDA APIs directly - all GPU operations go through the C++ backend
5. The C++ shared library is loaded via pybind11

## Project Structure

```
PyGPUkit/
├── src/pygpukit/       # Python API (NumPy-compatible)
├── core/               # C++ CUDA Runtime wrapper
├── memory/             # Memory management (Rust optional)
├── scheduler/          # GPU scheduling (Rust state + C++ kernel launch)
├── jit/                # C++ NVRTC wrapper
└── python/             # pybind11 bindings
```

## Current State (v0.1)

- Python prototype with CPU simulation backend
- C++ backend not yet implemented
- All 73 tests pass using CPU simulation
- Real GPU execution requires C++ backend implementation
