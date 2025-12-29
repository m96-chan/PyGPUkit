# PyGPUkit Project Overview

## Purpose
PyGPUkit is a minimal GPU runtime for Python that provides:
- High-performance GPU kernels (matmul, attention, etc.)
- NumPy-like API for GPU arrays
- LLM inference engine (Qwen, LLaMA via SafeTensors)
- Memory management and GPU scheduling

## Tech Stack
- **Python**: High-level API (NumPy-compatible)
- **Rust**: Core scheduling, memory management (pygpukit-core, pygpukit-python via PyO3)
- **C++/CUDA**: GPU kernels, CUDA Driver/Runtime API, NVRTC JIT

## Architecture
```
Python (API) -> Rust (scheduler/memory) -> C++ (CUDA kernels)
```

## Directory Structure
- `src/pygpukit/`: Python API
- `native/`: C++/CUDA code (kernels in `native/ops/matmul/`)
- `rust/`: Rust runtime (memory pool, scheduler)
- `.claude/skills/`: Development workflow automation
- `.claude/logs/build/`: Build logs (auto-saved)

## Target GPUs
- Supported: SM 80+ (Ampere, Ada, Hopper, Blackwell)
- Unsupported: Below SM80

## LLM Models Location
`F:/LLM/` - All LLM models for inference testing
