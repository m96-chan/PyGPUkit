# Style and Conventions

## General Rules
- NO emoji or non-ASCII in source code (cp932/Shift-JIS compatibility)
- Python is ONLY high-level orchestration; core logic in Rust
- All GPU kernels compiled at runtime with NVRTC

## Python
- Ruff for linting/formatting
- Mypy for type checking
- NumPy-like API for user-facing code

## C++/CUDA
- CUDA Driver/Runtime API (NOT cuda-python)
- Prefer L2-friendly patterns over shared-memory tiling
- Target SM 80+ only

## Rust
- pygpukit-core: MemoryPool, Scheduler, LRU
- pygpukit-python: PyO3 bindings (thin wrappers only)

## Commit Messages
```
type(scope): summary

Benchmark results (if applicable):
- 2048x2048: XX.XX TFLOPS
```
Types: feat, fix, perf, refactor, docs, test, chore
