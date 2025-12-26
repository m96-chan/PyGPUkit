# Contributing to PyGPUkit

---

## 0. Core Principles

These principles are **non-negotiable**. Every contribution must align with them.

### Mission

PyGPUkit makes GPU programming feel like using a standard Python library: pip-installable, minimal setup, no mandatory CUDA Toolkit.

### Philosophy

1. **Explicit over implicit** - GPU operations are visible, not hidden
2. **Performance is a prerequisite** - Slower than cuBLAS requires justification
3. **NumPy-like semantics** - `C = A @ B`, not opaque operator graphs
4. **GPU as a schedulable resource** - Kubernetes-inspired admission control

### Language Boundaries

```
Python  - High-level orchestration ONLY
Rust    - Memory pool, scheduler, GPU coordination
C++     - CUDA Driver/Runtime API, NVRTC, kernel launch
```

**Python must remain a thin wrapper.** Performance-critical logic belongs in Rust or C++.

---

## 1. What We Accept / What We Reject

### We Accept

| Type | Examples |
|------|----------|
| Performance improvements | Faster kernels, better memory patterns |
| New GPU operations | Ops that fit the GPUArray model |
| Bug fixes | Correctness issues, memory leaks |
| SM architecture support | New GPU generations (with benchmarks) |
| Documentation | Clarifications, examples, typo fixes |

### We Reject

| Type | Reason |
|------|--------|
| Python CUDA wrappers | No cuda-python, numba.cuda, cupy.cuda |
| Training features | Autograd, optimizers, training loops |
| Legacy GPU support | SM < 80 (Turing and below) |
| Magic/implicit behavior | Hidden allocations, undocumented heuristics |
| Over-engineering | Features for hypothetical future needs |

### Gray Areas (Discuss First)

- New module additions (e.g., vision, TTS)
- Alternative backends (ROCm, Metal)
- Breaking API changes

---

## 2. Architectural Invariants

These rules **cannot be violated**. PRs that break them will be rejected.

### Layer Model

```
Python API --> pybind11 --> C++ --> CUDA Driver/Runtime/NVRTC
                |
                +--> PyO3 --> Rust (memory, scheduler)
```

### Required Rust Components

These **MUST NOT** be removed or reimplemented in Python:

1. Memory pool with LRU eviction (`rust/pygpukit-core/src/memory/`)
2. GPU scheduler state machine (`rust/pygpukit-core/src/scheduler/`)
3. Async GPU memory transfer engine
4. Kernel dispatch controller

### Module Boundaries

| Module | Modality | Input | Output |
|--------|----------|-------|--------|
| `ops/` | Tensors | GPUArray | GPUArray |
| `llm/` | Text | Tokens | Tokens |
| `asr/` | Audio | Waveform | Text |

Modules are separated by **modality**, not architecture.

### File Ownership

| Path | Language | Owner |
|------|----------|-------|
| `src/pygpukit/` | Python | API surface only |
| `native/ops/` | C++/CUDA | Kernel implementations |
| `native/core/` | C++ | CUDA utilities |
| `rust/pygpukit-core/` | Rust | Runtime core |

---

## 3. Performance & Safety Rules

### Performance Requirements

| Metric | Requirement |
|--------|-------------|
| Regression | Not allowed without explicit justification |
| New kernels | Must include benchmark results |
| TensorCore | Required for FP16/BF16/TF32 on SM >= 80 |
| Memory | No hidden allocations in hot paths |

### Target Architectures

- **Supported**: SM 80+ (Ampere, Ada, Hopper, Blackwell)
- **Build default**: SM 80, 86, 89, 90, 100, 120a
- **Unsupported**: SM < 80

### Kernel Guidelines

```cpp
// DO: L2-friendly, coalesced, vectorized
float4 data = *reinterpret_cast<float4*>(&input[idx]);

// DON'T: Complex shared-memory tiling for Pascal/Turing
__shared__ float tile[32][32];  // Often slower on Ampere
```

### Safety Rules

- No `cuda-python` or external Python CUDA dependencies
- No secrets in code (API keys, tokens, passwords)
- No force push to main/master
- No skipping pre-commit hooks

---

## 4. How to Propose Changes

### Before You Start

1. **Check existing issues** - Your idea may already be discussed
2. **Read CLAUDE.md** - Understand architecture and constraints
3. **Small changes**: Just open a PR
4. **Large changes**: Open an issue first to discuss approach

### Development Workflow

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/PyGPUkit.git
cd PyGPUkit

# 2. Create feature branch
git checkout -b feature/your-feature

# 3. Build (Git Bash)
./build.sh 86    # or 120a for RTX 5090

# 4. Make changes, then run checks
git ls-files "*.py" | xargs python -m ruff check --fix
git ls-files "*.py" | xargs python -m ruff format
python -m mypy src/ --ignore-missing-imports \
  --disable-error-code=union-attr \
  --disable-error-code=no-redef \
  --disable-error-code=no-any-return \
  --disable-error-code=attr-defined \
  --disable-error-code=assignment \
  --disable-error-code=arg-type \
  --disable-error-code=index \
  --disable-error-code=misc

# 5. Run tests
python -m pytest tests/ -v

# 6. For kernel changes, run benchmarks
python scripts/benchmark.py --quick

# 7. Commit
git commit -m "feat(scope): description"

# 8. Push and create PR
git push origin feature/your-feature
```

### Commit Message Format

```
type(scope): short description

Longer description if needed.

For kernel changes:
Benchmark results (RTX 3090 Ti):
- 2048x2048: XX.XX TFLOPS
- 4096x4096: XX.XX TFLOPS
- 8192x8192: XX.XX TFLOPS

Correctness: PASS
```

**Types**: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `wip`, `bench`

### PR Requirements

- [ ] All CI checks pass (lint, typecheck, tests)
- [ ] No performance regressions (for kernel changes)
- [ ] Benchmark results included (for kernel changes)
- [ ] Documentation updated if needed
- [ ] No breaking changes without discussion

---

## 5. Review Criteria

PRs are evaluated on these criteria:

### Must Pass

| Criterion | Check |
|-----------|-------|
| CI green | Lint, typecheck, tests pass |
| Architecture | Follows layer model and module boundaries |
| No regressions | Performance equal or better |
| Correctness | Tests pass, no silent failures |

### Evaluated

| Criterion | Weight | Notes |
|-----------|--------|-------|
| Performance | High | Benchmark numbers required for kernels |
| Code quality | Medium | Clear, minimal, no over-engineering |
| Documentation | Medium | Updated if behavior changes |
| Test coverage | Medium | New features need tests |

### Automatic Rejection

- Violates architectural invariants
- Introduces cuda-python or similar dependencies
- Performance regression without justification
- Skips pre-commit checks
- Targets SM < 80

### Review Process

1. **Automated checks** - CI must pass
2. **Maintainer review** - Architecture and code quality
3. **Benchmark verification** - For kernel changes
4. **Merge** - Squash or rebase, clean history

---

## Questions?

- Open an issue for discussion
- Check CLAUDE.md for detailed architecture docs
- Review existing PRs for examples
