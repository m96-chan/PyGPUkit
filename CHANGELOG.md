# Changelog

All notable changes to PyGPUkit will be documented in this file.

For detailed release notes with code examples, see [README.md](README.md).

## [0.2.17] - 2025-12-28

### Added
- **Triton Backend MVP**: Optional Triton backend for rapid kernel prototyping
- **pygpukit.triton module**: TritonArray wrapper, from_gpuarray, triton_available
- **Triton Kernels**: RMSNorm, LayerNorm, Softmax, Rotary
- **Hybrid Execution**: Mix Triton + Native CUDA in same model
- **examples/chat_cli_triton.py**: Hybrid chat example demonstrating Triton + CUDA

### Fixed
- TritonArray dtype mapping: Support both PascalCase and lowercase dtype strings

## [0.2.16] - 2025-12-28

### Added
- **MoE (Mixture of Experts)**: Full Mixtral support with TopK routing, grouped GEMM
- **Thinking Model**: Qwen3 `<think>...</think>` block parsing
- **GEMV Kernels (SM120)**: FP8/FP8 (W8A8), NVF4/NVF4 (W4A4), Int4
- **GEMM Kernels (SM120)**: W8A16, Int8 native (dp4a), Int4 via Int8, Grouped GEMM v2
- **Claude Code Skills**: Build, benchmark, lint, test automation
- **Subagents**: kernel-reviewer, perf-analyzer, api-designer

### Changed
- Kernel directory restructure: `{gemm|gemv}/{input}/{output}/{arch}/`
- Removed redundant slow kernels (FP8 GEMV basic, Int8 via FP8)

## [0.2.15] - 2025-12-26

### Added
- **Whisper ASR Module**: Full encoder/decoder, preprocessing, streaming transcription
- **GEMV Kernels**: BF16 (vectorized BF16x2), NVF4 (pre-scaled LUT)
- **FP8 I/O GEMM (SM120)**: Pure FP8 E4M3 input/output with blockwise scaling
- **Pure NVF4 GEMM**: 446 TFLOPS with GPU-side quantization (170x vs CPU)
- **GPUArray improvements**: Scalar arithmetic, transpose, reshape, indexing
- **GPU Transpose Kernels**: 2D, 3D (0,2,1), 4D (0,2,1,3), 4D (0,1,3,2)
- **Math operations**: sin, cos, sqrt, rsqrt, abs, neg, clamp, where, sigmoid, tanh, argmax, min, sum_axis
- **uint8/int8 NumPy support**: `from_numpy` for FP8 data handling

### Changed
- Linear layer uses GEMV for M=1 decode (1.3-2.4x faster than matmul)
- SM120 compatibility via CUTLASS fork with alignment fixes

## [0.2.14] - 2025-12-23

### Fixed
- Windows wheel RECORD file: Missing `licenses/LICENSE` entry

## [0.2.13] - 2025-12-23

### Fixed
- RECORD file generation: Dynamic dist-info folder detection

### Changed
- Moved benchmark/demo files to `benchmarks/` and `examples/`

## [0.2.12] - 2025-12-22

### Added
- **GPU Audio Processing** (no cuFFT dependency):
  - Time-Frequency: `istft`, `griffin_lim`
  - Spectral: `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`
  - Pitch: `detect_pitch_yin`, `detect_pitch_yin_frames`, `autocorrelation`
  - Music: `cqt`, `chroma_stft`, `chroma_cqt`, `zero_crossing_rate`
  - Source Separation: `hpss`, `harmonic`, `percussive`
  - Time/Pitch: `time_stretch`, `pitch_shift`

## [0.2.11] - 2025-12-21

### Added
- **Batch decode**: Near-linear speedup (6.83x at batch=8)
- **Decode strategies**: DecodeM1, DecodeM1Graph, DecodeBatch, DecodeJacobi
- **Driver API**: Async memory ops, pinned malloc
- **RTX 5090 (SM120)**: Full support via CUDA 13.x
- **Qwen2 architecture**: `QWEN2_SPEC` for Qwen2/2.5
- **Audio ops**: STFT, Mel filterbank, MFCC, VAD, streaming

### Fixed
- CUDA Graph stream fix (RoPE/SDPA properly captured)

## [0.2.10] - 2025-12-19

### Added
- **Dynamic cuBLASLt loading**: True driver-only deployment
- **Descriptor caching**: 2.67x faster (395ms -> 148ms for 224 matmuls)

### Changed
- CUDA Graph optimizations (eliminated GPU allocations)

## [0.2.9] - 2025-12-17

### Added
- **Unified LLM Interface**: `CausalTransformerModel` with `ModelSpec`
- **Architecture support**: GPT-2, LLaMA 2/3, Qwen2/2.5, Qwen3
- **Hybrid attention**: Auto CPU/GPU switching
- **LLM operations**: `sdpa_causal`, `rope_inplace`, `silu`, `rmsnorm`
- **Sharded models**: Auto-load split safetensors

## [0.2.7] - 2025-12-14

### Added
- **CUTLASS epilogue fusion**: Linear + Bias + GELU
- **Multi-SM kernels**: SM80/86/89/90/100/120 optimized
- **Operations**: `transpose`, `bias_add_inplace`, `linear_bias_gelu`

### Changed
- Complete public API exports with snake_case naming

## [0.2.6] - 2025-12-12

### Added
- **CUTLASS backend**: Default GEMM (TF32: 31 TFLOPS, FP16/BF16: 63 TFLOPS)
- **Multi-LLM scheduling**: Concurrent execution with VRAM budgets
- **FP16/BF16 TensorCore**: Via CUTLASS

## [0.2.5] - 2025-12-10

### Added
- **FP16/BF16 support**: Half and brain float types
- **Type conversion**: `astype()` method
- **Reduction ops**: `sum`, `mean`, `max`
- **Operator overloads**: `+`, `-`, `*`, `/`, `@`

## [0.2.4] - 2025-12-08

### Added
- **Driver-only mode**: No cudart dependency
- **Dynamic NVRTC**: JIT loaded at runtime
- **TF32 TensorCore GEMM**: PTX mma.sync with cp.async pipeline

---

For full release notes with code examples, see the README.md "What's New" sections.
