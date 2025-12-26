# Changelog

All notable changes to PyGPUkit will be documented in this file.

## [0.2.15] - 2025-12-26

### Added
- **FP8 I/O GEMM (SM120)**: Pure FP8 E4M3 input/output GEMM for FP8 model inference
  - `matmul_fp8_fp8_sm120`: FP8 GEMM with unity scaling
  - `matmul_fp8_fp8_blockwise_sm120`: FP8 GEMM with per-block scale factors
  - `fp8_fp8_get_scale_sizes`: Get required scale factor sizes for (M, N, K)
  - `fp8_fp8_sm120_available`: Check SM120 FP8 I/O availability
- **Pure NVF4 GEMM**: GPU-side BF16->NVF4 quantization with 3-stage pipeline (446 TFLOPS)
- **New math operations**: sin, cos, sqrt, rsqrt, abs, neg
- **New comparison operations**: clamp, where
- **New activation functions**: sigmoid, tanh
- **New reduction operations**: argmax, min, sum_axis
- **uint8/int8 NumPy support**: `from_numpy` now supports uint8 and int8 arrays

### Changed
- Renamed `matmul_fp8_sm120.cu` to `matmul_fp8_fp32_sm120.cu` for clarity (FP8 compute, FP32 output)

## [0.2.14] - 2025-12-23

### Fixed
- **Windows wheel RECORD file**: Fixed missing `licenses/LICENSE` entry in RECORD file
  - Root cause: PowerShell `Get-ChildItem` was not using `-Recurse` flag, causing files in `dist-info/licenses/` subdirectory to be omitted from RECORD
  - This caused PyPI deprecation warnings about RECORD file mismatch

## [0.2.13] - 2025-12-23

### Fixed
- **RECORD file generation**: Made version detection dynamic in release workflow
  - Replaced hardcoded `pygpukit-0.2.11.dist-info` with dynamic folder detection
  - Linux: Added dist-info files to RECORD (was only including `pygpukit/` files)
  - Windows: Added `-Recurse` for pygpukit directory scanning

### Changed
- Organized project structure:
  - Moved benchmark files (`bench_*.py`, `benchmark_*.py`, `profile_blocks.py`) to `benchmarks/`
  - Moved demo files (`demo_*.py`) to `examples/`
  - Moved GPU integration tests to `benchmarks/` (these require GPU hardware and local model paths)

## [0.2.12] - 2025-12-22

### Added
- **GPU Audio Processing** (Driver-Only mode, no cuFFT dependency):
  - Time-Frequency: `istft`, `griffin_lim`
  - Spectral Features: `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`
  - Pitch Detection: `detect_pitch_yin`, `detect_pitch_yin_frames`, `autocorrelation`
  - Music Analysis: `cqt`, `chroma_stft`, `chroma_cqt`, `zero_crossing_rate`
  - Source Separation: `hpss`, `harmonic`, `percussive`
  - Time/Pitch Modification: `time_stretch`, `pitch_shift`

## [0.2.11] - 2025-12-21

### Added
- Batch decode support with near-linear speedup (up to 6.83x at batch=8)
- Decode strategy framework (M1, Batch, Jacobi, Speculative)
- GPU-side Lookahead KV Cache
- CUDA Events API
- Voice Activity Detection (VAD)
- Streaming audio API with ring buffer and windowing
- Basic GPU audio processing ops (STFT, Mel filterbank, MFCC)

### Fixed
- CUDA Graph stream fix (RoPE/SDPA now properly captured)

---

For feature details, see README.md "What's New" sections.
