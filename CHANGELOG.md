# Changelog

All notable changes to PyGPUkit will be documented in this file.

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
