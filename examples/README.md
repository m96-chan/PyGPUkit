# PyGPUkit Examples

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- Built native module (`_pygpukit_native`)

## Examples

### demo_gpu.py
Basic GPU operations demo using the native C++ backend directly.

```bash
python examples/demo_gpu.py
```

### demo_optimized.py
Performance comparison showing zero-copy optimizations.

```bash
python examples/demo_optimized.py
```

### demo_v01.py
Simple v0.1 feature demonstration (CPU simulation fallback).

```bash
python examples/demo_v01.py
```

## Building Native Module

```bash
cd native
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Copy the built module to `src/pygpukit/`:
- Linux: `_pygpukit_native.cpython-3xx-x86_64-linux-gnu.so`
- Windows: `_pygpukit_native.cp3xx-win_amd64.pyd`
