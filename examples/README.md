# PyGPUkit Examples

## Directory Structure

```
examples/
├── benchmarks/           # Performance benchmarks
├── chat/                 # Chat CLI applications
├── demos/archived/       # Version-specific demos (historical)
├── demo_*.py             # Current feature demos
├── tts.py                # Text-to-speech example
└── whisper_realtime_stt.py  # Speech-to-text example
```

## Requirements

- NVIDIA GPU with SM >= 80 (Ampere or newer)
- CUDA Toolkit 12.x or 13.x
- Built native module (`_pygpukit_native`)

## Quick Start

### Chat CLI

```bash
# Standard chat (Qwen)
python examples/chat/chat_cli.py

# With Triton backend
python examples/chat/chat_cli_triton.py

# MoE models (Qwen3)
python examples/chat/chat_cli_moe.py

# Thinking mode (Qwen3-8B-Thinking)
python examples/chat/chat_cli_thinking.py
```

### Demos

```bash
# Basic GPU operations
python examples/demo_gpu.py

# CUDA Graph for LLM inference
python examples/demo_cuda_graph.py

# End-to-end LLM demo
python examples/demo_llm_e2e.py

# Qwen3 model demo
python examples/demo_qwen3.py
```

### Benchmarks

```bash
# Matrix multiplication benchmark
python examples/benchmarks/benchmark_matmul.py

# CUDA Graph LLM benchmark
python examples/benchmarks/bench_cuda_graph_llm.py

# Compare with cuBLAS
python examples/benchmarks/benchmark_compare.py
```

### Speech/Audio

```bash
# Text-to-speech (Kokoro)
python examples/tts.py

# Real-time speech-to-text (Whisper)
python examples/whisper_realtime_stt.py
```

## Building Native Module

```bash
# From project root using build script
./build.sh 86      # RTX 3090 Ti
./build.sh 120a    # RTX 5090

# Or manually with pip
pip install -e . -v
```
