---
name: chat-test
description: Run LLM inference tests with Qwen or other models. Use when testing model loading, inference, CUDA Graph, or generation quality.
---

# LLM Chat Test

Test LLM inference with PyGPUkit.

## Usage

```bash
# Basic chat CLI
python examples/chat_cli.py --model /path/to/model

# Chat with thinking mode
python examples/chat_cli_thinking.py --model /path/to/model

# MoE model (Qwen3-8B etc.)
python examples/chat_cli_moe.py --model /path/to/model
```

## Test Models

Local test models:
- Qwen3-8B: `/c/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/`
- TinyLlama-1.1B: `/c/Users/y_har/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/`

## Instructions

1. Ensure project is built
2. Run the appropriate chat CLI
3. Test generation quality and performance
4. Report:
   - Model loading success
   - First token latency
   - Tokens per second
   - Any errors or issues

## CUDA Graph Testing

```bash
# Enable CUDA Graph for decode
python examples/chat_cli_moe.py --model /path/to/model --use-cuda-graph
```

## Notes

- Use HuggingFace tokenizers (not built-in)
- Large models require significant VRAM
- CUDA Graph provides ~1.2x speedup for decode
