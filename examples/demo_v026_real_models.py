#!/usr/bin/env python3
"""
PyGPUkit v0.2.6 Real Model Parallel Execution Demo

Downloads and runs actual GPT-2 models in parallel using PyGPUkit's
multi-LLM scheduler with asyncio.

Models:
- GPT-2 (124M params) - Text generation
- DistilGPT-2 (82M params) - Lighter text generation

Demonstrates:
- Real model parallel inference
- Independent CUDA streams per model
- asyncio.gather for concurrent execution
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# PyGPUkit scheduler
try:
    from pygpukit.scheduler import (
        GB,
        HAS_MULTI_LLM,
        context_session,
        create_context,
        destroy_context,
        initialize,
        reset,
        stats,
    )
except ImportError:
    HAS_MULTI_LLM = False


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


class RealLLM:
    """Wrapper for real HuggingFace LLM."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading {self.model_name}...")
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        elapsed = time.perf_counter() - start
        print(f"  {self.model_name} loaded in {elapsed:.2f}s")

        # Print model size
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {params / 1e6:.1f}M")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> tuple[str, float]:
        """Generate text and return (output, time)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Decode
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text, elapsed


def demo_sequential():
    """Run models sequentially (baseline)."""
    section("Sequential Execution (Baseline)")

    # Load models
    print("\nLoading models...")
    gpt2 = RealLLM("gpt2")
    distilgpt2 = RealLLM("distilgpt2")

    gpt2.load()
    distilgpt2.load()

    # Prompts
    prompt1 = "The future of artificial intelligence is"
    prompt2 = "In a world where robots"

    print("\nPrompts:")
    print(f"  GPT-2: '{prompt1}'")
    print(f"  DistilGPT-2: '{prompt2}'")

    # Run sequentially
    print("\nRunning sequentially...")
    total_start = time.perf_counter()

    text1, time1 = gpt2.generate(prompt1, max_new_tokens=30)
    text2, time2 = distilgpt2.generate(prompt2, max_new_tokens=30)

    total_elapsed = time.perf_counter() - total_start

    print("\nResults:")
    print(f"  GPT-2 ({time1*1000:.1f}ms): {text1[:100]}...")
    print(f"  DistilGPT-2 ({time2*1000:.1f}ms): {text2[:100]}...")
    print(f"\n  Total time: {total_elapsed*1000:.1f}ms")

    return total_elapsed, gpt2, distilgpt2


async def demo_parallel(gpt2: RealLLM, distilgpt2: RealLLM):
    """Run models in parallel with PyGPUkit scheduler."""
    section("Parallel Execution with PyGPUkit (v0.2.6)")

    if not HAS_MULTI_LLM:
        print("\n  [SKIP] Multi-LLM scheduler not available")
        return 0.0

    # Initialize scheduler
    reset()
    initialize(device_id=0)

    # Create execution contexts
    print("\nCreating execution contexts...")
    gpt2_ctx = create_context("gpt2", max_vram=2 * GB)
    distil_ctx = create_context("distilgpt2", max_vram=1 * GB)

    print(f"  GPT-2 context: stream_id={gpt2_ctx.stream_id}")
    print(f"  DistilGPT-2 context: stream_id={distil_ctx.stream_id}")

    # Prompts
    prompt1 = "The future of artificial intelligence is"
    prompt2 = "In a world where robots"

    print("\nPrompts:")
    print(f"  GPT-2: '{prompt1}'")
    print(f"  DistilGPT-2: '{prompt2}'")

    # Define async tasks
    async def run_gpt2():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: gpt2.generate(prompt1, max_new_tokens=30)
        )

    async def run_distil():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: distilgpt2.generate(prompt2, max_new_tokens=30)
        )

    # Run in parallel with context sessions
    print("\nRunning in parallel with asyncio.gather...")
    total_start = time.perf_counter()

    async with context_session(gpt2_ctx), context_session(distil_ctx):
        results = await asyncio.gather(run_gpt2(), run_distil())

    total_elapsed = time.perf_counter() - total_start

    (text1, time1), (text2, time2) = results

    print("\nResults:")
    print(f"  GPT-2 ({time1*1000:.1f}ms): {text1[:100]}...")
    print(f"  DistilGPT-2 ({time2*1000:.1f}ms): {text2[:100]}...")
    print(f"\n  Total time: {total_elapsed*1000:.1f}ms")

    # Show scheduler stats
    s = stats()
    print("\nScheduler stats:")
    print(f"  Active contexts: {s.context_count}")

    # Cleanup
    destroy_context("gpt2")
    destroy_context("distilgpt2")

    return total_elapsed


async def demo_three_models():
    """Run three models in parallel."""
    section("Three Models Parallel (GPT-2 + DistilGPT-2 + GPT-2)")

    if not HAS_MULTI_LLM:
        print("\n  [SKIP] Multi-LLM scheduler not available")
        return

    # Load three models
    print("\nLoading three model instances...")
    model_a = RealLLM("gpt2")
    model_b = RealLLM("distilgpt2")
    model_c = RealLLM("gpt2")  # Another GPT-2 instance

    model_a.load()
    model_b.load()
    model_c.load()

    # Initialize scheduler
    reset()
    initialize(device_id=0)

    # Create contexts
    ctx_a = create_context("model_a", max_vram=2 * GB)
    ctx_b = create_context("model_b", max_vram=1 * GB)
    ctx_c = create_context("model_c", max_vram=2 * GB)

    print(f"\nContexts: A={ctx_a.stream_id}, B={ctx_b.stream_id}, C={ctx_c.stream_id}")

    # Prompts
    prompts = [
        "Once upon a time",
        "The scientist discovered",
        "In the year 2050",
    ]

    async def run_model(model, prompt):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: model.generate(prompt, max_new_tokens=20)
        )

    # Run all three
    print("\nRunning 3 models in parallel...")
    total_start = time.perf_counter()

    async with context_session(ctx_a), context_session(ctx_b), context_session(ctx_c):
        results = await asyncio.gather(
            run_model(model_a, prompts[0]),
            run_model(model_b, prompts[1]),
            run_model(model_c, prompts[2]),
        )

    total_elapsed = time.perf_counter() - total_start

    print("\nResults:")
    for i, ((text, t), prompt) in enumerate(zip(results, prompts)):
        print(f"  Model {chr(65+i)} ({t*1000:.1f}ms): {text[:80]}...")

    print(f"\n  Total time: {total_elapsed*1000:.1f}ms")

    # Cleanup
    reset()


def main():
    print("=" * 70)
    print(" PyGPUkit v0.2.6 - Real Model Parallel Execution Demo")
    print("=" * 70)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Multi-LLM scheduler: {HAS_MULTI_LLM}")

    # Run sequential baseline
    seq_time, gpt2, distilgpt2 = demo_sequential()

    # Run parallel
    par_time = asyncio.run(demo_parallel(gpt2, distilgpt2))

    # Summary
    if par_time > 0:
        section("Comparison")
        print(f"\n  Sequential: {seq_time*1000:.1f}ms")
        print(f"  Parallel:   {par_time*1000:.1f}ms")
        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"  Speedup:    {speedup:.2f}x")

    # Three models demo
    asyncio.run(demo_three_models())

    section("Demo Complete")
    print("\nKey takeaways:")
    print("  - Real GPT-2 models running in parallel")
    print("  - PyGPUkit scheduler manages stream isolation")
    print("  - asyncio.gather enables concurrent execution")
    print("  - Each model gets independent VRAM budget")


if __name__ == "__main__":
    main()
