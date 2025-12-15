#!/usr/bin/env python3
"""
PyGPUkit v0.2.6 Multi-LLM Async Execution Demo

Demonstrates running multiple LLM-like workloads concurrently on a single GPU
using PyGPUkit's native LLM module (GPT2Model with MLP blocks).

Each workload runs on a separate CUDA stream with independent VRAM budgets.
Uses Python asyncio for non-blocking parallel execution.

Key differences from PyTorch-based demo:
- Uses PyGPUkit's native matmul (CUTLASS TF32)
- Uses PyGPUkit's native layernorm, gelu
- Real transformer block structure (LayerNorm -> MLP -> Residual)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# Check if multi-LLM scheduler is available
try:
    from pygpukit.scheduler import (
        GB,
        HAS_MULTI_LLM,
        MB,
        context_session,
        create_context,
        destroy_context,
        initialize,
        reset,
        stats,
    )
except ImportError:
    HAS_MULTI_LLM = False

# Check if GPU operations are available
try:
    import pygpukit as gpk
    from pygpukit.llm import MLP, LayerNorm, TransformerBlock

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


# =============================================================================
# Real LLM Workloads using PyGPUkit's GPT2Model
# =============================================================================


class PyGPUkitLLM:
    """LLM using PyGPUkit's native GPT2Model structure."""

    def __init__(
        self,
        name: str,
        n_embd: int = 768,
        n_layer: int = 6,
        n_inner: int | None = None,
    ):
        self.name = name
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner or (4 * n_embd)
        self.blocks: list[TransformerBlock] = []
        self.ln_f: LayerNorm | None = None

    def load_weights(self) -> None:
        """Initialize random weights (simulating model loading)."""
        if not HAS_GPU:
            return

        print(
            f"  [{self.name}] Loading GPT2-style model (embd={self.n_embd}, layers={self.n_layer})"
        )

        # Create transformer blocks with random weights
        self.blocks = []
        for i in range(self.n_layer):
            # LayerNorm weights
            ln_weight = gpk.from_numpy(np.ones(self.n_embd, dtype=np.float32))
            ln_bias = gpk.from_numpy(np.zeros(self.n_embd, dtype=np.float32))

            # MLP weights: fc1 [n_inner, n_embd], fc2 [n_embd, n_inner]
            c_fc_weight = gpk.from_numpy(
                (np.random.randn(self.n_inner, self.n_embd) * 0.02).astype(np.float32)
            )
            c_fc_bias = gpk.from_numpy(np.zeros(self.n_inner, dtype=np.float32))
            c_proj_weight = gpk.from_numpy(
                (np.random.randn(self.n_embd, self.n_inner) * 0.02).astype(np.float32)
            )
            c_proj_bias = gpk.from_numpy(np.zeros(self.n_embd, dtype=np.float32))

            mlp = MLP(c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias)
            block = TransformerBlock(ln_weight, ln_bias, mlp)
            self.blocks.append(block)

        # Final LayerNorm
        self.ln_f = LayerNorm(
            gpk.from_numpy(np.ones(self.n_embd, dtype=np.float32)),
            gpk.from_numpy(np.zeros(self.n_embd, dtype=np.float32)),
        )

        # Calculate model size
        params = self.n_layer * (
            self.n_embd  # ln weight
            + self.n_embd  # ln bias
            + self.n_inner * self.n_embd  # c_fc weight
            + self.n_inner  # c_fc bias
            + self.n_embd * self.n_inner  # c_proj weight
            + self.n_embd  # c_proj bias
        )
        print(f"  [{self.name}] Parameters: {params / 1e6:.1f}M")

    def forward(self, batch_size: int = 128, seq_len: int = 512) -> np.ndarray:
        """Run forward pass through transformer blocks.

        This simulates the MLP portion of transformer inference.
        Each block: LayerNorm -> MLP (fc1 -> gelu -> fc2) -> Residual
        """
        if not HAS_GPU or not self.blocks:
            time.sleep(0.1)
            return np.zeros((batch_size, self.n_embd), dtype=np.float32)

        # Create input hidden states [batch_size, n_embd]
        hidden = gpk.from_numpy(np.random.randn(batch_size, self.n_embd).astype(np.float32) * 0.1)

        # Apply transformer blocks
        for block in self.blocks:
            hidden = block(hidden)

        # Final LayerNorm
        if self.ln_f:
            hidden = self.ln_f(hidden)

        return hidden.to_numpy()


# =============================================================================
# Demo Functions
# =============================================================================


def demo_sequential() -> float:
    """Run workloads sequentially (baseline)."""
    section("Sequential Execution (Baseline)")

    # Create models with different sizes (simulating different LLMs)
    llm_large = PyGPUkitLLM("llm-large", n_embd=1024, n_layer=12)  # ~50M MLP params
    llm_medium = PyGPUkitLLM("llm-medium", n_embd=768, n_layer=6)  # ~14M MLP params
    llm_small = PyGPUkitLLM("llm-small", n_embd=512, n_layer=4)  # ~4M MLP params

    print("\nLoading models...")
    llm_large.load_weights()
    llm_medium.load_weights()
    llm_small.load_weights()

    # Warmup
    print("\nWarmup...")
    llm_large.forward(batch_size=64)
    llm_medium.forward(batch_size=64)
    llm_small.forward(batch_size=64)

    print("\nRunning sequentially (3 iterations)...")
    times = []
    for i in range(3):
        start = time.perf_counter()

        # Run one after another (simulating sequential inference requests)
        result_large = llm_large.forward(batch_size=128)
        result_medium = llm_medium.forward(batch_size=128)
        result_small = llm_small.forward(batch_size=128)

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i + 1}: {elapsed * 1000:.2f} ms")

    avg_elapsed = sum(times) / len(times)

    print("\nResults:")
    print(f"  Large output shape: {result_large.shape}")
    print(f"  Medium output shape: {result_medium.shape}")
    print(f"  Small output shape: {result_small.shape}")
    print(f"\n  Average time: {avg_elapsed * 1000:.2f} ms")

    return avg_elapsed


async def demo_parallel_async() -> float:
    """Run workloads in parallel using asyncio."""
    section("Parallel Async Execution (v0.2.6)")

    if not HAS_MULTI_LLM:
        print("\n  [SKIP] Multi-LLM scheduler not available")
        print("  Rebuild PyGPUkit with Rust backend to enable")
        return 0.0

    # Initialize scheduler
    initialize(device_id=0)

    # Create execution contexts with VRAM budgets
    print("\nCreating execution contexts...")
    ctx_large = create_context("llm-large", max_vram=4 * GB)
    ctx_medium = create_context("llm-medium", max_vram=2 * GB)
    ctx_small = create_context("llm-small", max_vram=1 * GB)

    print(
        f"  Large context: stream_id={ctx_large.stream_id}, max_vram={ctx_large.max_vram / GB:.1f} GB"
    )
    print(
        f"  Medium context: stream_id={ctx_medium.stream_id}, max_vram={ctx_medium.max_vram / GB:.1f} GB"
    )
    print(
        f"  Small context: stream_id={ctx_small.stream_id}, max_vram={ctx_small.max_vram / GB:.1f} GB"
    )

    # Create models
    llm_large = PyGPUkitLLM("llm-large", n_embd=1024, n_layer=12)
    llm_medium = PyGPUkitLLM("llm-medium", n_embd=768, n_layer=6)
    llm_small = PyGPUkitLLM("llm-small", n_embd=512, n_layer=4)

    print("\nLoading models...")
    llm_large.load_weights()
    llm_medium.load_weights()
    llm_small.load_weights()

    # Define async workloads
    async def run_large() -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: llm_large.forward(batch_size=128))

    async def run_medium() -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: llm_medium.forward(batch_size=128))

    async def run_small() -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: llm_small.forward(batch_size=128))

    # Warmup
    print("\nWarmup...")
    async with context_session(ctx_large), context_session(ctx_medium), context_session(ctx_small):
        await asyncio.gather(run_large(), run_medium(), run_small())

    print("\nRunning in parallel (3 iterations)...")
    times = []
    for i in range(3):
        start = time.perf_counter()

        async with (
            context_session(ctx_large),
            context_session(ctx_medium),
            context_session(ctx_small),
        ):
            result_large, result_medium, result_small = await asyncio.gather(
                run_large(),
                run_medium(),
                run_small(),
            )

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i + 1}: {elapsed * 1000:.2f} ms")

    avg_elapsed = sum(times) / len(times)

    print("\nResults:")
    print(f"  Large output shape: {result_large.shape}")
    print(f"  Medium output shape: {result_medium.shape}")
    print(f"  Small output shape: {result_small.shape}")
    print(f"\n  Average time: {avg_elapsed * 1000:.2f} ms")

    # Show scheduler stats
    s = stats()
    print("\nScheduler stats:")
    print(f"  Contexts: {s.context_count}")
    print(f"  VRAM used: {s.used_vram / MB:.1f} MB")

    # Cleanup
    destroy_context("llm-large")
    destroy_context("llm-medium")
    destroy_context("llm-small")

    return avg_elapsed


def demo_context_session_api():
    """Demonstrate the context_session API."""
    section("Context Session API Demo")

    if not HAS_MULTI_LLM:
        print("\n  [SKIP] Multi-LLM scheduler not available")
        return

    reset()  # Clean slate
    initialize(device_id=0)

    print("\nTarget API pattern:")
    print("""
    async with context_session(llm_ctx), context_session(tts_ctx):
        llm_f = llm_ctx.dispatch_async(llm_req)
        tts_f = tts_ctx.dispatch_async(tts_req)
        text, audio = await asyncio.gather(llm_f, tts_f)
    """)

    # Create contexts
    ctx1 = create_context("model_a", max_vram=2 * GB)
    ctx2 = create_context("model_b", max_vram=2 * GB)

    print("Sync usage (with statement):")
    print("  with context_session(ctx1), context_session(ctx2):")

    with context_session(ctx1), context_session(ctx2):
        print(f"    ctx1.is_session_active() = {ctx1.is_session_active()}")
        print(f"    ctx2.is_session_active() = {ctx2.is_session_active()}")

    print("  After exiting:")
    print(f"    ctx1.is_session_active() = {ctx1.is_session_active()}")
    print(f"    ctx2.is_session_active() = {ctx2.is_session_active()}")

    # Cleanup
    reset()


def demo_speedup_comparison():
    """Compare sequential vs parallel execution times."""
    section("Speedup Comparison")

    if not HAS_GPU:
        print("\n  [SKIP] GPU not available, speedup demo requires GPU")
        return

    # Run sequential
    seq_time = demo_sequential()

    # Run parallel
    par_time = asyncio.run(demo_parallel_async())

    if par_time > 0:
        section("Summary")
        print(f"\n  Sequential: {seq_time * 1000:.2f} ms")
        print(f"  Parallel:   {par_time * 1000:.2f} ms")
        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"  Speedup:    {speedup:.2f}x")

        if speedup < 1.0:
            print("\n  Note: Single-GPU parallel execution has overhead.")
            print("  Speedup improves with:")
            print("    - Multi-GPU setups (true parallelism)")
            print("    - I/O-bound workloads (async overlapping)")
            print("    - CPU preprocessing overlapping GPU compute")


def main():
    print("=" * 70)
    print(" PyGPUkit v0.2.6 - Multi-LLM Async Execution Demo")
    print(" Using native PyGPUkit LLM module (CUTLASS TF32 matmul)")
    print("=" * 70)

    print("\nBackend status:")
    print(f"  GPU available: {HAS_GPU}")
    print(f"  Multi-LLM scheduler: {HAS_MULTI_LLM}")

    if HAS_GPU:
        import pygpukit as gpk

        print(f"  CUDA available: {gpk.is_cuda_available()}")

    if not HAS_GPU:
        print("\n  [WARNING] No GPU available, running in CPU simulation mode")

    # Demo the API
    demo_context_session_api()

    # Run comparison
    demo_speedup_comparison()

    section("Demo Complete")
    print("\nPyGPUkit Multi-LLM features:")
    print("  - Native GPT2-style transformer blocks")
    print("  - CUTLASS TF32 TensorCore matmul (31+ TFLOPS)")
    print("  - Native layernorm, gelu operations")
    print("  - Separate CUDA streams per context")
    print("  - Independent VRAM budgets")
    print("  - asyncio-compatible execution")


if __name__ == "__main__":
    main()
