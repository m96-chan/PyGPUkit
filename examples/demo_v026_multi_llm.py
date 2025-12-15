#!/usr/bin/env python3
"""
PyGPUkit v0.2.6 Multi-LLM Async Execution Demo

Demonstrates running multiple "LLM-like" workloads concurrently on a single GPU:
- TTS (Text-to-Speech simulation)
- LLM (Language Model simulation)
- Vision (Image processing simulation)

Each workload runs on a separate CUDA stream with independent VRAM budgets.
Uses Python asyncio for non-blocking parallel execution.

Target API:
    async with context_session(llm_ctx), context_session(tts_ctx):
        llm_f = llm_ctx.dispatch_async(llm_req)
        tts_f = tts_ctx.dispatch_async(tts_req)
        text, audio = await asyncio.gather(llm_f, tts_f)
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

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def section(title: str) -> None:
    """Print section header."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


# =============================================================================
# Simulated LLM Workloads (using matmul as proxy)
# =============================================================================


class SimulatedLLM:
    """Simulated LLM using matmul operations."""

    def __init__(self, name: str, hidden_size: int = 1024, n_layers: int = 4):
        self.name = name
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self._weights = None

    def load_weights(self) -> None:
        """Load random weights (simulating model loading)."""
        if not HAS_GPU:
            return
        print(f"  [{self.name}] Loading weights (hidden={self.hidden_size}, layers={self.n_layers})")
        self._weights = [
            gpk.from_numpy(np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32))
            for _ in range(self.n_layers)
        ]

    def forward(self, batch_size: int = 32) -> np.ndarray:
        """Run forward pass (simulated with matmul chain)."""
        if not HAS_GPU or self._weights is None:
            # CPU fallback: just sleep
            time.sleep(0.1)
            return np.zeros((batch_size, self.hidden_size), dtype=np.float32)

        # Create input
        x = gpk.from_numpy(np.random.randn(batch_size, self.hidden_size).astype(np.float32))

        # Chain of matmuls (simulating transformer layers)
        for w in self._weights:
            x = x @ w

        return x.to_numpy()


class SimulatedTTS:
    """Simulated TTS using matmul operations."""

    def __init__(self, name: str = "tts"):
        self.name = name
        self._encoder = None
        self._decoder = None

    def load_weights(self) -> None:
        """Load random weights."""
        if not HAS_GPU:
            return
        print(f"  [{self.name}] Loading TTS weights")
        self._encoder = gpk.from_numpy(np.random.randn(512, 512).astype(np.float32))
        self._decoder = gpk.from_numpy(np.random.randn(512, 1024).astype(np.float32))

    def synthesize(self, text_len: int = 64) -> np.ndarray:
        """Synthesize audio (simulated)."""
        if not HAS_GPU or self._encoder is None:
            time.sleep(0.05)
            return np.zeros((text_len * 16,), dtype=np.float32)

        x = gpk.from_numpy(np.random.randn(text_len, 512).astype(np.float32))
        x = x @ self._encoder
        x = x @ self._decoder
        return x.to_numpy().flatten()


class SimulatedVision:
    """Simulated Vision model using matmul operations."""

    def __init__(self, name: str = "vision"):
        self.name = name
        self._backbone = None

    def load_weights(self) -> None:
        """Load random weights."""
        if not HAS_GPU:
            return
        print(f"  [{self.name}] Loading Vision weights")
        self._backbone = [
            gpk.from_numpy(np.random.randn(256, 256).astype(np.float32))
            for _ in range(3)
        ]

    def process(self, patches: int = 196) -> np.ndarray:
        """Process image (simulated)."""
        if not HAS_GPU or self._backbone is None:
            time.sleep(0.03)
            return np.zeros((patches, 256), dtype=np.float32)

        x = gpk.from_numpy(np.random.randn(patches, 256).astype(np.float32))
        for w in self._backbone:
            x = x @ w
        return x.to_numpy()


# =============================================================================
# Demo Functions
# =============================================================================


def demo_sequential() -> float:
    """Run workloads sequentially (baseline)."""
    section("Sequential Execution (Baseline)")

    llm = SimulatedLLM("llm", hidden_size=1024, n_layers=4)
    tts = SimulatedTTS("tts")
    vision = SimulatedVision("vision")

    print("\nLoading models...")
    llm.load_weights()
    tts.load_weights()
    vision.load_weights()

    print("\nRunning sequentially...")
    start = time.perf_counter()

    # Run one after another
    llm_result = llm.forward(batch_size=32)
    tts_result = tts.synthesize(text_len=64)
    vision_result = vision.process(patches=196)

    elapsed = time.perf_counter() - start

    print("\nResults:")
    print(f"  LLM output shape: {llm_result.shape}")
    print(f"  TTS output shape: {tts_result.shape}")
    print(f"  Vision output shape: {vision_result.shape}")
    print(f"\n  Total time: {elapsed * 1000:.2f} ms")

    return elapsed


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
    llm_ctx = create_context("llm", max_vram=4 * GB)
    tts_ctx = create_context("tts", max_vram=2 * GB)
    vision_ctx = create_context("vision", max_vram=1 * GB)

    print(f"  LLM context: stream_id={llm_ctx.stream_id}, max_vram={llm_ctx.max_vram / GB:.1f} GB")
    print(f"  TTS context: stream_id={tts_ctx.stream_id}, max_vram={tts_ctx.max_vram / GB:.1f} GB")
    print(f"  Vision context: stream_id={vision_ctx.stream_id}, max_vram={vision_ctx.max_vram / GB:.1f} GB")

    # Create simulated models
    llm = SimulatedLLM("llm", hidden_size=1024, n_layers=4)
    tts = SimulatedTTS("tts")
    vision = SimulatedVision("vision")

    print("\nLoading models...")
    llm.load_weights()
    tts.load_weights()
    vision.load_weights()

    # Define async workloads
    async def run_llm() -> np.ndarray:
        """Run LLM in executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: llm.forward(batch_size=32))

    async def run_tts() -> np.ndarray:
        """Run TTS in executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: tts.synthesize(text_len=64))

    async def run_vision() -> np.ndarray:
        """Run Vision in executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: vision.process(patches=196))

    print("\nRunning in parallel with asyncio.gather...")
    start = time.perf_counter()

    # Run all workloads in parallel
    async with context_session(llm_ctx), context_session(tts_ctx), context_session(vision_ctx):
        llm_result, tts_result, vision_result = await asyncio.gather(
            run_llm(),
            run_tts(),
            run_vision(),
        )

    elapsed = time.perf_counter() - start

    print("\nResults:")
    print(f"  LLM output shape: {llm_result.shape}")
    print(f"  TTS output shape: {tts_result.shape}")
    print(f"  Vision output shape: {vision_result.shape}")
    print(f"\n  Total time: {elapsed * 1000:.2f} ms")

    # Show scheduler stats
    s = stats()
    print("\nScheduler stats:")
    print(f"  Contexts: {s.context_count}")
    print(f"  VRAM used: {s.used_vram / MB:.1f} MB")

    # Cleanup
    destroy_context("llm")
    destroy_context("tts")
    destroy_context("vision")

    return elapsed


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


def main():
    print("=" * 60)
    print(" PyGPUkit v0.2.6 - Multi-LLM Async Execution Demo")
    print("=" * 60)

    print("\nBackend status:")
    print(f"  GPU available: {HAS_GPU}")
    print(f"  Multi-LLM scheduler: {HAS_MULTI_LLM}")

    if not HAS_GPU:
        print("\n  [WARNING] No GPU available, running in CPU simulation mode")

    # Demo the API
    demo_context_session_api()

    # Run comparison
    demo_speedup_comparison()

    section("Demo Complete")
    print("Multi-LLM async execution demonstrated successfully!")
    print("\nKey features:")
    print("  - Separate CUDA streams per LLM context")
    print("  - Independent VRAM budgets")
    print("  - asyncio-compatible execution")
    print("  - Non-blocking parallel execution with asyncio.gather")


if __name__ == "__main__":
    main()
