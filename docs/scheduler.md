# Multi-LLM Scheduler Guide

PyGPUkit includes a Rust-powered scheduler for running multiple AI models concurrently on a single GPU.

## Overview

The scheduler provides:
- **Execution Contexts** - Isolated environments with VRAM budgets
- **Stream Isolation** - Independent CUDA streams per model
- **QoS Policies** - Guaranteed, Burstable, BestEffort tiers
- **asyncio Integration** - Native Python async/await support

> **Note:** On a single GPU, concurrent execution doesn't make compute-bound workloads faster.
> The benefit is safe resource sharing and simplified orchestration.

---

## Basic Usage

### Creating Contexts

```python
from pygpukit.scheduler import (
    create_context,
    context_session,
    initialize,
    GB, MB,
)

# Initialize scheduler
initialize(device_id=0)

# Create execution contexts with VRAM budgets
llm_ctx = create_context("llm", max_vram=4 * GB)
tts_ctx = create_context("tts", max_vram=2 * GB)
vision_ctx = create_context("vision", max_vram=1 * GB)
```

### Running with Context Sessions

```python
from pygpukit.scheduler import context_session

# Synchronous usage
with context_session(llm_ctx):
    # All GPU operations here use llm_ctx's stream and VRAM budget
    result = run_llm_inference(prompt)

with context_session(tts_ctx):
    # TTS uses its own isolated stream
    audio = run_tts(text)
```

### Async Concurrent Execution

```python
import asyncio
from pygpukit.scheduler import context_session

async def run_llm(prompt):
    async with context_session(llm_ctx):
        return await llm_inference_async(prompt)

async def run_tts(text):
    async with context_session(tts_ctx):
        return await tts_synthesis_async(text)

async def main():
    # Run both models concurrently
    text_task = asyncio.create_task(run_llm("Hello"))
    audio_task = asyncio.create_task(run_tts("Welcome"))

    text, audio = await asyncio.gather(text_task, audio_task)
    return text, audio

result = asyncio.run(main())
```

---

## QoS Policies

### Policy Tiers

| Tier | Guarantees | Use Case |
|------|------------|----------|
| **Guaranteed** | Reserved VRAM, priority scheduling | Production LLM |
| **Burstable** | Base allocation, can burst higher | Interactive apps |
| **BestEffort** | No guarantees, uses spare capacity | Background tasks |

### Setting QoS Policy

```python
from pygpukit.scheduler import create_context, QoSPolicy, GB

# Guaranteed: Always has 4GB reserved
llm_ctx = create_context(
    "llm",
    max_vram=4 * GB,
    qos=QoSPolicy.GUARANTEED,
)

# Burstable: Base 1GB, can burst to 2GB
tts_ctx = create_context(
    "tts",
    max_vram=2 * GB,
    base_vram=1 * GB,
    qos=QoSPolicy.BURSTABLE,
)

# BestEffort: Uses whatever is available
bg_ctx = create_context(
    "background",
    max_vram=1 * GB,
    qos=QoSPolicy.BEST_EFFORT,
)
```

---

## Memory Management

### VRAM Budgeting

```python
from pygpukit.scheduler import (
    create_context,
    get_available_vram,
    get_context_usage,
    GB,
)

# Check available VRAM
available = get_available_vram()
print(f"Available VRAM: {available / GB:.1f} GB")

# Create context with budget
ctx = create_context("model", max_vram=4 * GB)

# Check context usage
usage = get_context_usage(ctx)
print(f"Used: {usage.used / GB:.1f} GB")
print(f"Max: {usage.max / GB:.1f} GB")
```

### Memory Pressure Handling

```python
from pygpukit.scheduler import (
    create_context,
    on_memory_pressure,
    GB,
)

def handle_pressure(ctx, requested, available):
    print(f"Context {ctx.name}: requested {requested}, available {available}")
    # Return True to allow allocation, False to reject
    return available > requested * 0.5

# Register handler
on_memory_pressure(handle_pressure)
```

---

## Stream Management

### Explicit Stream Control

```python
from pygpukit.scheduler import create_context, get_stream

ctx = create_context("model", max_vram=4 * GB)

# Get CUDA stream for context
stream = get_stream(ctx)
print(f"Stream ID: {stream.id}")

# Synchronize stream
stream.synchronize()
```

### Stream Events

```python
from pygpukit.scheduler import create_context, record_event, wait_event

ctx1 = create_context("producer", max_vram=2 * GB)
ctx2 = create_context("consumer", max_vram=2 * GB)

# Record event after producer finishes
with context_session(ctx1):
    result = produce_data()
    event = record_event(ctx1)

# Consumer waits for producer
with context_session(ctx2):
    wait_event(ctx2, event)
    consume_data(result)
```

---

## Rust Scheduler API

For advanced use cases, access the Rust scheduler directly:

```python
import _pygpukit_rust as rust

# Memory Pool with LRU eviction
pool = rust.MemoryPool(
    quota=100 * 1024 * 1024,  # 100 MB
    enable_eviction=True,
)

# Allocate and free
block = pool.allocate(4096)
pool.free(block)

# Check stats
stats = pool.stats()
print(f"Allocated: {stats.allocated_bytes}")
print(f"Free blocks: {stats.free_blocks}")
```

### QoS Policy Evaluator

```python
import _pygpukit_rust as rust

# Create evaluator
evaluator = rust.QosPolicyEvaluator(
    total_memory=8 * 1024**3,  # 8 GB
    total_bandwidth=1.0,
)

# Create task metadata
task = rust.QosTaskMeta.guaranteed(
    "inference-1",
    "LLM Inference",
    256 * 1024 * 1024,  # 256 MB
)

# Evaluate admission
result = evaluator.evaluate(task)
if result.admitted:
    print(f"Task admitted with priority {result.priority}")
else:
    print(f"Task rejected: {result.reason}")
```

### GPU Partitioning

```python
import _pygpukit_rust as rust

# Create partition manager
config = rust.PartitionConfig(total_memory=8 * 1024**3)
manager = rust.PartitionManager(config)

# Create partitions
manager.create_partition(
    "inference",
    "Inference Partition",
    rust.PartitionLimits()
        .memory(4 * 1024**3)
        .compute(0.5),
)

manager.create_partition(
    "training",
    "Training Partition",
    rust.PartitionLimits()
        .memory(4 * 1024**3)
        .compute(0.5),
)

# Get partition info
info = manager.get_partition("inference")
print(f"Memory limit: {info.memory_limit}")
print(f"Compute limit: {info.compute_limit}")
```

---

## Complete Example

```python
"""Run LLM + TTS + Vision concurrently."""
import asyncio
from pygpukit.scheduler import (
    create_context,
    context_session,
    initialize,
    GB,
)

# Initialize
initialize(device_id=0)

# Create contexts
llm_ctx = create_context("llm", max_vram=4 * GB)
tts_ctx = create_context("tts", max_vram=2 * GB)
vision_ctx = create_context("vision", max_vram=2 * GB)

async def process_request(user_input: str):
    """Process user request with multiple AI models."""

    async def llm_task():
        async with context_session(llm_ctx):
            # Generate text response
            return await generate_response(user_input)

    async def vision_task():
        async with context_session(vision_ctx):
            # Analyze any images in input
            return await analyze_images(user_input)

    async def tts_task(text):
        async with context_session(tts_ctx):
            # Convert text to speech
            return await synthesize_speech(text)

    # Run LLM and Vision in parallel
    text_response, image_analysis = await asyncio.gather(
        llm_task(),
        vision_task(),
    )

    # Then run TTS on the combined response
    combined_text = f"{text_response}. {image_analysis}"
    audio = await tts_task(combined_text)

    return {
        "text": combined_text,
        "audio": audio,
    }

# Run
result = asyncio.run(process_request("Describe this image and tell me a story"))
```

---

## Best Practices

1. **Set appropriate VRAM budgets** - Don't over-allocate
2. **Use async/await** for concurrent I/O-bound operations
3. **Use QoS policies** to prioritize critical workloads
4. **Monitor memory usage** with `get_context_usage()`
5. **Synchronize streams** when sharing data between contexts

---

## Limitations

- Single GPU only (multi-GPU support planned for v0.3)
- Compute-bound workloads don't benefit from concurrency
- No automatic memory defragmentation
