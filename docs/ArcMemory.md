oso
# PyGPUkit Memory Manager Architecture — Deep Technical Notes for LLM Implementation

This document explains the internal architecture, constraints, and required hacks to implement the **PyGPUkit Memory Manager**.
It is written specifically for **AI code generators (ClaudeCode / ChatGPT Code)** to avoid common CUDA misunderstandings and produce **correct, safe, deterministic GPU memory handling**.

---

# 1. Overview

The Memory Manager provides:

- **Memory reservations** (hard / soft)
- **GPU memory pools per task**
- **Overcommit / soft allocation model**
- **GPU→Host eviction system (LRU-based)**
- **Rehydration on access**
- **Pinned host memory cache**
- **Per-slice memory quotas (used by Scheduler)**

CUDA does **not** support per-task or per-stream memory limits.
Everything must be implemented fully in software.

---

# 2. CUDA Memory Model Constraints (LLM MUST understand)

### ❌ CUDA cannot:
- Limit memory usage per task
- Track memory ownership
- Evict GPU memory automatically
- Provide “private GPU memory spaces”
- Enforce quotas

### ✔ CUDA *can*:
- Allocate/free raw memory
- Copy to/from host (sync or async)
- Report total/free memory (`cudaMemGetInfo`)

Therefore the Memory Manager must simulate:

- isolation
- quotas
- eviction
- overcommit

This simulation is the core of PyGPUkit’s Memory Engine.

---

# 3. Memory Reservation Model

## 3.1 Hard Reservation
Allocate requested memory immediately.

```cpp
cudaMalloc(&ptr, size);
```

If allocation fails → **reject task**.

This ensures strict QoS.

---

## 3.2 Soft Reservation (Overcommit)
Soft reservation does **not** immediately allocate the full requested memory.
Instead:

- Task declares reservation (metadata only)
- Actual allocation happens lazily (on GPUArray creation)
- If memory pressure occurs → eviction system is triggered

Soft reservation = “promise to try”.

---

# 4. Memory Pool Architecture

Each task/slice receives an independent **GPU memory pool**, tracked by:

- total reserved
- current usage
- free blocks
- eviction debt (memory overcommit level)

Data structure:

```python
class MemoryPool:
    total_quota: int
    used: int
    blocks: Dict[id, MemoryBlock]
```

A **MemoryBlock**:

```python
class MemoryBlock:
    ptr: CUdeviceptr | None      # None when evicted
    host_ptr: void* | None       # pinned host memory
    size: int
    last_access: timestamp
    on_gpu: bool
```

LLMs must track `on_gpu` and `last_access` properly.

---

# 5. LRU Eviction System (GPU → Host)

When GPU memory is insufficient:

```
if free_gpu_memory < needed:
    evict LRU blocks to host
```

Eviction steps:

1. Allocate pinned host memory (`cudaHostAlloc`)
2. Copy GPU → Host
3. Free GPU memory (`cudaFree`)
4. Mark block as `on_gpu = False`

Pseudo-code:

```python
def evict_lru():
    block = pick_block_with_oldest(last_access)
    allocate_pinned_host(block.size)
    cudaMemcpy(host, block.ptr)
    cudaFree(block.ptr)
    block.ptr = None
    block.on_gpu = False
```

Pinned memory recommended for rehydration speed.

---

# 6. Rehydration (Fault-In Mechanism)

When a task accesses a GPUArray that is evicted:

```
if block.on_gpu == False:
    allocate GPU memory
    copy Host → GPU
```

Pseudo-code:

```python
def rehydrate(block):
    cudaMalloc(&block.ptr, block.size)
    cudaMemcpy(block.ptr, block.host_ptr)
    block.on_gpu = True
```

This process must be **transparent** to the user API.

---

# 7. Interaction with Scheduler

Memory Manager provides:

### → `can_reserve(memory, policy)`
Used during **admission control**.

### → `pressure_level()`
Used for scheduling decisions when bandwidth throttling interacts with memory load.

### → `evict_for_bandwidth()`
If Scheduler needs to reduce memory footprint to ensure other tasks can run.

Scheduler and Memory Manager communicate via:

- memory quota
- slice assignment
- eviction priority
- task lifecycle hooks

---

# 8. GPUArray Metadata

Each GPUArray must store:

```python
class GPUArray:
    pool: MemoryPool
    block: MemoryBlock
    shape: tuple
    dtype: DataType
    on_gpu: bool
```

### ✔ LLM MUST ensure:
- All operations update `last_access`
- Array slicing does not duplicate GPU memory (views share blocks)
- Copies allocate new blocks
- Deleting array frees block (refcount zero)

---

# 9. Allocator Strategies (LLM Must Implement One)

There are two acceptable designs:

## **9.1 Simple Bump Allocator (recommended for v0.1)**
- Linear allocations
- No fragmentation handling
- Eviction recovers space

Easy to implement and sufficient for early versions.

---

## **9.2 Free-List Allocator (future)**
- Buddy allocator
- Best-fit / first-fit strategies
- Coalescing of free blocks

LLMs should not attempt implementing this prematurely unless asked.

---

# 10. Memory Pressure Detection

Memory pressure happens when:

```
total_used / slice_quota > threshold
```

Default thresholds:

- Soft threshold: 0.80
- Hard threshold: 0.95

Actions:

| Pressure | Action |
|----------|--------|
| Soft | begin preemptive eviction |
| Hard | block allocations, force eviction |
| Critical | notify scheduler to throttle tasks |

---

# 11. LLM Pitfalls (Critical to Avoid)

### ❌ Pitfall 1: Assuming CUDA supports per-task memory control
→ Must simulate with pools.

### ❌ Pitfall 2: Forgetting to update `last_access`
→ LRU eviction becomes ineffective.

### ❌ Pitfall 3: Evicting arrays that are *currently in kernel use*
→ Must block or mark “locked”.

### ❌ Pitfall 4: Allocating pinned host memory too often
→ Must reuse pinned buffers when possible.

### ❌ Pitfall 5: Assuming unified memory solves this
→ Unified memory has unpredictable migration latency → unacceptable.

### ❌ Pitfall 6: Eviction deadlocks
→ Must ensure eviction always frees enough memory or fail gracefully.

---

# 12. Required Public API

```python
pool = memory_manager.create_pool(quota=int)

ptr = memory_manager.allocate(pool, size)
memory_manager.free(pool, ptr)

memory_manager.evict(pool)
memory_manager.rehydrate(block)

stats = memory_manager.stats()
```

Scheduler integrates via:

```python
memory_manager.can_reserve(memory, policy)
memory_manager.reserve(pool, memory)
```

---

# 13. Suggested Internal Pseudocode for LLMs

### Allocate
```python
def allocate(pool, size):
    if pool.used + size > pool.quota:
        evict_until_free(size)

    ptr = cudaMalloc(size)
    block = MemoryBlock(ptr, size)
    pool.blocks[block.id] = block
    pool.used += size
    return block
```

### Evict
```python
def evict_until_free(size):
    while free_gpu() < size:
        block = pick_lru_block()
        evict(block)
```

### Rehydrate
```python
def access(block):
    block.last_access = now()
    if not block.on_gpu:
        rehydrate(block)
    return block.ptr
```

---

# 14. Summary

The PyGPUkit Memory Manager is a **software-emulated GPU memory virtualization system** built from:

- memory pools
- soft/hard reservations
- LRU eviction
- pinned-memory backing store
- rehydration
- per-slice quotas

CUDA does **not** provide any of these primitives; all must be implemented on the host side.

---

# End of ArcMemory.md
