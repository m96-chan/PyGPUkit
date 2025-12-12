
# PyGPUkit Scheduler Architecture — Deep Technical Notes for LLM Implementation

This document is specifically written to enable AI code generators (ClaudeCode / ChatGPT Code) to implement the PyGPUkit Scheduler with full awareness of CUDA’s limitations and the software hacks required to simulate multi-tenant GPU scheduling.

---

## 1. Overview

The PyGPUkit Scheduler provides **Kubernetes‑style resource control** on a single GPU:

- Memory reservation (hard / soft)
- Compute bandwidth guarantees
- Logical GPU partitioning
- Stream priority control
- Kernel micro‑slicing
- Pacing / throttling of long kernels
- Admission control
- Monitoring and introspection

CUDA does **not** provide these features natively; everything must be done using host‑side scheduling and kernel structuring.

---

## 2. Key CUDA Limitations (LLM MUST understand)

| Feature | Native CUDA Support | Notes |
|--------|----------------------|-------|
| SM partitioning | ❌ No | Only MIG supports this. Not available on normal GPUs. |
| Kernel preemption | ❌ No | Long kernels will block all other work. |
| Bandwidth throttling | ❌ No | Must be simulated. |
| Memory hard limits | ❌ No | Must simulate using pools + eviction. |
| Infinite stream priorities | ❌ No | CUDA supports only 2 priority levels. |

LLMs **must not** generate code assuming these exist.

---

## 3. Essential Hacks Required

### 3.1 Stream Priority Hack
Use CUDA’s limited priority feature:

```cpp
cudaStreamCreateWithPriority(&stream, flags, priority);
```

Only two effective levels exist:
- High priority
- Low priority

Used for coarse task classification, not exact bandwidth guarantees.

---

### 3.2 Kernel Pacing (Time‑Based Throttling)

Since CUDA cannot throttle GPU usage, pacing is performed on host:

```cpp
if (now - last_launch < pacing_interval)
    return;  // skip launch to enforce bandwidth cap
launch_kernel();
last_launch = now;
```

Bandwidth → pacing interval:

```
allowed = bandwidth * window
pacing_interval = window - allowed
```

---

### 3.3 Kernel Micro‑Slicing (Tile Execution)

To avoid long-running monolithic kernels, we divide workloads:

Original:
```cpp
kernel<<<G, B>>>(ptr, N);
```

Micro‑sliced:
```cpp
for (int i = 0; i < N; i += tile) {
    kernel<<<Gt, Bt>>>(ptr + i, tile);
    scheduler->yield();
}
```

LLMs MUST generate micro‑kernel wrappers.

---

### 3.4 Cooperative Yield Points
Every micro‑kernel iteration must call into the scheduler:

```python
scheduler.yield_task(task_id)
```

This enables:
- pacing
- slice switching
- preemption simulation
- stats collection

---

### 3.5 Persistent Kernel Dispatcher (Future Enhancement)
A persistent kernel can:
- stay resident
- dispatch tile kernels dynamically
- allow GPU-driven scheduling

This is optional and not required in v0.2.

---

## 4. Memory Reservation Architecture

### 4.1 Hard Reservation
Allocate a dedicated pool:

```cpp
cudaMalloc(&pool, size);
```

If allocation fails → reject task.

---

### 4.2 Soft Reservation (Overcommit)

Simulate overcommit using:

1. Track all GPUArray memory blocks
2. Store last-access timestamps
3. Evict unused blocks to pinned host memory
4. Reload on demand

Eviction pseudo‑code:

```python
if gpu_free < needed:
    evict_lru_arrays()
```

Reload:

```python
if array.on_host:
    copy_host_to_gpu(array)
```

---

## 5. GPU Bandwidth Scheduling Model

A scheduling window is defined (default **10 ms**).

For bandwidth = 0.20 (20%):

```
allowed_time = window * 0.20
pacing_interval = window - allowed_time
```

Kernel launches are spaced accordingly.

---

## 6. Logical GPU Partitioning

Each slice defines:

```json
{
  "memory_quota": 1073741824,
  "bandwidth_quota": 0.3,
  "priority": 0,
  "streams": [stream0, stream1]
}
```

Tasks are pinned to a slice:
- Memory allocations come from slice pool
- Bandwidth enforced using pacing
- Kernels launched on slice's streams

---

## 7. Admission Control Logic

Pseudocode:

```python
def admit(task):
    if task.memory > free_memory:
        if task.policy == "guaranteed":
            return REJECT
        return QUEUED

    if task.bandwidth > free_bandwidth:
        if task.policy == "guaranteed":
            return REJECT

    return ADMIT
```

---

## 8. QoS Policy Model

| QoS | Memory Guarantee | Bandwidth Guarantee | Notes |
|-----|------------------|---------------------|-------|
| Guaranteed | Hard | Hard | Highest priority |
| Burstable | Hard | Soft | May throttle |
| BestEffort | Soft | Soft | Only uses leftovers |

---

## 9. Scheduler Loop (LLM-targeted Blueprint)

```python
while True:
    now = time.now()

    for task in runnable_tasks:
        if scheduler.should_run(task, now):
            task.run_micro_kernel()

    sleep(SCHED_TICK)   # ~1ms
```

Scheduler MUST implement:
- time‑based pacing
- micro‑kernel iteration
- slice enforcement
- memory pressure checks

---

## 10. Telemetry / Monitoring

Tasks expose:
- memory usage
- bandwidth usage
- pacing delay count
- kernel tiles executed
- eviction count
- queue position

Used for runtime optimization or autoscaling.

---

## 11. LLM Pitfalls (Things NOT allowed)

### ❌ Do NOT assume:
- SM partition APIs exist
- Kernel preemption is possible
- Unlimited stream priorities
- GPU-side scheduling without persistent kernels
- cudaMalloc can be intercepted per task

### ❌ Do NOT generate kernels with:
- unbounded loops
- internal synchronization across entire grid
- no opportunities to micro‑slice

---

## 12. Expected Public API

```python
scheduler.submit(fn, memory=None, bandwidth=None, policy="best_effort")
scheduler.step()                # Called by runtime
scheduler.stats(task_id)        # Monitoring
task.run_micro_kernel()         # Auto-generated wrapper
```

The scheduler orchestrates:
- admission
- pacing
- memory allocation
- slice enforcement
- execution order

---

## 13. Summary for LLM Implementers

PyGPUkit Scheduler =
**A software-emulated, Kubernetes-like GPU scheduler built entirely via micro-slicing, pacing, stream priority, and memory pools.**

There is:
- **No SM partitioning**
- **No preemption**
- **No hardware throttling**

Everything is host-driven.

---

# End of ArcScheduler.md
