"""Chrome trace format export for profiling data.

Exports profiling records to Chrome's trace event format for visualization
in chrome://tracing or Perfetto UI.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pygpukit.profiling.profiler import KernelRecord
    from pygpukit.profiling.memory import MemorySnapshot


def export_chrome_trace(
    records: list[KernelRecord],
    path: str,
    *,
    memory_snapshots: list[MemorySnapshot] | None = None,
    process_name: str = "PyGPUkit",
    thread_name: str = "GPU",
) -> None:
    """Export profiling data to Chrome trace format.

    The output can be viewed in:
    - chrome://tracing (paste the file)
    - Perfetto UI (https://ui.perfetto.dev)

    Args:
        records: List of KernelRecord from Profiler.
        path: Output file path (usually .json).
        memory_snapshots: Optional list of MemorySnapshot for memory events.
        process_name: Name shown for the process in trace viewer.
        thread_name: Name shown for the thread in trace viewer.
    """
    events: list[dict[str, Any]] = []

    # Add metadata events
    events.append({
        "name": "process_name",
        "ph": "M",
        "pid": 1,
        "args": {"name": process_name},
    })
    events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": 1,
        "tid": 1,
        "args": {"name": thread_name},
    })

    # Add kernel duration events
    current_ts = 0.0  # microseconds
    for record in records:
        event = {
            "name": record.name,
            "cat": "kernel",
            "ph": "X",  # Complete event (duration)
            "ts": current_ts,
            "dur": record.elapsed_us,
            "pid": 1,
            "tid": 1,
            "args": {
                "elapsed_ms": record.elapsed_ms,
            },
        }

        # Add optional metrics
        if record.flops is not None:
            event["args"]["flops"] = record.flops
            if record.tflops is not None:
                event["args"]["tflops"] = record.tflops

        if record.bytes_transferred is not None:
            event["args"]["bytes"] = record.bytes_transferred
            if record.bandwidth_gb_s is not None:
                event["args"]["bandwidth_gb_s"] = record.bandwidth_gb_s

        events.append(event)
        current_ts += record.elapsed_us

    # Add memory snapshot events as instant events
    if memory_snapshots:
        for snap in memory_snapshots:
            # Convert timestamp to microseconds relative to first snapshot
            if memory_snapshots:
                base_ts = memory_snapshots[0].timestamp
                ts_us = (snap.timestamp - base_ts) * 1_000_000
            else:
                ts_us = 0

            events.append({
                "name": snap.name,
                "cat": "memory",
                "ph": "i",  # Instant event
                "ts": ts_us,
                "pid": 1,
                "tid": 2,
                "s": "g",  # Global scope
                "args": {
                    "used_mb": snap.used_mb,
                    "cached_mb": snap.cached_mb,
                    "active_blocks": snap.active_blocks,
                    "reuse_rate": snap.reuse_rate,
                },
            })

        # Add memory thread metadata
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": 2,
            "args": {"name": "Memory"},
        })

    # Write trace file
    trace_data = {"traceEvents": events}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)


def export_combined_trace(
    profiler: Any,
    memory_profiler: Any,
    path: str,
    *,
    process_name: str = "PyGPUkit",
) -> None:
    """Export combined kernel and memory profiling data.

    Args:
        profiler: Profiler instance with kernel records.
        memory_profiler: MemoryProfiler instance with snapshots.
        path: Output file path.
        process_name: Name shown for the process.
    """
    from pygpukit.profiling.profiler import Profiler
    from pygpukit.profiling.memory import MemoryProfiler

    records = profiler.records if isinstance(profiler, Profiler) else []
    snapshots = (
        memory_profiler.snapshots
        if isinstance(memory_profiler, MemoryProfiler)
        else None
    )

    export_chrome_trace(
        records,
        path,
        memory_snapshots=snapshots,
        process_name=process_name,
    )
