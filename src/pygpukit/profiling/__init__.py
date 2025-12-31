"""GPU kernel profiling and memory analysis tools.

This module provides:
- Profiler: CUDA Event-based kernel timing and TFLOPS calculation
- MemoryProfiler: Memory pool statistics and allocation tracking
- Chrome trace export for timeline visualization

Example:
    >>> from pygpukit.profiling import Profiler, MemoryProfiler
    >>>
    >>> # Kernel timing
    >>> with Profiler() as prof:
    ...     result = matmul(A, B)
    >>> print(f"Time: {prof.elapsed_ms:.3f} ms, TFLOPS: {prof.tflops:.2f}")
    >>>
    >>> # Memory analysis
    >>> mem_prof = MemoryProfiler()
    >>> mem_prof.snapshot("before_forward")
    >>> output = model.forward(input)
    >>> mem_prof.snapshot("after_forward")
    >>> mem_prof.print_report()
"""

from __future__ import annotations

from pygpukit.profiling.memory import MemoryProfiler, MemorySnapshot
from pygpukit.profiling.profiler import (
    KernelRecord,
    Profiler,
    ProfilerContext,
)
from pygpukit.profiling.trace import export_chrome_trace

__all__ = [
    "Profiler",
    "ProfilerContext",
    "KernelRecord",
    "MemoryProfiler",
    "MemorySnapshot",
    "export_chrome_trace",
]
