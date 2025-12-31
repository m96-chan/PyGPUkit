"""CUDA Event-based kernel profiler.

Provides high-precision GPU timing using CUDA Events API.
When the native module is available, uses C++ ScopedTimer for
accurate timing with minimal Python overhead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray

# Try to import native module for CUDA Events and Profiler
_native_module: Any = None
_has_native_profiler: bool = False


def _get_native() -> Any:
    """Get native module with CUDA Event support."""
    global _native_module, _has_native_profiler
    if _native_module is None:
        try:
            from pygpukit.core.backend import get_native_module

            _native_module = get_native_module()
            # Check if native profiler is available
            _has_native_profiler = hasattr(_native_module, "ScopedTimer")
        except ImportError:
            _native_module = False
            _has_native_profiler = False
    return _native_module if _native_module else None


def _has_native() -> bool:
    """Check if native profiler is available."""
    _get_native()  # Ensure initialized
    return _has_native_profiler


@dataclass
class KernelRecord:
    """Record of a single kernel execution."""

    name: str
    elapsed_ms: float
    elapsed_us: float
    flops: int | None = None
    bytes_transferred: int | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def tflops(self) -> float | None:
        """Calculate TFLOPS if flops is set."""
        if self.flops is None or self.elapsed_ms <= 0:
            return None
        return (self.flops / 1e12) / (self.elapsed_ms / 1000)

    @property
    def bandwidth_gb_s(self) -> float | None:
        """Calculate bandwidth in GB/s if bytes_transferred is set."""
        if self.bytes_transferred is None or self.elapsed_ms <= 0:
            return None
        return (self.bytes_transferred / 1e9) / (self.elapsed_ms / 1000)

    @classmethod
    def from_native(cls, native_record: Any) -> KernelRecord:
        """Create from native KernelRecord."""
        return cls(
            name=native_record.name,
            elapsed_ms=native_record.elapsed_ms,
            elapsed_us=native_record.elapsed_us,
            flops=native_record.flops if native_record.flops >= 0 else None,
            bytes_transferred=native_record.bytes if native_record.bytes >= 0 else None,
            timestamp=native_record.timestamp,
        )


class ProfilerContext:
    """Context manager for profiling a single operation.

    When native module is available, uses C++ ScopedTimer for
    accurate GPU timing with minimal Python overhead.

    Example:
        >>> with ProfilerContext("matmul") as ctx:
        ...     result = matmul(A, B)
        >>> print(f"Elapsed: {ctx.elapsed_ms:.3f} ms")
    """

    def __init__(
        self,
        name: str = "kernel",
        *,
        flops: int | None = None,
        bytes_transferred: int | None = None,
    ) -> None:
        self.name = name
        self.flops = flops
        self.bytes_transferred = bytes_transferred
        self._native_timer: Any = None
        self._start_event: Any = None
        self._stop_event: Any = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._elapsed_ms: float | None = None
        self._elapsed_us: float | None = None

    def __enter__(self) -> ProfilerContext:
        native = _get_native()
        if native is not None and _has_native():
            # Use native ScopedTimer for accurate timing
            flops_val = self.flops if self.flops is not None else -1
            bytes_val = self.bytes_transferred if self.bytes_transferred is not None else -1
            self._native_timer = native.ScopedTimer(self.name, flops_val, bytes_val)
        elif native is not None:
            # Fallback to CudaEvent (old behavior)
            self._start_event = native.CudaEvent()
            self._stop_event = native.CudaEvent()
            self._start_event.record()
        else:
            # CPU fallback
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._native_timer is not None:
            # Native timer stops and syncs
            self._native_timer.stop()
            self._elapsed_ms = self._native_timer.elapsed_ms()
            self._elapsed_us = self._native_timer.elapsed_us()
        elif self._stop_event is not None:
            native = _get_native()
            self._stop_event.record()
            self._stop_event.synchronize()
            self._elapsed_ms = native.event_elapsed_ms(
                self._start_event, self._stop_event
            )
            self._elapsed_us = native.event_elapsed_us(
                self._start_event, self._stop_event
            )
        else:
            self._end_time = time.perf_counter()
            elapsed_sec = self._end_time - self._start_time
            self._elapsed_ms = elapsed_sec * 1000
            self._elapsed_us = elapsed_sec * 1_000_000

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self._elapsed_ms if self._elapsed_ms is not None else 0.0

    @property
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self._elapsed_us if self._elapsed_us is not None else 0.0

    @property
    def tflops(self) -> float | None:
        """Calculate TFLOPS if flops was specified."""
        if self.flops is None or self.elapsed_ms <= 0:
            return None
        return (self.flops / 1e12) / (self.elapsed_ms / 1000)

    @property
    def bandwidth_gb_s(self) -> float | None:
        """Calculate bandwidth in GB/s if bytes_transferred was specified."""
        if self.bytes_transferred is None or self.elapsed_ms <= 0:
            return None
        return (self.bytes_transferred / 1e9) / (self.elapsed_ms / 1000)

    def to_record(self) -> KernelRecord:
        """Convert to KernelRecord."""
        return KernelRecord(
            name=self.name,
            elapsed_ms=self.elapsed_ms,
            elapsed_us=self.elapsed_us,
            flops=self.flops,
            bytes_transferred=self.bytes_transferred,
        )


class Profiler:
    """GPU kernel profiler using CUDA Events.

    When native module is available, uses C++ KernelProfiler for
    accurate timing with minimal overhead.

    Example:
        >>> profiler = Profiler()
        >>>
        >>> # Profile individual operations
        >>> with profiler.record("matmul", flops=2*M*N*K):
        ...     C = matmul(A, B)
        >>>
        >>> with profiler.record("softmax"):
        ...     out = softmax(C)
        >>>
        >>> # Print summary
        >>> profiler.print_summary()
        >>>
        >>> # Export to Chrome trace
        >>> profiler.export_chrome_trace("profile.json")
    """

    def __init__(self, *, use_native: bool = True) -> None:
        """Create a kernel profiler.

        Args:
            use_native: If True and native module available, use C++ profiler.
        """
        self._native_profiler: Any = None
        self._records: list[KernelRecord] = []
        self._active_context: ProfilerContext | None = None

        # Try to use native profiler
        if use_native:
            native = _get_native()
            if native is not None and hasattr(native, "KernelProfiler"):
                self._native_profiler = native.KernelProfiler()

    @property
    def using_native(self) -> bool:
        """Check if using native C++ profiler."""
        return self._native_profiler is not None

    def record(
        self,
        name: str = "kernel",
        *,
        flops: int | None = None,
        bytes_transferred: int | None = None,
    ) -> _RecordingContext:
        """Create a profiling context for an operation.

        Args:
            name: Name of the operation being profiled.
            flops: Number of floating-point operations (for TFLOPS calculation).
            bytes_transferred: Bytes transferred (for bandwidth calculation).

        Returns:
            A context manager that profiles the enclosed code.
        """
        ctx = ProfilerContext(
            name, flops=flops, bytes_transferred=bytes_transferred
        )
        self._active_context = ctx
        return _RecordingContext(self, ctx)

    def _add_record(self, record: KernelRecord) -> None:
        """Add a kernel record to the profiler."""
        if self._native_profiler is not None:
            # Convert to native record and add
            native = _get_native()
            native_record = native.KernelRecord()
            native_record.name = record.name
            native_record.elapsed_ms = record.elapsed_ms
            native_record.elapsed_us = record.elapsed_us
            native_record.flops = record.flops if record.flops is not None else -1
            native_record.bytes = (
                record.bytes_transferred if record.bytes_transferred is not None else -1
            )
            native_record.timestamp = record.timestamp
            self._native_profiler.add_record(native_record)
        else:
            self._records.append(record)

    @property
    def records(self) -> list[KernelRecord]:
        """Get all recorded kernel executions."""
        if self._native_profiler is not None:
            return [
                KernelRecord.from_native(r)
                for r in self._native_profiler.records()
            ]
        return self._records.copy()

    def clear(self) -> None:
        """Clear all recorded data."""
        if self._native_profiler is not None:
            self._native_profiler.clear()
        else:
            self._records.clear()

    @property
    def total_time_ms(self) -> float:
        """Total profiled time in milliseconds."""
        if self._native_profiler is not None:
            return self._native_profiler.total_time_ms()
        return sum(r.elapsed_ms for r in self._records)

    def summary_by_name(self) -> dict[str, dict[str, float]]:
        """Get summary statistics grouped by kernel name.

        Returns:
            Dict mapping kernel name to stats (count, total_ms, avg_ms, min_ms, max_ms).
        """
        if self._native_profiler is not None:
            # Use native summary
            native_summary = self._native_profiler.summary_by_name()
            return {
                s["name"]: {
                    "count": s["count"],
                    "total_ms": s["total_ms"],
                    "avg_ms": s["avg_ms"],
                    "min_ms": s["min_ms"],
                    "max_ms": s["max_ms"],
                }
                for s in native_summary
            }

        from collections import defaultdict

        by_name: dict[str, list[float]] = defaultdict(list)
        for r in self._records:
            by_name[r.name].append(r.elapsed_ms)

        result = {}
        for name, times in by_name.items():
            result[name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
        return result

    def print_summary(self) -> None:
        """Print a summary of all profiled operations."""
        records = self.records
        if not records:
            print("No records to summarize.")
            return

        print(f"\n{'='*60}")
        print("Profiler Summary")
        if self.using_native:
            print("(using native C++ profiler)")
        print(f"{'='*60}")
        print(f"Total records: {len(records)}")
        print(f"Total time: {self.total_time_ms:.3f} ms")
        print()

        summary = self.summary_by_name()
        print(f"{'Kernel':<30} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>12}")
        print("-" * 62)
        for name, stats in sorted(
            summary.items(), key=lambda x: x[1]["total_ms"], reverse=True
        ):
            print(
                f"{name:<30} {stats['count']:>8} "
                f"{stats['total_ms']:>12.3f} {stats['avg_ms']:>12.3f}"
            )
        print()

    def export_chrome_trace(self, path: str) -> None:
        """Export profiling data to Chrome trace format.

        Args:
            path: Output file path (usually .json extension).
        """
        from pygpukit.profiling.trace import export_chrome_trace

        export_chrome_trace(self.records, path)


class _RecordingContext:
    """Internal wrapper that adds record to profiler on exit."""

    def __init__(self, profiler: Profiler, ctx: ProfilerContext) -> None:
        self._profiler = profiler
        self._ctx = ctx

    def __enter__(self) -> ProfilerContext:
        self._ctx.__enter__()
        return self._ctx

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._ctx.__exit__(exc_type, exc_val, exc_tb)
        self._profiler._add_record(self._ctx.to_record())


def profile_matmul(
    M: int,
    N: int,
    K: int,
    A: GPUArray,
    B: GPUArray,
    matmul_fn: Any,
    warmup: int = 3,
    iterations: int = 10,
) -> tuple[GPUArray, KernelRecord]:
    """Profile a matrix multiplication operation.

    Args:
        M, N, K: Matrix dimensions (A is MxK, B is KxN, C is MxN).
        A, B: Input matrices.
        matmul_fn: Function that performs matmul(A, B) -> C.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Tuple of (result, KernelRecord with average timing).
    """
    # Warmup
    for _ in range(warmup):
        result = matmul_fn(A, B)

    # Synchronize before timing
    native = _get_native()
    if native is not None:
        native.synchronize()

    # Timed iterations
    total_ms = 0.0
    for _ in range(iterations):
        with ProfilerContext() as ctx:
            result = matmul_fn(A, B)
        total_ms += ctx.elapsed_ms

    avg_ms = total_ms / iterations
    flops = 2 * M * N * K  # FMA = 2 ops per element

    record = KernelRecord(
        name="matmul",
        elapsed_ms=avg_ms,
        elapsed_us=avg_ms * 1000,
        flops=flops,
    )

    return result, record


def get_global_profiler() -> Any:
    """Get the global C++ kernel profiler instance.

    Returns None if native module is not available.
    """
    native = _get_native()
    if native is not None and hasattr(native, "get_global_profiler"):
        return native.get_global_profiler()
    return None
