"""Memory profiler for GPU memory analysis.

Tracks memory pool statistics, allocation patterns, and peak usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySnapshot:
    """Snapshot of memory pool state at a point in time."""

    name: str
    timestamp: float
    quota: int
    used: int
    cached: int
    available: int
    active_blocks: int
    free_blocks: int
    allocation_count: int
    reuse_count: int
    cudamalloc_count: int

    @property
    def used_mb(self) -> float:
        """Used memory in MB."""
        return self.used / (1024 * 1024)

    @property
    def cached_mb(self) -> float:
        """Cached memory in MB."""
        return self.cached / (1024 * 1024)

    @property
    def available_mb(self) -> float:
        """Available memory in MB."""
        return self.available / (1024 * 1024)

    @property
    def utilization(self) -> float:
        """Memory utilization as fraction (0.0 to 1.0)."""
        if self.quota == 0:
            return 0.0
        return self.used / self.quota

    @property
    def reuse_rate(self) -> float:
        """Block reuse rate (reuse / total allocations)."""
        total = self.reuse_count + self.cudamalloc_count
        if total == 0:
            return 0.0
        return self.reuse_count / total


def _get_pool_stats() -> dict[str, Any] | None:
    """Get current memory pool stats from Rust backend."""
    try:
        from pygpukit.memory import get_default_pool

        pool = get_default_pool()
        if pool is None:
            return None

        stats = pool.stats()
        return {
            "quota": stats.quota,
            "used": stats.used,
            "cached": stats.cached,
            "available": stats.available,
            "active_blocks": stats.active_blocks,
            "free_blocks": stats.free_blocks,
            "allocation_count": stats.allocation_count,
            "reuse_count": stats.reuse_count,
            "cudamalloc_count": stats.cudamalloc_count,
        }
    except (ImportError, AttributeError):
        return None


class MemoryProfiler:
    """GPU memory profiler tracking allocations and pool statistics.

    Example:
        >>> mem_prof = MemoryProfiler()
        >>>
        >>> mem_prof.snapshot("initial")
        >>> x = from_numpy(np.zeros((1024, 1024), dtype=np.float32))
        >>> mem_prof.snapshot("after_alloc")
        >>>
        >>> mem_prof.print_report()
        >>> mem_prof.print_diff("initial", "after_alloc")
    """

    def __init__(self) -> None:
        self._snapshots: list[MemorySnapshot] = []
        self._peak_used: int = 0

    def snapshot(self, name: str = "") -> MemorySnapshot | None:
        """Take a snapshot of current memory state.

        Args:
            name: Label for this snapshot.

        Returns:
            MemorySnapshot if pool stats available, None otherwise.
        """
        stats = _get_pool_stats()
        if stats is None:
            # Return a dummy snapshot for CPU-only mode
            snap = MemorySnapshot(
                name=name or f"snapshot_{len(self._snapshots)}",
                timestamp=time.time(),
                quota=0,
                used=0,
                cached=0,
                available=0,
                active_blocks=0,
                free_blocks=0,
                allocation_count=0,
                reuse_count=0,
                cudamalloc_count=0,
            )
            self._snapshots.append(snap)
            return snap

        snap = MemorySnapshot(
            name=name or f"snapshot_{len(self._snapshots)}",
            timestamp=time.time(),
            **stats,
        )
        self._snapshots.append(snap)

        # Track peak usage
        if snap.used > self._peak_used:
            self._peak_used = snap.used

        return snap

    @property
    def snapshots(self) -> list[MemorySnapshot]:
        """Get all recorded snapshots."""
        return self._snapshots.copy()

    @property
    def peak_used_bytes(self) -> int:
        """Peak memory usage in bytes."""
        return self._peak_used

    @property
    def peak_used_mb(self) -> float:
        """Peak memory usage in MB."""
        return self._peak_used / (1024 * 1024)

    def get_snapshot(self, name: str) -> MemorySnapshot | None:
        """Get a snapshot by name."""
        for snap in self._snapshots:
            if snap.name == name:
                return snap
        return None

    def diff(
        self, name1: str, name2: str
    ) -> dict[str, int | float] | None:
        """Calculate difference between two snapshots.

        Args:
            name1: Name of first (earlier) snapshot.
            name2: Name of second (later) snapshot.

        Returns:
            Dict with differences, or None if snapshots not found.
        """
        snap1 = self.get_snapshot(name1)
        snap2 = self.get_snapshot(name2)
        if snap1 is None or snap2 is None:
            return None

        return {
            "used_delta": snap2.used - snap1.used,
            "cached_delta": snap2.cached - snap1.cached,
            "active_blocks_delta": snap2.active_blocks - snap1.active_blocks,
            "free_blocks_delta": snap2.free_blocks - snap1.free_blocks,
            "allocation_delta": snap2.allocation_count - snap1.allocation_count,
            "time_delta": snap2.timestamp - snap1.timestamp,
        }

    def clear(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
        self._peak_used = 0

    def print_report(self) -> None:
        """Print a summary report of all snapshots."""
        if not self._snapshots:
            print("No snapshots recorded.")
            return

        print(f"\n{'='*70}")
        print("Memory Profiler Report")
        print(f"{'='*70}")
        print(f"Total snapshots: {len(self._snapshots)}")
        print(f"Peak memory used: {self.peak_used_mb:.2f} MB")
        print()

        print(
            f"{'Snapshot':<20} {'Used (MB)':>12} {'Cached (MB)':>12} "
            f"{'Active':>8} {'Reuse %':>10}"
        )
        print("-" * 70)

        for snap in self._snapshots:
            reuse_pct = snap.reuse_rate * 100
            print(
                f"{snap.name:<20} {snap.used_mb:>12.2f} {snap.cached_mb:>12.2f} "
                f"{snap.active_blocks:>8} {reuse_pct:>9.1f}%"
            )
        print()

    def print_diff(self, name1: str, name2: str) -> None:
        """Print difference between two snapshots."""
        diff_data = self.diff(name1, name2)
        if diff_data is None:
            print(f"Snapshots '{name1}' or '{name2}' not found.")
            return

        used_mb = diff_data["used_delta"] / (1024 * 1024)
        cached_mb = diff_data["cached_delta"] / (1024 * 1024)

        print(f"\nMemory diff: {name1} -> {name2}")
        print("-" * 40)
        print(f"Used:           {used_mb:+.2f} MB")
        print(f"Cached:         {cached_mb:+.2f} MB")
        print(f"Active blocks:  {diff_data['active_blocks_delta']:+d}")
        print(f"Free blocks:    {diff_data['free_blocks_delta']:+d}")
        print(f"Allocations:    {diff_data['allocation_delta']:+d}")
        print(f"Time elapsed:   {diff_data['time_delta']*1000:.2f} ms")
        print()


def get_current_memory_usage() -> dict[str, Any]:
    """Get current GPU memory usage.

    Returns:
        Dict with current memory statistics, or empty dict if unavailable.
    """
    stats = _get_pool_stats()
    if stats is None:
        return {}

    return {
        "used_bytes": stats["used"],
        "used_mb": stats["used"] / (1024 * 1024),
        "cached_bytes": stats["cached"],
        "cached_mb": stats["cached"] / (1024 * 1024),
        "available_bytes": stats["available"],
        "available_mb": stats["available"] / (1024 * 1024),
        "active_blocks": stats["active_blocks"],
        "free_blocks": stats["free_blocks"],
    }


def print_memory_summary() -> None:
    """Print current GPU memory summary."""
    stats = get_current_memory_usage()
    if not stats:
        print("Memory pool not available (GPU not initialized or CPU mode).")
        return

    print(f"\nGPU Memory Summary")
    print("-" * 40)
    print(f"Used:       {stats['used_mb']:.2f} MB")
    print(f"Cached:     {stats['cached_mb']:.2f} MB")
    print(f"Available:  {stats['available_mb']:.2f} MB")
    print(f"Active blocks: {stats['active_blocks']}")
    print(f"Free blocks:   {stats['free_blocks']}")
    print()
