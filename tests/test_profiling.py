"""Tests for the profiling module."""

import json
import os
import tempfile
import time

import numpy as np
import pytest

from pygpukit import from_numpy, profiling
from pygpukit.profiling import (
    KernelRecord,
    MemoryProfiler,
    MemorySnapshot,
    Profiler,
    ProfilerContext,
    export_chrome_trace,
)


class TestKernelRecord:
    """Test KernelRecord dataclass."""

    def test_basic_record(self):
        """Test basic record creation."""
        record = KernelRecord(
            name="test_kernel",
            elapsed_ms=1.5,
            elapsed_us=1500.0,
        )
        assert record.name == "test_kernel"
        assert record.elapsed_ms == 1.5
        assert record.elapsed_us == 1500.0
        assert record.flops is None
        assert record.bytes_transferred is None

    def test_tflops_calculation(self):
        """Test TFLOPS calculation."""
        # 2 TFLOPS = 2e12 ops in 1000ms = 2e9 ops/ms
        record = KernelRecord(
            name="matmul",
            elapsed_ms=1000.0,
            elapsed_us=1_000_000.0,
            flops=2_000_000_000_000,  # 2e12 flops
        )
        assert record.tflops == pytest.approx(2.0, rel=1e-6)

    def test_bandwidth_calculation(self):
        """Test bandwidth calculation."""
        # 100 GB/s = 100e9 bytes in 1000ms
        record = KernelRecord(
            name="copy",
            elapsed_ms=1000.0,
            elapsed_us=1_000_000.0,
            bytes_transferred=100_000_000_000,  # 100 GB
        )
        assert record.bandwidth_gb_s == pytest.approx(100.0, rel=1e-6)

    def test_none_metrics_when_zero_time(self):
        """Test that metrics are None when elapsed time is zero."""
        record = KernelRecord(
            name="test",
            elapsed_ms=0.0,
            elapsed_us=0.0,
            flops=1000,
            bytes_transferred=1000,
        )
        assert record.tflops is None
        assert record.bandwidth_gb_s is None


class TestProfilerContext:
    """Test ProfilerContext context manager."""

    def test_basic_timing(self):
        """Test that timing works."""
        with ProfilerContext("test") as ctx:
            # Do some actual work (not just sleep, which only affects CPU)
            _ = [i * i for i in range(10000)]

        # Timing should be non-negative (CUDA Events measure GPU time,
        # which may be very small if no GPU work is done)
        assert ctx.elapsed_ms >= 0
        assert ctx.elapsed_us >= 0

    def test_with_flops(self):
        """Test context with flops specified."""
        with ProfilerContext("matmul", flops=2_000_000_000) as ctx:
            time.sleep(0.001)  # 1ms

        assert ctx.flops == 2_000_000_000
        assert ctx.tflops is not None
        assert ctx.tflops > 0

    def test_to_record(self):
        """Test conversion to KernelRecord."""
        with ProfilerContext("test", flops=1000) as ctx:
            pass

        record = ctx.to_record()
        assert isinstance(record, KernelRecord)
        assert record.name == "test"
        assert record.flops == 1000


class TestProfiler:
    """Test Profiler class."""

    def test_record_operations(self):
        """Test recording multiple operations."""
        profiler = Profiler()

        with profiler.record("op1"):
            time.sleep(0.001)

        with profiler.record("op2"):
            time.sleep(0.001)

        assert len(profiler.records) == 2
        assert profiler.records[0].name == "op1"
        assert profiler.records[1].name == "op2"

    def test_total_time(self):
        """Test total time calculation."""
        profiler = Profiler()

        with profiler.record("op1"):
            _ = [i * i for i in range(10000)]

        with profiler.record("op2"):
            _ = [i * i for i in range(10000)]

        # Total time should be non-negative and sum of records
        assert profiler.total_time_ms >= 0
        assert len(profiler.records) == 2

    def test_summary_by_name(self):
        """Test summary grouped by kernel name."""
        profiler = Profiler()

        for _ in range(3):
            with profiler.record("kernel_a"):
                pass

        for _ in range(2):
            with profiler.record("kernel_b"):
                pass

        summary = profiler.summary_by_name()
        assert summary["kernel_a"]["count"] == 3
        assert summary["kernel_b"]["count"] == 2

    def test_clear(self):
        """Test clearing records."""
        profiler = Profiler()

        with profiler.record("test"):
            pass

        assert len(profiler.records) == 1

        profiler.clear()
        assert len(profiler.records) == 0

    def test_print_summary(self, capsys):
        """Test that print_summary works without errors."""
        profiler = Profiler()

        with profiler.record("test"):
            pass

        profiler.print_summary()
        captured = capsys.readouterr()
        assert "Profiler Summary" in captured.out
        assert "test" in captured.out


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_basic_snapshot(self):
        """Test basic snapshot creation."""
        snap = MemorySnapshot(
            name="test",
            timestamp=time.time(),
            quota=1_000_000_000,
            used=100_000_000,
            cached=50_000_000,
            available=850_000_000,
            active_blocks=10,
            free_blocks=5,
            allocation_count=100,
            reuse_count=80,
            cudamalloc_count=20,
        )

        assert snap.used_mb == pytest.approx(100_000_000 / (1024 * 1024))
        assert snap.utilization == pytest.approx(0.1)
        assert snap.reuse_rate == pytest.approx(0.8)


class TestMemoryProfiler:
    """Test MemoryProfiler class."""

    def test_snapshot(self):
        """Test taking memory snapshots."""
        mem_prof = MemoryProfiler()

        snap1 = mem_prof.snapshot("initial")
        assert snap1 is not None
        assert snap1.name == "initial"

        snap2 = mem_prof.snapshot("after")
        assert snap2 is not None
        assert snap2.name == "after"

        assert len(mem_prof.snapshots) == 2

    def test_get_snapshot(self):
        """Test retrieving snapshots by name."""
        mem_prof = MemoryProfiler()

        mem_prof.snapshot("test1")
        mem_prof.snapshot("test2")

        snap = mem_prof.get_snapshot("test1")
        assert snap is not None
        assert snap.name == "test1"

        assert mem_prof.get_snapshot("nonexistent") is None

    def test_diff(self):
        """Test calculating difference between snapshots."""
        mem_prof = MemoryProfiler()

        mem_prof.snapshot("before")
        time.sleep(0.01)
        mem_prof.snapshot("after")

        diff = mem_prof.diff("before", "after")
        assert diff is not None
        assert "time_delta" in diff
        assert diff["time_delta"] > 0

    def test_clear(self):
        """Test clearing snapshots."""
        mem_prof = MemoryProfiler()
        mem_prof.snapshot("test")
        assert len(mem_prof.snapshots) == 1

        mem_prof.clear()
        assert len(mem_prof.snapshots) == 0

    def test_print_report(self, capsys):
        """Test that print_report works without errors."""
        mem_prof = MemoryProfiler()
        mem_prof.snapshot("test")

        mem_prof.print_report()
        captured = capsys.readouterr()
        assert "Memory Profiler Report" in captured.out


class TestChromeTrace:
    """Test Chrome trace export."""

    def test_export_basic(self):
        """Test basic trace export."""
        records = [
            KernelRecord(
                name="kernel1",
                elapsed_ms=1.0,
                elapsed_us=1000.0,
                flops=1_000_000,
            ),
            KernelRecord(
                name="kernel2",
                elapsed_ms=2.0,
                elapsed_us=2000.0,
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            export_chrome_trace(records, path)

            with open(path, "r") as f:
                data = json.load(f)

            assert "traceEvents" in data
            events = data["traceEvents"]

            # Should have metadata events + kernel events
            assert len(events) >= 4  # 2 metadata + 2 kernels

            # Find kernel events
            kernel_events = [e for e in events if e.get("cat") == "kernel"]
            assert len(kernel_events) == 2
            assert kernel_events[0]["name"] == "kernel1"
            assert kernel_events[1]["name"] == "kernel2"

        finally:
            os.unlink(path)

    def test_export_with_memory(self):
        """Test trace export with memory snapshots."""
        records = [
            KernelRecord(name="kernel", elapsed_ms=1.0, elapsed_us=1000.0),
        ]

        base_time = time.time()
        snapshots = [
            MemorySnapshot(
                name="snap1",
                timestamp=base_time,
                quota=1000,
                used=100,
                cached=50,
                available=850,
                active_blocks=1,
                free_blocks=0,
                allocation_count=1,
                reuse_count=0,
                cudamalloc_count=1,
            ),
            MemorySnapshot(
                name="snap2",
                timestamp=base_time + 0.001,
                quota=1000,
                used=200,
                cached=50,
                available=750,
                active_blocks=2,
                free_blocks=0,
                allocation_count=2,
                reuse_count=0,
                cudamalloc_count=2,
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            export_chrome_trace(records, path, memory_snapshots=snapshots)

            with open(path, "r") as f:
                data = json.load(f)

            events = data["traceEvents"]

            # Should have memory events
            memory_events = [e for e in events if e.get("cat") == "memory"]
            assert len(memory_events) == 2

        finally:
            os.unlink(path)


class TestProfilerIntegration:
    """Integration tests with actual GPU arrays."""

    def test_profile_with_gpu_array(self):
        """Test profiling with actual GPU array operations."""
        profiler = Profiler()

        x = from_numpy(np.random.randn(100, 100).astype(np.float32))
        y = from_numpy(np.random.randn(100, 100).astype(np.float32))

        with profiler.record("add"):
            z = x + y

        assert len(profiler.records) == 1
        assert profiler.records[0].name == "add"
        assert profiler.records[0].elapsed_ms >= 0

    def test_memory_profiler_with_allocation(self):
        """Test memory profiler with GPU allocation."""
        mem_prof = MemoryProfiler()

        mem_prof.snapshot("before")

        # Allocate some memory
        x = from_numpy(np.zeros((1024, 1024), dtype=np.float32))

        mem_prof.snapshot("after")

        # The snapshots should be recorded
        assert len(mem_prof.snapshots) == 2


class TestModuleExports:
    """Test that module exports are correct."""

    def test_profiling_module_import(self):
        """Test that profiling module is importable."""
        from pygpukit import profiling

        assert hasattr(profiling, "Profiler")
        assert hasattr(profiling, "MemoryProfiler")
        assert hasattr(profiling, "export_chrome_trace")

    def test_direct_imports(self):
        """Test direct imports from profiling submodule."""
        from pygpukit.profiling import (
            KernelRecord,
            MemoryProfiler,
            MemorySnapshot,
            Profiler,
            ProfilerContext,
            export_chrome_trace,
        )

        assert Profiler is not None
        assert MemoryProfiler is not None
        assert export_chrome_trace is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
