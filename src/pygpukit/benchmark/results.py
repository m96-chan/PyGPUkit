"""Benchmark result classes and comparison utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    name: str
    category: str  # gemm, gemv, attention, inference
    params: dict[str, Any]  # M, K, N, dtype, etc.
    median_us: float  # Median time in microseconds
    min_us: float
    max_us: float
    std_us: float
    tflops: float | None = None  # For compute benchmarks
    bandwidth_gbps: float | None = None  # For memory benchmarks
    correct: bool = True
    rel_error: float = 0.0
    iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(**d)


@dataclass
class GPUInfo:
    """GPU information."""

    name: str
    sm_major: int
    sm_minor: int
    memory_gb: float
    driver_version: str = ""
    cuda_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report with multiple results."""

    gpu: GPUInfo
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "0.2.19"

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        data = {
            "version": self.version,
            "timestamp": self.timestamp,
            "gpu": self.gpu.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        """Load report from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        gpu = GPUInfo(**data["gpu"])
        results = [BenchmarkResult.from_dict(r) for r in data["results"]]
        return cls(
            gpu=gpu,
            results=results,
            timestamp=data.get("timestamp", ""),
            version=data.get("version", "unknown"),
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "gpu": self.gpu.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class Regression:
    """Regression information."""

    result: BenchmarkResult
    baseline: BenchmarkResult
    delta_percent: float  # Negative = regression


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark reports."""

    current: BenchmarkReport
    baseline: BenchmarkReport
    regressions: list[Regression] = field(default_factory=list)
    improvements: list[Regression] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # In baseline but not current
    new: list[str] = field(default_factory=list)  # In current but not baseline

    def has_regression(self, threshold: float = 0.05) -> bool:
        """Check if any regression exceeds threshold."""
        return any(r.delta_percent < -threshold * 100 for r in self.regressions)

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("Benchmark Comparison")
        lines.append("=" * 60)
        lines.append(f"Baseline: {self.baseline.timestamp}")
        lines.append(f"Current:  {self.current.timestamp}")
        lines.append("")

        if self.regressions:
            lines.append("REGRESSIONS:")
            for r in sorted(self.regressions, key=lambda x: x.delta_percent):
                lines.append(
                    f"  {r.result.name}: {r.baseline.median_us:.1f} -> "
                    f"{r.result.median_us:.1f} us ({r.delta_percent:+.1f}%)"
                )
            lines.append("")

        if self.improvements:
            lines.append("IMPROVEMENTS:")
            for r in sorted(self.improvements, key=lambda x: -x.delta_percent)[:5]:
                lines.append(
                    f"  {r.result.name}: {r.baseline.median_us:.1f} -> "
                    f"{r.result.median_us:.1f} us ({r.delta_percent:+.1f}%)"
                )
            lines.append("")

        return "\n".join(lines)


def compare_reports(
    current: BenchmarkReport,
    baseline: BenchmarkReport,
    threshold: float = 0.05,
) -> ComparisonResult:
    """Compare two benchmark reports."""
    result = ComparisonResult(current=current, baseline=baseline)

    # Build lookup by name+params
    def key(r: BenchmarkResult) -> str:
        params_str = json.dumps(r.params, sort_keys=True)
        return f"{r.category}:{r.name}:{params_str}"

    baseline_map = {key(r): r for r in baseline.results}
    current_map = {key(r): r for r in current.results}

    for k, curr in current_map.items():
        if k in baseline_map:
            base = baseline_map[k]
            if base.median_us > 0:
                delta = (curr.median_us - base.median_us) / base.median_us * 100
                reg = Regression(result=curr, baseline=base, delta_percent=-delta)
                if delta > threshold * 100:
                    result.regressions.append(reg)
                elif delta < -threshold * 100:
                    result.improvements.append(reg)
        else:
            result.new.append(k)

    for k in baseline_map:
        if k not in current_map:
            result.missing.append(k)

    return result
