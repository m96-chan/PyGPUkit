"""CLI interface for benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import BenchmarkReport, BenchmarkSuite


def main() -> int:
    """Main entry point for benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="PyGPUkit Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pygpukit.benchmark                    # Run default benchmarks
  python -m pygpukit.benchmark --quick            # Quick benchmarks
  python -m pygpukit.benchmark --save results.json
  python -m pygpukit.benchmark --compare baseline.json
  python -m pygpukit.benchmark --gemm --sizes 4096,8192
  python -m pygpukit.benchmark --gemv --dtypes bf16,fp8
""",
    )

    # Output options
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--compare",
        type=str,
        metavar="FILE",
        help="Compare with baseline JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression threshold (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regression detected",
    )

    # Benchmark selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--gemm",
        action="store_true",
        help="Run GEMM benchmarks",
    )
    parser.add_argument(
        "--gemv",
        action="store_true",
        help="Run GEMV benchmarks",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Run attention benchmarks",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Include FP8 benchmarks (SM120+)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--sizes",
        type=str,
        help="GEMM sizes: comma-separated (e.g., 2048,4096,8192)",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        help="Dtypes: comma-separated (e.g., fp32,tf32,bf16)",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        help="Attention seq lengths: comma-separated (e.g., 512,1024,2048)",
    )

    # Performance options
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Benchmark iterations (default: 50)",
    )

    # Output format
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output results as markdown table",
    )

    args = parser.parse_args()

    # Create suite
    suite = BenchmarkSuite(
        warmup=args.warmup,
        iterations=args.iterations,
        quick=args.quick,
    )

    # Parse sizes
    sizes = None
    if args.sizes:
        size_list = [int(s.strip()) for s in args.sizes.split(",")]
        sizes = [(s, s, s) for s in size_list]  # Square matrices

    # Parse dtypes
    dtypes = None
    if args.dtypes:
        dtypes = [d.strip() for d in args.dtypes.split(",")]

    # Parse seq lens
    seq_lens = None
    if args.seq_lens:
        seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]

    # Add benchmarks
    if args.all:
        suite.add_all()
        if args.fp8:
            suite.add_fp8_gemm()
            suite.add_w8a8_gemv()
    else:
        # Default: add gemm and gemv if nothing specified
        has_selection = args.gemm or args.gemv or args.attention
        if not has_selection:
            suite.add_gemm(sizes=sizes, dtypes=dtypes)
            suite.add_gemv(dtypes=dtypes)
        else:
            if args.gemm:
                suite.add_gemm(sizes=sizes, dtypes=dtypes)
                if args.fp8:
                    suite.add_fp8_gemm(sizes=sizes)
            if args.gemv:
                suite.add_gemv(dtypes=dtypes)
                if args.fp8:
                    suite.add_w8a8_gemv()
            if args.attention:
                suite.add_attention(seq_lens=seq_lens)
                suite.add_gqa(seq_lens=seq_lens)

    # Run benchmarks
    verbose = not args.quiet
    if args.compare:
        comparison = suite.compare(
            args.compare,
            threshold=args.threshold,
            verbose=verbose,
        )
        if args.fail_on_regression and comparison.has_regression(args.threshold):
            print("\nERROR: Performance regression detected!")
            return 1
        report = comparison.current
    else:
        report = suite.run(verbose=verbose)

    # Save results
    if args.save:
        report.save(args.save)
        if verbose:
            print(f"Results saved to {args.save}")

    # Print markdown table
    if args.markdown:
        print_markdown_table(report)

    return 0


def print_markdown_table(report: BenchmarkReport) -> None:
    """Print results as markdown table."""

    print("\n## Benchmark Results\n")
    print(f"GPU: {report.gpu.name}")
    print(f"SM: {report.gpu.sm_major}.{report.gpu.sm_minor}")
    print()

    # Group by category
    by_category: dict[str, list] = {}
    for r in report.results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)

    for category, results in by_category.items():
        print(f"### {category.upper()}\n")
        print("| Name | Time (us) | TFLOPS | Correct |")
        print("|------|-----------|--------|---------|")
        for r in results:
            tflops = f"{r.tflops:.1f}" if r.tflops else "-"
            correct = "Yes" if r.correct else "No"
            print(f"| {r.name} | {r.median_us:.1f} | {tflops} | {correct} |")
        print()


if __name__ == "__main__":
    sys.exit(main())
