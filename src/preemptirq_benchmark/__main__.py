from __future__ import annotations

import argparse
import subprocess
import sys

from preemptirq_benchmark.benchmarks import (
    BenchmarkResult,
    check_all_prerequisites,
    get_benchmark,
    import_all,
    resolve_benchmarks,
)
from preemptirq_benchmark.compare import compare_reports
from preemptirq_benchmark.perf_stat import is_available as perf_available
from preemptirq_benchmark.perf_stat import run_with_perf_stat
from preemptirq_benchmark.report import (
    build_report,
    display_report,
    load_report,
    save_report,
)

FORMAT_CHOICES = ["ascii", "txt", "markdown", "json"]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for preemptirq-benchmark.

    Args:
        argv: Command-line arguments.  Defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        prog="preemptirq-benchmark",
        description="Benchmark suite for Linux kernel preemptirq tracepoint overhead",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_run_parser(subparsers)
    add_show_parser(subparsers)
    add_compare_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "compare":
        cmd_compare(args)


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'run' subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    run = subparsers.add_parser("run", help="Run benchmarks")

    run.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override default iteration count for all benchmarks",
    )
    run.add_argument(
        "--include",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to run",
    )
    run.add_argument(
        "--all",
        action="store_true",
        dest="all_flag",
        help="Run all benchmarks (default)",
    )
    run.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to exclude",
    )
    run.add_argument("--perf-stat", action="store_true", help="Wrap benchmarks with perf stat")
    run.add_argument(
        "--kernel-src",
        type=str,
        default=None,
        help="Kernel source tree for kernel-compile benchmark",
    )
    run.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )

    run.add_argument("--samples", type=int, default=None, help="tracerbench: nr_samples")
    run.add_argument("--highest", type=int, default=None, help="tracerbench: nr_highest")
    run.add_argument(
        "--percentile",
        type=int,
        default=None,
        help="tracerbench: percentile to compute",
    )


def add_show_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'show' subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    show = subparsers.add_parser("show", help="Display a saved report")
    show.add_argument("report", help="Path to JSON report file")
    show.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="ascii",
        dest="fmt",
        help="Output format (default: ascii)",
    )


def add_compare_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'compare' subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    cmp = subparsers.add_parser("compare", help="Compare reports")
    cmp.add_argument("reports", nargs="+", help="JSON report files (first = base)")
    cmp.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="ascii",
        dest="fmt",
        help="Output format (default: ascii)",
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    import_all()

    names = resolve_benchmarks(args.include, args.exclude, args.all_flag)

    if "kernel-compile" in names and args.kernel_src is None:
        if args.include and "kernel-compile" in [b.strip() for b in args.include.split(",")]:
            print(
                "Error: --kernel-src is required when kernel-compile is included",
                file=sys.stderr,
            )
            raise SystemExit(1)
        else:
            names.remove("kernel-compile")
            print(
                "Warning: Skipping kernel-compile because --kernel-src was not provided",
                file=sys.stderr,
            )

    benchmarks = []
    for name in names:
        bench = get_benchmark(name)
        bench.configure(
            kernel_src=args.kernel_src,
            nr_samples=args.samples,
            nr_highest=args.highest,
            percentile=args.percentile,
        )
        benchmarks.append(bench)

    check_all_prerequisites(benchmarks)

    use_perf = args.perf_stat
    if use_perf and not perf_available():
        print("Warning: perf not found, running without perf stat", file=sys.stderr)
        use_perf = False

    results: list[BenchmarkResult] = []
    total = len(benchmarks)

    for idx, bench in enumerate(benchmarks, 1):
        iters = args.iterations if args.iterations is not None else bench.default_iterations
        result = BenchmarkResult(
            name=bench.name,
            units=bench.get_units(),
        )

        bench.setup()
        try:
            for i in range(iters):
                try:
                    metrics = bench.run_once()

                    for mname, mval in metrics.items():
                        result.metrics.setdefault(mname, []).append(mval)
                    result.iterations += 1
                except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
                    print(
                        f"\nWarning: Iteration {i+1} of {bench.name} failed: {e}",
                        file=sys.stderr,
                    )

                print_progress(bench.name, i + 1, iters, idx, total)

            if use_perf and bench.supports_perf_stat and result.iterations > 0:
                cmd = bench.get_command()
                if cmd:
                    print(f"  Collecting perf stat for {bench.name}...")
                    _, counters = run_with_perf_stat(cmd)
                    for cname, cval in counters.items():
                        result.perf_counters[cname] = [cval]
        finally:
            bench.cleanup()

        if result.iterations == 0:
            print(
                f"\nWarning: all iterations of {bench.name} failed, skipping",
                file=sys.stderr,
            )
            continue

        results.append(result)

    print()

    tracerbench_config = None
    if "tracerbench" in names:
        tracerbench_config = {}
        if args.samples is not None:
            tracerbench_config["nr_samples"] = args.samples
        if args.highest is not None:
            tracerbench_config["nr_highest"] = args.highest
        if args.percentile is not None:
            tracerbench_config["percentile_nth"] = args.percentile

    report = build_report(results, tracerbench_config)
    path = save_report(report, args.output)
    print(f"Report saved to: {path}")
    print()

    display_report(report, "ascii")


def cmd_show(args: argparse.Namespace) -> None:
    """Execute the 'show' subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    report = load_report(args.report)
    display_report(report, args.fmt)


def cmd_compare(args: argparse.Namespace) -> None:
    """Execute the 'compare' subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    compare_reports(args.reports, args.fmt)


def print_progress(
    name: str,
    current: int,
    total_iters: int,
    bench_idx: int,
    bench_total: int,
) -> None:
    """Print a progress line for the current benchmark iteration.

    Args:
        name: Benchmark name.
        current: Current iteration number (1-based).
        total_iters: Total iterations for this benchmark.
        bench_idx: Current benchmark index (1-based).
        bench_total: Total number of benchmarks.
    """
    bar_width = min(total_iters, 40)
    filled = int(bar_width * current / total_iters)
    bar = "." * filled + " " * (bar_width - filled)
    print(
        f"\r[{bench_idx}/{bench_total}] {name} {bar} {current}/{total_iters}",
        end="",
        flush=True,
    )
    if current == total_iters:
        print()


if __name__ == "__main__":
    main()
