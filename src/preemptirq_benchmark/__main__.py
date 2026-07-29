from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from pathlib import Path
from typing import cast

from preemptirq_benchmark.benchmarks import (
    ALL_BENCHMARK_NAMES,
    BENCHMARK_DESCRIPTIONS,
    REGISTRY,
    BenchmarkResult,
    check_all_prerequisites,
    get_benchmark,
    import_all,
    resolve_benchmarks,
)
from preemptirq_benchmark.compare import (
    compare_reports,
    display_comparison_data,
    is_comparison_data,
)
from preemptirq_benchmark.formatters import format_table
from preemptirq_benchmark.perf_stat import DEFAULT_EVENTS
from preemptirq_benchmark.perf_stat import is_available as perf_available
from preemptirq_benchmark.perf_stat import run_with_perf_stat
from preemptirq_benchmark.report import (
    build_report,
    display_report,
    load_report,
    save_report,
)
from preemptirq_benchmark.types import Report

FORMAT_CHOICES = ["ascii", "txt", "markdown", "json"]

EXT_TO_FORMAT: dict[str, str] = {
    ".ascii": "ascii",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
}


def infer_format(output: str) -> str:
    """Infer output format from a file extension.

    Args:
        output: Output file path.

    Returns:
        A format string from FORMAT_CHOICES, or "ascii" if the
        extension is not recognized.
    """
    ext = Path(output).suffix.lower()
    return EXT_TO_FORMAT.get(ext, "ascii")


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
    add_list_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list":
        cmd_list(args)


def _ci_percentage(value: str) -> float:
    f = float(value)
    if not (0 < f < 100):
        raise argparse.ArgumentTypeError(f"must be between 0 and 100 exclusive, got {f}")
    return f


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
        "--perf-stat-events",
        type=str,
        default=None,
        help="Comma-separated perf events to add to the defaults",
    )
    run.add_argument(
        "--kernel-src",
        type=str,
        default=None,
        help="Kernel source tree for kernel-compile benchmark",
    )
    run.add_argument(
        "--bpf-bench",
        type=str,
        default=None,
        help="Path to BPF bench binary (default: 'bench' from $PATH)",
    )
    run.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )

    run.add_argument(
        "--confidence-interval",
        type=_ci_percentage,
        default=95.0,
        help="Confidence interval percentage, 0 < ci < 100 (default: 95)",
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
        default=None,
        dest="fmt",
        help="Output format (default: ascii, or inferred from -o extension)",
    )
    show.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write output to file (format inferred from extension if --format not given)",
    )
    show.add_argument(
        "--tracerbench-exclude-stats",
        type=str,
        default=None,
        help="Comma-separated tracerbench statistics to exclude (e.g., 'median,max')",
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
        default=None,
        dest="fmt",
        help="Output format (default: ascii, or inferred from -o extension)",
    )
    cmp.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write output to file (format inferred from extension if --format not given)",
    )
    cmp.add_argument(
        "--tracerbench-exclude-stats",
        type=str,
        default=None,
        help="Comma-separated tracerbench statistics to exclude (e.g., 'median,max')",
    )


def add_list_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'list' subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    subparsers.add_parser("list", help="List available benchmarks")


def cmd_list(_args: argparse.Namespace) -> None:
    """Execute the 'list' subcommand.

    Args:
        _args: Parsed arguments from argparse (unused).
    """
    import_all()

    headers = ["Name", "Description", "Default Iterations"]
    rows = [
        [name, BENCHMARK_DESCRIPTIONS.get(name, ""), str(REGISTRY[name].default_iterations)]
        for name in ALL_BENCHMARK_NAMES
    ]
    print(format_table("Available benchmarks", headers, rows, "ascii"))


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    import_all()

    names = resolve_benchmarks(args.include, args.exclude, args.all_flag)

    benchmarks = []
    for name in names:
        bench = get_benchmark(name)
        bench.configure(
            kernel_src=args.kernel_src,
            bpf_bench=args.bpf_bench,
            nr_samples=args.samples,
            nr_highest=args.highest,
            percentile=args.percentile,
        )
        benchmarks.append(bench)

    check_all_prerequisites(benchmarks)

    use_perf = args.perf_stat or bool(args.perf_stat_events)
    if use_perf and not perf_available():
        print("Warning: perf not found, running without perf stat", file=sys.stderr)
        use_perf = False

    extra_perf_events: list[str] = []
    if args.perf_stat_events:
        extra_perf_events = [e.strip() for e in args.perf_stat_events.split(",") if e.strip()]

    # Use dict.fromkeys to deduplicate the event list while preserving insertion order
    perf_events = list(dict.fromkeys(list(DEFAULT_EVENTS) + extra_perf_events))

    results: list[BenchmarkResult] = []
    total = len(benchmarks)

    for idx, bench in enumerate(benchmarks, 1):
        iters = args.iterations if args.iterations is not None else bench.default_iterations
        result = BenchmarkResult(
            name=bench.name,
            units=bench.get_units(),
        )

        try:
            bench.setup()
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
                    _, counters = run_with_perf_stat(cmd, events=perf_events)
                    for cname, cval in counters.items():
                        result.perf_counters[cname] = [cval]
        except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
            print(f"\nWarning: {bench.name} failed: {e}", file=sys.stderr)
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

    report = build_report(results, tracerbench_config, ci_pct=args.confidence_interval)
    path = save_report(report, args.output)
    print(f"Report saved to: {path}")
    print()

    display_report(report, "ascii")


def resolve_output_format(args: argparse.Namespace) -> str:
    """Resolve the output format from --format and -o flags.

    Precedence: --format flag wins; otherwise infer from -o file
    extension; otherwise fall back to "ascii".

    Args:
        args: Parsed arguments with ``fmt`` and ``output`` attributes.

    Returns:
        A format string from FORMAT_CHOICES.
    """
    if args.fmt is not None:
        return args.fmt
    if args.output is not None:
        return infer_format(args.output)
    return "ascii"


@contextlib.contextmanager
def managed_output(output_path: str | None):
    """Context manager for file output redirection.

    Args:
        output_path: Path to the output file. If None, yields without redirection.
    """
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
                yield
        except OSError as e:
            print(f"Error: cannot write to {output_path}: {e}", file=sys.stderr)
            raise SystemExit(1) from e
        print(f"Output written to: {output_path}")
    else:
        yield


def cmd_show(args: argparse.Namespace) -> None:
    """Execute the 'show' subcommand.

    Accepts either a benchmark report (from ``run``) or comparison
    data (from ``compare --format json``) and dispatches to the
    matching display function.

    Args:
        args: Parsed arguments from argparse.
    """
    fmt = resolve_output_format(args)
    data = load_report(args.report)
    exclude_stats_raw = getattr(args, "tracerbench_exclude_stats", None)
    exclude_stats = (
        [s.strip() for s in exclude_stats_raw.split(",")]
        if exclude_stats_raw
        else None
    )
    with managed_output(args.output):
        if is_comparison_data(data):
            display_comparison_data(data, fmt, tracerbench_exclude_stats=exclude_stats)
        else:
            display_report(cast(Report, data), fmt, tracerbench_exclude_stats=exclude_stats)


def cmd_compare(args: argparse.Namespace) -> None:
    """Execute the 'compare' subcommand.

    Args:
        args: Parsed arguments from argparse.
    """
    fmt = resolve_output_format(args)
    exclude_stats_raw = getattr(args, "tracerbench_exclude_stats", None)
    exclude_stats = (
        [s.strip() for s in exclude_stats_raw.split(",")]
        if exclude_stats_raw
        else None
    )
    with managed_output(args.output):
        compare_reports(args.reports, fmt, tracerbench_exclude_stats=exclude_stats)


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
