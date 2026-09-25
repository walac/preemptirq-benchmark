from __future__ import annotations

import json
import os
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preemptirq_benchmark.benchmarks import (
    BENCHMARK_DESCRIPTIONS,
    BenchmarkResult,
)
from preemptirq_benchmark.formatters import format_table
from preemptirq_benchmark.stats import compute_stats
from preemptirq_benchmark.types import BenchmarkEntry, MetricData, Report

REPORT_VERSION = 3


def perf_counter_is_integer(values: list[Any]) -> bool:
    """Check whether a perf counter's raw samples are all integer counts.

    Most perf events (cycles, instructions, ...) report whole-number
    counts, but some (e.g. "task-clock") report fractional values.

    Args:
        values: The perf counter's raw per-iteration samples.

    Returns:
        True if there is at least one sample and all samples are
        ``int``, False otherwise (including when *values* is empty).
    """
    return bool(values) and all(isinstance(v, int) for v in values)


def format_perf_mean_value(mean: float, is_integer: bool) -> str:
    """Format a perf counter mean value, preserving fractional precision.

    Truncating a fractional counter's mean (e.g. "task-clock") to an
    integer would silently discard precision, so integer-typed and
    fractional counters use different decimal precision.

    Args:
        mean: The counter's mean value.
        is_integer: Whether the counter's raw samples are all integer
            counts (see :func:`perf_counter_is_integer`).

    Returns:
        The formatted mean, with decimals only for fractional counters.
    """
    if is_integer:
        return f"{mean:.0f}"
    return f"{mean:.4f}"


def format_perf_counter_mean(cdata: Mapping[str, Any]) -> str:
    """Format a perf counter's mean, preserving fractional precision.

    Args:
        cdata: The perf counter's report entry.

    Returns:
        The formatted mean, with decimals only when the underlying
        samples are fractional.
    """
    return format_perf_mean_value(cdata["mean"], perf_counter_is_integer(cdata["values"]))


def should_exclude_tracerbench_metric(
    metric_name: str,
    exclude_stats: list[str],
) -> bool:
    """Check if a tracerbench metric should be excluded.

    Args:
        metric_name: Metric name in "test_type/stat_name" format
            (e.g., "irq/median").
        exclude_stats: List of statistic names to exclude.

    Returns:
        True if the metric should be excluded, False otherwise.
    """
    if "/" not in metric_name:
        return False
    _, stat_name = metric_name.rsplit("/", 1)
    return stat_name in exclude_stats


def build_report(
    results: list[BenchmarkResult],
    tracerbench_config: dict[str, int] | None = None,
    ci_pct: float = 95.0,
) -> Report:
    """Build a full report dict from benchmark results.

    Args:
        results: List of BenchmarkResult objects from completed
            benchmark runs.
        tracerbench_config: Optional dict with nr_samples, nr_highest,
            and percentile_nth for the tracerbench module.
        ci_pct: Confidence interval percentage (default 95.0).

    Returns:
        A JSON-serializable dict containing version, metadata,
        benchmark names, and per-benchmark statistics.
    """
    report: Report = {
        "version": REPORT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "kernel_version": platform.release(),
        "nr_cpus": os.cpu_count(),
        "ci_pct": ci_pct,
        "benchmarks_run": [r.name for r in results],
        "results": {},
    }

    for result in results:
        entry: BenchmarkEntry = {
            "iterations": result.iterations,
            "metrics": {},
            "perf_counters": {},
        }

        if result.name == "tracerbench" and tracerbench_config:
            entry["config"] = tracerbench_config

        for metric_name, values in result.metrics.items():
            if not values:
                continue
            stats = compute_stats(values, ci_pct=ci_pct)
            unit = result.units.get(metric_name, "")
            entry["metrics"][metric_name] = MetricData(
                unit=unit,
                values=values,
                mean=stats.mean,
                median=stats.median,
                stddev=stats.stddev,
                ci_low=stats.ci_low,
                ci_high=stats.ci_high,
                ci_pct=stats.ci_pct,
                n=stats.n,
            )

        for counter_name, counts in result.perf_counters.items():
            if not counts:
                continue
            mean = sum(counts) / len(counts)
            entry["perf_counters"][counter_name] = {
                "values": counts,
                "mean": mean,
                "sample_count": len(counts),
            }

        report["results"][result.name] = entry

    return report


def save_report(report: Report, output: str | None = None) -> Path:
    """Save a report dict to a JSON file.

    Args:
        report: The report dict from :func:`build_report`.
        output: Explicit output file path, or None to generate a
            default filename from the kernel version and timestamp.

    Returns:
        Path to the written JSON file.
    """
    if output:
        path = Path(output)
    else:
        kernel = report["kernel_version"]
        dt = datetime.fromisoformat(report["timestamp"])
        ts = dt.strftime("%Y%m%d-%H%M%S")
        path = Path(f"preemptirq-benchmark-{kernel}-{ts}.json")

    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a report from a JSON file.

    The file may hold either a benchmark report from :func:`build_report`
    or comparison data from :func:`preemptirq_benchmark.compare.build_comparison_data`;
    callers are responsible for checking the shape before treating it as
    one or the other.

    Args:
        path: Path to the JSON report file.

    Returns:
        Parsed JSON as a dict.

    Raises:
        SystemExit: If the file does not exist or is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Error: file not found: {p}")
    try:
        with open(p, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Error: invalid JSON in {p}: {e}") from e
    except OSError as e:
        raise SystemExit(f"Error: cannot read file {p}: {e}") from e


def display_report(
    report: Report,
    fmt: str,
    tracerbench_exclude_stats: list[str] | None = None,
) -> None:
    """Print a report to stdout in the requested format.

    Each benchmark gets its own table with one row per metric
    showing Mean, Median, StdDev, and confidence interval.

    Args:
        report: A report dict from :func:`build_report` or
            :func:`load_report`.
        fmt: Output format — "ascii", "txt", "markdown", or "json".
        tracerbench_exclude_stats: List of statistic names to exclude
            from tracerbench metrics (e.g., ["median", "max"]).
    """
    if fmt == "json":
        print(json.dumps(report, indent=2))
        return

    print_header(report, fmt)

    ci_pct = report.get("ci_pct", 95.0)

    for bench_name in report["benchmarks_run"]:
        bench_data = report["results"][bench_name]
        desc = BENCHMARK_DESCRIPTIONS.get(bench_name, "")
        title = f"{bench_name} ({desc})"

        headers = ["Metric", "Mean", "Median", "StdDev", f"{ci_pct:g}% CI"]
        rows: list[list[str]] = []

        for metric_name, mdata in bench_data["metrics"].items():
            if (
                bench_name == "tracerbench"
                and tracerbench_exclude_stats
                and should_exclude_tracerbench_metric(metric_name, tracerbench_exclude_stats)
            ):
                continue
            unit = mdata.get("unit", "")
            suffix = f" {unit}" if unit else ""
            rows.append(
                [
                    metric_name,
                    f"{mdata['mean']:.2f}{suffix}",
                    f"{mdata['median']:.2f}{suffix}",
                    f"{mdata['stddev']:.3f}" if mdata["stddev"] is not None else "N/A",
                    (
                        f"[{mdata['ci_low']:.2f}, {mdata['ci_high']:.2f}]"
                        if mdata["ci_low"] is not None and mdata["ci_high"] is not None
                        else "N/A"
                    ),
                ]
            )

        if bench_data.get("perf_counters"):
            for cname, cdata in bench_data["perf_counters"].items():
                rows.append(
                    [
                        f"perf:{cname}",
                        format_perf_counter_mean(cdata),
                        "",
                        "",
                        "",
                    ]
                )

        print(format_table(title, headers, rows, fmt))


def print_header(report: Report, fmt: str) -> None:
    """Print the report metadata header.

    Args:
        report: The full report dict.
        fmt: Output format for style adjustments.
    """
    lines = [
        f"Host: {report['hostname']}",
        f"Kernel: {report['kernel_version']}",
        f"CPUs: {report['nr_cpus']}",
        f"Date: {report['timestamp']}",
        f"Benchmarks: {', '.join(report['benchmarks_run'])}",
    ]
    if fmt == "markdown":
        print("## Preemptirq Benchmark Report")
        print()
        for line in lines:
            print(f"- **{line}**")
        print()
    else:
        for line in lines:
            print(line)
        print()
