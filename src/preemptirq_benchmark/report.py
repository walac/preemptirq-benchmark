from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preemptirq_benchmark.benchmarks import (
    BENCHMARK_DESCRIPTIONS,
    BenchmarkResult,
)
from preemptirq_benchmark.formatters import format_table
from preemptirq_benchmark.stats import compute_stats

REPORT_VERSION = 2


def build_report(
    results: list[BenchmarkResult],
    tracerbench_config: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a full report dict from benchmark results.

    Args:
        results: List of BenchmarkResult objects from completed
            benchmark runs.
        tracerbench_config: Optional dict with nr_samples, nr_highest,
            and percentile_nth for the tracerbench module.

    Returns:
        A JSON-serializable dict containing version, metadata,
        benchmark names, and per-benchmark statistics.
    """
    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "kernel_version": platform.release(),
        "nr_cpus": os.cpu_count(),
        "benchmarks_run": [r.name for r in results],
        "results": {},
    }

    for result in results:
        entry: dict[str, Any] = {
            "iterations": result.iterations,
            "metrics": {},
            "perf_counters": {},
        }

        if result.name == "tracerbench" and tracerbench_config:
            entry["config"] = tracerbench_config

        for metric_name, values in result.metrics.items():
            if not values:
                continue
            stats = compute_stats(values)
            unit = result.units.get(metric_name, "")
            entry["metrics"][metric_name] = {
                "unit": unit,
                "values": values,
                **asdict(stats),
            }

        for counter_name, counts in result.perf_counters.items():
            if not counts:
                continue
            float_counts = [float(c) for c in counts]
            mean = sum(float_counts) / len(float_counts)
            entry["perf_counters"][counter_name] = {
                "values": counts,
                "mean": mean,
                "sample_count": len(counts),
            }

        report["results"][result.name] = entry

    return report


def save_report(report: dict[str, Any], output: str | None = None) -> Path:
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

    Args:
        path: Path to the JSON report file.

    Returns:
        Parsed report dict.

    Raises:
        SystemExit: If the file does not exist or is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Error: file not found: {p}")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"Error: invalid JSON in {p}: {e}") from e


def display_report(report: dict[str, Any], fmt: str) -> None:
    """Print a report to stdout in the requested format.

    Each benchmark gets its own table with one row per metric
    showing Mean, Median, StdDev, and 95% CI.

    Args:
        report: A report dict from :func:`build_report` or
            :func:`load_report`.
        fmt: Output format — "ascii", "txt", "markdown", or "json".
    """
    if fmt == "json":
        print(json.dumps(report, indent=2))
        return

    print_header(report, fmt)

    for bench_name in report["benchmarks_run"]:
        bench_data = report["results"][bench_name]
        desc = BENCHMARK_DESCRIPTIONS.get(bench_name, "")
        title = f"{bench_name} ({desc})"

        headers = ["Metric", "Mean", "Median", "StdDev", "95% CI"]
        rows: list[list[str]] = []

        for metric_name, mdata in bench_data["metrics"].items():
            unit = mdata.get("unit", "")
            suffix = f" {unit}" if unit else ""
            rows.append(
                [
                    metric_name,
                    f"{mdata['mean']:.2f}{suffix}",
                    f"{mdata['median']:.2f}{suffix}",
                    f"{mdata['stddev']:.3f}",
                    f"[{mdata['ci_95_low']:.2f}, {mdata['ci_95_high']:.2f}]",
                ]
            )

        if bench_data.get("perf_counters"):
            for cname, cdata in bench_data["perf_counters"].items():
                rows.append(
                    [
                        f"perf:{cname}",
                        f"{cdata['mean']:.0f}",
                        "",
                        "",
                        "",
                    ]
                )

        print(format_table(title, headers, rows, fmt))


def print_header(report: dict[str, Any], fmt: str) -> None:
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
