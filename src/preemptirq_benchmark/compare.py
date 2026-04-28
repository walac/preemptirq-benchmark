from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from preemptirq_benchmark.benchmarks import BENCHMARK_DESCRIPTIONS
from preemptirq_benchmark.formatters import format_table
from preemptirq_benchmark.report import load_report
from preemptirq_benchmark.stats import (
    compute_delta_pct,
    format_delta_pct,
    mann_whitney,
)


def compare_reports(
    paths: list[str],
    fmt: str,
) -> None:
    """Load multiple reports and print comparison tables.

    The first report is treated as the baseline.  Subsequent reports
    show their values as percentage deltas relative to the baseline,
    annotated with Mann-Whitney U significance.

    Args:
        paths: List of paths to JSON report files (minimum 2).
        fmt: Output format — "ascii", "txt", "markdown", or "json".

    Raises:
        SystemExit: If fewer than 2 paths are given or files are invalid.
    """
    if len(paths) < 2:
        raise SystemExit("Error: compare requires at least 2 report files")

    reports = []
    labels = []
    for p in paths:
        reports.append(load_report(p))
        labels.append(Path(p).stem)

    if fmt == "json":
        print(json.dumps(build_comparison_data(reports, labels), indent=2))
        return

    print_comparison_header(reports, labels, fmt)

    base = reports[0]
    all_benchmarks = set(base.get("benchmarks_run", []))
    for r in reports[1:]:
        all_benchmarks |= set(r.get("benchmarks_run", []))

    for bench_name in sorted(list(all_benchmarks)):

        desc = BENCHMARK_DESCRIPTIONS.get(bench_name, "")
        title = f"{bench_name} ({desc})"
        headers = ["Metric"] + labels
        rows: list[list[str]] = []

        base_metrics = base.get("results", {}).get(bench_name, {}).get("metrics", {})

        # We need a list of all metrics across all reports for this benchmark
        all_metrics = set(base_metrics.keys())
        for r in reports[1:]:
            all_metrics |= set(r.get("results", {}).get(bench_name, {}).get("metrics", {}).keys())

        for metric_name in sorted(list(all_metrics)):
            row = [metric_name]

            base_mdata = base_metrics.get(metric_name)
            if base_mdata:
                unit = base_mdata.get("unit", "")
                suffix = f" {unit}" if unit else ""
                row.append(f"{base_mdata['mean']:.2f}{suffix}")
                base_values = base_mdata.get("values", [])
            else:
                row.append("N/A")
                base_values = []

            for other in reports[1:]:
                other_mdata = (
                    other.get("results", {}).get(bench_name, {}).get("metrics", {}).get(metric_name)
                )
                if other_mdata is None:
                    row.append("N/A")
                    continue

                if base_mdata is None:
                    unit = other_mdata.get("unit", "")
                    suffix = f" {unit}" if unit else ""
                    row.append(f"{other_mdata['mean']:.2f}{suffix}")
                    continue

                pct = compute_delta_pct(base_mdata["mean"], other_mdata["mean"])
                other_values = other_mdata.get("values", [])
                sig = mann_whitney(base_values, other_values)
                row.append(f"{format_delta_pct(pct)} {sig.label}")

            rows.append(row)

        base_perf = base.get("results", {}).get(bench_name, {}).get("perf_counters", {})
        all_perf = set(base_perf.keys())
        for r in reports[1:]:
            all_perf |= set(
                r.get("results", {}).get(bench_name, {}).get("perf_counters", {}).keys()
            )

        if all_perf:
            for cname in sorted(list(all_perf)):
                row = [f"perf:{cname}"]
                base_cdata = base_perf.get(cname)

                if base_cdata:
                    row.append(f"{base_cdata['mean']:.0f}")
                else:
                    row.append("N/A")

                for other in reports[1:]:
                    other_cdata = (
                        other.get("results", {})
                        .get(bench_name, {})
                        .get("perf_counters", {})
                        .get(cname)
                    )
                    if other_cdata is None:
                        row.append("N/A")
                        continue

                    if base_cdata is None:
                        row.append(f"{other_cdata['mean']:.0f}")
                        continue

                    pct = compute_delta_pct(
                        base_cdata["mean"],
                        other_cdata["mean"],
                    )
                    row.append(f"{format_delta_pct(pct)}")

                rows.append(row)

        print(format_table(title, headers, rows, fmt))


def build_comparison_data(
    reports: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable comparison structure.

    Args:
        reports: List of loaded report dicts, first is baseline.
        labels: Display names for each report (from filenames).

    Returns:
        Dict with per-benchmark, per-metric delta percentages
        and significance results.
    """
    base = reports[0]
    data: dict[str, Any] = {
        "base": labels[0],
        "compared": labels[1:],
        "benchmarks": {},
    }

    all_benchmarks = set(base.get("benchmarks_run", []))
    for r in reports[1:]:
        all_benchmarks |= set(r.get("benchmarks_run", []))

    for bench_name in sorted(all_benchmarks):
        bench_data: dict[str, Any] = {}
        base_metrics = base.get("results", {}).get(bench_name, {}).get("metrics", {})

        all_metrics = set(base_metrics.keys())
        for r in reports[1:]:
            all_metrics |= set(r.get("results", {}).get(bench_name, {}).get("metrics", {}).keys())

        for metric_name in sorted(all_metrics):
            base_mdata = base_metrics.get(metric_name)
            metric_cmp: dict[str, Any] = {
                "base_mean": base_mdata["mean"] if base_mdata else None,
                "unit": base_mdata.get("unit", "") if base_mdata else "",
                "comparisons": {},
            }

            base_values = base_mdata.get("values", []) if base_mdata else []
            for i, other in enumerate(reports[1:]):
                other_mdata = (
                    other.get("results", {}).get(bench_name, {}).get("metrics", {}).get(metric_name)
                )
                if other_mdata is None:
                    continue
                if base_mdata is None:
                    metric_cmp["comparisons"][labels[i + 1]] = {
                        "other_mean": other_mdata["mean"],
                    }
                    continue
                pct = compute_delta_pct(base_mdata["mean"], other_mdata["mean"])
                other_values = other_mdata.get("values", [])
                sig = mann_whitney(base_values, other_values)
                metric_cmp["comparisons"][labels[i + 1]] = {
                    "delta_pct": round(pct, 2),
                    "other_mean": other_mdata["mean"],
                    "p_value": round(sig.p_value, 4),
                    "significant": sig.label,
                }

            bench_data[metric_name] = metric_cmp

        data["benchmarks"][bench_name] = bench_data

    return data


def print_comparison_header(
    reports: list[dict[str, Any]],
    labels: list[str],
    fmt: str,
) -> None:
    """Print metadata header for comparison output.

    Args:
        reports: List of loaded report dicts.
        labels: Display names for each report.
        fmt: Output format for style adjustments.
    """
    base = reports[0]
    lines = [
        f"Base: {labels[0]} (kernel {base['kernel_version']})",
        f"Compared: {', '.join(labels[1:])}",
        "(ns) = not significant, (*) = p<0.05, (**) = p<0.01",
    ]
    if fmt == "markdown":
        print("## Benchmark Comparison")
        print()
        for line in lines:
            print(f"- {line}")
        print()
    else:
        for line in lines:
            print(line)
        print()
