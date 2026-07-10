from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from preemptirq_benchmark.benchmarks import BENCHMARK_DESCRIPTIONS
from preemptirq_benchmark.formatters import format_table
from preemptirq_benchmark.report import load_report
from preemptirq_benchmark.stats import (
    compute_delta_pct,
    format_delta_pct,
    mann_whitney,
)
from preemptirq_benchmark.types import Report


def _fmt_metric(mdata: dict[str, Any]) -> str:
    unit = mdata.get("unit", "")
    suffix = f" {unit}" if unit else ""
    return f"{mdata['mean']:.2f}{suffix}"


def _build_comparison_rows(
    base: Report,
    others: list[Report],
    bench_name: str,
    section: str,
    *,
    label_fn: Callable[[str], str],
    format_base: Callable[[dict[str, Any]], str],
    format_delta: Callable[[dict[str, Any], dict[str, Any]], str],
    format_abs: Callable[[dict[str, Any]], str],
) -> list[list[str]]:
    base_section = base.get("results", {}).get(bench_name, {}).get(section, {})
    all_keys = set(base_section.keys())
    for r in others:
        all_keys |= set(r.get("results", {}).get(bench_name, {}).get(section, {}).keys())

    rows: list[list[str]] = []
    for key in sorted(all_keys):
        row = [label_fn(key)]
        base_data = base_section.get(key)
        row.append(format_base(base_data) if base_data else "N/A")

        for other in others:
            other_data = other.get("results", {}).get(bench_name, {}).get(section, {}).get(key)
            if other_data is None:
                row.append("N/A")
            elif base_data is None:
                row.append(format_abs(other_data))
            else:
                row.append(format_delta(base_data, other_data))

        rows.append(row)
    return rows


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
        SystemExit: If fewer than 2 paths are given, a file is invalid,
            or a file holds comparison output instead of a report.
    """
    if len(paths) < 2:
        raise SystemExit("Error: compare requires at least 2 report files")

    reports = []
    labels = []
    for p in paths:
        data = load_report(p)
        if is_comparison_data(data):
            raise SystemExit(f"Error: {p} is comparison output, not a benchmark report")
        reports.append(cast(Report, data))
        labels.append(Path(p).stem)

    if fmt == "json":
        print(json.dumps(build_comparison_data(reports, labels), indent=2))
        return

    print_comparison_header(reports, labels, fmt)

    base = reports[0]
    all_benchmarks = set(base.get("benchmarks_run", []))
    for r in reports[1:]:
        all_benchmarks |= set(r.get("benchmarks_run", []))

    for bench_name in sorted(all_benchmarks):

        desc = BENCHMARK_DESCRIPTIONS.get(bench_name, "")
        title = f"{bench_name} ({desc})"
        headers = ["Metric"] + labels
        rows: list[list[str]] = []

        rows.extend(
            _build_comparison_rows(
                base,
                reports[1:],
                bench_name,
                "metrics",
                label_fn=lambda name: name,
                format_base=_fmt_metric,
                format_delta=lambda bd, od: (
                    f"{format_delta_pct(compute_delta_pct(bd['mean'], od['mean']))} "
                    f"{mann_whitney(bd.get('values', []), od.get('values', [])).label}"
                ),
                format_abs=_fmt_metric,
            )
        )
        rows.extend(
            _build_comparison_rows(
                base,
                reports[1:],
                bench_name,
                "perf_counters",
                label_fn=lambda name: f"perf:{name}",
                format_base=lambda d: f"{d['mean']:.0f}",
                format_delta=lambda bd, od: format_delta_pct(
                    compute_delta_pct(bd["mean"], od["mean"])
                ),
                format_abs=lambda d: f"{d['mean']:.0f}",
            )
        )

        print(format_table(title, headers, rows, fmt))


def build_comparison_data(
    reports: list[Report],
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
            unit = base_mdata.get("unit", "") if base_mdata else ""
            metric_cmp: dict[str, Any] = {
                "base_mean": base_mdata["mean"] if base_mdata else None,
                "unit": unit,
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
                    if not metric_cmp["unit"]:
                        # Metric doesn't exist in the baseline at all, so
                        # there's no base unit to report — take the first
                        # non-empty unit among compared reports instead of
                        # leaving it blank.
                        metric_cmp["unit"] = other_mdata.get("unit", "")
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
    reports: list[Report],
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


def is_comparison_data(data: Mapping[str, Any]) -> bool:
    """Return True if data is comparison JSON from :func:`build_comparison_data`.

    Args:
        data: A dict loaded from a report or comparison JSON file.

    Returns:
        True if the dict has the comparison shape rather than the
        single-report shape produced by :func:`preemptirq_benchmark.report.build_report`.
    """
    return "compared" in data and "benchmarks" in data


def display_comparison_data(data: Mapping[str, Any], fmt: str) -> None:
    """Print previously saved comparison JSON as formatted tables.

    This re-displays the output of :func:`build_comparison_data` (e.g.
    from ``compare --format json``) without needing the original
    reports, since the deltas and significance results are already
    baked into the data.

    Args:
        data: Comparison dict as produced by :func:`build_comparison_data`.
        fmt: Output format — "ascii", "txt", "markdown", or "json".
    """
    if fmt == "json":
        print(json.dumps(data, indent=2))
        return

    base_label = data.get("base", "base")
    compared_labels = data.get("compared", [])

    lines = [
        f"Base: {base_label}",
        f"Compared: {', '.join(compared_labels)}",
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

    for bench_name in sorted(data.get("benchmarks", {})):
        bench_data = data["benchmarks"][bench_name]
        desc = BENCHMARK_DESCRIPTIONS.get(bench_name, "")
        title = f"{bench_name} ({desc})"
        headers = ["Metric", base_label] + compared_labels
        rows: list[list[str]] = []

        for metric_name in sorted(bench_data):
            mcmp = bench_data[metric_name]
            unit = mcmp.get("unit", "")
            suffix = f" {unit}" if unit else ""
            base_mean = mcmp.get("base_mean")
            row = [metric_name, f"{base_mean:.2f}{suffix}" if base_mean is not None else "N/A"]

            comparisons = mcmp.get("comparisons", {})
            for label in compared_labels:
                entry = comparisons.get(label)
                if entry is None:
                    row.append("N/A")
                elif "delta_pct" in entry:
                    row.append(f"{format_delta_pct(entry['delta_pct'])} {entry['significant']}")
                else:
                    row.append(f"{entry['other_mean']:.2f}{suffix}")

            rows.append(row)

        print(format_table(title, headers, rows, fmt))
