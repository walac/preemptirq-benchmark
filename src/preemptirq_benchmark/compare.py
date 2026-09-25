from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from preemptirq_benchmark.benchmarks import BENCHMARK_DESCRIPTIONS
from preemptirq_benchmark.formatters import format_table
from preemptirq_benchmark.report import (
    format_perf_counter_mean,
    format_perf_mean_value,
    load_report,
    perf_counter_is_integer,
    should_exclude_tracerbench_metric,
)
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
    exclude_stats: list[str] | None = None,
) -> list[list[str]]:
    base_section = base.get("results", {}).get(bench_name, {}).get(section, {})
    all_keys = set(base_section.keys())
    for r in others:
        all_keys |= set(r.get("results", {}).get(bench_name, {}).get(section, {}).keys())

    rows: list[list[str]] = []
    for key in sorted(all_keys):
        if (
            bench_name == "tracerbench"
            and section == "metrics"
            and exclude_stats
            and should_exclude_tracerbench_metric(key, exclude_stats)
        ):
            continue
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
    tracerbench_exclude_stats: list[str] | None = None,
) -> None:
    """Load multiple reports and print comparison tables.

    The first report is treated as the baseline.  Subsequent reports
    show their values as percentage deltas relative to the baseline,
    annotated with Mann-Whitney U significance.

    Args:
        paths: List of paths to JSON report files (minimum 2).
        fmt: Output format — "ascii", "txt", "markdown", or "json".
        tracerbench_exclude_stats: List of statistic names to exclude
            from tracerbench metrics (e.g., ["median", "max"]).

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
        print(
            json.dumps(
                build_comparison_data(reports, labels, tracerbench_exclude_stats),
                indent=2,
            )
        )
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
                exclude_stats=tracerbench_exclude_stats,
            )
        )
        rows.extend(
            _build_comparison_rows(
                base,
                reports[1:],
                bench_name,
                "perf_counters",
                label_fn=lambda name: f"perf:{name}",
                format_base=format_perf_counter_mean,
                format_delta=lambda bd, od: format_delta_pct(
                    compute_delta_pct(bd["mean"], od["mean"])
                ),
                format_abs=format_perf_counter_mean,
                exclude_stats=None,
            )
        )

        print(format_table(title, headers, rows, fmt))


def build_comparison_data(
    reports: list[Report],
    labels: list[str],
    tracerbench_exclude_stats: list[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable comparison structure.

    Args:
        reports: List of loaded report dicts, first is baseline.
        labels: Display names for each report (from filenames).
        tracerbench_exclude_stats: List of statistic names to exclude
            from tracerbench metrics (e.g., ["median", "max"]).

    Returns:
        Dict with per-benchmark, per-metric delta percentages and
        significance results.  Perf counters are included under keys
        prefixed with "perf:" (e.g. "perf:cycles") with a delta
        percentage but, matching the table output, no significance
        test.
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
            if (
                bench_name == "tracerbench"
                and tracerbench_exclude_stats
                and should_exclude_tracerbench_metric(metric_name, tracerbench_exclude_stats)
            ):
                continue
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
                    # Metric doesn't exist in the baseline, so its
                    # "other_mean" is displayed directly rather than as a
                    # delta -- unit must come from this specific report's
                    # own data, not a shared/first-seen guess, since
                    # different compared reports could disagree.
                    metric_cmp["comparisons"][labels[i + 1]] = {
                        "other_mean": other_mdata["mean"],
                        "other_unit": other_mdata.get("unit", ""),
                    }
                    continue
                pct = compute_delta_pct(base_mdata["mean"], other_mdata["mean"])
                other_values = other_mdata.get("values", [])
                sig = mann_whitney(base_values, other_values)
                metric_cmp["comparisons"][labels[i + 1]] = {
                    "delta_pct": round(pct, 2),
                    "other_mean": other_mdata["mean"],
                    "p_value": round(sig.p_value, 4) if sig.p_value is not None else None,
                    "significant": sig.label,
                    "test_status": sig.status,
                }

            bench_data[metric_name] = metric_cmp

        base_counters = base.get("results", {}).get(bench_name, {}).get("perf_counters", {})
        all_counters = set(base_counters.keys())
        for r in reports[1:]:
            all_counters |= set(
                r.get("results", {}).get(bench_name, {}).get("perf_counters", {}).keys()
            )

        for counter_name in sorted(all_counters):
            base_cdata = base_counters.get(counter_name)
            base_is_integer = perf_counter_is_integer(base_cdata["values"]) if base_cdata else None
            counter_cmp: dict[str, Any] = {
                "base_mean": base_cdata["mean"] if base_cdata else None,
                "is_integer": base_is_integer if base_is_integer is not None else True,
                "comparisons": {},
            }

            for i, other in enumerate(reports[1:]):
                other_cdata = (
                    other.get("results", {})
                    .get(bench_name, {})
                    .get("perf_counters", {})
                    .get(counter_name)
                )
                if other_cdata is None:
                    continue
                if base_cdata is None:
                    # Counter doesn't exist in the baseline, so its
                    # "other_mean" is displayed directly rather than as a
                    # delta -- typing must come from this specific report's
                    # own samples, not a shared/first-seen guess, since
                    # different compared reports could disagree.
                    counter_cmp["comparisons"][labels[i + 1]] = {
                        "other_mean": other_cdata["mean"],
                        "other_is_integer": perf_counter_is_integer(other_cdata["values"]),
                    }
                    continue
                pct = compute_delta_pct(base_cdata["mean"], other_cdata["mean"])
                counter_cmp["comparisons"][labels[i + 1]] = {
                    "delta_pct": round(pct, 2),
                    "other_mean": other_cdata["mean"],
                }

            bench_data[f"perf:{counter_name}"] = counter_cmp

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
        "(insufficient samples) = no test, (unavailable) = test failed",
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


def display_comparison_data(
    data: Mapping[str, Any],
    fmt: str,
    tracerbench_exclude_stats: list[str] | None = None,
) -> None:
    """Print previously saved comparison JSON as formatted tables.

    This re-displays the output of :func:`build_comparison_data` (e.g.
    from ``compare --format json``) without needing the original
    reports, since the deltas and significance results are already
    baked into the data.

    Args:
        data: Comparison dict as produced by :func:`build_comparison_data`.
        fmt: Output format — "ascii", "txt", "markdown", or "json".
        tracerbench_exclude_stats: List of statistic names to exclude
            from tracerbench metrics (e.g., ["median", "max"]).
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
        "(insufficient samples) = no test, (unavailable) = test failed",
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

        # Metrics are listed before perf counters, matching the ordering
        # produced directly by compare_reports (metrics section, then
        # perf_counters section) instead of one alphabetical sort that
        # would interleave "perf:*" keys with metric names.
        metric_names = sorted(k for k in bench_data if not k.startswith("perf:"))
        perf_names = sorted(k for k in bench_data if k.startswith("perf:"))

        for metric_name in metric_names + perf_names:
            if (
                bench_name == "tracerbench"
                and not metric_name.startswith("perf:")
                and tracerbench_exclude_stats
                and should_exclude_tracerbench_metric(metric_name, tracerbench_exclude_stats)
            ):
                continue
            mcmp = bench_data[metric_name]
            comparisons = mcmp.get("comparisons", {})

            if metric_name.startswith("perf:"):
                # Perf counters have no unit and no significance test
                # (see build_comparison_data), and need is_integer-aware
                # formatting to avoid truncating fractional counters
                # like "task-clock".
                is_integer = mcmp.get("is_integer", True)
                base_mean = mcmp.get("base_mean")
                row = [
                    metric_name,
                    (
                        format_perf_mean_value(base_mean, is_integer)
                        if base_mean is not None
                        else "N/A"
                    ),
                ]
                for label in compared_labels:
                    entry = comparisons.get(label)
                    if entry is None:
                        row.append("N/A")
                    elif "delta_pct" in entry:
                        row.append(format_delta_pct(entry["delta_pct"]))
                    else:
                        # No baseline to diff against for this counter, so
                        # its own typing (not the row-level one, which
                        # reflects only the base report) is used to format
                        # "other_mean" directly.
                        entry_is_integer = entry.get("other_is_integer", is_integer)
                        row.append(format_perf_mean_value(entry["other_mean"], entry_is_integer))
                rows.append(row)
                continue

            unit = mcmp.get("unit", "")
            suffix = f" {unit}" if unit else ""
            base_mean = mcmp.get("base_mean")
            row = [metric_name, f"{base_mean:.2f}{suffix}" if base_mean is not None else "N/A"]

            for label in compared_labels:
                entry = comparisons.get(label)
                if entry is None:
                    row.append("N/A")
                elif "delta_pct" in entry:
                    row.append(f"{format_delta_pct(entry['delta_pct'])} {entry['significant']}")
                else:
                    # No baseline to diff against for this metric, so its
                    # own unit (not the row-level one, which reflects only
                    # the base report) is used to format "other_mean"
                    # directly.
                    entry_unit = entry.get("other_unit", unit)
                    entry_suffix = f" {entry_unit}" if entry_unit else ""
                    row.append(f"{entry['other_mean']:.2f}{entry_suffix}")

            rows.append(row)

        print(format_table(title, headers, rows, fmt))
