from __future__ import annotations

from typing import Any, TypedDict


class MetricData(TypedDict):
    unit: str
    values: list[float]
    mean: float
    median: float
    stddev: float
    ci_low: float
    ci_high: float
    ci_pct: float
    n: int


class PerfCounterData(TypedDict):
    values: list[int]
    mean: float
    sample_count: int


class _BenchmarkEntryRequired(TypedDict):
    iterations: int
    metrics: dict[str, MetricData]
    perf_counters: dict[str, PerfCounterData]


class BenchmarkEntry(_BenchmarkEntryRequired, total=False):
    config: dict[str, Any]


class Report(TypedDict):
    version: int
    timestamp: str
    hostname: str
    kernel_version: str
    nr_cpus: int | None
    ci_pct: float
    benchmarks_run: list[str]
    results: dict[str, BenchmarkEntry]
