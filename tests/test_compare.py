from __future__ import annotations

import json

import pytest

from preemptirq_benchmark.benchmarks import BenchmarkResult
from preemptirq_benchmark.compare import build_comparison_data, compare_reports
from preemptirq_benchmark.report import build_report, save_report
from preemptirq_benchmark.types import Report


def make_report(
    name: str = "hackbench",
    values: list[float] | None = None,
    units: dict[str, str] | None = None,
    perf_counters: dict[str, list[int]] | None = None,
) -> Report:
    result = BenchmarkResult(
        name=name,
        metrics={"time_seconds": values or [1.0, 1.1, 1.2]},
        units=units or {"time_seconds": "s"},
        perf_counters=perf_counters or {},
        iterations=len(values) if values else 3,
    )
    return build_report([result])


class TestBuildComparisonData:
    def test_basic_comparison(self):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])

        data = build_comparison_data(
            [base, patched],
            ["baseline", "patched"],
        )

        assert data["base"] == "baseline"
        assert data["compared"] == ["patched"]
        assert "hackbench" in data["benchmarks"]

        ts = data["benchmarks"]["hackbench"]["time_seconds"]
        assert ts["base_mean"] is not None
        assert "patched" in ts["comparisons"]
        cmp = ts["comparisons"]["patched"]
        assert cmp["delta_pct"] > 0
        assert "p_value" in cmp
        assert cmp["significant"] in ["(**)", "(*)", "(ns)"]

    def test_three_way_comparison(self):
        base = make_report(values=[1.0, 1.1, 1.2])
        v1 = make_report(values=[1.3, 1.4, 1.5])
        v2 = make_report(values=[0.8, 0.9, 1.0])

        data = build_comparison_data(
            [base, v1, v2],
            ["baseline", "v1", "v2"],
        )

        assert data["compared"] == ["v1", "v2"]
        ts = data["benchmarks"]["hackbench"]["time_seconds"]
        assert "v1" in ts["comparisons"]
        assert "v2" in ts["comparisons"]
        assert ts["comparisons"]["v1"]["delta_pct"] > 0
        assert ts["comparisons"]["v2"]["delta_pct"] < 0

    def test_missing_metric_in_other(self):
        base_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.0, 1.1, 1.2], "extra": [5.0, 5.1]},
            units={"time_seconds": "s", "extra": "x"},
            iterations=3,
        )
        base = build_report([base_result])

        other = make_report(values=[1.3, 1.4, 1.5])

        data = build_comparison_data([base, other], ["base", "other"])

        assert "extra" in data["benchmarks"]["hackbench"]
        assert "other" not in data["benchmarks"]["hackbench"]["extra"]["comparisons"]

    def test_missing_metric_in_base(self):
        base = make_report(values=[1.0, 1.1, 1.2])

        other_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5], "new_metric": [2.0, 2.1]},
            units={"time_seconds": "s", "new_metric": "y"},
            iterations=3,
        )
        other = build_report([other_result])

        data = build_comparison_data([base, other], ["base", "other"])

        nm = data["benchmarks"]["hackbench"]["new_metric"]
        assert nm["base_mean"] is None
        assert "other" in nm["comparisons"]
        assert "other_mean" in nm["comparisons"]["other"]


class TestCompareReports:
    def test_too_few_reports(self):
        with pytest.raises(SystemExit, match="at least 2"):
            compare_reports(["only_one.json"], "ascii")

    def test_ascii_output(self, tmp_path, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(patched, str(tmp_path / "patched.json"))

        compare_reports([str(p1), str(p2)], "ascii")

        captured = capsys.readouterr()
        assert "time_seconds" in captured.out

    def test_markdown_output(self, tmp_path, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(patched, str(tmp_path / "patched.json"))

        compare_reports([str(p1), str(p2)], "markdown")

        captured = capsys.readouterr()
        assert "## Benchmark Comparison" in captured.out
        assert "%" in captured.out

    def test_json_output(self, tmp_path, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(patched, str(tmp_path / "patched.json"))

        compare_reports([str(p1), str(p2)], "json")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "benchmarks" in data
        assert "hackbench" in data["benchmarks"]

    def test_perf_counter_comparison(self, tmp_path, capsys):
        base = make_report(perf_counters={"cycles": [1000000]})
        patched = make_report(perf_counters={"cycles": [1100000]})

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(patched, str(tmp_path / "patched.json"))

        compare_reports([str(p1), str(p2)], "txt")

        captured = capsys.readouterr()
        assert "perf:cycles" in captured.out

    def test_benchmark_only_in_other(self, tmp_path, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])

        other_result = BenchmarkResult(
            name="fio",
            metrics={"iops": [5000.0, 5100.0, 4900.0]},
            units={"iops": "ops/s"},
            iterations=3,
        )
        other = build_report([other_result])

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(other, str(tmp_path / "other.json"))

        compare_reports([str(p1), str(p2)], "txt")

        captured = capsys.readouterr()
        assert "fio" in captured.out
