from __future__ import annotations

import json

import pytest

from preemptirq_benchmark.benchmarks import BenchmarkResult
from preemptirq_benchmark.report import (
    REPORT_VERSION,
    build_report,
    display_report,
    load_report,
    save_report,
)


def make_result(
    name: str = "hackbench",
    metrics: dict[str, list[float]] | None = None,
    units: dict[str, str] | None = None,
    perf_counters: dict[str, list[int]] | None = None,
    iterations: int = 3,
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        metrics=metrics or {"time_seconds": [1.0, 1.1, 1.2]},
        units=units or {"time_seconds": "s"},
        perf_counters=perf_counters or {},
        iterations=iterations,
    )


class TestBuildReport:
    def test_basic_structure(self):
        result = make_result()
        report = build_report([result])

        assert report["version"] == REPORT_VERSION
        assert "timestamp" in report
        assert "hostname" in report
        assert "kernel_version" in report
        assert "nr_cpus" in report
        assert report["benchmarks_run"] == ["hackbench"]
        assert "hackbench" in report["results"]

    def test_metrics_have_stats(self):
        result = make_result()
        report = build_report([result])
        mdata = report["results"]["hackbench"]["metrics"]["time_seconds"]

        assert "mean" in mdata
        assert "median" in mdata
        assert "stddev" in mdata
        assert "ci_low" in mdata
        assert "ci_high" in mdata
        assert "values" in mdata
        assert mdata["unit"] == "s"
        assert mdata["n"] == 3

    def test_ci_pct_propagated(self):
        result = make_result()
        report = build_report([result], ci_pct=99.0)

        assert report["ci_pct"] == 99.0
        mdata = report["results"]["hackbench"]["metrics"]["time_seconds"]
        assert mdata["ci_pct"] == 99.0

    def test_perf_counters(self):
        result = make_result(perf_counters={"cycles": [1000000]})
        report = build_report([result])
        cdata = report["results"]["hackbench"]["perf_counters"]["cycles"]

        assert cdata["mean"] == 1000000.0
        assert cdata["sample_count"] == 1
        assert cdata["values"] == [1000000]

    def test_multiple_benchmarks(self):
        r1 = make_result(name="hackbench")
        r2 = make_result(name="fio", metrics={"iops": [5000.0, 5100.0, 4900.0]}, units={"iops": "ops/s"})
        report = build_report([r1, r2])

        assert report["benchmarks_run"] == ["hackbench", "fio"]
        assert "hackbench" in report["results"]
        assert "fio" in report["results"]

    def test_tracerbench_config(self):
        result = make_result(name="tracerbench")
        config = {"nr_samples": 50000, "nr_highest": 250}
        report = build_report([result], tracerbench_config=config)

        assert report["results"]["tracerbench"]["config"] == config

    def test_empty_values_skipped(self):
        result = make_result(metrics={"good": [1.0, 2.0], "empty": []})
        report = build_report([result])

        assert "good" in report["results"]["hackbench"]["metrics"]
        assert "empty" not in report["results"]["hackbench"]["metrics"]


class TestSaveLoadReport:
    def test_round_trip(self, tmp_path):
        result = make_result()
        report = build_report([result])

        path = save_report(report, str(tmp_path / "report.json"))
        loaded = load_report(str(path))

        assert loaded["version"] == report["version"]
        assert loaded["benchmarks_run"] == report["benchmarks_run"]
        assert loaded["results"]["hackbench"]["metrics"]["time_seconds"]["mean"] == pytest.approx(
            report["results"]["hackbench"]["metrics"]["time_seconds"]["mean"]
        )

    def test_auto_generated_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = make_result()
        report = build_report([result])

        path = save_report(report)
        assert path.name.startswith("preemptirq-benchmark-")
        assert path.name.endswith(".json")
        assert path.exists()

    def test_load_missing_file(self):
        with pytest.raises(SystemExit, match="file not found"):
            load_report("/nonexistent/report.json")

    def test_load_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json{{{")
        with pytest.raises(SystemExit, match="invalid JSON"):
            load_report(str(bad))


class TestDisplayReport:
    def test_ascii_output(self, capsys):
        result = make_result()
        report = build_report([result])
        display_report(report, "ascii")

        captured = capsys.readouterr()
        assert "hackbench" in captured.out
        assert "time_seconds" in captured.out

    def test_markdown_output(self, capsys):
        result = make_result()
        report = build_report([result])
        display_report(report, "markdown")

        captured = capsys.readouterr()
        assert "## Preemptirq Benchmark Report" in captured.out
        assert "| time_seconds" in captured.out

    def test_txt_output(self, capsys):
        result = make_result()
        report = build_report([result])
        display_report(report, "txt")

        captured = capsys.readouterr()
        assert "time_seconds" in captured.out
        assert "+" in captured.out

    def test_json_output(self, capsys):
        result = make_result()
        report = build_report([result])
        display_report(report, "json")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["version"] == REPORT_VERSION
