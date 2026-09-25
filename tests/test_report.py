from __future__ import annotations

import json

import pytest

from preemptirq_benchmark.benchmarks import BenchmarkResult
from preemptirq_benchmark.report import (
    REPORT_VERSION,
    build_report,
    display_report,
    format_perf_counter_mean,
    load_report,
    save_report,
)


def make_result(
    name: str = "hackbench",
    metrics: dict[str, list[float]] | None = None,
    units: dict[str, str] | None = None,
    perf_counters: dict[str, list[int | float]] | None = None,
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
    def test_latency_workload_config_survives_save_and_load(self, tmp_path):
        result = make_result(name="cyclictest")
        result.config = {
            "duration": "5m",
            "isolated_cpus_only": True,
            "cpu_selection": "isolated",
            "cpus": [2, 3],
        }

        path = save_report(build_report([result]), str(tmp_path / "latency.json"))

        assert load_report(path)["results"]["cyclictest"]["config"] == result.config

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

    def test_single_observation_has_unavailable_uncertainty(self):
        result = make_result(
            name="cyclictest",
            metrics={"max_latency_us": [42.0]},
            units={"max_latency_us": "us"},
            iterations=1,
        )
        report = build_report([result])
        metric = report["results"]["cyclictest"]["metrics"]["max_latency_us"]

        assert metric["mean"] == 42.0
        assert metric["median"] == 42.0
        assert metric["stddev"] is None
        assert metric["ci_low"] is None
        assert metric["ci_high"] is None

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

    def test_perf_counters_have_descriptive_stats(self):
        result = make_result(perf_counters={"cycles": [1000000, 1000100]})
        report = build_report([result])
        cdata = report["results"]["hackbench"]["perf_counters"]["cycles"]

        assert cdata["median"] == 1000050.0
        assert cdata["stddev"] == pytest.approx(70.71067811865476)
        assert cdata["ci_low"] is not None
        assert cdata["ci_high"] is not None
        assert cdata["ci_pct"] == 95.0
        assert cdata["n"] == 2

    def test_multiple_benchmarks(self):
        r1 = make_result(name="hackbench")
        r2 = make_result(
            name="fio", metrics={"iops": [5000.0, 5100.0, 4900.0]}, units={"iops": "ops/s"}
        )
        report = build_report([r1, r2])

        assert report["benchmarks_run"] == ["hackbench", "fio"]
        assert "hackbench" in report["results"]
        assert "fio" in report["results"]

    def test_tracerbench_config(self):
        result = make_result(name="tracerbench")
        config = {"nr_samples": 50000, "nr_highest": 250}
        report = build_report([result], tracerbench_config=config)

        assert report["results"]["tracerbench"].get("config") == config

    def test_empty_values_skipped(self):
        result = make_result(metrics={"good": [1.0, 2.0], "empty": []})
        report = build_report([result])

        assert "good" in report["results"]["hackbench"]["metrics"]
        assert "empty" not in report["results"]["hackbench"]["metrics"]

    def test_fractional_perf_counter(self):
        result = make_result(perf_counters={"task-clock": [123.456789, 130.111]})
        report = build_report([result])
        cdata = report["results"]["hackbench"]["perf_counters"]["task-clock"]

        assert cdata["values"] == [123.456789, 130.111]
        assert cdata["mean"] == pytest.approx(126.7838945)


class TestFormatPerfCounterMean:
    def test_integer_counter_has_no_decimals(self):
        cdata = {"values": [1000000, 1000200], "mean": 1000100.0, "sample_count": 2}

        assert format_perf_counter_mean(cdata) == "1000100"

    def test_fractional_counter_keeps_decimals(self):
        cdata = {"values": [123.456789, 130.111], "mean": 126.7838945, "sample_count": 2}

        assert format_perf_counter_mean(cdata) == "126.7839"


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
    def test_single_observation_json_has_null_uncertainty(self, capsys):
        result = make_result(
            name="cyclictest",
            metrics={"max_latency_us": [42.0]},
            units={"max_latency_us": "us"},
            iterations=1,
        )
        display_report(build_report([result]), "json")

        metric = json.loads(capsys.readouterr().out)["results"]["cyclictest"]["metrics"][
            "max_latency_us"
        ]
        assert metric["stddev"] is None
        assert metric["ci_low"] is None
        assert metric["ci_high"] is None

    @pytest.mark.parametrize("fmt", ["ascii", "txt", "markdown"])
    def test_single_observation_renders_uncertainty_unavailable(self, capsys, fmt):
        result = make_result(
            name="cyclictest",
            metrics={"max_latency_us": [42.0]},
            units={"max_latency_us": "us"},
            iterations=1,
        )
        display_report(build_report([result]), fmt)

        output = capsys.readouterr().out
        assert output.count("N/A") == 2
        assert "42.00 us" in output

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

    def test_ascii_output_preserves_fractional_perf_counter(self, capsys):
        result = make_result(perf_counters={"task-clock": [123.456789]})
        report = build_report([result])
        display_report(report, "ascii")

        captured = capsys.readouterr()
        assert "123.4568" in captured.out

    def test_ascii_output_includes_perf_counter_uncertainty(self, capsys):
        result = make_result(perf_counters={"cycles": [1000000, 1000100]})
        report = build_report([result])
        display_report(report, "ascii")

        captured = capsys.readouterr()
        assert "70.711" in captured.out
