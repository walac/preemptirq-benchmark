from __future__ import annotations

import json

import pytest

from preemptirq_benchmark.benchmarks import BenchmarkResult
from preemptirq_benchmark.compare import (
    build_comparison_data,
    compare_reports,
    display_comparison_data,
    is_comparison_data,
)
from preemptirq_benchmark.report import build_report, save_report
from preemptirq_benchmark.types import Report


def make_report(
    name: str = "hackbench",
    values: list[float] | None = None,
    units: dict[str, str] | None = None,
    perf_counters: dict[str, list[int | float]] | None = None,
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
    def test_rejects_duplicate_labels_before_json_keys_collide(self):
        reports = [make_report(values=[1.0]), make_report(values=[2.0])]

        with pytest.raises(ValueError, match="unique"):
            build_comparison_data(reports, ["report", "report"])

    def test_warns_about_mismatched_latency_workloads_in_saved_data(self, capsys):
        base = make_report(name="cyclictest")
        other = make_report(name="cyclictest")
        base["results"]["cyclictest"]["config"] = {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [0, 1],
        }
        other["results"]["cyclictest"]["config"] = {
            "duration": "5m",
            "isolated_cpus_only": True,
            "cpu_selection": "isolated",
            "cpus": [2, 3],
        }

        data = build_comparison_data([base, other], ["base", "other"])

        assert len(data["warnings"]) == 1
        assert "cyclictest" in data["warnings"][0]
        assert "duration" in data["warnings"][0]
        assert "cpus" in data["warnings"][0]

        display_comparison_data(json.loads(json.dumps(data)), "txt")
        assert data["warnings"][0] in capsys.readouterr().out

    def test_warns_when_old_latency_report_lacks_config(self):
        base = make_report(name="rtla")
        other = make_report(name="rtla")
        other["results"]["rtla"]["config"] = {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "all_online",
            "cpus": [0, 1],
        }

        data = build_comparison_data([base, other], ["base", "other"])

        assert len(data["warnings"]) == 1
        assert "cannot verify" in data["warnings"][0]

    def test_same_latency_config_has_no_warning(self):
        base = make_report(name="cyclictest")
        other = make_report(name="cyclictest")
        config = {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [0, 1],
        }
        base["results"]["cyclictest"]["config"] = config
        other["results"]["cyclictest"]["config"] = config.copy()

        assert build_comparison_data([base, other], ["base", "other"])["warnings"] == []

    def test_equivalent_duration_and_cpu_order_have_no_warning(self):
        base = make_report(name="cyclictest")
        other = make_report(name="cyclictest")
        base["results"]["cyclictest"]["config"] = {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [0, 1],
        }
        other["results"]["cyclictest"]["config"] = {
            "duration": "30s",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [1, 0],
        }

        assert build_comparison_data([base, other], ["base", "other"])["warnings"] == []

    def test_single_sample_comparison_has_no_test_result(self):
        base = make_report(name="cyclictest", values=[1.0])
        other = make_report(name="cyclictest", values=[100.0])

        data = build_comparison_data([base, other], ["base", "other"])
        comparison = data["benchmarks"]["cyclictest"]["time_seconds"]["comparisons"]["other"]

        assert comparison["delta_pct"] == 9900.0
        assert comparison["p_value"] is None
        assert comparison["test_status"] == "insufficient_samples"
        assert comparison["significant"] == "(insufficient samples: n=1v1)"

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
        assert cmp["significant"] == "(insufficient samples: n=3v3)"

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
        assert nm["unit"] == ""
        assert "other" in nm["comparisons"]
        assert "other_mean" in nm["comparisons"]["other"]
        assert nm["comparisons"]["other"]["other_unit"] == "y"

    def test_missing_metric_in_base_each_report_keeps_own_unit(self):
        # Regression test: when two compared reports both carry a metric
        # that's absent from the baseline and disagree on unit, each
        # report's own unit must be used to format its own value --
        # not a single unit locked in from whichever report was seen
        # first, which would misrepresent the other report's value.
        base = make_report(values=[1.0, 1.1, 1.2])

        v1_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5], "new_metric": [2.0, 2.1]},
            units={"time_seconds": "s", "new_metric": "y1"},
            iterations=3,
        )
        v1 = build_report([v1_result])

        v2_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.6, 1.7, 1.8], "new_metric": [2.2, 2.3]},
            units={"time_seconds": "s", "new_metric": "y2"},
            iterations=3,
        )
        v2 = build_report([v2_result])

        data = build_comparison_data([base, v1, v2], ["base", "v1", "v2"])

        nm = data["benchmarks"]["hackbench"]["new_metric"]
        assert nm["base_mean"] is None
        assert nm["comparisons"]["v1"]["other_unit"] == "y1"
        assert nm["comparisons"]["v2"]["other_unit"] == "y2"

    def test_empty_base_unit_not_overridden_by_compared_report(self):
        # Regression test: when a metric IS present in the baseline but
        # its unit is legitimately empty (e.g. a raw count), the unit
        # fallback must not kick in just because the stored value is
        # falsy -- it's gated on the metric being entirely absent from
        # the baseline, not on an empty unit string.
        base_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.0, 1.1, 1.2], "count": [5.0, 6.0]},
            units={"time_seconds": "s", "count": ""},
            iterations=3,
        )
        base = build_report([base_result])

        other_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5], "count": [7.0, 8.0]},
            units={"time_seconds": "s", "count": "items"},
            iterations=3,
        )
        other = build_report([other_result])

        data = build_comparison_data([base, other], ["base", "other"])

        count = data["benchmarks"]["hackbench"]["count"]
        assert count["base_mean"] is not None
        assert count["unit"] == ""

    def test_perf_counter_comparison_includes_significance(self):
        base = make_report(perf_counters={"cycles": [1000000, 1000100, 1000200, 1000300]})
        patched = make_report(perf_counters={"cycles": [1100000, 1100100, 1100200, 1100300]})

        data = build_comparison_data([base, patched], ["baseline", "patched"])

        cycles = data["benchmarks"]["hackbench"]["perf:cycles"]
        assert cycles["base_mean"] == 1000150.0
        assert cycles["comparisons"]["patched"]["other_mean"] == 1100150.0
        assert cycles["comparisons"]["patched"]["delta_pct"] > 0
        assert cycles["comparisons"]["patched"]["p_value"] == pytest.approx(0.0286)
        assert cycles["comparisons"]["patched"]["significant"] == "(*)"
        assert cycles["comparisons"]["patched"]["test_status"] == "tested"

    def test_fractional_perf_counter_preserved(self):
        base = make_report(perf_counters={"task-clock": [123.456789]})
        patched = make_report(perf_counters={"task-clock": [130.111]})

        data = build_comparison_data([base, patched], ["baseline", "patched"])

        tc = data["benchmarks"]["hackbench"]["perf:task-clock"]
        assert tc["base_mean"] == pytest.approx(123.456789)
        assert tc["comparisons"]["patched"]["other_mean"] == pytest.approx(130.111)

    def test_perf_counter_missing_in_base(self):
        base = make_report()
        other_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5]},
            perf_counters={"cycles": [1100000]},
            iterations=3,
        )
        other = build_report([other_result])

        data = build_comparison_data([base, other], ["base", "other"])

        cycles = data["benchmarks"]["hackbench"]["perf:cycles"]
        assert cycles["base_mean"] is None
        assert cycles["comparisons"]["other"]["other_mean"] == 1100000.0

    def test_perf_counter_missing_in_base_with_mixed_typing(self, capsys):
        # Regression test: when a counter is absent from the baseline and
        # present in more than one compared report, each report's own
        # int/float typing must be used to format its "other_mean" --
        # not a single flag locked in from whichever report was seen
        # first, which would misformat the others if their typing
        # differs.
        base = make_report()

        int_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5]},
            perf_counters={"cycles": [1100000]},
            iterations=3,
        )
        int_report = build_report([int_result])

        float_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.6, 1.7, 1.8]},
            perf_counters={"cycles": [123.456789]},
            iterations=3,
        )
        float_report = build_report([float_result])

        data = build_comparison_data([base, int_report, float_report], ["base", "ints", "floats"])

        cycles = data["benchmarks"]["hackbench"]["perf:cycles"]
        assert cycles["comparisons"]["ints"]["other_is_integer"] is True
        assert cycles["comparisons"]["floats"]["other_is_integer"] is False

        display_comparison_data(data, "ascii")

        # No exception, and each report's mean keeps its own precision.
        captured = capsys.readouterr()
        assert "1100000" in captured.out
        assert "123.4568" in captured.out


class TestIsComparisonData:
    def test_comparison_data_detected(self):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        assert is_comparison_data(data) is True

    def test_report_not_comparison_data(self):
        report = make_report()

        assert is_comparison_data(report) is False


class TestDisplayComparisonData:
    def test_direct_comparison_warns_about_latency_workload(self, tmp_path, capsys):
        base = make_report(name="cyclictest")
        other = make_report(name="cyclictest")
        base["results"]["cyclictest"]["config"] = {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [0, 1],
        }
        other["results"]["cyclictest"]["config"] = {
            "duration": "60",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [0, 1],
        }
        base_path = save_report(base, str(tmp_path / "base.json"))
        other_path = save_report(other, str(tmp_path / "other.json"))

        compare_reports([str(base_path), str(other_path)], "txt")

        assert "Warning: cyclictest workload differs" in capsys.readouterr().out

    def test_single_sample_comparison_table_says_insufficient_samples(self, tmp_path, capsys):
        base = make_report(name="cyclictest", values=[1.0])
        other = make_report(name="cyclictest", values=[100.0])
        base_path = save_report(base, str(tmp_path / "base.json"))
        other_path = save_report(other, str(tmp_path / "other.json"))

        compare_reports([str(base_path), str(other_path)], "txt")
        direct_output = capsys.readouterr().out
        assert "+9900.0% (insufficient samples: n=1v1)" in direct_output
        assert "+9900.0% (ns)" not in direct_output
        # [BUG-ST-01] The legend must describe the dynamic "n=AvB" label
        # actually printed above, not the old fixed "(insufficient
        # samples)" string.
        assert "(insufficient samples: n=AvB) = no test" in direct_output

        data = build_comparison_data([base, other], ["base", "other"])
        display_comparison_data(json.loads(json.dumps(data)), "txt")
        saved_output = capsys.readouterr().out
        assert "+9900.0% (insufficient samples: n=1v1)" in saved_output
        assert "(insufficient samples: n=AvB) = no test" in saved_output

    def test_ascii_output(self, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        display_comparison_data(data, "ascii")

        captured = capsys.readouterr()
        assert "Base: baseline" in captured.out
        assert "Compared: patched" in captured.out
        assert "baseline" in captured.out
        assert "patched" in captured.out
        assert "time_seconds" in captured.out

    def test_markdown_output(self, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        display_comparison_data(data, "markdown")

        captured = capsys.readouterr()
        assert "## Benchmark Comparison" in captured.out
        assert "%" in captured.out

    def test_json_output_round_trips(self, capsys):
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        display_comparison_data(data, "json")

        captured = capsys.readouterr()
        assert json.loads(captured.out) == data

    def test_perf_counters_ordered_after_metrics_like_compare_reports(self, tmp_path, capsys):
        # Regression test: compare_reports always lists all metrics
        # before all perf counters (two separate table sections), but
        # display_comparison_data used to sort every bench_data key
        # (metric names and "perf:*" names) together alphabetically,
        # which could interleave them -- e.g. a metric name starting
        # with a letter after "p" would sort after "perf:cycles" even
        # though compare_reports always puts it first. Use such a
        # metric name to prove the two paths now agree on ordering.
        base_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.0, 1.1, 1.2], "zzz_metric": [1.0, 2.0]},
            units={"time_seconds": "s", "zzz_metric": "x"},
            perf_counters={"cycles": [1000000]},
            iterations=3,
        )
        base = build_report([base_result])
        other_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5], "zzz_metric": [1.5, 2.5]},
            units={"time_seconds": "s", "zzz_metric": "x"},
            perf_counters={"cycles": [1100000]},
            iterations=3,
        )
        other = build_report([other_result])

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(other, str(tmp_path / "other.json"))
        compare_reports([str(p1), str(p2)], "txt")
        direct_out = capsys.readouterr().out
        direct_zzz_pos = direct_out.index("zzz_metric")
        direct_perf_pos = direct_out.index("perf:cycles")
        assert direct_zzz_pos < direct_perf_pos

        data = build_comparison_data([base, other], ["base", "other"])
        display_comparison_data(data, "txt")
        redisplay_out = capsys.readouterr().out
        redisplay_zzz_pos = redisplay_out.index("zzz_metric")
        redisplay_perf_pos = redisplay_out.index("perf:cycles")
        assert redisplay_zzz_pos < redisplay_perf_pos

    def test_metric_missing_from_base_shows_other_mean(self, capsys):
        # Regression test: build_comparison_data omits "delta_pct" for
        # metrics absent from the baseline (only "other_mean" is set),
        # so display_comparison_data must not KeyError on "delta_pct".
        base = make_report(values=[1.0, 1.1, 1.2])

        other_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5], "new_metric": [2.0, 2.1]},
            units={"time_seconds": "s", "new_metric": "y"},
            iterations=3,
        )
        other = build_report([other_result])

        data = build_comparison_data([base, other], ["base", "other"])

        display_comparison_data(data, "ascii")

        captured = capsys.readouterr()
        assert "new_metric" in captured.out
        assert "2.05 y" in captured.out

    def test_metric_missing_from_base_display_keeps_each_reports_own_unit(self, capsys):
        # Regression test: when a metric is absent from the baseline and
        # compared reports disagree on unit (e.g. "ms" vs "s"), each
        # report's own value must be displayed with its own unit --
        # a shared row-level unit here would silently misrepresent one
        # report's value under the other's unit.
        base = make_report(values=[1.0, 1.1, 1.2])

        ms_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5], "latency": [50.0]},
            units={"time_seconds": "s", "latency": "ms"},
            iterations=3,
        )
        ms_report = build_report([ms_result])

        s_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.6, 1.7, 1.8], "latency": [0.05]},
            units={"time_seconds": "s", "latency": "s"},
            iterations=3,
        )
        s_report = build_report([s_result])

        data = build_comparison_data([base, ms_report, s_report], ["base", "ms_run", "s_run"])
        display_comparison_data(data, "ascii")

        captured = capsys.readouterr()
        assert "50.00 ms" in captured.out
        assert "0.05 s" in captured.out

    def test_tracerbench_exclude_stats_does_not_affect_perf_counters(self, capsys):
        # Regression test: the tracerbench exclude-stats filter matches
        # a metric name's "test_type/stat_name" suffix (e.g. dropping
        # "irq/max" when "max" is excluded). Raw perf event names can
        # also legitimately contain "/" (e.g. PMU syntax like
        # "cpu/event=0x3c/max"), so a filter that doesn't distinguish
        # perf:* rows from tracerbench metric rows would incorrectly
        # drop such a counter. compare_reports already guards this via
        # _build_comparison_rows's section=="metrics" check; this test
        # locks in the same guard in display_comparison_data.
        base_result = BenchmarkResult(
            name="tracerbench",
            metrics={"irq/median": [100.0], "irq/max": [120.0]},
            units={"irq/median": "cycles", "irq/max": "cycles"},
            perf_counters={"cpu/event=0x3c/max": [1000000]},
            iterations=1,
        )
        base = build_report([base_result])
        other_result = BenchmarkResult(
            name="tracerbench",
            metrics={"irq/median": [110.0], "irq/max": [130.0]},
            units={"irq/median": "cycles", "irq/max": "cycles"},
            perf_counters={"cpu/event=0x3c/max": [1100000]},
            iterations=1,
        )
        other = build_report([other_result])

        data = build_comparison_data(
            [base, other], ["base", "other"], tracerbench_exclude_stats=["max"]
        )
        display_comparison_data(data, "ascii", tracerbench_exclude_stats=["max"])

        captured = capsys.readouterr()
        assert "irq/median" in captured.out
        assert "irq/max" not in captured.out
        assert "perf:cpu/event=0x3c/max" in captured.out

    def test_perf_counter_delta_does_not_crash(self, capsys):
        # Regression test: a one-sample perf counter has an explicit
        # insufficient-samples result, which display_comparison_data must
        # render without a KeyError.
        base = make_report(perf_counters={"cycles": [1000000]})
        patched = make_report(perf_counters={"cycles": [1100000]})
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        display_comparison_data(data, "ascii")

        captured = capsys.readouterr()
        assert "perf:cycles" in captured.out

    def test_perf_counter_missing_from_base_shows_other_mean(self):
        base = make_report()
        other_result = BenchmarkResult(
            name="hackbench",
            metrics={"time_seconds": [1.3, 1.4, 1.5]},
            perf_counters={"cycles": [1100000]},
            iterations=3,
        )
        other = build_report([other_result])

        data = build_comparison_data([base, other], ["base", "other"])

        display_comparison_data(data, "ascii")

    def test_fractional_perf_counter_display_preserves_precision(self, capsys):
        base = make_report(perf_counters={"task-clock": [123.456789]})
        patched = make_report(perf_counters={"task-clock": [130.111]})
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        display_comparison_data(data, "ascii")

        captured = capsys.readouterr()
        assert "123.4568" in captured.out

    def test_header_includes_base_and_compared_labels(self, capsys):
        # Regression test: the header row must include the base label
        # as its own column (not just the compared labels), otherwise
        # it misaligns with the base-mean cell in each row.
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])
        data = build_comparison_data([base, patched], ["baseline", "patched"])

        display_comparison_data(data, "txt")

        captured = capsys.readouterr()
        header_line = next(line for line in captured.out.splitlines() if "Metric" in line)
        assert "baseline" in header_line
        assert "patched" in header_line


class TestCompareReports:
    def test_duplicate_stems_keep_all_targets_in_saved_json(self, tmp_path, capsys):
        paths = []
        for parent, value in [("base", 1.0), ("v1", 2.0), ("v2", 3.0)]:
            directory = tmp_path / parent
            directory.mkdir()
            paths.append(
                str(save_report(make_report(values=[value]), str(directory / "report.json")))
            )

        compare_reports(paths, "json")
        data = json.loads(capsys.readouterr().out)

        assert data["base"] == "base/report"
        assert data["compared"] == ["v1/report", "v2/report"]
        comparisons = data["benchmarks"]["hackbench"]["time_seconds"]["comparisons"]
        assert set(comparisons) == {"v1/report", "v2/report"}
        assert comparisons["v1/report"]["other_mean"] == 2.0
        assert comparisons["v2/report"]["other_mean"] == 3.0

        display_comparison_data(json.loads(json.dumps(data)), "txt")
        output = capsys.readouterr().out
        assert "v1/report" in output
        assert "v2/report" in output

        compare_reports(paths, "txt")
        direct_output = capsys.readouterr().out
        assert "v1/report" in direct_output
        assert "v2/report" in direct_output

    def test_same_parent_name_uses_longer_unique_suffix(self, tmp_path, capsys):
        paths = []
        for parent in ("a", "b"):
            directory = tmp_path / parent / "v1"
            directory.mkdir(parents=True)
            paths.append(str(save_report(make_report(), str(directory / "report.json"))))

        compare_reports(paths, "json")

        data = json.loads(capsys.readouterr().out)
        assert data["base"] == "a/v1/report"
        assert data["compared"] == ["b/v1/report"]

    def test_same_stem_and_directory_uses_extension(self, tmp_path, capsys):
        base = str(save_report(make_report(values=[1.0]), str(tmp_path / "run.json")))
        other = str(save_report(make_report(values=[2.0]), str(tmp_path / "run.txt")))

        compare_reports([base, other], "json")

        data = json.loads(capsys.readouterr().out)
        assert data["base"] == "run.json"
        assert data["compared"] == ["run.txt"]
        comparisons = data["benchmarks"]["hackbench"]["time_seconds"]["comparisons"]
        assert list(comparisons) == ["run.txt"]

    def test_repeated_input_path_is_rejected(self, tmp_path):
        path = str(save_report(make_report(), str(tmp_path / "report.json")))

        with pytest.raises(SystemExit, match="cannot distinguish"):
            compare_reports([path, path], "json")

    def test_too_few_reports(self):
        with pytest.raises(SystemExit, match="at least 2"):
            compare_reports(["only_one.json"], "ascii")

    def test_comparison_file_rejected_as_input(self, tmp_path):
        # Regression test: feeding compare's own JSON output back in as
        # one of the reports to compare must fail loudly instead of
        # silently producing a nonsensical comparison.
        base = make_report(values=[1.0, 1.1, 1.2])
        patched = make_report(values=[1.3, 1.4, 1.5])

        p1 = save_report(base, str(tmp_path / "base.json"))

        comparison_path = tmp_path / "comparison.json"
        comparison_path.write_text(
            json.dumps(build_comparison_data([base, patched], ["base", "patched"]))
        )

        with pytest.raises(SystemExit, match="comparison output"):
            compare_reports([str(p1), str(comparison_path)], "ascii")

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

    def test_perf_counter_comparison_json(self, tmp_path, capsys):
        base = make_report(perf_counters={"cycles": [1000000]})
        patched = make_report(perf_counters={"cycles": [1100000]})

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(patched, str(tmp_path / "patched.json"))

        compare_reports([str(p1), str(p2)], "json")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "perf:cycles" in data["benchmarks"]["hackbench"]

    def test_fractional_perf_counter_comparison_keeps_precision(self, tmp_path, capsys):
        base = make_report(perf_counters={"task-clock": [123.456789]})
        patched = make_report(perf_counters={"task-clock": [130.111]})

        p1 = save_report(base, str(tmp_path / "base.json"))
        p2 = save_report(patched, str(tmp_path / "patched.json"))

        compare_reports([str(p1), str(p2)], "txt")

        captured = capsys.readouterr()
        assert "123.4568" in captured.out

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
