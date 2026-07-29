"""Test tracerbench metric filtering."""

from __future__ import annotations

from preemptirq_benchmark.benchmarks import BenchmarkResult
from preemptirq_benchmark.report import (
    build_report,
    display_report,
    should_exclude_tracerbench_metric,
)


class TestShouldExcludeTracerbenchMetric:
    """Test the should_exclude_tracerbench_metric function."""

    def test_excludes_matching_stat(self):
        assert should_exclude_tracerbench_metric("irq/median", ["median"])

    def test_excludes_matching_stat_different_test_type(self):
        assert should_exclude_tracerbench_metric("preempt/median", ["median"])

    def test_does_not_exclude_non_matching_stat(self):
        assert not should_exclude_tracerbench_metric("irq/average", ["median"])

    def test_excludes_multiple_stats(self):
        assert should_exclude_tracerbench_metric("irq/median", ["median", "max"])
        assert should_exclude_tracerbench_metric("irq/max", ["median", "max"])

    def test_non_tracerbench_metric_not_excluded(self):
        # Non-tracerbench metrics don't have "/" in the name
        assert not should_exclude_tracerbench_metric("simple_metric", ["median"])


class TestTracerbenchFilteringInDisplay:
    """Test that filtering works in display_report."""

    def test_filters_excluded_stats(self, capsys):
        result = BenchmarkResult(
            name="tracerbench",
            units={
                "irq/median": "cycles",
                "irq/average": "cycles",
                "irq/max": "cycles",
                "preempt/median": "cycles",
                "preempt/average": "cycles",
            },
        )
        result.metrics = {
            "irq/median": [100.0],
            "irq/average": [105.0],
            "irq/max": [120.0],
            "preempt/median": [80.0],
            "preempt/average": [85.0],
        }
        result.iterations = 1

        report = build_report([result])
        display_report(report, "ascii", tracerbench_exclude_stats=["median"])

        captured = capsys.readouterr()

        # Should show average and max, but not median
        assert "irq/average" in captured.out
        assert "irq/max" in captured.out
        assert "preempt/average" in captured.out

        # Should not show median
        assert "irq/median" not in captured.out
        assert "preempt/median" not in captured.out

    def test_no_filtering_when_none(self, capsys):
        result = BenchmarkResult(
            name="tracerbench",
            units={"irq/median": "cycles", "irq/average": "cycles"},
        )
        result.metrics = {
            "irq/median": [100.0],
            "irq/average": [105.0],
        }
        result.iterations = 1

        report = build_report([result])
        display_report(report, "ascii", tracerbench_exclude_stats=None)

        captured = capsys.readouterr()

        # Should show all metrics
        assert "irq/median" in captured.out
        assert "irq/average" in captured.out

    def test_filters_multiple_stats(self, capsys):
        result = BenchmarkResult(
            name="tracerbench",
            units={
                "irq/median": "cycles",
                "irq/average": "cycles",
                "irq/max": "cycles",
                "irq/percentile": "cycles",
            },
        )
        result.metrics = {
            "irq/median": [100.0],
            "irq/average": [105.0],
            "irq/max": [120.0],
            "irq/percentile": [110.0],
        }
        result.iterations = 1

        report = build_report([result])
        display_report(report, "ascii", tracerbench_exclude_stats=["median", "max"])

        captured = capsys.readouterr()

        # Should show average and percentile
        assert "irq/average" in captured.out
        assert "irq/percentile" in captured.out

        # Should not show median or max
        assert "irq/median" not in captured.out
        assert "irq/max" not in captured.out

    def test_does_not_filter_other_benchmarks(self, capsys):
        result = BenchmarkResult(
            name="other-benchmark",
            units={"median": "ns", "average": "ns"},
        )
        result.metrics = {
            "median": [100.0],
            "average": [105.0],
        }
        result.iterations = 1

        report = build_report([result])
        display_report(report, "ascii", tracerbench_exclude_stats=["median"])

        captured = capsys.readouterr()

        # Should show all metrics for non-tracerbench benchmarks
        assert "median" in captured.out
        assert "average" in captured.out
