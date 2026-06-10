from __future__ import annotations

from unittest.mock import patch

import pytest

from preemptirq_benchmark.stats import (
    DescriptiveStats,
    SignificanceResult,
    compute_delta_pct,
    compute_stats,
    format_delta_pct,
    mann_whitney,
)


class TestComputeStats:
    def test_odd_count(self):
        values = [10.0, 12.0, 14.0, 15.0, 18.0]
        s = compute_stats(values, ci_pct=95.0)

        assert s.mean == pytest.approx(13.8)
        assert s.median == 14.0
        assert s.n == 5
        assert s.stddev == pytest.approx(3.03315, abs=1e-4)
        assert s.ci_low < s.mean
        assert s.ci_high > s.mean
        assert s.ci_pct == 95.0

    def test_even_count(self):
        values = [10.0, 20.0, 30.0, 40.0]
        s = compute_stats(values)

        assert s.mean == 25.0
        assert s.median == 25.0
        assert s.n == 4

    def test_single_element(self):
        s = compute_stats([42.0])

        assert s.mean == 42.0
        assert s.median == 42.0
        assert s.stddev == 0.0
        assert s.ci_low == 42.0
        assert s.ci_high == 42.0
        assert s.n == 1

    def test_two_elements(self):
        s = compute_stats([10.0, 20.0])

        assert s.mean == 15.0
        assert s.median == 15.0
        assert s.n == 2
        assert s.stddev > 0
        assert s.ci_low < s.mean
        assert s.ci_high > s.mean

    def test_identical_values(self):
        s = compute_stats([5.0, 5.0, 5.0])

        assert s.mean == 5.0
        assert s.median == 5.0
        assert s.stddev == 0.0
        assert s.ci_low == 5.0
        assert s.ci_high == 5.0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot compute statistics on empty list"):
            compute_stats([])

    @pytest.mark.parametrize("ci_pct", [0.0, 100.0, -5.0, 105.0])
    def test_invalid_ci_raises(self, ci_pct):
        with pytest.raises(ValueError, match="ci_pct must be between 0 and 100 exclusive"):
            compute_stats([1.0, 2.0], ci_pct=ci_pct)

    def test_custom_ci_99(self):
        values = [10.0, 12.0, 14.0, 15.0, 18.0]
        s95 = compute_stats(values, ci_pct=95.0)
        s99 = compute_stats(values, ci_pct=99.0)

        assert s99.ci_pct == 99.0
        assert s99.ci_low < s95.ci_low
        assert s99.ci_high > s95.ci_high

    def test_returns_descriptive_stats(self):
        s = compute_stats([1.0, 2.0, 3.0])
        assert isinstance(s, DescriptiveStats)


class TestComputeDeltaPct:
    @pytest.mark.parametrize(
        "base, other, expected",
        [
            (100.0, 110.0, 10.0),
            (100.0, 90.0, -10.0),
            (100.0, 100.0, 0.0),
            (50.0, 75.0, 50.0),
            (-100.0, -90.0, 10.0),
        ],
    )
    def test_normal_deltas(self, base, other, expected):
        assert compute_delta_pct(base, other) == pytest.approx(expected)

    def test_zero_base_positive(self):
        assert compute_delta_pct(0.0, 10.0) == float("inf")

    def test_zero_base_negative(self):
        assert compute_delta_pct(0.0, -10.0) == float("-inf")

    def test_zero_both(self):
        assert compute_delta_pct(0.0, 0.0) == 0.0


class TestFormatDeltaPct:
    @pytest.mark.parametrize(
        "pct, expected",
        [
            (12.34, "+12.3%"),
            (-5.67, "-5.7%"),
            (0.0, "0.0%"),
            (-0.0, "0.0%"),
            (float("inf"), "+inf%"),
            (float("-inf"), "-inf%"),
            (float("nan"), "N/A"),
            (0.04, "+0.0%"),
            (-0.04, "-0.0%"),
            (100.0, "+100.0%"),
        ],
    )
    def test_format(self, pct, expected):
        assert format_delta_pct(pct) == expected


class TestMannWhitney:
    def test_insufficient_base_samples(self):
        result = mann_whitney([1.0, 2.0], [1.0, 2.0, 3.0])

        assert not result.significant_05
        assert not result.significant_01
        assert result.label == "(ns)"
        assert result.p_value == 1.0

    def test_insufficient_other_samples(self):
        result = mann_whitney([1.0, 2.0, 3.0], [1.0])

        assert result.label == "(ns)"
        assert result.p_value == 1.0

    def test_identical_samples(self):
        result = mann_whitney([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])

        assert not result.significant_05
        assert result.label == "(ns)"

    def test_returns_significance_result(self):
        result = mann_whitney([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert isinstance(result, SignificanceResult)

    def test_clearly_different_distributions(self):
        base = [1.0, 2.0, 3.0, 4.0, 5.0]
        other = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = mann_whitney(base, other)

        assert result.p_value < 0.05
        assert result.significant_05

    def test_label_boundaries_via_mock(self):
        with patch("preemptirq_benchmark.stats.mannwhitneyu") as mock_mwu:
            mock_mwu.return_value = (10.0, 0.005)
            res = mann_whitney([1, 2, 3], [4, 5, 6])
            assert res.label == "(**)"
            assert res.significant_01
            assert res.significant_05

            mock_mwu.return_value = (10.0, 0.03)
            res = mann_whitney([1, 2, 3], [4, 5, 6])
            assert res.label == "(*)"
            assert not res.significant_01
            assert res.significant_05

            mock_mwu.return_value = (10.0, 0.06)
            res = mann_whitney([1, 2, 3], [4, 5, 6])
            assert res.label == "(ns)"
            assert not res.significant_01
            assert not res.significant_05

    def test_exact_boundary_p_001(self):
        with patch("preemptirq_benchmark.stats.mannwhitneyu") as mock_mwu:
            mock_mwu.return_value = (10.0, 0.01)
            res = mann_whitney([1, 2, 3], [4, 5, 6])
            assert res.label == "(*)"

    def test_exact_boundary_p_005(self):
        with patch("preemptirq_benchmark.stats.mannwhitneyu") as mock_mwu:
            mock_mwu.return_value = (10.0, 0.05)
            res = mann_whitney([1, 2, 3], [4, 5, 6])
            assert res.label == "(ns)"

    def test_value_error_fallback(self):
        with patch(
            "preemptirq_benchmark.stats.mannwhitneyu",
            side_effect=ValueError("all values identical"),
        ):
            res = mann_whitney([1, 2, 3], [4, 5, 6])
            assert res.label == "(ns)"
            assert res.p_value == 1.0
