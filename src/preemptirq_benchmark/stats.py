from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import mannwhitneyu, t


@dataclass
class DescriptiveStats:
    """Summary statistics for a series of measurements.

    Attributes:
        mean: Arithmetic mean.
        median: Middle value (or average of two middle values).
        stddev: Sample standard deviation (Bessel's correction).
        ci_95_low: Lower bound of the 95% confidence interval.
        ci_95_high: Upper bound of the 95% confidence interval.
        n: Number of observations.
    """

    mean: float
    median: float
    stddev: float
    ci_95_low: float
    ci_95_high: float
    n: int


@dataclass
class SignificanceResult:
    """Result of a Mann-Whitney U significance test.

    Attributes:
        u_statistic: The U statistic from the test.
        p_value: Two-sided p-value.
        significant_05: True if p < 0.05.
        significant_01: True if p < 0.01.
        label: Human-readable label — "(**)" for p < 0.01,
            "(*)" for p < 0.05, "(ns)" for not significant.
    """

    u_statistic: float
    p_value: float
    significant_05: bool
    significant_01: bool
    label: str


def compute_stats(values: list[float]) -> DescriptiveStats:
    """Compute descriptive statistics for a list of measurements.

    Args:
        values: List of numeric observations (must not be empty).

    Returns:
        A DescriptiveStats instance with mean, median, stddev,
        95% confidence interval bounds, and sample count.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("cannot compute statistics on empty list")

    n = len(values)
    mean = sum(values) / n

    sorted_v = sorted(values)
    if n % 2 == 1:
        median = sorted_v[n // 2]
    else:
        median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2

    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    stddev = math.sqrt(variance)

    if n > 1:
        stderr = stddev / math.sqrt(n)
        t_crit = float(t.ppf(1 - 0.025, df=n - 1))
        margin = t_crit * stderr
    else:
        margin = 0.0

    ci_95_low = mean - margin
    ci_95_high = mean + margin

    return DescriptiveStats(
        mean=mean,
        median=median,
        stddev=stddev,
        ci_95_low=ci_95_low,
        ci_95_high=ci_95_high,
        n=n,
    )


def compute_delta_pct(base: float, other: float) -> float:
    """Compute the percentage change from base to other.

    Args:
        base: The reference value.
        other: The comparison value.

    Returns:
        Percentage change as a float.  Returns float('inf') or float('-inf') if base is zero and other is not. Returns 0.0 if both are 0.
    """
    if base == 0.0:
        if other > 0:
            return float("inf")
        elif other < 0:
            return float("-inf")
        else:
            return 0.0
    return ((other - base) / abs(base)) * 100


def format_delta_pct(pct: float) -> str:
    """Format a percentage delta as a signed string.

    Args:
        pct: Percentage change value.

    Returns:
        A string like "+2.3%" or "-1.5%" or "0.0%".
    """
    if math.isinf(pct):
        return "+inf%" if pct > 0 else "-inf%"
    if math.isnan(pct):
        return "N/A"
    if pct > 0:
        return f"+{pct:.1f}%"
    if pct < 0:
        return f"{pct:.1f}%"
    return "0.0%"


def mann_whitney(base: list[float], other: list[float]) -> SignificanceResult:
    """Run a two-sided Mann-Whitney U test between two sample sets.

    Args:
        base: Observations from the baseline condition.
        other: Observations from the comparison condition.

    Returns:
        A SignificanceResult with the U statistic, p-value, boolean
        significance flags at 0.05 and 0.01 levels, and a human-readable
        label.  Returns a not-significant result if either sample has
        fewer than 3 observations.
    """
    if len(base) < 3 or len(other) < 3:
        return SignificanceResult(
            u_statistic=0.0,
            p_value=1.0,
            significant_05=False,
            significant_01=False,
            label="(ns)",
        )

    try:
        stat, p = mannwhitneyu(base, other, alternative="two-sided")
    except ValueError:
        return SignificanceResult(
            u_statistic=0.0,
            p_value=1.0,
            significant_05=False,
            significant_01=False,
            label="(ns)",
        )

    if p < 0.01:
        label = "(**)"
    elif p < 0.05:
        label = "(*)"
    else:
        label = "(ns)"

    return SignificanceResult(
        u_statistic=float(stat),
        p_value=float(p),
        significant_05=p < 0.05,
        significant_01=p < 0.01,
        label=label,
    )
