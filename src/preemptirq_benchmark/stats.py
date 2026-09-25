from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import mannwhitneyu, t


@dataclass
class DescriptiveStats:
    """Summary statistics for a series of measurements.

    Attributes:
        mean: Arithmetic mean.
        median: Middle value (or average of two middle values).
        stddev: Sample standard deviation, unavailable for one observation.
        ci_low: Lower confidence bound, unavailable for one observation.
        ci_high: Upper confidence bound, unavailable for one observation.
        ci_pct: Confidence level as a percentage (e.g. 95.0).
        n: Number of observations.
    """

    mean: float
    median: float
    stddev: float | None
    ci_low: float | None
    ci_high: float | None
    ci_pct: float
    n: int


@dataclass
class SignificanceResult:
    """Result of a Mann-Whitney U significance test.

    Attributes:
        u_statistic: The U statistic, or None when no test ran.
        p_value: Two-sided p-value, or None when no test ran.
        significant_05: True if p < 0.05, or None when no test ran.
        significant_01: True if p < 0.01, or None when no test ran.
        label: Human-readable label — "(**)" for p < 0.01,
            "(*)" for p < 0.05, "(ns)" for not significant,
            or an explicit unavailable reason.
        status: Whether the test ran, lacked samples, or failed.
    """

    u_statistic: float | None
    p_value: float | None
    significant_05: bool | None
    significant_01: bool | None
    label: str
    status: Literal["tested", "insufficient_samples", "unavailable"]


def compute_stats(values: list[float], ci_pct: float = 95.0) -> DescriptiveStats:
    """Compute descriptive statistics for a list of measurements.

    Args:
        values: List of numeric observations (must not be empty).
        ci_pct: Confidence level as a percentage (default 95.0).
            Must be between 0 and 100 exclusive.

    Returns:
        A DescriptiveStats instance with mean, median, stddev,
        confidence interval bounds, and sample count.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("cannot compute statistics on empty list")
    if not (0 < ci_pct < 100):
        raise ValueError(f"ci_pct must be between 0 and 100 exclusive, got {ci_pct}")

    n = len(values)
    mean = sum(values) / n

    sorted_v = sorted(values)
    if n % 2 == 1:
        median = sorted_v[n // 2]
    else:
        median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2

    if n == 1:
        return DescriptiveStats(
            mean=mean,
            median=median,
            stddev=None,
            ci_low=None,
            ci_high=None,
            ci_pct=ci_pct,
            n=n,
        )

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    stddev = math.sqrt(variance)

    stderr = stddev / math.sqrt(n)
    alpha = (1 - ci_pct / 100) / 2
    t_crit = float(t.ppf(1 - alpha, df=n - 1))
    margin = t_crit * stderr

    ci_low = mean - margin
    ci_high = mean + margin

    return DescriptiveStats(
        mean=mean,
        median=median,
        stddev=stddev,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_pct=ci_pct,
        n=n,
    )


def compute_delta_pct(base: float, other: float) -> float:
    """Compute the percentage change from base to other.

    Args:
        base: The reference value.
        other: The comparison value.

    Returns:
        Percentage change as a float.  Returns float('inf') or
        float('-inf') if base is zero and other is a nonzero, non-NaN
        value of the corresponding sign. Returns 0.0 if both are 0.
        Returns float('nan') if either input is NaN.
    """
    if math.isnan(base) or math.isnan(other):
        return float("nan")
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


# [BUG-ST-01] Smallest C(n1+n2, n1) at which a two-sided exact Mann-Whitney
# U test can attain p < 0.05 (min p = 2 / C(n1+n2, n1)). Below this, every
# possible outcome yields p >= 0.05, so "(ns)" would misreport a test that
# was mathematically incapable of ever finding significance as one that
# ran and found no effect.
_MIN_COMB_FOR_TESTABLE = 40


def _unavailable_significance() -> SignificanceResult:
    return SignificanceResult(
        u_statistic=None,
        p_value=None,
        significant_05=None,
        significant_01=None,
        label="(unavailable)",
        status="unavailable",
    )


def mann_whitney(base: list[float], other: list[float]) -> SignificanceResult:
    """Run a two-sided Mann-Whitney U test between two sample sets.

    Args:
        base: Observations from the baseline condition.
        other: Observations from the comparison condition.

    Returns:
        A SignificanceResult with the U statistic, p-value, boolean
        significance flags at 0.05 and 0.01 levels, and a human-readable
        label.  Returns an insufficient-samples result if the combined
        sample sizes are too small for a two-sided exact test to ever
        attain p < 0.05, regardless of how well-separated the samples are.
    """
    n1, n2 = len(base), len(other)
    if math.comb(n1 + n2, n1) <= _MIN_COMB_FOR_TESTABLE:
        return SignificanceResult(
            u_statistic=None,
            p_value=None,
            significant_05=None,
            significant_01=None,
            label=f"(insufficient samples: n={n1}v{n2})",
            status="insufficient_samples",
        )

    try:
        stat, p = mannwhitneyu(base, other, alternative="two-sided", method="exact")
    except ValueError:
        return _unavailable_significance()

    if not math.isfinite(float(stat)) or not math.isfinite(float(p)):
        return _unavailable_significance()

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
        status="tested",
    )
