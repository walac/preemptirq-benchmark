from __future__ import annotations

import shutil

from preemptirq_benchmark.benchmarks import BenchmarkBase, register, run_command
from preemptirq_benchmark.cpu_isolation import (
    format_cpu_list,
    get_isolated_cpus,
    get_online_cpus,
)

_EXPECTED_TIMERLAT_ALL_BLOCK_WIDTH = 3


@register
class RtlaBenchmark(BenchmarkBase):
    """RT latency benchmark using rtla timerlat and osnoise."""

    name = "rtla"
    description = "RT latency (timerlat + osnoise)"
    default_iterations = 1
    fixed_iterations = True

    def __init__(self) -> None:
        self.duration = "30"
        self.isolated_cpus_only = False
        self._last_cpus: list[int] | None = None

    def configure(self, **kwargs: object) -> None:
        """Accept the duration and isolated_cpus_only CLI parameters.

        Args:
            kwargs: Optional keys "duration" (str, in rtla's own
                ``s``/``m``/``h``/``d``-suffixed format) and
                "isolated_cpus_only" (bool).
        """
        if kwargs.get("duration") is not None:
            self.duration = str(kwargs["duration"])
        if kwargs.get("isolated_cpus_only") is not None:
            self.isolated_cpus_only = bool(kwargs["isolated_cpus_only"])
        self._last_cpus = None

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that rtla is installed and isolated CPUs exist if requested.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if not shutil.which("rtla"):
            return False, "rtla not found (install: dnf install rtla or kernel-tools)"
        if self.isolated_cpus_only and not get_isolated_cpus():
            return False, "no isolated CPUs found (set isolcpus= kernel parameter)"
        return True, ""

    def _cpu_args(self) -> list[str]:
        """Return the ``-c`` argument restricting rtla to isolated CPUs.

        Returns:
            ``["-c", cpu-list]`` if isolated_cpus_only is set, else [].
        """
        if not self.isolated_cpus_only:
            return []
        if self._last_cpus is None:
            self._last_cpus = get_isolated_cpus()
        return ["-c", format_cpu_list(self._last_cpus)]

    def run_once(self) -> dict[str, float]:
        """Run rtla timerlat and osnoise, parsing summary output.

        Returns:
            Dict with timerlat_max_us and osnoise_max_single_us.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        metrics: dict[str, float] = {}

        tl = run_command(
            self.get_command(),
            capture_output=True,
            text=True,
            check=True,
        )
        max_lat = parse_timerlat_max_from_output(tl.stdout)
        metrics["timerlat_max_us"] = max_lat

        on = run_command(
            ["rtla", "osnoise", "top", "-d", self.duration, "-q", *self._cpu_args()],
            capture_output=True,
            text=True,
            check=True,
        )
        max_noise = parse_osnoise_max_from_output(on.stdout)
        metrics["osnoise_max_single_us"] = max_noise

        return metrics

    def get_command(self) -> list[str]:
        """Return the rtla timerlat command for perf stat wrapping.

        ``run_once`` reuses this same command for its timerlat
        measurement (rather than hardcoding a separate copy), so the
        hardware counters collected via ``perf stat`` always
        correspond to the timerlat sub-measurement specifically, not
        the separate osnoise measurement also reported by
        ``run_once``.

        Returns:
            The rtla timerlat command as a list of strings.
        """
        if not self.isolated_cpus_only and self._last_cpus is None:
            self._last_cpus = get_online_cpus()
        return ["rtla", "timerlat", "top", "-d", self.duration, "-q", *self._cpu_args()]

    def get_workload_config(self) -> dict[str, object] | None:
        """Return the duration and CPU population used by rtla."""
        if self._last_cpus is None:
            return None
        return {
            "duration": self.duration,
            "isolated_cpus_only": self.isolated_cpus_only,
            "cpu_selection": "isolated" if self.isolated_cpus_only else "all_online",
            "cpus": self._last_cpus.copy(),
        }

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for rtla metrics.

        Returns:
            Dict mapping each metric to "us".
        """
        return {
            "timerlat_max_us": "us",
            "osnoise_max_single_us": "us",
        }


def parse_timerlat_max_from_output(output: str) -> float:
    """Extract the maximum latency from rtla timerlat summary output.

    Args:
        output: Stdout from ``rtla timerlat top``.

    Returns:
        Maximum numeric value found in the ALL row.

    Raises:
        RuntimeError: if the max latency cannot be parsed
    """
    for line in output.splitlines():
        if line.startswith("ALL"):
            blocks = [part.split() for part in line.split("|")[1:] if part.strip()]
            if len(blocks) < 2:
                break
            widths = {len(block) for block in blocks}
            if widths != {_EXPECTED_TIMERLAT_ALL_BLOCK_WIDTH}:
                raise RuntimeError(
                    "unexpected rtla timerlat ALL-row layout "
                    f"(block widths {sorted(widths)}, expected "
                    f"{_EXPECTED_TIMERLAT_ALL_BLOCK_WIDTH}); refusing to guess which "
                    f"column is max. line={line!r}"
                )
            try:
                maxima = [float(block[-1]) for block in blocks if block[-1] != "-"]
            except ValueError as e:
                raise RuntimeError("could not parse timerlat max latency from output") from e
            if maxima:
                return max(maxima)
    raise RuntimeError("could not parse timerlat max latency from output")


def parse_osnoise_max_from_output(output: str) -> float:
    """Extract the maximum numeric value from rtla osnoise summary output.

    Args:
        output: Stdout from ``rtla osnoise top``.

    Returns:
        Maximum numeric value found in Max Single column.

    Raises:
        RuntimeError: if the max noise cannot be parsed
    """
    max_noise = None
    for line in output.splitlines():
        if (
            line.startswith("duration:")
            or line.startswith("CPU Period")
            or "Operating System Noise" in line
        ):
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                # 0: CPU, 1: Period (#1000), 2: Runtime, 3: Noise, 4: % CPU, 5: Max Noise, 6: Max Single
                val = float(parts[6])
                if max_noise is None or val > max_noise:
                    max_noise = val
            except ValueError:
                pass
    if max_noise is None:
        raise RuntimeError("could not parse osnoise max latency from output")
    return max_noise
