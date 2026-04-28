from __future__ import annotations

import shutil
import subprocess

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class RtlaBenchmark(BenchmarkBase):
    """RT latency benchmark using rtla timerlat and osnoise."""

    name = "rtla"
    default_iterations = 30

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that rtla is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("rtla"):
            return True, ""
        return False, "rtla not found (install: dnf install rtla or kernel-tools)"

    def run_once(self) -> dict[str, float]:
        """Run rtla timerlat and osnoise, parsing summary output.

        Returns:
            Dict with timerlat_max_us and osnoise_max_single_us.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        metrics: dict[str, float] = {}

        tl = subprocess.run(
            ["rtla", "timerlat", "top", "-d", "30", "-q"],
            capture_output=True,
            text=True,
            check=True,
        )
        max_lat = parse_timerlat_max_from_output(tl.stdout)
        metrics["timerlat_max_us"] = max_lat

        on = subprocess.run(
            ["rtla", "osnoise", "top", "-d", "30", "-q"],
            capture_output=True,
            text=True,
            check=True,
        )
        max_noise = parse_osnoise_max_from_output(on.stdout)
        metrics["osnoise_max_single_us"] = max_noise

        return metrics

    def get_command(self) -> list[str]:
        """Return the rtla timerlat command for perf stat wrapping.

        Returns:
            The rtla timerlat command as a list of strings.
        """
        return ["rtla", "timerlat", "top", "-d", "30", "-q"]

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
            parts = line.split("|")
            if len(parts) >= 3:
                irq_stats = parts[1].split()
                thread_stats = parts[2].split()
                if len(irq_stats) >= 3 and len(thread_stats) >= 3:
                    return max(float(irq_stats[2]), float(thread_stats[2]))
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
