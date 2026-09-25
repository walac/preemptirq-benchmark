from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from preemptirq_benchmark.benchmarks import BenchmarkBase, register
from preemptirq_benchmark.cpu_isolation import format_cpu_list, get_isolated_cpus


@register
class CyclictestBenchmark(BenchmarkBase):
    """RT scheduling latency benchmark using cyclictest."""

    name = "cyclictest"
    description = "RT scheduling latency"
    default_iterations = 1
    fixed_iterations = True

    def __init__(self) -> None:
        self.duration = "30"
        self.isolated_cpus_only = False

    def configure(self, **kwargs: object) -> None:
        """Accept the duration and isolated_cpus_only CLI parameters.

        Args:
            kwargs: Optional keys "duration" (str, in cyclictest's own
                ``s``/``m``/``h``/``d``-suffixed format) and
                "isolated_cpus_only" (bool).
        """
        if kwargs.get("duration") is not None:
            self.duration = str(kwargs["duration"])
        if kwargs.get("isolated_cpus_only") is not None:
            self.isolated_cpus_only = bool(kwargs["isolated_cpus_only"])

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that cyclictest is installed and isolated CPUs exist if requested.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if not shutil.which("cyclictest"):
            return False, "cyclictest not found (install: dnf install rt-tests)"
        if self.isolated_cpus_only and not get_isolated_cpus():
            return False, "no isolated CPUs found (set isolcpus= kernel parameter)"
        return True, ""

    def _cpu_args(self) -> list[str]:
        """Return the thread count and affinity for isolated CPUs.

        Returns:
            One thread per isolated CPU, pinned in CPU-list order, or [].
        """
        if not self.isolated_cpus_only:
            return []
        cpus = get_isolated_cpus()
        return ["-t", str(len(cpus)), "-a", format_cpu_list(cpus)]

    def _base_command(self) -> list[str]:
        """Return the cyclictest command shared by run_once() and get_command().

        Kept as a single source of truth (per the perf-stat re-run
        pitfall documented in AGENTS.md) so duration/CPU-affinity
        options can't drift between the measurement run and the
        standalone perf-stat re-run.

        Returns:
            The cyclictest command as a list of strings, without
            ``--json`` (added by ``run_once`` only).
        """
        return [
            "cyclictest",
            "-m",
            *([] if self.isolated_cpus_only else ["-S"]),
            "-p",
            "98",
            "-i",
            "1000",
            "-D",
            self.duration,
            "-q",
            *self._cpu_args(),
        ]

    def run_once(self) -> dict[str, float]:
        """Run a single cyclictest iteration and parse JSON output.

        Returns:
            Dict with min_latency_us, avg_latency_us, and
            max_latency_us (worst across all CPUs).
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json_path = tmp.name

        try:
            subprocess.run(
                self._base_command() + [f"--json={json_path}"],
                capture_output=True,
                text=True,
                check=True,
            )
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(f"cannot parse cyclictest JSON output: {e}") from e
        finally:
            Path(json_path).unlink(missing_ok=True)

        min_lat = float("inf")
        avg_total = 0.0
        max_lat = 0.0
        n_threads = 0

        for _, thread in data.get("thread", {}).items():
            try:
                min_lat = min(min_lat, thread["min"])
                avg_total += thread["avg"]
                max_lat = max(max_lat, thread["max"])
                n_threads += 1
            except KeyError as e:
                raise RuntimeError(f"cyclictest JSON thread data missing keys: {e}") from e

        if n_threads == 0:
            raise RuntimeError("cyclictest returned no thread data")

        avg_lat = avg_total / n_threads

        return {
            "min_latency_us": float(min_lat),
            "avg_latency_us": avg_lat,
            "max_latency_us": float(max_lat),
        }

    def get_command(self) -> list[str]:
        """Return the cyclictest command for perf stat wrapping.

        Returns:
            The cyclictest command as a list of strings.
        """
        return self._base_command()

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for cyclictest metrics.

        Returns:
            Dict mapping each latency metric to "us".
        """
        return {
            "min_latency_us": "us",
            "avg_latency_us": "us",
            "max_latency_us": "us",
        }
