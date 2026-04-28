from __future__ import annotations

import json
import shutil
import subprocess

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class CyclictestBenchmark(BenchmarkBase):
    """RT scheduling latency benchmark using cyclictest."""

    name = "cyclictest"
    default_iterations = 30

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that cyclictest is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("cyclictest"):
            return True, ""
        return False, "cyclictest not found (install: dnf install rt-tests)"

    def run_once(self) -> dict[str, float]:
        """Run a single cyclictest iteration and parse JSON output.

        Returns:
            Dict with min_latency_us, avg_latency_us, and
            max_latency_us (worst across all CPUs).
        """
        proc = subprocess.run(
            [
                "cyclictest",
                "-m",
                "-S",
                "-p",
                "98",
                "-i",
                "1000",
                "-l",
                "100000",
                "-q",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cannot parse cyclictest JSON output: {proc.stdout[:200]}") from e

        min_lat = float("inf")
        avg_total = 0.0
        max_lat = 0.0
        n_threads = 0

        for _tid, thread in data.get("thread", {}).items():
            try:
                min_lat = min(min_lat, thread["min"])
                avg_total += thread["avg"]
                max_lat = max(max_lat, thread["max"])
                n_threads += 1
            except KeyError as e:
                raise RuntimeError(
                    f"cyclictest JSON thread data missing keys: {e}. Output: {proc.stdout[:200]}"
                ) from e

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
        return [
            "cyclictest",
            "-m",
            "-S",
            "-p",
            "98",
            "-i",
            "1000",
            "-l",
            "100000",
            "-q",
        ]

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
