from __future__ import annotations

import re
import shutil
import subprocess

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class HackbenchBenchmark(BenchmarkBase):
    """Scheduler/IPC stress benchmark using hackbench."""

    name = "hackbench"
    default_iterations = 10

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that hackbench is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("hackbench"):
            return True, ""
        return False, "hackbench not found (install: dnf install rt-tests)"

    def run_once(self) -> dict[str, float]:
        """Run a single hackbench iteration.

        Returns:
            Dict with "time_seconds" as the measured wall-clock time.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            ["hackbench", "-s", "4096", "-l", "1000", "-g", "16", "-f", "20", "-P"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Time:\s+([\d.]+)", proc.stdout)
        if not match:
            raise RuntimeError(f"cannot parse hackbench output: {proc.stdout}")
        return {"time_seconds": float(match.group(1))}

    def get_command(self) -> list[str]:
        """Return the hackbench command for perf stat wrapping.

        Returns:
            The hackbench command as a list of strings.
        """
        return ["hackbench", "-s", "4096", "-l", "1000", "-g", "16", "-f", "20", "-P"]

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for hackbench metrics.

        Returns:
            Dict mapping "time_seconds" to "s".
        """
        return {"time_seconds": "s"}
