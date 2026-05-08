from __future__ import annotations

import os
import re
import shutil
import subprocess

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class StressNgBenchmark(BenchmarkBase):
    """Context switch saturation benchmark using stress-ng."""

    name = "stress-ng"
    default_iterations = 10

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that stress-ng is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("stress-ng"):
            return True, ""
        return False, "stress-ng not found (install: dnf install stress-ng)"

    def run_once(self) -> dict[str, float]:
        """Run a single stress-ng context-switch iteration.

        Returns:
            Dict with "context_switches_per_sec" as bogo ops/sec.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.get_command(),
            capture_output=True,
            text=True,
            check=True,
        )
        output = proc.stderr + proc.stdout
        match = re.search(
            # [bogo ops] [real time] [usr time] [sys time] [bogo ops/s] [bogo ops]
            r"context"
            r"\s+[\d.]+"  # [bogo ops]
            r"\s+[\d.]+"  # [real time]
            r"\s+[\d.]+"  # [usr time]
            r"\s+[\d.]+"  # [sys time]
            r"\s+([\d.]+)",  # [bogo ops/s]
            output,
        )
        if not match:
            raise RuntimeError(f"cannot parse stress-ng output: {output[:200]}")
        return {"context_switches_per_sec": float(match.group(1))}

    def get_command(self) -> list[str]:
        """Return the stress-ng command for perf stat wrapping.

        Returns:
            The stress-ng command as a list of strings.
        """
        return [
            "stress-ng",
            "--context",
            "0",
            "-t",
            "1m",
            "--metrics-brief",
            "--cpu",
            f"{max(1, (os.cpu_count() or 1) - 1)}",
            "--cpu-method",
            "all",
        ]

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for stress-ng metrics.

        Returns:
            Dict mapping "context_switches_per_sec" to "ops/s".
        """
        return {"context_switches_per_sec": "ops/s"}
