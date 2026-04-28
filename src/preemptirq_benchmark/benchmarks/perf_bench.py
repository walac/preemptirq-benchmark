from __future__ import annotations

import re
import shutil
import subprocess

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class PerfBenchBenchmark(BenchmarkBase):
    """In-tree scheduler benchmarks using perf bench sched."""

    name = "perf-bench"
    default_iterations = 10

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that perf is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("perf"):
            return True, ""
        return False, "perf not found (install: dnf install perf)"

    def run_once(self) -> dict[str, float]:
        """Run perf bench sched pipe and messaging.

        Returns:
            Dict with pipe_usecs_per_op, pipe_ops_per_sec, and
            messaging_time_seconds.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        metrics: dict[str, float] = {}

        pipe = subprocess.run(
            ["perf", "bench", "sched", "pipe"],
            capture_output=True,
            text=True,
            check=True,
        )
        ops_match = re.search(
            r"([\d.]+)\s+ops/sec",
            pipe.stdout,
        )
        usec_match = re.search(
            r"([\d.]+)\s+usecs/op",
            pipe.stdout,
        )
        if ops_match:
            metrics["pipe_ops_per_sec"] = float(ops_match.group(1))
        if usec_match:
            metrics["pipe_usecs_per_op"] = float(usec_match.group(1))

        msg = subprocess.run(
            ["perf", "bench", "sched", "messaging"],
            capture_output=True,
            text=True,
            check=True,
        )
        time_match = re.search(
            r"Total time:\s+([\d.]+)\s+\[sec\]",
            msg.stdout,
        )
        if time_match:
            metrics["messaging_time_seconds"] = float(time_match.group(1))

        expected = {"pipe_ops_per_sec", "pipe_usecs_per_op", "messaging_time_seconds"}
        missing = expected - metrics.keys()
        if missing:
            raise RuntimeError(
                f"perf bench partial parse failure, missing: {', '.join(sorted(missing))}. "
                f"pipe={pipe.stdout[:200]!r} msg={msg.stdout[:200]!r}"
            )

        return metrics

    def get_command(self) -> list[str]:
        """Return the perf bench pipe command for perf stat wrapping.

        Returns:
            The perf bench sched pipe command.
        """
        return ["perf", "bench", "sched", "pipe"]

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for perf bench metrics.

        Returns:
            Dict mapping each metric to its unit string.
        """
        return {
            "pipe_usecs_per_op": "us/op",
            "pipe_ops_per_sec": "ops/s",
            "messaging_time_seconds": "s",
        }
