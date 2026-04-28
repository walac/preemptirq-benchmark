from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from preemptirq_benchmark.benchmarks import BenchmarkBase, register

NULLB_DEV = Path("/dev/nullb0")


@register
class FioBenchmark(BenchmarkBase):
    """I/O interrupt stress benchmark using fio with null_blk and io_uring."""

    name = "fio"
    default_iterations = 10

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that fio is installed and /dev/nullb0 is available.

        Returns:
            (True, "") if ready, or (False, install/setup hint) otherwise.
        """
        if not shutil.which("fio"):
            return False, "fio not found (install: dnf install fio)"
        if not NULLB_DEV.exists():
            return False, ("/dev/nullb0 not available " "(run: modprobe null_blk irqmode=1)")
        return True, ""

    def run_once(self) -> dict[str, float]:
        """Run a single fio iteration with JSON output.

        Returns:
            Dict with iops, bandwidth_kbs, lat_avg_usec, and
            lat_p99_usec.

        Raises:
            RuntimeError: If fio JSON output cannot be parsed.
        """
        proc = subprocess.run(
            [
                "fio",
                "--name=nullblk",
                "--ioengine=io_uring",
                f"--filename={NULLB_DEV}",
                "--direct=1",
                "--bs=4k",
                "--iodepth=64",
                "--rw=randread",
                "--runtime=30",
                "--time_based",
                "--output-format=json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cannot parse fio JSON output: {proc.stdout[:200]}") from e
        try:
            job = data["jobs"][0]["read"]
            return {
                "iops": job["iops"],
                "bandwidth_kbs": job["bw"],
                "lat_avg_usec": job["lat_ns"]["mean"] / 1000,
                "lat_p99_usec": (job["clat_ns"]["percentile"]["99.000000"] / 1000),
            }
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"fio JSON missing expected keys/data: {e}. Output: {proc.stdout[:200]}"
            ) from e

    def get_command(self) -> list[str]:
        """Return the fio command for perf stat wrapping.

        Returns:
            The fio command as a list of strings.
        """
        return [
            "fio",
            "--name=nullblk",
            "--ioengine=io_uring",
            f"--filename={NULLB_DEV}",
            "--direct=1",
            "--bs=4k",
            "--iodepth=64",
            "--rw=randread",
            "--runtime=30",
            "--time_based",
            "--output-format=json",
        ]

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for fio metrics.

        Returns:
            Dict mapping each metric to its unit string.
        """
        return {
            "iops": "ops/s",
            "bandwidth_kbs": "KB/s",
            "lat_avg_usec": "us",
            "lat_p99_usec": "us",
        }
