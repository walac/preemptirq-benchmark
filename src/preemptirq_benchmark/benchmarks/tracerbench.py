from __future__ import annotations

import subprocess
from pathlib import Path

from preemptirq_benchmark.benchmarks import BenchmarkBase, register

DEBUGFS_BASE = Path("/sys/kernel/debug/tracerbench")

TEST_TYPES = ["irq", "preempt", "irq_save"]
STAT_NAMES = ["median", "average", "max", "max_avg", "percentile"]


@register
class TracerbenchBenchmark(BenchmarkBase):
    """Kernel module micro-benchmark measuring CPU cycles for
    local_irq_disable/enable, preempt_disable/enable, and
    local_irq_save/restore.

    Interacts with the tracerbench kernel module via debugfs.
    Does not support perf stat wrapping — the module already measures
    at cycle granularity.
    """

    name = "tracerbench"
    description = "Kernel module micro-benchmark (CPU cycles)"
    default_iterations = 5
    supports_perf_stat = False

    def __init__(self) -> None:
        self.nr_samples: int | None = None
        self.nr_highest: int | None = None
        self.percentile: int | None = None

    def configure(self, **kwargs: object) -> None:
        """Accept tracerbench-specific CLI parameters.

        Args:
            kwargs: Optional keys "nr_samples", "nr_highest", and
                "percentile" as integers.
        """
        if kwargs.get("nr_samples") is not None:
            self.nr_samples = int(str(kwargs["nr_samples"]))
        if kwargs.get("nr_highest") is not None:
            self.nr_highest = int(str(kwargs["nr_highest"]))
        if kwargs.get("percentile") is not None:
            self.percentile = int(str(kwargs["percentile"]))

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that the tracerbench module is loaded, attempting modprobe.

        If the debugfs directory does not exist, attempts
        ``modprobe tracerbench``.  If modprobe itself fails, that is
        reported distinctly from the case where modprobe succeeds but
        the expected debugfs directory still does not appear (e.g.
        debugfs is not mounted).

        Returns:
            (True, "") if the module is loaded and debugfs is accessible,
            or (False, error message) otherwise.
        """
        if DEBUGFS_BASE.is_dir():
            return True, ""

        result = subprocess.run(
            ["modprobe", "tracerbench"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, (
                "tracerbench module not loaded and modprobe failed "
                f"({result.stderr.strip()}). "
                "Load manually: insmod tracerbench.ko"
            )

        if DEBUGFS_BASE.is_dir():
            return True, ""

        return False, (
            f"tracerbench module loaded but debugfs path {DEBUGFS_BASE} "
            "was not found. Is debugfs mounted? Try: "
            "mount -t debugfs none /sys/kernel/debug"
        )

    def setup(self) -> None:
        """Write configuration parameters to debugfs before benchmarking."""
        try:
            if self.nr_samples is not None:
                (DEBUGFS_BASE / "nr_samples").write_text(str(self.nr_samples))
            if self.nr_highest is not None:
                (DEBUGFS_BASE / "nr_highest").write_text(str(self.nr_highest))
            if self.percentile is not None:
                (DEBUGFS_BASE / "nth_percentile").write_text(str(self.percentile))
        except OSError as e:
            raise RuntimeError(f"cannot write tracerbench config to {DEBUGFS_BASE}: {e}") from e

    def run_once(self) -> dict[str, float]:
        """Trigger a benchmark run and read all results from debugfs.

        Writes "1" to the benchmark file, then reads 3 test types x
        5 statistics = 15 values.

        Returns:
            Dict with keys like "irq_median", "preempt_average", etc.
        """
        try:
            (DEBUGFS_BASE / "benchmark").write_text("1")
        except OSError as e:
            raise RuntimeError(f"cannot trigger tracerbench benchmark: {e}") from e

        metrics: dict[str, float] = {}
        for test_type in TEST_TYPES:
            for stat_name in STAT_NAMES:
                path = DEBUGFS_BASE / test_type / stat_name
                try:
                    value = int(path.read_text().strip())
                except (FileNotFoundError, PermissionError, ValueError) as e:
                    raise RuntimeError(f"cannot read tracerbench metric {path}: {e}") from e
                metrics[f"{test_type}/{stat_name}"] = float(value)
        return metrics

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for tracerbench metrics.

        Returns:
            Dict mapping each metric (15 total) to "cycles".
        """
        units: dict[str, str] = {}
        for test_type in TEST_TYPES:
            for stat_name in STAT_NAMES:
                units[f"{test_type}/{stat_name}"] = "cycles"
        return units
