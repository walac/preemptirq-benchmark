from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class KernelCompileBenchmark(BenchmarkBase):
    """Kernel build throughput benchmark using make -j$(nproc).

    Each iteration runs ``make clean``, ``make defconfig``, and
    ``make -j$(nproc)`` on the kernel source tree specified via
    ``--kernel-src``.
    """

    name = "kernel-compile"
    default_iterations = 10

    def __init__(self) -> None:
        self.kernel_src: Path | None = None

    def configure(self, **kwargs: object) -> None:
        """Accept the kernel_src CLI parameter.

        Args:
            kwargs: Must include "kernel_src" as a string or Path
                when this benchmark is selected.
        """
        src = kwargs.get("kernel_src")
        if src is not None:
            self.kernel_src = Path(str(src))

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that a kernel source directory is configured and valid.

        Returns:
            (True, "") if the Makefile exists at the configured path,
            or (False, error message) otherwise.
        """
        if self.kernel_src is None:
            return False, "--kernel-src is required for kernel-compile"
        makefile = self.kernel_src / "Makefile"
        if not makefile.exists():
            return False, f"no Makefile found in {self.kernel_src}"
        if not shutil.which("time") and not Path("/usr/bin/time").exists():
            return False, "/usr/bin/time not found (install: dnf install time)"
        return True, ""

    def run_once(self) -> dict[str, float]:
        """Run a single kernel build iteration.

        Runs ``make clean``, ``make defconfig``, then
        ``make -j$(nproc)`` and measures wall, user, and system time.

        Returns:
            Dict with wall_time_seconds, user_time_seconds, and
            sys_time_seconds.
        """
        if self.kernel_src is None:
            raise RuntimeError("kernel_src not configured; call configure() first")
        nproc = os.cpu_count() or 1
        src = str(self.kernel_src)

        subprocess.run(
            ["make", "-C", src, "clean"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["make", "-C", src, "defconfig"],
            capture_output=True,
            check=True,
        )

        start = time.monotonic()
        proc = subprocess.run(
            ["/usr/bin/time", "-v", "make", "-C", src, f"-j{nproc}"],
            capture_output=True,
            text=True,
            check=True,
        )
        wall = time.monotonic() - start

        user_match = re.search(
            r"User time(?:\s*\(seconds\))?:\s*([\d.]+)", proc.stderr, re.IGNORECASE
        )
        sys_match = re.search(
            r"System time(?:\s*\(seconds\))?:\s*([\d.]+)", proc.stderr, re.IGNORECASE
        )

        if not user_match or not sys_match:
            raise RuntimeError(f"Failed to parse time output. Stderr: {proc.stderr[:200]}")

        user = float(user_match.group(1))
        sys_ = float(sys_match.group(1))

        return {
            "wall_time_seconds": wall,
            "user_time_seconds": user,
            "sys_time_seconds": sys_,
        }

    def get_command(self) -> list[str] | None:
        """Return the full build command for perf stat wrapping.

        Includes clean, defconfig, and build steps so that perf stat
        captures a representative compilation rather than a no-op
        against an already-built tree.

        Returns:
            A shell command as a list of strings, or None if
            kernel_src is not configured.
        """
        if self.kernel_src is None:
            return None
        nproc = os.cpu_count() or 1
        src = shlex.quote(str(self.kernel_src))
        return [
            "sh",
            "-c",
            f"make -C {src} clean && make -C {src} defconfig && make -C {src} -j{nproc}",
        ]

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for kernel compile metrics.

        Returns:
            Dict mapping each time metric to "s".
        """
        return {
            "wall_time_seconds": "s",
            "user_time_seconds": "s",
            "sys_time_seconds": "s",
        }
