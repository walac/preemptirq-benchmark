from __future__ import annotations

import os
import re
import shutil
import subprocess

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


class BpfBenchBase(BenchmarkBase):
    """Base class for benchmarks that wrap the BPF selftests bench tool.

    Subclasses set :attr:`bench_name` to select which bench sub-benchmark
    to run (e.g. ``trig-fentry``).

    Attributes:
        bench_name: The bench sub-benchmark name passed on the command line.
        bpf_bench_path: Path to the bench binary.
        affinity: Whether to pin producer threads to CPUs (``-a``).
        producers: Number of producer threads (``-p``).
        duration: Benchmark duration in seconds (``-d``).
        warmup: Warmup duration in seconds (``-w``).
    """

    bench_name: str
    default_iterations: int = 5
    bpf_bench_path: str = "bench"
    supports_perf_stat: bool = False
    duration: int = 10
    warmup: int = 1
    affinity: bool = False
    producers: int = 1

    def configure(self, **kwargs: object) -> None:
        """Accept the --bpf-bench CLI option.

        Args:
            kwargs: Keyword arguments from the CLI parser.
        """
        path = kwargs.get("bpf_bench")
        if path is not None:
            self.bpf_bench_path = str(path)

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that the bench binary is available.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which(self.bpf_bench_path):
            return True, ""
        return (
            False,
            f"'{self.bpf_bench_path}' not found"
            " (build: make -C tools/testing/selftests/bpf bench)",
        )

    def bench_cmd(self) -> list[str]:
        """Build the bench command line.

        Returns:
            Command list with duration, warmup, producer count,
            optional affinity, and bench name.
        """
        cmd = [
            self.bpf_bench_path,
            f"-d{self.duration}",
            f"-w{self.warmup}",
        ]
        if self.affinity:
            cmd.append("-a")
        cmd += [f"-p{self.producers}", self.bench_name]
        return cmd

    def run_once(self) -> dict[str, float]:
        """Run a single bench iteration.

        Returns:
            Dict with "hits_m_per_sec" from the bench Summary line.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.bench_cmd(),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Summary:\s+hits\s+([\d.]+)\s*", proc.stdout)
        if not match:
            raise RuntimeError(f"cannot parse bench output: {proc.stdout}")
        return {"hits_m_per_sec": float(match.group(1))}

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for bench metrics.

        Returns:
            Dict mapping "hits_m_per_sec" to "M/s".
        """
        return {"hits_m_per_sec": "M/s"}


@register
class BpfFentryBenchmark(BpfBenchBase):
    """BPF fentry trampoline overhead benchmark."""

    name = "bpf-fentry"
    bench_name = "trig-fentry"
    affinity = True


@register
class BpfTpBenchmark(BpfBenchBase):
    """BPF tracepoint overhead benchmark."""

    name = "bpf-tp"
    bench_name = "trig-tp"
    affinity = True


@register
class BpfKprobeBenchmark(BpfBenchBase):
    """BPF kprobe overhead benchmark."""

    name = "bpf-kprobe"
    bench_name = "trig-kprobe"
    affinity = True


@register
class BpfLocalStorageBenchmark(BpfBenchBase):
    """BPF local storage benchmark (exercises irq save/restore)."""

    name = "bpf-local-storage"
    bench_name = "local-storage-cache-seq-get"

    def run_once(self) -> dict[str, float]:
        """Run a single local-storage bench iteration.

        The local-storage benchmark reports throughput as
        ``Summary: hits throughput X.XXX +- Y.YYY M ops/s``.

        Returns:
            Dict with "throughput_m_ops_per_sec".

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.bench_cmd(),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"hits throughput\s+([\d.]+)\s*", proc.stdout)
        if not match:
            raise RuntimeError(f"cannot parse bench output: {proc.stdout}")
        return {"throughput_m_ops_per_sec": float(match.group(1))}

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for local-storage metrics.

        Returns:
            Dict mapping metric to unit string.
        """
        return {"throughput_m_ops_per_sec": "M ops/s"}


@register
class BpfHashmapBenchmark(BpfBenchBase):
    """BPF hashmap full update benchmark (exercises spin lock)."""

    name = "bpf-hashmap"
    bench_name = "bpf-hashmap-full-update"
    producers = max(1, (os.cpu_count() or 1) - 1)

    def run_once(self) -> dict[str, float]:
        """Run a single hashmap bench iteration.

        The hashmap benchmark measures spinlock contention during
        concurrent map updates, so it runs with one producer per CPU
        (matching ``run_bench_bpf_hashmap_full_update.sh``).  The output
        has per-CPU lines ``N:hash_map_full_perf XXXXX events per sec``;
        this method sums events across all CPUs.

        Returns:
            Dict with "events_per_sec".

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.bench_cmd(),
            capture_output=True,
            text=True,
            check=True,
        )
        total = 0.0
        for match in re.finditer(r"hash_map_full_perf\s+(\d+)\s+events per sec", proc.stdout):
            total += float(match.group(1))
        if total == 0.0:
            raise RuntimeError(f"cannot parse bench output: {proc.stdout}")
        return {"events_per_sec": total}

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for hashmap metrics.

        Returns:
            Dict mapping metric to unit string.
        """
        return {"events_per_sec": "events/s"}


@register
class BpfKernelCountBenchmark(BpfBenchBase):
    """BPF in-kernel counting benchmark (baseline control)."""

    name = "bpf-kernel-count"
    bench_name = "trig-kernel-count"
    affinity = True
