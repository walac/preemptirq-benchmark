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
    description = "BPF fentry trampoline overhead"
    bench_name = "trig-fentry"
    affinity = True


@register
class BpfTpBenchmark(BpfBenchBase):
    """BPF tracepoint overhead benchmark."""

    name = "bpf-tp"
    description = "BPF tracepoint overhead"
    bench_name = "trig-tp"
    affinity = True


@register
class BpfKprobeBenchmark(BpfBenchBase):
    """BPF kprobe overhead benchmark."""

    name = "bpf-kprobe"
    description = "BPF kprobe overhead"
    bench_name = "trig-kprobe"
    affinity = True


@register
class BpfLocalStorageBenchmark(BpfBenchBase):
    """BPF local storage benchmark (exercises irq save/restore)."""

    name = "bpf-local-storage"
    description = "BPF local storage (irq save/restore)"
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
    description = "BPF hashmap update (spin lock)"
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
    description = "BPF in-kernel counting (baseline)"
    bench_name = "trig-kernel-count"
    affinity = True


@register
class BpfSyscallCountBenchmark(BpfBenchBase):
    """BPF syscall trigger rate benchmark.

    Measures raw syscall overhead by repeatedly calling
    ``syscall(__NR_getpgid)`` with a BPF kprobe attached.  Useful as a
    baseline to isolate BPF attachment cost from map operation costs.
    """

    name = "bpf-syscall-count"
    description = "BPF syscall trigger rate (baseline)"
    bench_name = "trig-syscall-count"
    affinity = True


@register
class BpfHashmapLookupBenchmark(BpfBenchBase):
    """BPF hashmap lookup benchmark.

    Exercises the ``bpf_map_lookup_elem`` and ``bpf_map_update_elem``
    syscall paths with configurable key/value sizes.  The bench tool
    reports per-CPU lines with ``lookup X.XXXM ± Y.YYYM events/sec``;
    this method sums events across all CPUs.
    """

    name = "bpf-hashmap-lookup"
    description = "BPF hashmap lookup/update syscall"
    bench_name = "bpf-hashmap-lookup"
    producers = max(1, (os.cpu_count() or 1) - 1)

    def run_once(self) -> dict[str, float]:
        """Run a single hashmap-lookup iteration.

        Returns:
            Dict with "lookup_m_events_per_sec".

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
        found = False
        for m in re.finditer(
            r"lookup\s+([\d.]+)M\s*(?:[±+-]*\s*[\d.]*M\s*)?events/sec",
            proc.stdout,
        ):
            total += float(m.group(1))
            found = True
        if not found:
            raise RuntimeError(f"cannot parse bench output: {proc.stdout}")
        return {"lookup_m_events_per_sec": total}

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for hashmap-lookup metrics."""
        return {"lookup_m_events_per_sec": "M events/s"}


@register
class BpfLocalStorageCreateBenchmark(BpfBenchBase):
    """BPF local storage creation benchmark.

    Exercises the ``bpf_map_create`` syscall path via socket local
    storage allocation.  Measures how many storage instances can be
    created per second.
    """

    name = "bpf-local-storage-create"
    description = "BPF local storage creation (map_create)"
    bench_name = "local-storage-create"

    def run_once(self) -> dict[str, float]:
        """Run a single local-storage-create iteration.

        Returns:
            Dict with "creates_k_per_sec".

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.bench_cmd(),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"creates\s+([\d.]+)\s*[±+-]*\s*[\d.]*\s*k/s", proc.stdout)
        if not match:
            raise RuntimeError(f"cannot parse bench output: {proc.stdout}")
        return {"creates_k_per_sec": float(match.group(1))}

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for local-storage-create metrics."""
        return {"creates_k_per_sec": "k/s"}


@register
class BpfHtabMemBenchmark(BpfBenchBase):
    """BPF hash table memory operations benchmark.

    Exercises hash table add/delete cycles via ``syscall(__NR_getpgid)``
    triggers, measuring per-producer operation throughput and memory
    usage.
    """

    name = "bpf-htab-mem"
    description = "BPF hash table memory ops (add/del)"
    bench_name = "htab-mem"

    def run_once(self) -> dict[str, float]:
        """Run a single htab-mem iteration.

        Returns:
            Dict with "ops_k_per_sec".

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.bench_cmd(),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"per-prod-op\s+([\d.]+)\s*[±+-]*\s*[\d.]*\s*k/s", proc.stdout)
        if not match:
            raise RuntimeError(f"cannot parse bench output: {proc.stdout}")
        return {"ops_k_per_sec": float(match.group(1))}

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for htab-mem metrics."""
        return {"ops_k_per_sec": "k/s"}


@register
class BpfLpmTrieLookupBenchmark(BpfBenchBase):
    """BPF LPM trie lookup benchmark.

    Exercises the LPM trie map lookup path, reporting throughput and
    per-operation latency.
    """

    name = "bpf-lpm-trie-lookup"
    description = "BPF LPM trie lookup"
    bench_name = "lpm-trie-lookup"
    nr_entries: int = 1000

    def bench_cmd(self) -> list[str]:
        cmd = super().bench_cmd()
        cmd.extend(["--nr_entries", str(self.nr_entries)])
        return cmd

    def run_once(self) -> dict[str, float]:
        """Run a single lpm-trie-lookup iteration.

        Returns:
            Dict with throughput and latency metrics.

        Raises:
            RuntimeError: If the output cannot be parsed.
        """
        proc = subprocess.run(
            self.bench_cmd(),
            capture_output=True,
            text=True,
            check=True,
        )
        result: dict[str, float] = {}
        m = re.search(
            r"throughput\s+([\d.]+)\s*[±+-]*\s*[\d.]*\s*([A-Za-z])\s*ops/s",
            proc.stdout,
        )
        if m:
            result["throughput_ops_per_sec"] = float(m.group(1))
        m = re.search(r"latency\s+([\d.]+)\s*([a-z]+)/op", proc.stdout)
        if m:
            result["latency_per_op"] = float(m.group(1))
        expected = {"throughput_ops_per_sec", "latency_per_op"}
        missing = expected - result.keys()
        if missing:
            raise RuntimeError(
                f"lpm-trie-lookup partial parse failure, "
                f"missing: {', '.join(sorted(missing))}. "
                f"output={proc.stdout[:200]!r}"
            )
        return result

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for lpm-trie-lookup metrics."""
        return {
            "throughput_ops_per_sec": "M ops/s",
            "latency_per_op": "ns/op",
        }
