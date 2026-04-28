from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

ALL_BENCHMARK_NAMES = [
    "hackbench",
    "iperf3",
    "fio",
    "stress-ng",
    "cyclictest",
    "perf-bench",
    "kernel-compile",
    "rtla",
    "tracerbench",
]

BENCHMARK_DESCRIPTIONS = {
    "hackbench": "Scheduler/IPC stress",
    "iperf3": "Networking throughput and jitter",
    "fio": "I/O interrupt stress (null_blk + io_uring)",
    "stress-ng": "Context switch saturation",
    "cyclictest": "RT scheduling latency",
    "perf-bench": "In-tree scheduler benchmarks",
    "kernel-compile": "Kernel build throughput (make -j)",
    "rtla": "RT latency (timerlat + osnoise)",
    "tracerbench": "Kernel module micro-benchmark (CPU cycles)",
}


@dataclass
class BenchmarkResult:
    """Collected data from one benchmark across all iterations.

    Attributes:
        name: Benchmark identifier (e.g. "hackbench").
        metrics: Mapping of metric name to list of per-iteration values.
        units: Mapping of metric name to its unit string (e.g. "s", "ns").
        perf_counters: Mapping of perf event name to list of per-iteration
            counts.  Empty when perf stat is disabled or not applicable.
        iterations: Number of completed iterations.
    """

    name: str
    metrics: dict[str, list[float]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    perf_counters: dict[str, list[int]] = field(default_factory=dict)
    iterations: int = 0


class BenchmarkBase(ABC):
    """Abstract base for all benchmark implementations.

    Subclasses must set :attr:`name` and :attr:`default_iterations` as
    class attributes and implement :meth:`check_prerequisites` and
    :meth:`run_once`.

    Attributes:
        name: Short identifier used in CLI flags and report keys.
        default_iterations: How many times to repeat when --iterations
            is not specified.
        supports_perf_stat: Whether this benchmark can be wrapped with
            ``perf stat``.  Defaults to True; tracerbench sets this to
            False.
    """

    name: str
    default_iterations: int
    supports_perf_stat: bool = True

    @abstractmethod
    def check_prerequisites(self) -> tuple[bool, str]:
        """Verify that all dependencies are satisfied.

        This method is called once before any benchmarks run.  If it
        returns False, the entire benchmark suite aborts.

        Returns:
            A tuple of (ok, message).  When *ok* is False, *message*
            explains what is missing and how to fix it.
        """

    def configure(self, **kwargs: object) -> None:
        """Accept CLI parameters relevant to this benchmark.

        Called after instantiation to pass benchmark-specific options
        such as ``kernel_src``.  The default implementation is a no-op.

        Args:
            kwargs: Keyword arguments from the CLI parser.
        """

    def setup(self) -> None:
        """One-time setup before the iteration loop.

        Override to start servers, load modules, etc.  The default
        implementation is a no-op.
        """

    @abstractmethod
    def run_once(self) -> dict[str, float]:
        """Execute a single iteration and return measured values.

        Returns:
            Dict mapping metric name to its numeric value for this
            iteration.  The keys must be consistent across calls.
        """

    def get_command(self) -> list[str] | None:
        """Return the shell command for perf stat wrapping.

        Returns:
            A command list suitable for :func:`subprocess.run`, or
            None if this benchmark does not use an external command
            (e.g. tracerbench interacts via debugfs).
        """
        return None

    def cleanup(self) -> None:
        """Cleanup after all iterations complete.

        Override to kill servers, unload modules, etc.  The default
        implementation is a no-op.
        """

    def get_units(self) -> dict[str, str]:
        """Return a mapping of metric name to unit string.

        Returns:
            Dict like ``{"time_seconds": "s", "iops": "ops/s"}``.
        """
        return {}


REGISTRY: dict[str, type[BenchmarkBase]] = {}


def register(cls: type[BenchmarkBase]) -> type[BenchmarkBase]:
    """Class decorator that adds a benchmark to the global registry.

    Args:
        cls: A concrete BenchmarkBase subclass with a ``name`` attribute.

    Returns:
        The same class, unmodified.
    """
    REGISTRY[cls.name] = cls
    return cls


def get_benchmark(name: str) -> BenchmarkBase:
    """Instantiate a registered benchmark by name.

    Args:
        name: Benchmark identifier (e.g. "hackbench").

    Returns:
        A new instance of the corresponding benchmark class.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    return REGISTRY[name]()


def check_all_prerequisites(
    benchmarks: list[BenchmarkBase],
) -> None:
    """Check prerequisites for all selected benchmarks and abort on failure.

    Prints a status line for each benchmark ([OK] or [FAIL]) and
    aborts the process if any prerequisite is not satisfied.

    Args:
        benchmarks: List of instantiated benchmark objects to validate.

    Raises:
        SystemExit: If any benchmark's prerequisites are not met.
    """
    failed = False
    print("Checking prerequisites...")
    for bench in benchmarks:
        ok, msg = bench.check_prerequisites()
        if ok:
            print(f"  [OK]   {bench.name}")
        else:
            print(f"  [FAIL] {bench.name}: {msg}")
            failed = True

    if failed:
        print(
            "\nAborting: fix the above prerequisites before running.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print()


def resolve_benchmarks(
    include: str | None,
    exclude: str | None,
    all_flag: bool,
) -> list[str]:
    """Resolve CLI flags into a list of benchmark names to run.

    Args:
        include: Comma-separated benchmark names, or None.
        exclude: Comma-separated benchmark names to remove, or None.
        all_flag: True if --all was explicitly passed.

    Returns:
        Ordered list of benchmark names.

    Raises:
        SystemExit: On invalid combinations or unknown benchmark names.
    """
    if include and all_flag:
        print("Error: --include and --all are mutually exclusive", file=sys.stderr)
        raise SystemExit(1)
    if include and exclude:
        print("Error: --include and --exclude are mutually exclusive", file=sys.stderr)
        raise SystemExit(1)

    if include:
        selected = [b.strip() for b in include.split(",")]
        invalid = set(selected) - set(ALL_BENCHMARK_NAMES)
        if invalid:
            print(
                f"Error: unknown benchmarks: {', '.join(sorted(invalid))}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return selected

    selected = ALL_BENCHMARK_NAMES[:]
    if exclude:
        to_remove = {b.strip() for b in exclude.split(",")}
        invalid = to_remove - set(ALL_BENCHMARK_NAMES)
        if invalid:
            print(
                f"Error: unknown benchmarks: {', '.join(sorted(invalid))}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        selected = [b for b in selected if b not in to_remove]
        if not selected:
            print("Error: --exclude removed all benchmarks", file=sys.stderr)
            raise SystemExit(1)

    return selected


def import_all() -> None:
    """Import all benchmark modules to trigger registration.

    Must be called once before :func:`get_benchmark` to ensure all
    :func:`register`-decorated classes are loaded.
    """
    from preemptirq_benchmark.benchmarks import (  # noqa: F401
        cyclictest,
        fio,
        hackbench,
        iperf3_bench,
        kernel_compile,
        perf_bench,
        rtla,
        stress_ng,
        tracerbench,
    )
