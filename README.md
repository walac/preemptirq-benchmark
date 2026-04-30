# preemptirq-benchmark

Benchmark suite for measuring Linux kernel preemptirq tracepoint overhead.
Runs 9 benchmarks across different kernel subsystems, collects statistical
results, saves JSON reports, and supports multi-report comparison with
significance testing.

## Benchmarks

| Name | Focus | Default Iterations |
|------|-------|--------------------|
| hackbench | Scheduler/IPC stress | 10 |
| iperf3 | Networking throughput and jitter | 10 |
| fio | I/O interrupt stress (null_blk + io_uring) | 10 |
| stress-ng | Context switch saturation | 10 |
| cyclictest | RT scheduling latency | 30 |
| perf-bench | In-tree scheduler benchmarks | 10 |
| kernel-compile | Kernel build throughput (make -j) | 10 |
| rtla | RT latency (timerlat + osnoise) | 30 |
| tracerbench | Kernel module micro-benchmark (CPU cycles) | 5 |

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- Linux with the benchmark tools installed (see below)
- Root access for benchmarks that require it (fio, cyclictest, rtla, tracerbench,
  perf stat)
- The [`tracerbench`](https://github.com/walac/tracer-benchmark/) kernel module for the tracerbench benchmark
- A configured kernel source tree for the kernel-compile benchmark

### RPM packages (Fedora/RHEL)

```bash
sudo dnf install \
    realtime-tests \
    iperf3 \
    fio \
    stress-ng \
    perf \
    rtla \
    time
```

| Package | Provides |
|---------|----------|
| `realtime-tests` | hackbench, cyclictest |
| `iperf3` | iperf3 |
| `fio` | fio |
| `stress-ng` | stress-ng |
| `perf` | perf bench, perf stat |
| `rtla` | rtla timerlat, rtla osnoise |
| `time` | /usr/bin/time (used by kernel-compile) |

The `kernel-compile` benchmark also requires kernel build dependencies
(`gcc`, `make`, `flex`, `bison`, `elfutils-libelf-devel`, etc.).

## Installation

```bash
uv sync
```

## Usage

### Run benchmarks

```bash
# Run all benchmarks (skips those with unmet prerequisites)
sudo uv run preemptirq-benchmark run

# Run specific benchmarks
sudo uv run preemptirq-benchmark run --include=hackbench,perf-bench

# Run all except some
sudo uv run preemptirq-benchmark run --exclude=kernel-compile,rtla

# Override iteration count
sudo uv run preemptirq-benchmark run --include=hackbench --iterations 20

# Collect hardware counters alongside benchmarks
sudo uv run preemptirq-benchmark run --include=hackbench --perf-stat

# Include kernel-compile (requires --kernel-src)
sudo uv run preemptirq-benchmark run --include=kernel-compile --kernel-src /path/to/linux

# Configure tracerbench module parameters
sudo uv run preemptirq-benchmark run --include=tracerbench \
    --samples 50000 --highest 250 --percentile 99
```

The `run` subcommand saves a JSON report to the current directory
(e.g., `preemptirq-benchmark-6.14.0-{timestamp}.json`) and prints an
ASCII summary to the terminal. Use `-o` to specify a custom report path.
Use `show` to re-display in a different format.

### Show a saved report

```bash
uv run preemptirq-benchmark show report.json
uv run preemptirq-benchmark show report.json --format markdown
uv run preemptirq-benchmark show report.json --format txt
uv run preemptirq-benchmark show report.json -o summary.md
```

### Compare reports

The first report is the baseline. Subsequent reports show percentage deltas
with Mann-Whitney U significance testing.

```bash
uv run preemptirq-benchmark compare baseline.json patched.json
uv run preemptirq-benchmark compare baseline.json v1.json v2.json --format markdown
uv run preemptirq-benchmark compare baseline.json patched.json -o comparison.txt
```

Comparison output shows:

- Baseline values in absolute units
- Other reports as +/-% relative to baseline
- Significance markers: `(**)` p<0.01, `(*)` p<0.05, `(ns)` not significant

## Output formats (show and compare)

The `--format` option is available on the `show` and `compare` subcommands.

Use `-o`/`--output` to write the output directly to a file. The format
is automatically inferred from the file extension (`.txt`, `.md`,
`.markdown`, `.json`, `.ascii`). If `--format` is explicitly provided,
it overrides the extension inference.

| Format | Description |
|--------|-------------|
| `ascii` | Rich tables with box-drawing characters and color coding (default) |
| `txt` | Plain-text tables using `+`, `-`, `\|` characters |
| `markdown` | GitHub-flavored markdown tables |
| `json` | Structured JSON output |

## Statistics

Each metric is reported with:

- **Mean** -- arithmetic mean across iterations
- **Median** -- middle value
- **StdDev** -- sample standard deviation (Bessel's correction)
- **95% CI** -- confidence interval using t-distribution

Comparisons include:

- **Delta %** -- percentage change from baseline
- **Mann-Whitney U** -- non-parametric significance test

## Benchmark selection rules

| Flags | Behavior |
|-------|----------|
| (none) | Run all benchmarks |
| `--all` | Same as default |
| `--include=a,b` | Run only listed benchmarks |
| `--exclude=a,b` | Run all except listed |
| `--include` + `--all` | Error (mutually exclusive) |
| `--include` + `--exclude` | Error (mutually exclusive) |

## Development

```bash
# Install with dev dependencies
uv sync

# Run linter
uv run ruff check src/

# Run type checker
uv run pyright src/
```

## License

GPL-2.0-only
