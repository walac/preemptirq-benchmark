# preemptirq-benchmark

Benchmark suite for measuring Linux kernel preemptirq tracepoint overhead.
Runs 15 benchmarks across different kernel subsystems, collects statistical
results, saves JSON reports, and supports multi-report comparison with
significance testing.

The tool works by wrapping standard kernel testing tools (hackbench, fio,
cyclictest, perf bench, etc.), running them with controlled iteration
counts, and computing descriptive statistics with confidence intervals.
Multi-report comparison uses the Mann-Whitney U test to flag statistically
significant regressions or improvements. A separate `codegen-overhead`
tool statically compares vmlinux binaries to measure per-function
instruction count overhead from tracepoint instrumentation.

## Quick start

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/walac/preemptirq-benchmark.git
cd preemptirq-benchmark
uv sync

# Run all benchmarks (skips those with unmet prerequisites)
sudo uv run preemptirq-benchmark run

# Compare a baseline against a patched kernel
uv run preemptirq-benchmark compare baseline.json patched.json
```

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
| bpf-fentry | BPF fentry trampoline overhead | 5 |
| bpf-tp | BPF tracepoint overhead | 5 |
| bpf-kprobe | BPF kprobe overhead | 5 |
| bpf-local-storage | BPF local storage (irq save/restore) | 5 |
| bpf-hashmap | BPF hashmap update (spin lock) | 5 |
| bpf-kernel-count | BPF in-kernel counting (baseline) | 5 |

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- Linux with the benchmark tools installed (see below)
- Root access for benchmarks that require it (fio, cyclictest, rtla, tracerbench,
  perf stat, BPF benchmarks)
- The [`tracerbench`](https://github.com/walac/tracer-benchmark/) kernel module for the tracerbench benchmark
- A configured kernel source tree for the kernel-compile benchmark
- The BPF selftests `bench` binary for BPF benchmarks (build from kernel source:
  `make -C tools/testing/selftests/bpf bench`)

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

[uv](https://github.com/astral-sh/uv) is a fast Python package manager
that creates an isolated virtual environment automatically.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv
uv sync                                            # install dependencies
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
# Note: silently skipped by tracerbench and bpf-* benchmarks
sudo uv run preemptirq-benchmark run --include=hackbench --perf-stat

# Use a 99% confidence interval instead of the default 95%
sudo uv run preemptirq-benchmark run --include=hackbench --confidence-interval 99

# Include kernel-compile (requires --kernel-src)
sudo uv run preemptirq-benchmark run --include=kernel-compile --kernel-src /path/to/linux

# Configure tracerbench module parameters
sudo uv run preemptirq-benchmark run --include=tracerbench \
    --samples 50000 --highest 250 --percentile 99

# Run BPF benchmarks (requires bench binary from kernel selftests)
sudo uv run preemptirq-benchmark run --include=bpf-fentry,bpf-tp,bpf-kprobe

# Specify a custom path to the bench binary
sudo uv run preemptirq-benchmark run --include=bpf-fentry \
    --bpf-bench /path/to/bench
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

### Analyze code generation overhead

The `codegen-overhead` tool statically compares two vmlinux binaries -- a
baseline without tracepoints and a target with `TRACE_PREEMPT_TOGGLE` /
`TRACE_IRQFLAGS_TOGGLE` enabled -- to measure the per-function instruction
overhead introduced by the instrumentation.

```bash
# Compare two builds (terminal output)
uv run codegen-overhead --base vmlinux.base --target vmlinux.target

# Export a pandoc-ready markdown report
uv run codegen-overhead --base vmlinux.base --target vmlinux.target -o report.md

# Cross-compile analysis (e.g. ARM64 kernels on x86)
uv run codegen-overhead --base vmlinux.base --target vmlinux.target \
    --cross-compile aarch64-linux-gnu-

# Filter out functions likely affected by compiler inlining differences
uv run codegen-overhead --base vmlinux.base --target vmlinux.target \
    --filter-inlining

# Sort by absolute instruction difference
uv run codegen-overhead --base vmlinux.base --target vmlinux.target \
    --sort diff

# Dump disassembly of specific functions for manual inspection
uv run codegen-overhead --base vmlinux.base --target vmlinux.target \
    --functions=schedule,__switch_to --output-asm-dir asm/

# Only dump assembly, skip the comparison analysis
uv run codegen-overhead --base vmlinux.base --target vmlinux.target \
    --functions=schedule --no-analysis

# Export as plain text or JSON
uv run codegen-overhead --base vmlinux.base --target vmlinux.target --format txt
uv run codegen-overhead --base vmlinux.base --target vmlinux.target --format json

# Override objdump options (e.g. interleave source)
uv run codegen-overhead --base vmlinux.base --target vmlinux.target \
    --objdump-extra-args='-S'
```

Functions that trigger inlining-difference heuristics are marked with `*`
in the output. Use `--filter-inlining` to exclude them instead.

## Output formats

The `--format` option is available on the `show`, `compare`, and
`codegen-overhead` subcommands.

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
- **CI** -- confidence interval using t-distribution (default 95%, configurable
  via `--confidence-interval`)

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

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov --cov-report=term-missing

# Run formatter check
uv run black --check src/

# Run import sort check
uv run isort --check-only --diff src/

# Run type checker
uv run pyright src/
```

## License

GPL-2.0-only
