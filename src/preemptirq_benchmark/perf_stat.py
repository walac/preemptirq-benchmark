from __future__ import annotations

import shutil
import subprocess
import sys

DEFAULT_EVENTS = [
    "cycles",
    "instructions",
    "L1-icache-load-misses",
    "branch-misses",
]


def is_available() -> bool:
    """Check whether the perf binary is installed and accessible.

    Returns:
        True if ``perf`` is found on PATH, False otherwise.
    """
    return shutil.which("perf") is not None


def run_with_perf_stat(
    cmd: list[str],
    events: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, int]]:
    """Execute a command under ``perf stat`` and return parsed counters.

    The command is prefixed with
    ``perf stat -e <events> -x ";" -- <cmd>`` so that perf writes
    machine-parseable CSV to stderr while the benchmark's own stdout
    is passed through unmodified.

    Args:
        cmd: The benchmark command and arguments to execute.
        events: Hardware counter event names to collect.  Defaults to
            :data:`DEFAULT_EVENTS` (cycles, instructions,
            L1-icache-load-misses, branch-misses).

    Returns:
        A tuple of (CompletedProcess, counters) where *counters* is a
        dict mapping event name to its integer count.  Events that perf
        could not measure (e.g. ``<not supported>``) are omitted.
    """
    if events is None:
        events = DEFAULT_EVENTS

    perf_cmd = [
        "perf",
        "stat",
        "-e",
        ",".join(events),
        "-x",
        ";",
        "--",
    ] + cmd

    proc = subprocess.run(perf_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            f"Warning: perf stat wrapped command exited with code {proc.returncode}",
            file=sys.stderr,
        )
    counters = parse_perf_csv(proc.stderr)
    return proc, counters


def parse_perf_csv(stderr: str) -> dict[str, int]:
    """Parse perf stat CSV output (``-x ";"`` format) into a dict.

    Each line has the form ``<count>;<unit>;<event>;...`` where *count*
    may be ``<not counted>`` or ``<not supported>`` for unavailable
    counters.

    Args:
        stderr: Raw stderr output from ``perf stat -x ";"``.

    Returns:
        Dict mapping event name to integer count.  Unparseable or
        unsupported counters are silently skipped.
    """
    counters: dict[str, int] = {}
    for line in stderr.strip().splitlines():
        parts = line.split(";")
        if len(parts) < 3:
            continue
        count_str = parts[0].strip()
        event_name = parts[2].strip()
        if not event_name or count_str.startswith("<"):
            continue
        try:
            counters[event_name] = int(count_str)
        except ValueError:
            continue
    return counters
