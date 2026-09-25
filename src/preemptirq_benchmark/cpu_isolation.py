from __future__ import annotations

from pathlib import Path

ISOLATED_CPUS_PATH = Path("/sys/devices/system/cpu/isolated")
ONLINE_CPUS_PATH = Path("/sys/devices/system/cpu/online")


def _parse_cpu_list(raw: str) -> list[int]:
    cpus: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(cpus)


def get_isolated_cpus() -> list[int]:
    """Read the set of CPUs isolated via the ``isolcpus=`` boot parameter.

    Returns:
        Sorted list of isolated CPU ids, parsed from
        ``/sys/devices/system/cpu/isolated`` (a comma-separated list of ids
        and/or ranges, e.g. ``"2-3,7"``). Empty if the file is missing,
        unreadable, or reports no isolated CPUs.
    """
    try:
        raw = ISOLATED_CPUS_PATH.read_text().strip()
    except OSError:
        return []

    if not raw:
        return []

    try:
        return _parse_cpu_list(raw)
    except ValueError:
        return []


def get_online_cpus() -> list[int]:
    """Return the CPUs monitored by rtla when no ``-c`` filter is used."""
    try:
        cpus = _parse_cpu_list(ONLINE_CPUS_PATH.read_text().strip())
    except (OSError, ValueError) as e:
        raise RuntimeError("cannot determine online CPUs for rtla report") from e
    if not cpus:
        raise RuntimeError("cannot determine online CPUs for rtla report")
    return cpus


def format_cpu_list(cpus: list[int]) -> str:
    """Format a list of CPU ids as a comma-separated CLI argument.

    Args:
        cpus: CPU ids, e.g. from :func:`get_isolated_cpus`.

    Returns:
        Comma-separated string suitable for rtla's ``-c`` or cyclictest's
        ``-a`` flag, e.g. ``"2,3,7"``.
    """
    return ",".join(str(c) for c in cpus)
