#!/usr/bin/env python3
"""Compare kernel vmlinux binaries for tracepoint code generation overhead.

Disassembles two vmlinux builds — a baseline without tracepoints and a target
with TRACE_PREEMPT_TOGGLE / TRACE_IRQFLAGS_TOGGLE — then reports per-function
instruction count deltas and trace call site counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from preemptirq_benchmark.formatters import format_table

# Trace helper symbols injected by the preemptirq patch series, mapped to
# short labels used in the breakdown column of the report.
TRACE_HELPERS = {
    "__trace_preempt_on": "p+",
    "__trace_preempt_off": "p-",
    "trace_local_irq_enable": "ie",
    "trace_local_irq_disable": "id",
    "trace_local_irq_save": "is",
    "trace_local_irq_restore": "ir",
    "trace_safe_halt": "sh",
}

# ---------------------------------------------------------------------------
# Objdump output patterns
# ---------------------------------------------------------------------------
#
# FUNC_RE — matches function header lines in objdump -d output:
#   ffffffff81f823a0 <__alloc_skb>:
#   ^^^^^^^^^^^^^^^    ^^^^^^^^^^^
#   hex address         function name (captured in group 1)
#
# [^>]+ captures any character except '>', so it handles all symbol names
# including those with underscores, dots (.cold, .isra.0), or other
# compiler-generated suffixes. The trailing ':$' anchors to end-of-line
# to avoid matching symbolic references within instruction operands.
FUNC_RE = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")

# INSN_RE — matches instruction lines (as opposed to blank lines,
# section headers, or source annotations):
#   ffffffff81f823a4:  call   ffffffff81390910 <__fentry__>
#   ^^^^^^^^^^^^^^^^
#   optional whitespace + hex address + colon
#
# The \s* prefix handles both vmlinux (no indent) and relocatable
# object files (indented addresses) across objdump versions.
INSN_RE = re.compile(r"^\s*[0-9a-f]+:")

# CALL_RE — matches call/branch-and-link instructions that target a
# named symbol, across multiple architectures:
#
#   x86:   call   ffffffff8151e020 <trace_local_irq_restore>
#   x86:   callq  ffffffff8151e020 <trace_local_irq_restore>
#   arm:   bl     ffff800080123456 <trace_local_irq_restore>
#   ppc:   bl     c000000000123456 <trace_local_irq_restore>
#   s390:  brasl  %r14,0000000000123456 <trace_local_irq_restore>
#   riscv: jal    ra,ffffffff80123456 <trace_local_irq_restore>
#   riscv: jalr   ra,0(a5)  — no symbol, won't match (indirect call)
#
# The \b word boundary after the mnemonic prevents partial matches
# (e.g. 'bla' or 'calls' won't match). The symbol name is captured
# in group 1 and checked against TRACE_HELPERS.
CALL_RE = re.compile(r"(?:callq?|bl|brasl|jalr?)\b.*<([^>]+)>")

# NOP_RE — matches x86 NOP instruction mnemonics used for alignment
# padding between functions.  These include single-byte ``nop``,
# multi-byte ``nopl``/``nopw`` variants, prefixed forms (``data16``,
# ``cs``), and the two-byte ``xchg %ax,%ax`` encoding.
#
# Used to strip trailing alignment padding from per-function instruction
# counts — the linker inserts NOPs between function boundaries to satisfy
# alignment constraints, and these must not inflate the count.
#
# Anchored with ^ to match only at the start of the mnemonic field
# (callers must strip the address prefix before matching).  Without
# anchoring, a ``search()`` over the full objdump line would false-
# positive on symbol names containing "nop" (e.g. ``<__kmalloc_noprof>``).
NOP_RE = re.compile(r"^(?:data16\s+)*(?:cs\s+)?nop[lwq]?\b|^xchg\s+%([a-d]x),%\1")

# Inlining-difference thresholds — functions exceeding these are flagged
# as likely artefacts of unrelated compiler inlining decisions rather
# than tracepoint overhead (excluded when --filter-inlining is used).
MAX_DIFF_PER_CALL = 20
MAX_PCT_CHANGE = 100
MAX_SHRINK_PER_CALL = 10
# Minimum base instruction count for the percentage-change heuristic.
# Functions smaller than this are not flagged by MAX_PCT_CHANGE alone,
# because a 3-instruction function gaining 4 instructions (+133%) is
# normal tracepoint overhead, not an inlining artefact.
MIN_BASE_FOR_PCT_CHECK = 10

DEFAULT_OBJDUMP_ARGS: tuple[str, ...] = ("-d", "--no-show-raw-insn")


@dataclass
class FuncTrace:
    """Trace call profile for a single function in the target build.

    Tracks instruction count and per-helper call counts, used to
    determine which functions are affected by the tracepoint
    instrumentation and by how much.
    """

    insn_count: int = 0
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total_calls(self) -> int:
        """Total number of trace helper call sites in this function."""
        return sum(self.calls.values())

    def breakdown(self) -> str:
        """Compact summary of which helpers are called, e.g. ``p+2 p-1 is1 ir1``."""
        parts = []
        for helper, label in TRACE_HELPERS.items():
            n = self.calls.get(helper, 0)
            if n > 0:
                parts.append(f"{label}{n}")
        return " ".join(parts)


@dataclass
class CompareRow:
    """One row in the comparison report — a single function present in both builds."""

    name: str
    base_insns: int
    target_insns: int
    diff: int
    pct: float
    total_calls: int
    avg_per_call: float
    breakdown: str
    inlining_suspect: bool = False


@dataclass
class Summary:
    """Aggregate statistics across all compared functions."""

    functions_analyzed: int
    functions_skipped_missing: int
    functions_filtered_inlining: int
    functions_flagged_inlining: int
    total_base: int
    total_target: int
    total_diff: int
    total_pct: float
    total_calls: int
    avg_per_call: float
    dist_min: float
    dist_p25: float
    dist_median: float
    dist_p75: float
    dist_p95: float
    dist_max: float


def stream_objdump(
    vmlinux: str,
    cross_compile: str = "",
    objdump_args: list[str] | None = None,
) -> Iterator[str]:
    """Stream disassembly output line-by-line from objdump.

    Yields lines without buffering the entire output into memory,
    which is critical for vmlinux binaries that produce ~7M lines.

    Args:
        vmlinux: Path to the ELF binary to disassemble.
        cross_compile: Toolchain prefix (e.g. ``aarch64-linux-gnu-``).
        objdump_args: Arguments passed to objdump. Defaults to
            :data:`DEFAULT_OBJDUMP_ARGS` if not specified.

    Yields:
        Individual lines from objdump stdout.

    Raises:
        FileNotFoundError: If *vmlinux* does not exist or objdump is
            not found in PATH.
        RuntimeError: If objdump exits with a non-zero status.
    """
    if not os.path.isfile(vmlinux):
        raise FileNotFoundError(f"vmlinux not found: {vmlinux}")
    objdump = f"{cross_compile}objdump"
    if not shutil.which(objdump):
        raise FileNotFoundError(f"{objdump} not found in PATH")

    args = objdump_args if objdump_args is not None else DEFAULT_OBJDUMP_ARGS
    proc = subprocess.Popen(
        [objdump, *args, vmlinux],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        yield from proc.stdout
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"objdump failed on {vmlinux} (exit {proc.returncode})")


def extract_function_data(
    vmlinux: str,
    progress: Progress,
    task: TaskID,
    cross_compile: str = "",
    objdump_args: list[str] | None = None,
    track_trace_calls: bool = False,
) -> dict[str, FuncTrace]:
    """Count instructions per function, optionally tracking trace helper calls.

    Single streaming pass over objdump output. For each function,
    counts total instructions. When *track_trace_calls* is True, also
    records which trace helpers are called and how many times, and only
    includes functions with at least one trace helper call in the result.

    Args:
        vmlinux: Path to the vmlinux ELF binary.
        progress: Rich progress instance for status updates.
        task: Progress task ID.
        cross_compile: Toolchain prefix for objdump.
        objdump_args: Custom objdump arguments.
        track_trace_calls: When True, detect calls to trace helpers and
            only return functions that have at least one. When False,
            return all functions with their instruction counts.

    Returns:
        Mapping of function name to its :class:`FuncTrace` data.
    """
    result: dict[str, FuncTrace] = {}
    current_func: str | None = None
    current_data = FuncTrace()
    trailing_nops = 0

    def save_current() -> None:
        current_data.insn_count -= trailing_nops
        if current_func and current_data.insn_count > 0:
            if not track_trace_calls or current_data.total_calls > 0:
                result[current_func] = current_data

    for line in stream_objdump(vmlinux, cross_compile, objdump_args):
        m = FUNC_RE.match(line)
        if m:
            save_current()
            current_func = m.group(1)
            current_data = FuncTrace()
            trailing_nops = 0
            progress.update(task, advance=1)
            continue

        if current_func and INSN_RE.match(line):
            current_data.insn_count += 1
            tab = line.find("\t")
            mnemonic = line[tab + 1 :] if tab >= 0 else line
            if NOP_RE.match(mnemonic):
                trailing_nops += 1
            else:
                trailing_nops = 0
            if track_trace_calls:
                cm = CALL_RE.search(line)
                if cm and cm.group(1) in TRACE_HELPERS:
                    current_data.calls[cm.group(1)] += 1

    save_current()

    return result


def build_comparison(
    target_data: dict[str, FuncTrace],
    base_data: dict[str, FuncTrace],
    *,
    filter_inlining: bool = False,
) -> tuple[list[CompareRow], Summary]:
    """Join target and base data, compute deltas, and detect outliers.

    For each function present in both builds, computes the instruction
    count difference and percentage change. Functions where the delta
    is disproportionate to the number of trace call sites are flagged
    as likely artefacts of unrelated compiler inlining decisions.

    When *filter_inlining* is True, flagged functions are excluded from
    the report. When False (the default), they are included and marked.

    Args:
        target_data: Per-function trace data from the target build.
        base_data: Per-function data from the base build.
        filter_inlining: When True, exclude inlining-suspect functions
            instead of marking them.

    Returns:
        A tuple of (comparison rows, aggregate summary statistics).
    """
    rows: list[CompareRow] = []
    skipped_missing = 0
    skipped_inlining = 0
    flagged_inlining = 0

    for name in sorted(target_data):
        td = target_data[name]
        if name not in base_data:
            skipped_missing += 1
            continue

        base = base_data[name].insn_count
        target = td.insn_count
        diff = target - base

        if base == 0:
            continue

        pct = (diff / base) * 100
        tc = td.total_calls

        suspect = False
        if tc > 0 and abs(diff) / tc > MAX_DIFF_PER_CALL:
            suspect = True
        elif abs(pct) > MAX_PCT_CHANGE and base >= MIN_BASE_FOR_PCT_CHECK:
            suspect = True
        elif diff < 0 and abs(diff) > tc * MAX_SHRINK_PER_CALL:
            suspect = True

        if suspect and filter_inlining:
            skipped_inlining += 1
            continue

        if suspect:
            flagged_inlining += 1

        apc = diff / tc if tc > 0 else 0.0
        rows.append(CompareRow(name, base, target, diff, pct, tc, apc, td.breakdown(), suspect))

    total_base = sum(r.base_insns for r in rows)
    total_target = sum(r.target_insns for r in rows)
    total_diff = total_target - total_base
    total_pct = (total_diff / total_base * 100) if total_base else 0
    total_calls = sum(r.total_calls for r in rows)
    avg_per_call = total_diff / total_calls if total_calls else 0

    per_call_vals = sorted(
        r.diff / r.total_calls for r in rows if r.total_calls > 0 and not r.inlining_suspect
    )
    n = len(per_call_vals)
    p25 = median = p75 = p95 = 0.0
    if n >= 2:
        q = statistics.quantiles(per_call_vals, n=20)
        p25, median, p75, p95 = q[4], q[9], q[14], q[18]
    elif n == 1:
        p25 = median = p75 = p95 = per_call_vals[0]

    summary = Summary(
        functions_analyzed=len(rows),
        functions_skipped_missing=skipped_missing,
        functions_filtered_inlining=skipped_inlining,
        functions_flagged_inlining=flagged_inlining,
        total_base=total_base,
        total_target=total_target,
        total_diff=total_diff,
        total_pct=total_pct,
        total_calls=total_calls,
        avg_per_call=avg_per_call,
        dist_min=per_call_vals[0] if n else 0,
        dist_p25=p25,
        dist_median=median,
        dist_p75=p75,
        dist_p95=p95,
        dist_max=per_call_vals[-1] if n else 0,
    )

    return rows, summary


def output_markdown(rows: list[CompareRow], summary: Summary, path: str) -> None:
    """Write the comparison report as pandoc-ready markdown.

    Produces a self-contained markdown file with YAML frontmatter
    configured for landscape PDF output via pandoc (extarticle, 8pt,
    longtable). Underscores and pipes in function names are escaped
    for correct rendering in both LaTeX/PDF and plain-text outputs.

    Args:
        rows: Comparison data rows.
        summary: Aggregate statistics.
        path: Destination file path.
    """
    with open(path, "w") as f:
        f.write("---\n")
        f.write("geometry: landscape,margin=1.5cm\n")
        f.write("documentclass: extarticle\n")
        f.write("fontsize: 8pt\n")
        f.write("header-includes:\n")
        f.write("  - \\usepackage{longtable}\n")
        f.write("  - \\usepackage{booktabs}\n")
        f.write("---\n\n")
        f.write("# Tracepoint Code Generation Overhead\n\n")
        f.write(
            "Breakdown legend: p+=preempt\\_on, p-=preempt\\_off, "
            "ie=irq\\_enable, id=irq\\_disable, is=irq\\_save, "
            "ir=irq\\_restore, sh=safe\\_halt.\n\n"
        )

        f.write("| Function | Base | Trace | Diff | Diff% | Calls | Avg/Call | Breakdown |\n")
        f.write("|:---------|-----:|------:|-----:|------:|------:|--------:|:----------|\n")

        has_suspects = False
        for r in rows:
            fn = r.name.replace("|", r"\|").replace("_", r"\_")
            if r.inlining_suspect:
                fn += " \\*"
                has_suspects = True
            sign = "+" if r.diff >= 0 else ""
            avg_call = f"{r.avg_per_call:.1f}" if r.total_calls > 0 else "-"
            f.write(
                f"| {fn} | {r.base_insns} | {r.target_insns} "
                f"| {sign}{r.diff} | {r.pct:+.1f}% "
                f"| {r.total_calls} | {avg_call} | {r.breakdown} |\n"
            )

        if has_suspects:
            f.write("\n\\* likely affected by compiler inlining differences.\n")

        s = summary
        f.write("\n## Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|:-------|------:|\n")
        f.write(f"| Functions analyzed | {s.functions_analyzed} |\n")
        f.write(f"| Functions skipped (not in both builds) | {s.functions_skipped_missing} |\n")
        if s.functions_filtered_inlining:
            f.write(f"| Functions filtered (inlining diffs) | {s.functions_filtered_inlining} |\n")
        if s.functions_flagged_inlining:
            f.write(f"| Functions flagged (inlining diffs) | {s.functions_flagged_inlining} |\n")
        f.write(f"| Total baseline instructions | {s.total_base:,} |\n")
        f.write(f"| Total traced instructions | {s.total_target:,} |\n")
        f.write(f"| Total difference | {s.total_diff:+,} ({s.total_pct:+.2f}%) |\n")
        f.write(f"| Total trace call sites | {s.total_calls:,} |\n")
        f.write(f"| Avg overhead per call site | {s.avg_per_call:.1f} insns |\n")

        f.write("\n## Overhead Distribution (instructions/call)\n\n")
        f.write("| Stat | Value |\n")
        f.write("|:-----|------:|\n")
        f.write(f"| Min | {s.dist_min:.1f} |\n")
        f.write(f"| P25 | {s.dist_p25:.1f} |\n")
        f.write(f"| Median | {s.dist_median:.1f} |\n")
        f.write(f"| P75 | {s.dist_p75:.1f} |\n")
        f.write(f"| P95 | {s.dist_p95:.1f} |\n")
        f.write(f"| Max | {s.dist_max:.1f} |\n")


def output_terminal(rows: list[CompareRow], summary: Summary) -> None:
    """Render the comparison report as a rich table to stdout.

    Diff values are colour-coded by severity: green for negative
    (function shrank), white for small positive (<=10%), yellow for
    moderate (<=30%), red for large (>30%).

    Args:
        rows: Comparison data rows.
        summary: Aggregate statistics.
    """
    console = Console()
    if console.width < 120:
        console = Console(width=120)
    s = summary

    table = Table(
        title="Tracepoint Code Generation Overhead",
        show_lines=False,
        pad_edge=False,
        expand=True,
    )
    table.add_column("Function", style="cyan", no_wrap=True, ratio=4)
    table.add_column("Base", justify="right")
    table.add_column("Trace", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("Diff%", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Avg/Call", justify="right")
    table.add_column("Breakdown", style="dim")

    for r in rows:
        if r.diff < 0:
            diff_style = "green"
        elif r.pct <= 10:
            diff_style = "white"
        elif r.pct <= 30:
            diff_style = "yellow"
        else:
            diff_style = "red"

        fn = f"{r.name} *" if r.inlining_suspect else r.name
        sign = "+" if r.diff >= 0 else ""
        avg_call = f"{r.avg_per_call:.1f}" if r.total_calls > 0 else "-"
        table.add_row(
            fn,
            str(r.base_insns),
            str(r.target_insns),
            f"[{diff_style}]{sign}{r.diff}[/]",
            f"[{diff_style}]{r.pct:+.1f}%[/]",
            str(r.total_calls),
            avg_call,
            r.breakdown,
        )

    console.print(table)
    if any(r.inlining_suspect for r in rows):
        console.print("[dim]* likely affected by compiler inlining differences[/]")
    console.print()

    summary_lines = [
        f"[bold]Functions analyzed:[/] {s.functions_analyzed}",
        f"[bold]Skipped (not in both builds):[/] {s.functions_skipped_missing}",
    ]
    if s.functions_filtered_inlining:
        summary_lines.append(f"[bold]Filtered (inlining diffs):[/] {s.functions_filtered_inlining}")
    if s.functions_flagged_inlining:
        summary_lines.append(f"[bold]Flagged (inlining diffs):[/] {s.functions_flagged_inlining}")
    summary_text = (
        "\n".join(summary_lines) + "\n"
        f"[bold]Total baseline instructions:[/] {s.total_base:,}\n"
        f"[bold]Total traced instructions:[/] {s.total_target:,}\n"
        f"[bold]Total difference:[/] {s.total_diff:+,} ({s.total_pct:+.2f}%)\n"
        f"[bold]Total trace call sites:[/] {s.total_calls:,}\n"
        f"[bold]Avg overhead per call site:[/] {s.avg_per_call:.1f} instructions"
    )
    console.print(Panel(summary_text, title="Summary"))

    dist_text = (
        f"Min: {s.dist_min:.1f}  "
        f"P25: {s.dist_p25:.1f}  "
        f"[bold]Median: {s.dist_median:.1f}[/]  "
        f"P75: {s.dist_p75:.1f}  "
        f"P95: {s.dist_p95:.1f}  "
        f"Max: {s.dist_max:.1f}"
    )
    console.print(Panel(dist_text, title="Overhead Distribution (instructions/call)"))


def disassemble_function(
    vmlinux: str,
    func_name: str,
    cross_compile: str = "",
    objdump_args: list[str] | None = None,
) -> str | None:
    """Disassemble a single function using ``objdump --disassemble=symbol``.

    Uses objdump's built-in symbol lookup instead of streaming the
    entire binary — fast (~80ms) regardless of binary size.

    Args:
        vmlinux: Path to the vmlinux ELF binary.
        func_name: Exact symbol name to disassemble.
        cross_compile: Toolchain prefix for objdump.
        objdump_args: Extra objdump arguments (e.g. ``["-S"]`` for
            source interleaving). ``--disassemble=`` is always added.

    Returns:
        The disassembly text, or None if the symbol was not found.
    """
    objdump = f"{cross_compile}objdump"
    extra = list(objdump_args) if objdump_args else []
    result = subprocess.run(
        [objdump, f"--disassemble={func_name}", "--no-show-raw-insn", *extra, vmlinux],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    # objdump outputs headers even for missing symbols — check for
    # an actual function header in the output.
    if f"<{func_name}>:" not in result.stdout:
        return None
    return result.stdout


def dump_functions(
    base: str,
    target: str,
    func_names: set[str],
    output_dir: str,
    cross_compile: str = "",
    objdump_args: list[str] | None = None,
    console: Console | None = None,
) -> None:
    """Dump disassembly of selected functions from both builds.

    Runs one targeted ``objdump --disassemble=symbol`` per function per
    build, which is fast (~80ms each) since objdump only processes the
    named symbol.

    Writes ``<func>.base.s`` and ``<func>.target.s`` files for each
    requested function.

    Args:
        base: Path to the baseline vmlinux.
        target: Path to the target vmlinux.
        func_names: Set of function names to dump.
        output_dir: Directory to write assembly files into.
        cross_compile: Toolchain prefix for objdump.
        objdump_args: Custom objdump arguments.
        console: Rich console for status messages.
    """
    os.makedirs(output_dir, exist_ok=True)
    con = console or Console(stderr=True)

    written = 0
    for name in sorted(func_names):
        safe_name = name.replace("/", "_")

        base_asm = disassemble_function(base, name, cross_compile, objdump_args)
        if base_asm:
            path = os.path.join(output_dir, f"{safe_name}.base.s")
            with open(path, "w") as f:
                f.write(base_asm)
            written += 1
        else:
            con.print(f"[yellow]Warning:[/] {name} not found in base build")

        target_asm = disassemble_function(target, name, cross_compile, objdump_args)
        if target_asm:
            path = os.path.join(output_dir, f"{safe_name}.target.s")
            with open(path, "w") as f:
                f.write(target_asm)
            written += 1
        else:
            con.print(f"[yellow]Warning:[/] {name} not found in target build")

    con.print(f"Wrote {written} assembly files to {output_dir}/")


def _rows_to_table_data(
    rows: list[CompareRow],
) -> tuple[list[str], list[list[str]]]:
    headers = ["Function", "Base", "Trace", "Diff", "Diff%", "Calls", "Avg/Call", "Breakdown"]
    table_rows = []
    for r in rows:
        fn = f"{r.name} *" if r.inlining_suspect else r.name
        sign = "+" if r.diff >= 0 else ""
        avg_call = f"{r.avg_per_call:.1f}" if r.total_calls > 0 else "-"
        table_rows.append(
            [
                fn,
                str(r.base_insns),
                str(r.target_insns),
                f"{sign}{r.diff}",
                f"{r.pct:+.1f}%",
                str(r.total_calls),
                avg_call,
                r.breakdown,
            ]
        )
    return headers, table_rows


def _summary_to_table_data(summary: Summary) -> tuple[list[str], list[list[str]]]:
    s = summary
    rows: list[list[str]] = [
        ["Functions analyzed", str(s.functions_analyzed)],
        ["Skipped (not in both builds)", str(s.functions_skipped_missing)],
    ]
    if s.functions_filtered_inlining:
        rows.append(["Filtered (inlining diffs)", str(s.functions_filtered_inlining)])
    if s.functions_flagged_inlining:
        rows.append(["Flagged (inlining diffs)", str(s.functions_flagged_inlining)])
    rows.extend(
        [
            ["Total baseline instructions", f"{s.total_base:,}"],
            ["Total traced instructions", f"{s.total_target:,}"],
            ["Total difference", f"{s.total_diff:+,} ({s.total_pct:+.2f}%)"],
            ["Total trace call sites", f"{s.total_calls:,}"],
            ["Avg overhead per call site", f"{s.avg_per_call:.1f} insns"],
        ]
    )
    return ["Metric", "Value"], rows


def output_txt(rows: list[CompareRow], summary: Summary) -> str:
    headers, table_rows = _rows_to_table_data(rows)
    result = format_table("Tracepoint Code Generation Overhead", headers, table_rows, "txt")
    sh, sr = _summary_to_table_data(summary)
    result += format_table("Summary", sh, sr, "txt")
    return result


def output_json(rows: list[CompareRow], summary: Summary) -> str:
    data = {
        "title": "Tracepoint Code Generation Overhead",
        "functions": [
            {
                "name": r.name,
                "base_insns": r.base_insns,
                "target_insns": r.target_insns,
                "diff": r.diff,
                "pct": round(r.pct, 1),
                "total_calls": r.total_calls,
                "avg_per_call": round(r.avg_per_call, 1),
                "breakdown": r.breakdown,
                "inlining_suspect": r.inlining_suspect,
            }
            for r in rows
        ],
        "summary": {
            "functions_analyzed": summary.functions_analyzed,
            "functions_skipped_missing": summary.functions_skipped_missing,
            "functions_filtered_inlining": summary.functions_filtered_inlining,
            "functions_flagged_inlining": summary.functions_flagged_inlining,
            "total_base": summary.total_base,
            "total_target": summary.total_target,
            "total_diff": summary.total_diff,
            "total_pct": round(summary.total_pct, 2),
            "total_calls": summary.total_calls,
            "avg_per_call": round(summary.avg_per_call, 1),
            "distribution": {
                "min": summary.dist_min,
                "p25": summary.dist_p25,
                "median": summary.dist_median,
                "p75": summary.dist_p75,
                "p95": summary.dist_p95,
                "max": summary.dist_max,
            },
        },
    }
    return json.dumps(data, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare kernel vmlinux binaries for tracepoint code generation overhead."
    )
    parser.add_argument("--base", required=True, help="Baseline vmlinux (without tracepoints)")
    parser.add_argument("--target", required=True, help="Target vmlinux (with tracepoints enabled)")
    parser.add_argument("-o", metavar="FILE", help="Output file (default: terminal display)")
    parser.add_argument(
        "--format",
        choices=["ascii", "txt", "markdown", "json"],
        default=None,
        dest="fmt",
        help="Output format (default: ascii, or inferred from -o extension)",
    )
    parser.add_argument(
        "--cross-compile",
        default="",
        metavar="PREFIX",
        help="Toolchain prefix for objdump (e.g. aarch64-linux-gnu-)",
    )
    parser.add_argument(
        "--functions",
        default="",
        metavar="LIST",
        help="Comma-separated list of functions to dump as .base.s / .target.s",
    )
    parser.add_argument(
        "--output-asm-dir",
        default=".",
        metavar="DIR",
        help="Directory for assembly dump files (default: current directory)",
    )
    parser.add_argument(
        "--objdump-args",
        default=None,
        metavar="ARGS",
        help=(
            "Replace default objdump options "
            f"({' '.join(DEFAULT_OBJDUMP_ARGS)}) with ARGS "
            "(use = syntax: --objdump-args='-d -S')"
        ),
    )
    parser.add_argument(
        "--objdump-extra-args",
        default=None,
        metavar="ARGS",
        help="Append extra arguments to objdump (use = syntax: --objdump-extra-args='-S')",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip the comparison analysis (use with --functions to only dump assembly)",
    )
    parser.add_argument(
        "--sort",
        default="name",
        choices=["name", "diff", "pct", "avg"],
        help="Sort results by function name (ascending), absolute diff, %% change, or avg overhead per call (default: name)",
    )
    parser.add_argument(
        "--filter-inlining",
        action="store_true",
        help="Filter out functions likely affected by compiler inlining differences",
    )
    return parser.parse_args()


def build_objdump_args(args: argparse.Namespace) -> list[str]:
    """Build the objdump argument list from CLI options."""
    if args.objdump_args is not None:
        result = shlex.split(args.objdump_args)
    else:
        result = list(DEFAULT_OBJDUMP_ARGS)
    if args.objdump_extra_args is not None:
        result.extend(shlex.split(args.objdump_extra_args))
    return result


def main() -> None:
    """CLI entry point.

    Orchestrates the two operating modes:

    1. **Function dump** (``--functions``): extracts disassembly of
       named functions from both builds into ``.base.s`` / ``.target.s``
       files. Runs independently of the analysis.

    2. **Analysis** (default): disassembles both builds, identifies
       functions affected by tracepoint instrumentation, computes
       instruction count deltas, detects inlining artefacts, and
       renders the comparison report to the terminal or a markdown file.

    ``--no-analysis`` skips mode 2, useful when only the assembly
    dump is needed.
    """
    args = parse_args()
    if args.no_analysis and not args.functions:
        raise SystemExit("error: --no-analysis requires --functions")
    console = Console(stderr=True)
    xc = args.cross_compile
    od_args = build_objdump_args(args)

    if args.functions:
        func_names = {f.strip() for f in args.functions.split(",") if f.strip()}
        dump_functions(
            args.base,
            args.target,
            func_names,
            args.output_asm_dir,
            xc,
            od_args,
            console,
        )

    if args.no_analysis:
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        try:
            task1 = progress.add_task("Disassembling target...", total=None)
            target_data = extract_function_data(
                args.target,
                progress,
                task1,
                xc,
                od_args,
                track_trace_calls=True,
            )
            progress.update(
                task1, description=f"Target: {len(target_data)} functions with trace calls"
            )
            progress.stop_task(task1)

            task2 = progress.add_task("Disassembling base...", total=None)
            base_data = extract_function_data(args.base, progress, task2, xc, od_args)
            progress.update(task2, description=f"Base: {len(base_data)} functions total")
            progress.stop_task(task2)
        except RuntimeError as e:
            progress.stop()
            console.print(f"[bold red]Error:[/] {e}")
            raise SystemExit(1)

    if not target_data:
        console.print(
            "[bold yellow]Warning:[/] no trace call sites found in target. "
            "Is this a vmlinux with TRACE_PREEMPT_TOGGLE / "
            "TRACE_IRQFLAGS_TOGGLE enabled?"
        )
        raise SystemExit(1)

    rows, summary = build_comparison(target_data, base_data, filter_inlining=args.filter_inlining)

    sort_keys = {
        "name": lambda r: r.name,
        "diff": lambda r: -abs(r.diff),
        "pct": lambda r: -abs(r.pct),
        "avg": lambda r: -abs(r.avg_per_call),
    }
    rows.sort(key=sort_keys[args.sort])

    if not rows:
        console.print(
            "[bold yellow]Warning:[/] all functions were filtered by inlining "
            "heuristics. Consider relaxing MAX_DIFF_PER_CALL / MAX_PCT_CHANGE."
        )
        raise SystemExit(1)

    fmt = args.fmt
    if fmt is None and args.o:
        ext_map = {".md": "markdown", ".markdown": "markdown", ".txt": "txt", ".json": "json"}
        ext = os.path.splitext(args.o)[1].lower()
        fmt = ext_map.get(ext, "ascii")
    if fmt is None:
        fmt = "ascii"

    if args.o:
        if fmt == "markdown":
            output_markdown(rows, summary, args.o)
        else:
            writers = {"ascii": output_txt, "txt": output_txt, "json": output_json}
            write_fn = writers[fmt]
            with open(args.o, "w") as f:
                f.write(write_fn(rows, summary) + "\n")
        console.print(f"Report written to {args.o}")
    else:
        if fmt == "json":
            print(output_json(rows, summary))
        elif fmt == "txt":
            print(output_txt(rows, summary))
        elif fmt == "markdown":
            raise SystemExit("error: markdown format requires -o FILE")
        else:
            output_terminal(rows, summary)


if __name__ == "__main__":
    main()
