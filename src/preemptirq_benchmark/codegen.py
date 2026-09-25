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
# to avoid matching symbolic references within instruction operands. The
# address is captured too, to find every symbol-table alias for the body.
FUNC_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")

# INSN_RE — matches address-prefixed disassembly lines (as opposed to
# blank lines, section headers, or source annotations). Byte-only
# continuation lines are filtered separately below:
#   ffffffff81f823a4:  call   ffffffff81390910 <__fentry__>
#   ^^^^^^^^^^^^^^^^
#   optional whitespace + hex address + colon
#
# The \s* prefix handles both vmlinux (no indent) and relocatable
# object files (indented addresses) across objdump versions.
INSN_RE = re.compile(r"^\s*[0-9a-f]+:")

# A long instruction's raw bytes can wrap onto another address-prefixed
# objdump line without a mnemonic. Such a line is not another instruction.
RAW_BYTES_RE = re.compile(r"(?:[0-9a-fA-F]{2}\s*)+")

# objdump may render a branch target as an offset into a helper or as a
# compiler-generated clone. Strip those decorations before helper lookup.
_SYM_SUFFIX_RE = re.compile(
    r"(?:\+0x[0-9a-f]+)?(?:\.(?:cold|isra|constprop|part|localalias)(?:\.\d+)*)*$"
)


def _canonical_symbol(symbol: str) -> str:
    """Return a trace-helper symbol without objdump/compiler decorations."""
    return _SYM_SUFFIX_RE.sub("", symbol)


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
# Also matches unconditional tail-call jumps to a named symbol: GCC/Clang
# routinely turn "return trace_helper(...);" into a sibling-call jump
# instead of call+ret at -O2, and a trace helper reached only via such a
# tail call would otherwise never be counted, silently dropping the whole
# calling function from the report if that was its only trace reference:
#
#   x86:     jmp  ffffffff8151e020 <trace_local_irq_restore>
#   x86:     jmpq ffffffff8151e020 <trace_local_irq_restore>
#   arm/ppc: b    ffff800080123456 <trace_local_irq_restore>
#   riscv:   j    ffffffff80123456 <trace_local_irq_restore>
#
# s390 also has its own unconditional-jump mnemonics ("j"/"jg"); "j" is
# already covered above (identical to riscv's mnemonic, and a genuine
# match there is correct, not a false positive). "jg" is deliberately
# left out of the alternation: it collides with x86's real conditional
# "jump if greater" mnemonic, which would misattribute an unrelated
# conditional branch as a trace-helper call if it ever happened to
# target a labeled address.
#
# The expression is matched at the start of the parsed instruction field,
# rather than searched across the whole objdump line. That keeps the
# alternatives in the mnemonic position: a %bl x86 register operand, or
# 'b' at the end of 'jb'/'movb', cannot be mistaken for a branch mnemonic
# because a later "# <symbol>" annotation appears on the same line.
#
# 'b' additionally excludes a following '.' via a negative lookahead:
# AArch64 spells its conditional branches "b.<cond>" (b.eq, b.ne, b.lt,
# ...) — since '.' is a non-word character, \b treats "b.eq" the same as
# "b " for boundary purposes, so without this exclusion the bare 'b'
# alternative would match the 'b' in every AArch64 conditional branch.
# The plain, dot-less unconditional "b <symbol>" form that AArch64 also
# has is unaffected by the lookahead and still matches.
# The symbol name is captured in group 1 and checked against TRACE_HELPERS.
CALL_RE = re.compile(r"^(?:callq?|bl|brasl|jalr?|jmpq?|b(?!\.)|j)\b.*?<([^>]+)>")

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

# INT3_RE — matches the ``int3`` trap instruction, used as alignment
# padding instead of ``nop`` on CONFIG_X86_KERNEL_IBT/CFI-hardened
# kernels (so stray control flow into the gap traps rather than
# executing silently). Handled separately from NOP_RE rather than
# folded into it: CONFIG_SLS (straight-line-speculation mitigation,
# `-mharden-sls=all`/`-mharden-sls=return`) makes the compiler emit a
# *genuine*, real ``int3`` immediately after every `ret` — commonly
# alongside IBT on hardened kernel configs — so a single trailing
# ``int3`` cannot be assumed to be padding. Only a *second or later*
# consecutive trailing ``int3`` is unambiguously alignment filler; the
# first one is conservatively kept as a real instruction.
INT3_RE = re.compile(r"^int3\b")

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
    aliases: frozenset[str] = field(default_factory=frozenset)

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
    functions_ambiguous_target: int
    functions_ambiguous_base: int
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


def _function_aliases(vmlinux: str, cross_compile: str = "") -> dict[int, frozenset[str]]:
    """Return function symbol names grouped by their ELF address."""
    aliases: defaultdict[int, set[str]] = defaultdict(set)
    for line in stream_objdump(vmlinux, cross_compile, ["-t"]):
        fields = line.split()
        if len(fields) < 6 or fields[2] != "F":
            continue
        try:
            address = int(fields[0], 16)
        except ValueError:
            continue
        aliases[address].add(fields[-1])
    return {address: frozenset(names) for address, names in aliases.items()}


def extract_function_data(
    vmlinux: str,
    progress: Progress,
    task: TaskID,
    cross_compile: str = "",
    objdump_args: list[str] | None = None,
    track_trace_calls: bool = False,
) -> tuple[dict[str, FuncTrace], int]:
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
        A tuple of (mapping of function name to its :class:`FuncTrace`
        data, count of symbol names excluded because they were not
        unique in this binary). Symbol names that appear more than once
        — e.g. same-named ``static`` functions from different
        translation units, which objdump prints as separate ``<name>:``
        blocks at different addresses — cannot be reliably attributed
        to a single instance, so all occurrences of such a name are
        excluded from the result rather than one silently overwriting
        another. A name is tracked as "seen" as soon as it produces a
        real function body, independent of *track_trace_calls* — so an
        ambiguous name is still detected even when only one of its
        colliding instances happens to have a trace-helper call.
    """
    result: dict[str, FuncTrace] = {}
    aliases_by_address = _function_aliases(vmlinux, cross_compile)
    seen_names: set[str] = set()
    ambiguous_names: set[str] = set()
    current_func: str | None = None
    current_aliases: frozenset[str] = frozenset()
    current_data = FuncTrace()
    trailing_nops = 0
    trailing_int3 = 0

    def save_current() -> None:
        current_data.insn_count -= trailing_nops
        current_data.insn_count -= max(0, trailing_int3 - 1)
        if not current_func or current_data.insn_count <= 0:
            return
        current_data.aliases = current_aliases
        if current_func in seen_names:
            ambiguous_names.add(current_func)
            result.pop(current_func, None)
            return
        seen_names.add(current_func)
        if not track_trace_calls or current_data.total_calls > 0:
            result[current_func] = current_data

    for line in stream_objdump(vmlinux, cross_compile, objdump_args):
        m = FUNC_RE.match(line)
        if m:
            save_current()
            name = m.group(2)
            current_func = name
            current_aliases = aliases_by_address.get(int(m.group(1), 16), frozenset()) | frozenset(
                {name}
            )
            current_data = FuncTrace()
            trailing_nops = 0
            trailing_int3 = 0
            progress.update(task, advance=1)
            continue

        insn_match = INSN_RE.match(line)
        if current_func and insn_match:
            if RAW_BYTES_RE.fullmatch(line[insn_match.end() :].strip()):
                continue
            current_data.insn_count += 1
            # Extract the mnemonic positionally rather than by rsplitting on
            # tabs: GNU objdump tabs between the address and the mnemonic
            # (optionally with a raw-bytes field in between), but
            # llvm-objdump instead tabs between the mnemonic and its
            # operands. rsplit("\t", 1)[-1] would then capture the operands
            # under llvm, which never matches NOP_RE/INT3_RE and silently
            # re-inflates the instruction count with alignment padding.
            tail = line[insn_match.end() :]
            fields = [f for f in tail.split("\t") if f.strip()]
            mnemonic = ""
            instruction = ""
            if fields:
                instruction = "\t".join(fields).strip()
                first = fields[0].strip()
                if len(fields) > 1 and RAW_BYTES_RE.fullmatch(first):
                    first = fields[1].strip()
                    instruction = "\t".join(fields[1:]).strip()
                mnemonic = first
            if INT3_RE.match(mnemonic):
                trailing_int3 += 1
                trailing_nops = 0
            elif NOP_RE.match(mnemonic):
                trailing_nops += 1
                trailing_int3 = 0
            else:
                trailing_nops = 0
                trailing_int3 = 0
            if track_trace_calls:
                cm = CALL_RE.match(instruction)
                if cm and (helper := _canonical_symbol(cm.group(1))) in TRACE_HELPERS:
                    current_data.calls[helper] += 1

    save_current()

    return result, len(ambiguous_names)


def _aggregate_function_data(data: dict[str, FuncTrace]) -> dict[tuple[str, ...], FuncTrace]:
    """Group compiler clones and ELF aliases under canonical symbol names."""
    aggregated: dict[tuple[str, ...], FuncTrace] = {}
    for name, trace in data.items():
        aliases = trace.aliases or frozenset({name})
        canonical_names = tuple(sorted({_canonical_symbol(alias) for alias in aliases}))
        aggregate = aggregated.setdefault(
            canonical_names, FuncTrace(aliases=frozenset(canonical_names))
        )
        aggregate.insn_count += trace.insn_count
        for helper, calls in trace.calls.items():
            aggregate.calls[helper] += calls
    return aggregated


def build_comparison(
    target_data: dict[str, FuncTrace],
    base_data: dict[str, FuncTrace],
    *,
    filter_inlining: bool = False,
    target_ambiguous: int = 0,
    base_ambiguous: int = 0,
) -> tuple[list[CompareRow], Summary]:
    """Join target and base data, compute deltas, and detect outliers.

    For each function present in both builds, computes the instruction
    count difference and percentage change. Functions where the delta
    is disproportionate to the number of trace call sites are flagged
    as likely artefacts of unrelated compiler inlining decisions.

    When *filter_inlining* is True, flagged functions are excluded from
    the report. When False (the default), they are included and marked
    in the per-row report, but the returned summary's aggregate totals
    (``total_base``, ``total_target``, ``total_calls``, ``total_diff``,
    ``total_pct``, ``avg_per_call``) and overhead distribution always
    exclude flagged functions regardless of *filter_inlining*, since
    they are by definition disproportionate outliers that would skew
    those aggregates.

    Args:
        target_data: Per-function trace data from the target build.
        base_data: Per-function data from the base build.
        filter_inlining: When True, exclude inlining-suspect functions
            instead of marking them.
        target_ambiguous: Count of target-build symbol names excluded
            because they were not unique (see :func:`extract_function_data`).
            Passed straight through into the returned summary.
        base_ambiguous: Same, for the base build.

    Returns:
        A tuple of (comparison rows, aggregate summary statistics).
    """
    rows: list[CompareRow] = []
    skipped_missing = 0
    skipped_inlining = 0
    flagged_inlining = 0

    target_functions = _aggregate_function_data(target_data)
    base_functions = _aggregate_function_data(base_data)

    for names in sorted(target_functions):
        td = target_functions[names]
        if names not in base_functions:
            skipped_missing += 1
            continue

        base = base_functions[names].insn_count
        target = td.insn_count
        diff = target - base

        if base == 0:
            continue

        pct = (diff / base) * 100
        tc = td.total_calls

        suspect = (
            (tc > 0 and abs(diff) / tc > MAX_DIFF_PER_CALL)
            or (abs(pct) > MAX_PCT_CHANGE and base >= MIN_BASE_FOR_PCT_CHECK)
            or (diff < 0 and abs(diff) > tc * MAX_SHRINK_PER_CALL)
        )

        if suspect and filter_inlining:
            skipped_inlining += 1
            continue

        if suspect:
            flagged_inlining += 1

        apc = diff / tc if tc > 0 else 0.0
        rows.append(
            CompareRow(", ".join(names), base, target, diff, pct, tc, apc, td.breakdown(), suspect)
        )

    non_suspect_rows = [r for r in rows if not r.inlining_suspect]
    total_base = sum(r.base_insns for r in non_suspect_rows)
    total_target = sum(r.target_insns for r in non_suspect_rows)
    total_diff = total_target - total_base
    total_pct = (total_diff / total_base * 100) if total_base else 0
    total_calls = sum(r.total_calls for r in non_suspect_rows)
    avg_per_call = total_diff / total_calls if total_calls else 0

    per_call_vals = sorted(r.diff / r.total_calls for r in non_suspect_rows if r.total_calls > 0)
    n = len(per_call_vals)
    p25 = median = p75 = p95 = 0.0
    if n >= 2:
        q = statistics.quantiles(per_call_vals, n=20, method="inclusive")
        p25, median, p75, p95 = q[4], q[9], q[14], q[18]
    elif n == 1:
        p25 = median = p75 = p95 = per_call_vals[0]

    summary = Summary(
        functions_analyzed=len(rows),
        functions_skipped_missing=skipped_missing,
        functions_filtered_inlining=skipped_inlining,
        functions_flagged_inlining=flagged_inlining,
        functions_ambiguous_target=target_ambiguous,
        functions_ambiguous_base=base_ambiguous,
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
        if s.functions_ambiguous_target:
            f.write(
                f"| Functions excluded (ambiguous name, target) | {s.functions_ambiguous_target} |\n"
            )
        if s.functions_ambiguous_base:
            f.write(
                f"| Functions excluded (ambiguous name, base) | {s.functions_ambiguous_base} |\n"
            )
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
    if s.functions_ambiguous_target:
        summary_lines.append(
            f"[bold]Excluded (ambiguous name, target):[/] {s.functions_ambiguous_target}"
        )
    if s.functions_ambiguous_base:
        summary_lines.append(
            f"[bold]Excluded (ambiguous name, base):[/] {s.functions_ambiguous_base}"
        )
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
    if s.functions_ambiguous_target:
        rows.append(["Excluded (ambiguous name, target)", str(s.functions_ambiguous_target)])
    if s.functions_ambiguous_base:
        rows.append(["Excluded (ambiguous name, base)", str(s.functions_ambiguous_base)])
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


def _distribution_to_table_data(summary: Summary) -> tuple[list[str], list[list[str]]]:
    s = summary
    rows: list[list[str]] = [
        ["Min", f"{s.dist_min:.1f}"],
        ["P25", f"{s.dist_p25:.1f}"],
        ["Median", f"{s.dist_median:.1f}"],
        ["P75", f"{s.dist_p75:.1f}"],
        ["P95", f"{s.dist_p95:.1f}"],
        ["Max", f"{s.dist_max:.1f}"],
    ]
    return ["Stat", "Value"], rows


def output_txt(rows: list[CompareRow], summary: Summary) -> str:
    headers, table_rows = _rows_to_table_data(rows)
    result = format_table("Tracepoint Code Generation Overhead", headers, table_rows, "txt")
    sh, sr = _summary_to_table_data(summary)
    result += format_table("Summary", sh, sr, "txt")
    dh, dr = _distribution_to_table_data(summary)
    result += format_table("Overhead Distribution (instructions/call)", dh, dr, "txt")
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
            "functions_ambiguous_target": summary.functions_ambiguous_target,
            "functions_ambiguous_base": summary.functions_ambiguous_base,
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


def empty_result_reasons(summary: Summary, filter_inlining: bool) -> list[str]:
    """Explain why a comparison produced zero reportable rows.

    ``build_comparison`` can end up with no rows for more than one
    reason, and they are not mutually exclusive. Enumerating the ones
    that actually apply (rather than assuming inlining filtering is
    always the cause) avoids sending the user to tune
    MAX_DIFF_PER_CALL/MAX_PCT_CHANGE when the real problem is that no
    function was present in both builds.

    Best-effort, not exhaustive: a function present in both builds
    with a zero base instruction count is also silently excluded from
    ``rows``, but that case has no dedicated counter in ``Summary``, so
    it only surfaces via the fallback reason below, and is omitted
    entirely when it coincides with one of the two tracked causes.
    """
    reasons = []
    if summary.functions_skipped_missing:
        reasons.append(
            f"{summary.functions_skipped_missing} function(s) not present in both builds"
        )
    if filter_inlining and summary.functions_filtered_inlining:
        reasons.append(
            f"{summary.functions_filtered_inlining} function(s) filtered by inlining "
            "heuristics (consider relaxing MAX_DIFF_PER_CALL / MAX_PCT_CHANGE)"
        )
    if not reasons:
        reasons.append("no function in target had a nonzero base instruction count")
    return reasons


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
            target_data, target_ambiguous = extract_function_data(
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
            base_data, base_ambiguous = extract_function_data(
                args.base, progress, task2, xc, od_args
            )
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

    rows, summary = build_comparison(
        target_data,
        base_data,
        filter_inlining=args.filter_inlining,
        target_ambiguous=target_ambiguous,
        base_ambiguous=base_ambiguous,
    )

    sort_keys = {
        "name": lambda r: r.name,
        "diff": lambda r: -abs(r.diff),
        "pct": lambda r: -abs(r.pct),
        "avg": lambda r: -abs(r.avg_per_call),
    }
    rows.sort(key=sort_keys[args.sort])

    if not rows:
        reasons = empty_result_reasons(summary, args.filter_inlining)
        console.print(f"[bold yellow]Warning:[/] no functions to report: {'; '.join(reasons)}.")
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
