from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from preemptirq_benchmark import codegen
from preemptirq_benchmark.codegen import (
    FuncTrace,
    Summary,
    build_comparison,
    empty_result_reasons,
    extract_function_data,
    output_txt,
)


def _extract(monkeypatch, lines, track_trace_calls=False):
    monkeypatch.setattr(codegen, "stream_objdump", lambda *a, **kw: iter(lines))
    return extract_function_data(
        "fake.vmlinux", MagicMock(), MagicMock(), track_trace_calls=track_trace_calls
    )


class TestPaddingStripping:
    def test_raw_byte_continuation_is_not_an_instruction(self, monkeypatch):
        lines = [
            "0000000000000000 <helper>:",
            "   0:\t48 b8 00 00 00 00 00 \tmovabs $0x0,%rax",
            "   7:\t00 00 00",
            "   a:\tc3                   \tret",
        ]

        result, _ = _extract(monkeypatch, lines)

        assert result["helper"].insn_count == 2

    def test_continuation_after_trailing_nop_does_not_reset_padding(self, monkeypatch):
        lines = [
            "0000000000000000 <helper>:",
            "   0:\tc3                   \tret",
            "   1:\t66 66 66 2e 0f 1f 84\tnopw 0x0(%rax,%rax,1)",
            "   8:\t00 00 00 00 00",
        ]

        result, _ = _extract(monkeypatch, lines)

        assert result["helper"].insn_count == 1

    def test_trailing_nop_stripped(self, monkeypatch):
        lines = [
            "0000000000000000 <helper>:",
            "   0:\tpush   %rbp",
            "   1:\tret",
            "   2:\tnop",
            "   3:\tnop",
        ]
        result, ambiguous = _extract(monkeypatch, lines)
        assert result["helper"].insn_count == 2
        assert ambiguous == 0

    def test_multiple_trailing_int3_stripped_but_one_kept(self, monkeypatch):
        # Two or more trailing int3 is unambiguous IBT/CFI alignment filler,
        # so all but one are stripped. A single trailing int3 is kept, since
        # CONFIG_SLS emits a real int3 immediately after ret.
        lines = [
            "0000000000000000 <helper>:",
            "   0:\tpush   %rbp",
            "   1:\tret",
            "   2:\tint3",
            "   3:\tint3",
            "   4:\tint3",
        ]
        result, _ = _extract(monkeypatch, lines)
        # push, ret, + 1 surviving int3 = 3
        assert result["helper"].insn_count == 3

    def test_single_trailing_int3_kept_as_real_instruction(self, monkeypatch):
        # A lone trailing int3 (the CONFIG_SLS trap after ret) must not be
        # stripped as if it were alignment padding.
        lines = [
            "0000000000000000 <helper>:",
            "   0:\tpush   %rbp",
            "   1:\tret",
            "   2:\tint3",
        ]
        result, _ = _extract(monkeypatch, lines)
        assert result["helper"].insn_count == 3

    def test_raw_insn_byte_layout_still_strips_padding(self, monkeypatch):
        # With raw instruction bytes shown, each line has two tabs:
        # "<addr>:\t<bytes>\t<mnemonic>". Mnemonic extraction must use the
        # last tab-delimited field, not the first.
        lines = [
            "0000000000000000 <helper>:",
            "   0:\t55                   \tpush   %rbp",
            "   1:\t48 89 e5             \tmov    %rsp,%rbp",
            "   4:\t90                   \tnop",
            "   5:\t90                   \tnop",
        ]
        result, _ = _extract(monkeypatch, lines)
        assert result["helper"].insn_count == 2


class TestAmbiguousNameExclusion:
    def test_duplicate_name_excluded_from_result(self, monkeypatch):
        lines = [
            "0000000000000000 <show>:",
            "   0:\tnop",
            "   1:\tret",
            "0000000000000002 <unique_func>:",
            "   2:\tret",
            "0000000000000003 <show>:",
            "   3:\tnop",
            "   4:\tnop",
            "   5:\tret",
        ]
        result, ambiguous = _extract(monkeypatch, lines)
        assert "show" not in result
        assert result["unique_func"].insn_count == 1
        assert ambiguous == 1

    def test_ambiguity_detected_even_when_only_one_instance_has_trace_calls(self, monkeypatch):
        # Regression test: name-collision detection must not depend on
        # which colliding instance happens to satisfy the
        # track_trace_calls>0 filter, or a name that collides 2+ times
        # with only one call-bearing instance would be silently treated
        # as unique again.
        lines = [
            "0000000000000000 <ambig>:",
            "   0:\tret",
            "0000000000000001 <ambig>:",
            "   1:\tcall   0000000000000002 <trace_local_irq_restore>",
            "   2:\tret",
            "0000000000000003 <ambig>:",
            "   3:\tret",
        ]
        result, ambiguous = _extract(monkeypatch, lines, track_trace_calls=True)
        assert "ambig" not in result
        assert ambiguous == 1

    def test_unrelated_function_unaffected_by_collision(self, monkeypatch):
        lines = [
            "0000000000000000 <dup>:",
            "   0:\tret",
            "0000000000000001 <dup>:",
            "   1:\tret",
            "0000000000000002 <fine>:",
            "   2:\tret",
        ]
        result, ambiguous = _extract(monkeypatch, lines)
        assert ambiguous == 1
        assert "fine" in result
        assert "dup" not in result


class TestTailCallDetection:
    def test_x86_tail_call_jmp_to_trace_helper_counted(self, monkeypatch):
        # GCC/Clang commonly turn "return trace_helper(...);" into a
        # sibling-call jmp instead of call+ret at -O2. Without recognizing
        # this, a function whose only trace reference is such a tail call
        # would be silently dropped from the report entirely.
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tpush   %rbp",
            "   1:\tjmp    0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert result["caller"].calls["trace_local_irq_restore"] == 1

    def test_arm_ppc_tail_call_b_to_trace_helper_counted(self, monkeypatch):
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tb      0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert result["caller"].calls["trace_local_irq_restore"] == 1

    def test_riscv_tail_call_j_to_trace_helper_counted(self, monkeypatch):
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tj      0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert result["caller"].calls["trace_local_irq_restore"] == 1

    def test_x86_conditional_jg_not_miscounted_as_call(self, monkeypatch):
        # "jg" (jump if greater) is a real, distinct x86 mnemonic that must
        # not be swallowed by the bare tail-call "j" alternative added for
        # riscv/s390 -- word-boundary matching must keep them separate.
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tjg     0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert "caller" not in result

    def test_conditional_branch_bne_not_miscounted_as_call(self, monkeypatch):
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tbne    0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert "caller" not in result

    def test_x86_jb_not_miscounted_as_tail_call(self, monkeypatch):
        # Regression test: "jb" (jump if below, a real unsigned-comparison
        # conditional jump) ends in the same letter as the bare tail-call
        # "j" alternative. A search() without a LEADING word boundary can
        # match just the trailing "b" of "jb" as if it were the standalone
        # arm/ppc "b" mnemonic, silently miscounting an unrelated
        # conditional jump as a trace-helper call.
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tjb     0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert "caller" not in result

    def test_size_suffixed_instruction_with_unrelated_symbol_comment_not_miscounted(
        self, monkeypatch
    ):
        # Regression test: objdump commonly annotates RIP-relative operands
        # on ordinary instructions with a "<symbol>" comment unrelated to
        # any call/jump. A search() without a leading word boundary can
        # match the trailing "b" of a size-suffixed mnemonic like "movb"
        # and misattribute the annotation as a trace-helper call.
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tmovb   $0x1,0x123(%rip)        # <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert "caller" not in result

    def test_aarch64_dotted_conditional_branch_not_miscounted_as_call(self, monkeypatch):
        # Regression test: AArch64 spells conditional branches "b.<cond>"
        # (b.eq, b.ne, ...). Since "." is a non-word character, a trailing
        # \b alone treats "b.eq" the same as a standalone "b " mnemonic,
        # letting the bare arm/ppc "b" alternative match the "b" in an
        # unrelated conditional branch.
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tb.eq   0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert "caller" not in result

    def test_aarch64_plain_unconditional_b_still_counted(self, monkeypatch):
        # The dot-exclusion added for b.<cond> must not also swallow
        # AArch64's plain, dot-less unconditional "b" tail-call form.
        lines = [
            "0000000000000000 <caller>:",
            "   0:\tb      0000000000000002 <trace_local_irq_restore>",
        ]
        result, _ = _extract(monkeypatch, lines, track_trace_calls=True)
        assert result["caller"].calls["trace_local_irq_restore"] == 1


class TestBuildComparisonAmbiguousSummary:
    def test_ambiguous_counts_flow_into_summary(self):
        target_data = {"f": FuncTrace(insn_count=10)}
        base_data = {"f": FuncTrace(insn_count=8)}
        _, summary = build_comparison(target_data, base_data, target_ambiguous=3, base_ambiguous=2)
        assert summary.functions_ambiguous_target == 3
        assert summary.functions_ambiguous_base == 2

    def test_ambiguous_counts_default_to_zero(self):
        _, summary = build_comparison({}, {})
        assert summary.functions_ambiguous_target == 0
        assert summary.functions_ambiguous_base == 0


class TestBuildComparisonTotalsExcludeSuspects:
    # Regression test: totals/avg_per_call previously summed over every
    # row, including inlining-suspect outliers, while the overhead
    # distribution right next to them already excluded those same rows
    # -- a self-contradictory report. Both must now agree.

    def test_totals_and_distribution_exclude_suspect_when_not_filtered(self):
        target_data = {
            # Flagged suspect: diff/call (300) exceeds MAX_DIFF_PER_CALL.
            "suspect": FuncTrace(insn_count=310, calls={"trace_x": 1}),
            # Ordinary, non-suspect overhead.
            "clean": FuncTrace(insn_count=105, calls={"trace_x": 1}),
        }
        base_data = {
            "suspect": FuncTrace(insn_count=10),
            "clean": FuncTrace(insn_count=100),
        }

        rows, summary = build_comparison(target_data, base_data, filter_inlining=False)

        # filter_inlining=False: suspect row is still present and marked
        # in the per-row report.
        assert len(rows) == 2
        assert any(r.name == "suspect" and r.inlining_suspect for r in rows)
        assert summary.functions_flagged_inlining == 1
        assert summary.functions_filtered_inlining == 0

        # But aggregate totals and the distribution reflect only the
        # non-suspect "clean" row.
        assert summary.total_base == 100
        assert summary.total_target == 105
        assert summary.total_diff == 5
        assert summary.total_pct == pytest.approx(5.0)
        assert summary.total_calls == 1
        assert summary.avg_per_call == pytest.approx(5.0)
        assert summary.dist_min == summary.dist_max == pytest.approx(5.0)


class TestEmptyResultReasons:
    # Regression tests: build_comparison() can return zero rows for
    # reasons that have nothing to do with inlining filtering (e.g. no
    # function was present in both builds). The CLI must name the
    # actual cause instead of always blaming inlining heuristics.

    def test_all_missing_from_base_reports_skipped_missing(self):
        target_data = {"only_in_target": FuncTrace(insn_count=5)}
        rows, summary = build_comparison(target_data, {})
        assert rows == []
        reasons = empty_result_reasons(summary, filter_inlining=False)
        assert reasons == ["1 function(s) not present in both builds"]

    def test_all_filtered_by_inlining_reports_inlining_cause(self):
        target_data = {"f": FuncTrace(insn_count=310, calls={"trace_x": 1})}
        base_data = {"f": FuncTrace(insn_count=10)}
        rows, summary = build_comparison(target_data, base_data, filter_inlining=True)
        assert rows == []
        assert summary.functions_skipped_missing == 0
        reasons = empty_result_reasons(summary, filter_inlining=True)
        assert reasons == [
            "1 function(s) filtered by inlining heuristics "
            "(consider relaxing MAX_DIFF_PER_CALL / MAX_PCT_CHANGE)"
        ]

    @pytest.mark.parametrize("filter_inlining", [False, True])
    def test_flagged_inlining_field_never_consulted(self, filter_inlining):
        # functions_flagged_inlining tracks suspects that were marked
        # (not dropped) and is unrelated to why rows might be empty;
        # only functions_filtered_inlining may ever be blamed. Regression
        # guard against confusing the two similarly-named fields.
        summary = build_comparison({}, {})[1]
        summary = Summary(**{**summary.__dict__, "functions_flagged_inlining": 5})
        reasons = empty_result_reasons(summary, filter_inlining=filter_inlining)
        assert reasons == ["no function in target had a nonzero base instruction count"]

    def test_inlining_count_ignored_when_filter_inlining_not_requested(self):
        # functions_filtered_inlining can only be nonzero in practice
        # when filter_inlining=True was passed to build_comparison, but
        # empty_result_reasons() still gates on filter_inlining
        # defensively. Construct the otherwise-impossible combination
        # by hand to prove that gate actually does something.
        summary = build_comparison({}, {})[1]
        summary = Summary(**{**summary.__dict__, "functions_filtered_inlining": 3})
        reasons = empty_result_reasons(summary, filter_inlining=False)
        assert reasons == ["no function in target had a nonzero base instruction count"]

    def test_both_causes_reported_together(self):
        target_data = {
            "missing": FuncTrace(insn_count=5),
            "suspect": FuncTrace(insn_count=310, calls={"trace_x": 1}),
        }
        base_data = {"suspect": FuncTrace(insn_count=10)}
        rows, summary = build_comparison(target_data, base_data, filter_inlining=True)
        assert rows == []
        reasons = empty_result_reasons(summary, filter_inlining=True)
        assert len(reasons) == 2
        assert any("not present in both builds" in r for r in reasons)
        assert any("filtered by inlining" in r for r in reasons)


class TestOutputTxt:
    def test_includes_overhead_distribution_table(self):
        # Two functions with distinct avg-per-call overhead (5.0 and 15.0)
        # so Min/Median/Max diverge, proving the printed values line up
        # with the right labels rather than all collapsing to one number.
        # Base sizes and diffs are kept below the inlining-suspect thresholds
        # (MAX_DIFF_PER_CALL / MAX_PCT_CHANGE) so both rows count toward the
        # distribution.
        target_data = {
            "low": FuncTrace(insn_count=40, calls={"trace_x": 2}),
            "high": FuncTrace(insn_count=60, calls={"trace_x": 2}),
        }
        base_data = {"low": FuncTrace(insn_count=30), "high": FuncTrace(insn_count=30)}
        rows, summary = build_comparison(target_data, base_data)
        assert summary.dist_min == pytest.approx(5.0)
        assert summary.dist_median == pytest.approx(10.0)
        assert summary.dist_max == pytest.approx(15.0)

        text = output_txt(rows, summary)
        assert "Overhead Distribution" in text
        assert "| Min    |   5.0 |" in text
        assert "| Median |  10.0 |" in text
        assert "| Max    |  15.0 |" in text
        assert f"{summary.dist_median:.1f}" in text


if __name__ == "__main__":
    pytest.main([__file__])
