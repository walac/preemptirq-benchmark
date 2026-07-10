from __future__ import annotations

import json

import pytest

from preemptirq_benchmark.formatters import (
    auto_style_cell,
    format_ascii,
    format_json,
    format_markdown,
    format_table,
    format_txt,
)


class TestFormatTable:
    def test_empty_rows(self):
        result = format_table("Test", ["A"], [], "ascii")
        assert "(no data)" in result

    def test_empty_headers(self):
        result = format_table("Test", [], [["a"]], "ascii")
        assert "(no data)" in result

    def test_dispatches_to_ascii(self):
        result = format_table("Title", ["Col"], [["val"]], "ascii")
        assert "val" in result

    def test_dispatches_to_txt(self):
        result = format_table("Title", ["Col"], [["val"]], "txt")
        assert "val" in result
        assert "+" in result

    def test_dispatches_to_markdown(self):
        result = format_table("Title", ["Col"], [["val"]], "markdown")
        assert "### Title" in result

    def test_dispatches_to_json(self):
        result = format_table("Title", ["Col"], [["val"]], "json")
        assert '"title": "Title"' in result

    def test_col_styles_applied(self):
        result = format_table(
            "Test",
            ["A", "B"],
            [["one", "two"]],
            "ascii",
            col_styles={1: "bold red"},
        )
        assert "two" in result


class TestAutoStyleCell:
    def test_highly_significant_suffix(self):
        text = auto_style_cell("+5.2% (**)")
        assert text.style == "bold yellow"

    def test_not_significant_suffix(self):
        text = auto_style_cell("+5.2% (ns)")
        assert text.style == "dim"

    def test_improvement_negative_pct(self):
        text = auto_style_cell("-3.1%")
        assert text.style == "green"

    def test_regression_positive_pct_only(self):
        text = auto_style_cell("+3.1%")
        assert text.style == "red"

    def test_marginal_significance(self):
        text = auto_style_cell("+2.0% (*)")
        assert text.style == "yellow"

    def test_plain_text(self):
        text = auto_style_cell("hackbench")
        assert text.style == ""

    def test_ns_suffix(self):
        text = auto_style_cell("+0.1% (ns)")
        assert text.style == "dim"


class TestRowLengthValidation:
    def test_format_ascii_raises_on_short_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 1 cell\(s\).*2 header\(s\)"):
            format_ascii("Title", ["A", "B"], [["one"]])

    def test_format_ascii_raises_on_long_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 3 cell\(s\).*2 header\(s\)"):
            format_ascii("Title", ["A", "B"], [["one", "two", "three"]])

    def test_format_txt_raises_on_short_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 1 cell\(s\).*2 header\(s\)"):
            format_txt("Title", ["A", "B"], [["one"]])

    def test_format_txt_raises_on_long_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 3 cell\(s\).*2 header\(s\)"):
            format_txt("Title", ["A", "B"], [["one", "two", "three"]])

    def test_format_markdown_raises_on_short_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 1 cell\(s\).*2 header\(s\)"):
            format_markdown("Title", ["A", "B"], [["one"]])

    def test_format_markdown_raises_on_long_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 3 cell\(s\).*2 header\(s\)"):
            format_markdown("Title", ["A", "B"], [["one", "two", "three"]])

    def test_format_json_raises_on_short_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 1 cell\(s\).*2 header\(s\)"):
            format_json("Title", ["A", "B"], [["one"]])

    def test_format_json_raises_on_long_row(self):
        with pytest.raises(ValueError, match=r"Row 0 has 3 cell\(s\).*2 header\(s\)"):
            format_json("Title", ["A", "B"], [["one", "two", "three"]])

    def test_format_table_raises_for_mismatched_row(self):
        with pytest.raises(ValueError, match=r"Row 1 has 1 cell\(s\).*2 header\(s\)"):
            format_table(
                "Title",
                ["A", "B"],
                [["one", "two"], ["only-one"]],
                "markdown",
            )


class TestFormatJson:
    def test_structure(self):
        result = format_json("Title", ["A", "B"], [["1", "2"], ["3", "4"]])
        data = json.loads(result)

        assert data["title"] == "Title"
        assert data["headers"] == ["A", "B"]
        assert len(data["rows"]) == 2
        assert data["rows"][0] == {"A": "1", "B": "2"}
