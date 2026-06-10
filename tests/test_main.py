from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from preemptirq_benchmark.__main__ import (
    EXT_TO_FORMAT,
    _ci_percentage,
    cmd_compare,
    cmd_show,
    infer_format,
    managed_output,
    resolve_output_format,
)
from preemptirq_benchmark.benchmarks import BenchmarkResult
from preemptirq_benchmark.report import build_report, save_report


def make_fixture_report(tmp_path: Path, values: list[float] | None = None) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = BenchmarkResult(
        name="hackbench",
        metrics={"time_seconds": values or [1.0, 1.1, 1.2]},
        units={"time_seconds": "s"},
        iterations=len(values) if values else 3,
    )
    report = build_report([result])
    path = save_report(report, str(tmp_path / "report.json"))
    return str(path)


class TestInferFormat:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("report.txt", "txt"),
            ("report.md", "markdown"),
            ("report.markdown", "markdown"),
            ("report.json", "json"),
            ("report.ascii", "ascii"),
        ],
    )
    def test_known_extensions(self, filename, expected):
        assert infer_format(filename) == expected

    def test_unknown_extension_defaults_ascii(self):
        assert infer_format("report.pdf") == "ascii"

    def test_no_extension_defaults_ascii(self):
        assert infer_format("report") == "ascii"

    def test_ext_to_format_completeness(self):
        assert set(EXT_TO_FORMAT.values()) == {"ascii", "txt", "markdown", "json"}


class TestResolveOutputFormat:
    def test_explicit_format_wins(self):
        args = argparse.Namespace(fmt="markdown", output="report.txt")
        assert resolve_output_format(args) == "markdown"

    def test_infer_from_output(self):
        args = argparse.Namespace(fmt=None, output="report.md")
        assert resolve_output_format(args) == "markdown"

    def test_fallback_ascii(self):
        args = argparse.Namespace(fmt=None, output=None)
        assert resolve_output_format(args) == "ascii"

    def test_explicit_format_overrides_extension(self):
        args = argparse.Namespace(fmt="json", output="report.txt")
        assert resolve_output_format(args) == "json"


class TestCiPercentage:
    def test_valid_value(self):
        assert _ci_percentage("95") == 95.0

    def test_valid_float(self):
        assert _ci_percentage("99.9") == 99.9

    @pytest.mark.parametrize("val", ["0", "100", "-5", "105"])
    def test_invalid_values(self, val):
        with pytest.raises(argparse.ArgumentTypeError):
            _ci_percentage(val)


class TestManagedOutput:
    def test_redirects_to_file(self, tmp_path, capsys):
        out = tmp_path / "output.txt"
        with managed_output(str(out)):
            print("hello world")

        assert out.read_text().strip() == "hello world"
        captured = capsys.readouterr()
        assert "Output written to" in captured.out

    def test_no_redirect_when_none(self, capsys):
        with managed_output(None):
            print("direct output")

        captured = capsys.readouterr()
        assert "direct output" in captured.out

    def test_invalid_path_exits(self):
        with pytest.raises(SystemExit):
            with managed_output("/nonexistent/dir/file.txt"):
                print("should fail")


class TestCmdShow:
    def test_ascii_output(self, tmp_path, capsys):
        report_path = make_fixture_report(tmp_path)
        args = argparse.Namespace(report=report_path, fmt=None, output=None)
        cmd_show(args)

        captured = capsys.readouterr()
        assert "hackbench" in captured.out
        assert "time_seconds" in captured.out

    def test_markdown_output(self, tmp_path, capsys):
        report_path = make_fixture_report(tmp_path)
        args = argparse.Namespace(report=report_path, fmt="markdown", output=None)
        cmd_show(args)

        captured = capsys.readouterr()
        assert "## Preemptirq Benchmark Report" in captured.out

    def test_json_output(self, tmp_path, capsys):
        report_path = make_fixture_report(tmp_path)
        args = argparse.Namespace(report=report_path, fmt="json", output=None)
        cmd_show(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["benchmarks_run"] == ["hackbench"]

    def test_output_to_file(self, tmp_path):
        report_path = make_fixture_report(tmp_path)
        out_path = str(tmp_path / "output.md")
        args = argparse.Namespace(report=report_path, fmt=None, output=out_path)
        cmd_show(args)

        content = (tmp_path / "output.md").read_text()
        assert "hackbench" in content


class TestCmdCompare:
    def test_ascii_output(self, tmp_path, capsys):
        p1 = make_fixture_report(tmp_path / "a", [1.0, 1.1, 1.2])
        p2 = make_fixture_report(tmp_path / "b", [1.3, 1.4, 1.5])

        args = argparse.Namespace(reports=[p1, p2], fmt=None, output=None)
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "time_seconds" in captured.out

    def test_markdown_output(self, tmp_path, capsys):
        p1 = make_fixture_report(tmp_path / "a", [1.0, 1.1, 1.2])
        p2 = make_fixture_report(tmp_path / "b", [1.3, 1.4, 1.5])

        args = argparse.Namespace(reports=[p1, p2], fmt="markdown", output=None)
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "## Benchmark Comparison" in captured.out
        assert "%" in captured.out

    def test_json_output(self, tmp_path, capsys):
        p1 = make_fixture_report(tmp_path / "a", [1.0, 1.1, 1.2])
        p2 = make_fixture_report(tmp_path / "b", [1.3, 1.4, 1.5])

        args = argparse.Namespace(reports=[p1, p2], fmt="json", output=None)
        cmd_compare(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "benchmarks" in data

    def test_output_to_file(self, tmp_path):
        p1 = make_fixture_report(tmp_path / "a", [1.0, 1.1, 1.2])
        p2 = make_fixture_report(tmp_path / "b", [1.3, 1.4, 1.5])

        out_path = str(tmp_path / "comparison.txt")
        args = argparse.Namespace(reports=[p1, p2], fmt=None, output=out_path)
        cmd_compare(args)

        content = (tmp_path / "comparison.txt").read_text()
        assert "time_seconds" in content
