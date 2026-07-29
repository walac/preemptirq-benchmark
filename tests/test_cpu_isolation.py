from __future__ import annotations

from pathlib import Path

import pytest

from preemptirq_benchmark.cpu_isolation import format_cpu_list, get_isolated_cpus


class TestGetIsolatedCpus:
    def test_empty_file_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "\n")
        assert get_isolated_cpus() == []

    def test_missing_file_returns_empty_list(self, monkeypatch):
        def raise_oserror(self, **kw):
            raise OSError("no such file")

        monkeypatch.setattr(Path, "read_text", raise_oserror)
        assert get_isolated_cpus() == []

    def test_single_ids(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "2,7\n")
        assert get_isolated_cpus() == [2, 7]

    def test_range(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "2-4\n")
        assert get_isolated_cpus() == [2, 3, 4]

    def test_single_cpu_range(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "2-2\n")
        assert get_isolated_cpus() == [2]

    def test_mixed_ids_and_ranges_sorted(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "7,2-4\n")
        assert get_isolated_cpus() == [2, 3, 4, 7]

    def test_malformed_content_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "abc,2-3\n")
        assert get_isolated_cpus() == []


class TestFormatCpuList:
    @pytest.mark.parametrize(
        "cpus, expected",
        [
            ([], ""),
            ([2], "2"),
            ([2, 3, 7], "2,3,7"),
        ],
    )
    def test_format(self, cpus, expected):
        assert format_cpu_list(cpus) == expected
