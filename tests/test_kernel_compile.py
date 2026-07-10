from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from preemptirq_benchmark.benchmarks.kernel_compile import KernelCompileBenchmark

TIME_V_STDERR = "User time (seconds): 1.23\nSystem time (seconds): 0.45\n"


def _make_recording_run(calls: list[list[str]]):
    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr=TIME_V_STDERR)

    return fake_run


class TestBuildCommand:
    def test_build_command_has_no_side_effects(self, monkeypatch):
        # _build_command() must be a pure getter: it is reused inside
        # run_once() right after that method has already run its own
        # "make clean", so it must never spawn a subprocess itself.
        def fail_if_called(*args, **kwargs):
            raise AssertionError("_build_command() must not spawn a subprocess")

        monkeypatch.setattr(subprocess, "run", fail_if_called)

        bench = KernelCompileBenchmark()
        bench.kernel_src = Path("/fake/kernel")

        cmd = bench._build_command()

        assert cmd[:5] == ["time", "-v", "make", "-C", "/fake/kernel"]
        assert cmd[5].startswith("-j")
        assert "clean" not in cmd


class TestGetCommand:
    """Regression tests for the perf-stat-wrap-omits-clean bug.

    ``__main__.py`` collects perf counters by calling
    ``bench.get_command()`` standalone and feeding the result straight
    into ``run_with_perf_stat()`` *after* all of ``run_once()``'s
    iterations have already completed and left the tree fully built.
    Without a clean step inside ``get_command()`` itself, that
    perf-stat-wrapped build measured a near-instant no-op incremental
    rebuild instead of a real compile.
    """

    def test_runs_clean_immediately_before_returning_build_command(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _make_recording_run(calls))

        bench = KernelCompileBenchmark()
        bench.kernel_src = Path("/fake/kernel")

        cmd = bench.get_command()

        assert calls == [["make", "-C", "/fake/kernel", "clean"]]
        assert cmd[:5] == ["time", "-v", "make", "-C", "/fake/kernel"]
        assert "clean" not in cmd

    def test_cleans_before_each_call_across_repeated_invocations(self, monkeypatch):
        # If the CLI runner ever collects perf stat once per measured
        # iteration, each call must be preceded by its own clean, not
        # just a single clean the first time get_command() runs.
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _make_recording_run(calls))

        bench = KernelCompileBenchmark()
        bench.kernel_src = Path("/fake/kernel")

        for _ in range(3):
            bench.get_command()

        assert calls == [["make", "-C", "/fake/kernel", "clean"]] * 3


class TestRunOnce:
    def test_cleans_before_the_measured_build_and_does_not_double_clean(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _make_recording_run(calls))

        bench = KernelCompileBenchmark()
        bench.kernel_src = Path("/fake/kernel")

        bench.run_once()

        # Order must be: defconfig, clean, then the timed build. Clean
        # must appear exactly once per iteration -- run_once() already
        # cleans explicitly, so it must build via the side-effect-free
        # _build_command() rather than get_command() (which would
        # clean a second time).
        assert len(calls) == 3
        assert calls[0] == ["make", "-C", "/fake/kernel", "defconfig"]
        assert calls[1] == ["make", "-C", "/fake/kernel", "clean"]
        assert calls[2][:3] == ["time", "-v", "make"]

    def test_multiple_iterations_clean_before_each_build(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _make_recording_run(calls))

        bench = KernelCompileBenchmark()
        bench.kernel_src = Path("/fake/kernel")

        for _ in range(3):
            bench.run_once()

        assert len(calls) == 9
        for i in range(3):
            defconfig_call, clean_call, build_call = calls[i * 3 : i * 3 + 3]
            assert defconfig_call == ["make", "-C", "/fake/kernel", "defconfig"]
            assert clean_call == ["make", "-C", "/fake/kernel", "clean"]
            assert build_call[:3] == ["time", "-v", "make"]
