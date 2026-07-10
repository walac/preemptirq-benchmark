from __future__ import annotations

import subprocess
from types import SimpleNamespace

from preemptirq_benchmark.benchmarks.rtla import RtlaBenchmark

TIMERLAT_OUTPUT = "ALL       | 1  2  10.5   | 1  2  8.3   |\n"

OSNOISE_OUTPUT = "0     1000  2000  3    0.05  6  6.5\n"


class TestGetCommand:
    def test_returns_timerlat_command(self):
        bench = RtlaBenchmark()

        assert bench.get_command() == ["rtla", "timerlat", "top", "-d", "30", "-q"]


class TestRunOnceUsesGetCommandForTimerlat:
    def test_timerlat_subprocess_call_matches_get_command(self, monkeypatch):
        # Regression test: run_once() previously hardcoded its own
        # copy of the timerlat command instead of reusing
        # get_command(), so the two could silently drift apart --
        # the same command-drift bug class fixed in perf_bench.py.
        # It also only ever measured "timerlat" for perf-stat
        # wrapping while run_once() reports metrics for a separate
        # "osnoise" run too, so the wrapped counters must be clearly
        # attributable to the timerlat measurement specifically.
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[1] == "timerlat":
                return SimpleNamespace(stdout=TIMERLAT_OUTPUT, stderr="", returncode=0)
            return SimpleNamespace(stdout=OSNOISE_OUTPUT, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = RtlaBenchmark()
        metrics = bench.run_once()

        assert len(calls) == 2
        assert calls[0] == bench.get_command()
        assert calls[0] == ["rtla", "timerlat", "top", "-d", "30", "-q"]
        assert calls[1] == ["rtla", "osnoise", "top", "-d", "30", "-q"]

        assert metrics["timerlat_max_us"] == 10.5
        assert metrics["osnoise_max_single_us"] == 6.5
