from __future__ import annotations

import subprocess
from types import SimpleNamespace

import preemptirq_benchmark.benchmarks.rtla as rtla_module
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


class TestConfigure:
    def test_default_duration_and_isolation(self):
        bench = RtlaBenchmark()
        assert bench.duration == "30"
        assert bench.isolated_cpus_only is False

    def test_duration_override(self):
        bench = RtlaBenchmark()
        bench.configure(duration="12h", isolated_cpus_only=None)
        assert bench.duration == "12h"
        assert bench.isolated_cpus_only is False

    def test_isolated_cpus_only_override(self):
        bench = RtlaBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)
        assert bench.isolated_cpus_only is True
        assert bench.duration == "30"

    def test_isolated_cpus_only_can_be_reset_to_false(self):
        bench = RtlaBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)
        bench.configure(duration=None, isolated_cpus_only=False)
        assert bench.isolated_cpus_only is False


class TestGetCommandWithOptions:
    def test_custom_duration(self):
        bench = RtlaBenchmark()
        bench.configure(duration="30m", isolated_cpus_only=None)

        assert bench.get_command() == ["rtla", "timerlat", "top", "-d", "30m", "-q"]

    def test_isolated_cpus_only_appends_cpu_flag(self, monkeypatch):
        monkeypatch.setattr(rtla_module, "get_isolated_cpus", lambda: [2, 3, 7])

        bench = RtlaBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)

        assert bench.get_command() == [
            "rtla",
            "timerlat",
            "top",
            "-d",
            "30",
            "-q",
            "-c",
            "2,3,7",
        ]


class TestCheckPrerequisites:
    def test_fails_when_isolated_cpus_only_and_none_found(self, monkeypatch):
        monkeypatch.setattr(rtla_module.shutil, "which", lambda name: "/usr/bin/rtla")
        monkeypatch.setattr(rtla_module, "get_isolated_cpus", lambda: [])

        bench = RtlaBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)

        ok, msg = bench.check_prerequisites()

        assert ok is False
        assert "isolated CPUs" in msg

    def test_passes_when_isolated_cpus_only_and_some_found(self, monkeypatch):
        monkeypatch.setattr(rtla_module.shutil, "which", lambda name: "/usr/bin/rtla")
        monkeypatch.setattr(rtla_module, "get_isolated_cpus", lambda: [2, 3])

        bench = RtlaBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)

        ok, _ = bench.check_prerequisites()

        assert ok is True


class TestRunOnceUsesConfiguredDurationAndCpus:
    def test_osnoise_call_uses_duration_and_cpu_args(self, monkeypatch):
        monkeypatch.setattr(rtla_module, "get_isolated_cpus", lambda: [4, 5])
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[1] == "timerlat":
                return SimpleNamespace(stdout=TIMERLAT_OUTPUT, stderr="", returncode=0)
            return SimpleNamespace(stdout=OSNOISE_OUTPUT, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = RtlaBenchmark()
        bench.configure(duration="1m", isolated_cpus_only=True)
        bench.run_once()

        assert calls[0] == ["rtla", "timerlat", "top", "-d", "1m", "-q", "-c", "4,5"]
        assert calls[1] == ["rtla", "osnoise", "top", "-d", "1m", "-q", "-c", "4,5"]
