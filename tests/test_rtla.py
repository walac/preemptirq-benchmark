from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

import preemptirq_benchmark.benchmarks.rtla as rtla_module
from preemptirq_benchmark.benchmarks.rtla import RtlaBenchmark, parse_timerlat_max_from_output

TIMERLAT_OUTPUT = "ALL       | 1  2  10.5   | 1  2  8.3   |\n"

OSNOISE_OUTPUT = "0     1000  2000  3    0.05  6  6.5\n"


class TestTimerlatSummaryParsing:
    def test_includes_userspace_return_latency(self):
        output = (
            "CPU COUNT | IRQ Timer Latency | Thread Timer Latency | Ret user Timer Latency\n"
            "ALL #10 e0 | 1 2 10 | 2 3 20 | 4 5 100\n"
        )

        assert parse_timerlat_max_from_output(output) == 100.0

    def test_skips_unavailable_userspace_return_latency(self):
        output = "ALL #10 e0 | 1 2 10 | 2 3 20 | - - -\n"

        assert parse_timerlat_max_from_output(output) == 20.0

    def test_rejects_incomplete_userspace_block(self):
        output = "ALL #10 e0 | 1 2 10 | 2 3 20 | 4 5\n"

        with pytest.raises(RuntimeError, match="could not parse timerlat"):
            parse_timerlat_max_from_output(output)


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

    def test_records_isolated_workload_from_command(self, monkeypatch):
        monkeypatch.setattr(rtla_module, "get_isolated_cpus", lambda: [2, 3, 7])
        bench = RtlaBenchmark()
        bench.configure(duration="5m", isolated_cpus_only=True)

        bench.get_command()

        assert bench.get_workload_config() == {
            "duration": "5m",
            "isolated_cpus_only": True,
            "cpu_selection": "isolated",
            "cpus": [2, 3, 7],
        }

    def test_records_default_online_workload(self, monkeypatch):
        monkeypatch.setattr(rtla_module, "get_online_cpus", lambda: [0, 2, 4])
        bench = RtlaBenchmark()

        bench.get_command()

        assert bench.get_workload_config() == {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "all_online",
            "cpus": [0, 2, 4],
        }


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
