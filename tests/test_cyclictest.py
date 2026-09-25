from __future__ import annotations

import json
import subprocess

import preemptirq_benchmark.benchmarks.cyclictest as cyclictest_module
from preemptirq_benchmark.benchmarks.cyclictest import CyclictestBenchmark

DEFAULT_COMMAND = [
    "cyclictest",
    "-m",
    "-S",
    "-p",
    "98",
    "-i",
    "1000",
    "-D",
    "30",
    "-q",
]


class TestGetCommand:
    def test_returns_default_command(self):
        bench = CyclictestBenchmark()
        assert bench.get_command() == DEFAULT_COMMAND

    def test_custom_duration(self):
        bench = CyclictestBenchmark()
        bench.configure(duration="30m", isolated_cpus_only=None)

        assert bench.get_command()[7:9] == ["-D", "30m"]

    def test_isolated_cpus_only_runs_one_pinned_thread_per_cpu(self, monkeypatch):
        monkeypatch.setattr(cyclictest_module, "get_isolated_cpus", lambda: [2, 3, 7])

        bench = CyclictestBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)

        assert bench.get_command() == [
            "cyclictest",
            "-m",
            "-p",
            "98",
            "-i",
            "1000",
            "-D",
            "30",
            "-q",
            "-t",
            "3",
            "-a",
            "2,3,7",
        ]

    def test_records_isolated_workload_from_command(self, monkeypatch):
        monkeypatch.setattr(cyclictest_module, "get_isolated_cpus", lambda: [2, 3, 7])
        bench = CyclictestBenchmark()
        bench.configure(duration="5m", isolated_cpus_only=True)

        bench.get_command()

        assert bench.get_workload_config() == {
            "duration": "5m",
            "isolated_cpus_only": True,
            "cpu_selection": "isolated",
            "cpus": [2, 3, 7],
        }

    def test_records_smp_workload_from_affinity(self, monkeypatch):
        monkeypatch.setattr(cyclictest_module.os, "sched_getaffinity", lambda pid: {0, 4})
        bench = CyclictestBenchmark()

        bench.get_command()

        assert bench.get_workload_config() == {
            "duration": "30",
            "isolated_cpus_only": False,
            "cpu_selection": "smp",
            "cpus": [0, 4],
        }


class TestConfigure:
    def test_defaults(self):
        bench = CyclictestBenchmark()
        assert bench.duration == "30"
        assert bench.isolated_cpus_only is False

    def test_isolated_cpus_only_can_be_reset_to_false(self):
        bench = CyclictestBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)
        bench.configure(duration=None, isolated_cpus_only=False)
        assert bench.isolated_cpus_only is False


class TestCheckPrerequisites:
    def test_fails_when_isolated_cpus_only_and_none_found(self, monkeypatch):
        monkeypatch.setattr(cyclictest_module.shutil, "which", lambda name: "/usr/bin/cyclictest")
        monkeypatch.setattr(cyclictest_module, "get_isolated_cpus", lambda: [])

        bench = CyclictestBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)

        ok, msg = bench.check_prerequisites()

        assert ok is False
        assert "isolated CPUs" in msg

    def test_passes_when_isolated_cpus_only_and_some_found(self, monkeypatch):
        monkeypatch.setattr(cyclictest_module.shutil, "which", lambda name: "/usr/bin/cyclictest")
        monkeypatch.setattr(cyclictest_module, "get_isolated_cpus", lambda: [2, 3])

        bench = CyclictestBenchmark()
        bench.configure(duration=None, isolated_cpus_only=True)

        ok, _ = bench.check_prerequisites()

        assert ok is True

    def test_fails_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(cyclictest_module.shutil, "which", lambda name: None)

        bench = CyclictestBenchmark()

        ok, msg = bench.check_prerequisites()

        assert ok is False
        assert "cyclictest not found" in msg


CYCLICTEST_JSON = {
    "thread": {
        "0": {"min": 1.0, "avg": 2.0, "max": 5.0},
        "1": {"min": 0.5, "avg": 3.0, "max": 6.0},
    }
}


class TestRunOnce:
    def test_uses_base_command_plus_json_flag(self, monkeypatch):
        calls: list[list[str]] = []
        json_path_holder: dict[str, str] = {}

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            json_arg = next(a for a in cmd if a.startswith("--json="))
            json_path = json_arg.split("=", 1)[1]
            json_path_holder["path"] = json_path
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(CYCLICTEST_JSON, f)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = CyclictestBenchmark()
        bench.configure(duration="1m", isolated_cpus_only=True)
        monkeypatch.setattr(cyclictest_module, "get_isolated_cpus", lambda: [4, 5])

        metrics = bench.run_once()

        assert len(calls) == 1
        base_len = len(bench._base_command())
        assert calls[0][:base_len] == bench._base_command()
        assert calls[0][base_len] == f"--json={json_path_holder['path']}"
        assert "-D" in calls[0] and "1m" in calls[0]
        assert "-a" in calls[0] and "4,5" in calls[0]
        assert "-S" not in calls[0]
        assert calls[0][calls[0].index("-t") + 1] == "2"

        assert metrics["min_latency_us"] == 0.5
        assert metrics["avg_latency_us"] == 2.5
        assert metrics["max_latency_us"] == 6.0

    def test_no_thread_data_raises(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            json_arg = next(a for a in cmd if a.startswith("--json="))
            json_path = json_arg.split("=", 1)[1]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"thread": {}}, f)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = CyclictestBenchmark()

        try:
            bench.run_once()
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "no thread data" in str(e)


class TestGetUnits:
    def test_units(self):
        bench = CyclictestBenchmark()
        assert bench.get_units() == {
            "min_latency_us": "us",
            "avg_latency_us": "us",
            "max_latency_us": "us",
        }
