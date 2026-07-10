from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from preemptirq_benchmark.benchmarks.tracerbench import (
    STAT_NAMES,
    TEST_TYPES,
    TracerbenchBenchmark,
)


class TestStatNames:
    def test_includes_max_avg(self):
        assert "max_avg" in STAT_NAMES

    def test_matches_debugfs_layout(self):
        # Mirrors the per-test-type debugfs files exposed by the
        # tracerbench kernel module (median, average, max, max_avg,
        # percentile).
        assert STAT_NAMES == ["median", "average", "max", "max_avg", "percentile"]


class TestRunOnce:
    def test_populates_max_avg_for_every_test_type(self, monkeypatch):
        monkeypatch.setattr(Path, "write_text", lambda self, data, **kw: None)
        monkeypatch.setattr(Path, "read_text", lambda self, **kw: "42")

        bench = TracerbenchBenchmark()
        metrics = bench.run_once()

        for test_type in TEST_TYPES:
            for stat_name in STAT_NAMES:
                assert metrics[f"{test_type}/{stat_name}"] == 42.0
            assert f"{test_type}/max_avg" in metrics


class TestGetUnits:
    def test_includes_max_avg_units(self):
        bench = TracerbenchBenchmark()
        units = bench.get_units()

        for test_type in TEST_TYPES:
            assert units[f"{test_type}/max_avg"] == "cycles"


class TestCheckPrerequisites:
    def test_debugfs_already_present_skips_modprobe(self, monkeypatch):
        monkeypatch.setattr(Path, "is_dir", lambda self: True)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("modprobe should not be invoked when debugfs exists")

        monkeypatch.setattr(subprocess, "run", fail_if_called)

        bench = TracerbenchBenchmark()
        ok, msg = bench.check_prerequisites()

        assert ok is True
        assert msg == ""

    def test_modprobe_failure_reports_modprobe_error(self, monkeypatch):
        monkeypatch.setattr(Path, "is_dir", lambda self: False)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=1, stderr="modprobe: FATAL: Module tracerbench not found"
            ),
        )

        bench = TracerbenchBenchmark()
        ok, msg = bench.check_prerequisites()

        assert ok is False
        assert "modprobe failed" in msg
        assert "Module tracerbench not found" in msg
        assert "debugfs" not in msg.lower()

    def test_modprobe_succeeds_but_debugfs_missing_reports_debugfs_error(self, monkeypatch):
        # modprobe returns success, but the debugfs directory never
        # appears (e.g. debugfs is not mounted). This must NOT be
        # reported as a modprobe failure.
        monkeypatch.setattr(Path, "is_dir", lambda self: False)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stderr=""),
        )

        bench = TracerbenchBenchmark()
        ok, msg = bench.check_prerequisites()

        assert ok is False
        assert "modprobe failed" not in msg
        assert "debugfs" in msg.lower()
        assert "mount" in msg.lower()

    def test_modprobe_succeeds_and_debugfs_appears(self, monkeypatch):
        calls = {"n": 0}

        def fake_is_dir(self):
            calls["n"] += 1
            # False on the pre-modprobe check, True afterwards.
            return calls["n"] > 1

        monkeypatch.setattr(Path, "is_dir", fake_is_dir)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stderr=""),
        )

        bench = TracerbenchBenchmark()
        ok, msg = bench.check_prerequisites()

        assert ok is True
        assert msg == ""


# Note: exercising the real modprobe/debugfs interaction (actually
# loading the tracerbench.ko kernel module and mounting debugfs)
# requires root and a matching kernel build; that path is not
# practical to cover here and is left to manual/hardware testing.
