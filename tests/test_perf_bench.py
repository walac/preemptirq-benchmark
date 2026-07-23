from __future__ import annotations

import subprocess
from types import SimpleNamespace

from preemptirq_benchmark.benchmarks.perf_bench import PerfBenchBenchmark
from preemptirq_benchmark.perf_stat import parse_perf_csv, run_with_perf_stat

PIPE_OUTPUT = "# Running 'sched/pipe' benchmark:\n" "4.321 usecs/op\n" "231234.567 ops/sec\n"

MESSAGING_OUTPUT = "# Running 'sched/messaging' benchmark:\n" "     Total time: 0.456 [sec]\n"


class TestGetCommand:
    def test_returns_messaging_not_pipe(self):
        # Regression test: get_command() previously returned the
        # "pipe" sub-benchmark command, so perf stat counters
        # collected via __main__'s post-loop wrapping were attributed
        # to "perf-bench" while actually only ever measuring "pipe",
        # even though "messaging" is the heavier scheduler sub-test
        # reported alongside them.
        bench = PerfBenchBenchmark()

        assert bench.get_command() == ["perf", "bench", "sched", "messaging"]
        assert bench.get_command() != ["perf", "bench", "sched", "pipe"]


class TestRunOnceUsesGetCommandForMessaging:
    def test_messaging_subprocess_call_matches_get_command(self, monkeypatch):
        # The command used to produce messaging_time_seconds must be
        # exactly the command perf-stat wraps (get_command()), so the
        # hardware counters end up keyed against the sub-benchmark
        # that actually produced them.
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[-1] == "pipe":
                return SimpleNamespace(stdout=PIPE_OUTPUT, stderr="", returncode=0)
            return SimpleNamespace(stdout=MESSAGING_OUTPUT, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = PerfBenchBenchmark()
        metrics = bench.run_once()

        assert len(calls) == 2
        assert calls[0] == ["perf", "bench", "sched", "pipe"]
        assert calls[1] == bench.get_command()
        assert calls[1] == ["perf", "bench", "sched", "messaging"]

        assert metrics["pipe_ops_per_sec"] == 231234.567
        assert metrics["pipe_usecs_per_op"] == 4.321
        assert metrics["messaging_time_seconds"] == 0.456


class TestPerfStatWrappingAttribution:
    def test_wrapped_command_and_counters_correspond_to_messaging(self, monkeypatch):
        # Simulate __main__'s post-loop perf-stat wrapping: it calls
        # run_with_perf_stat(bench.get_command()). Verify the wrapped
        # command is the messaging one and that parsed counters are
        # not silently mislabeled as belonging to "pipe".
        bench = PerfBenchBenchmark()
        cmd = bench.get_command()

        perf_stderr = "1234567;;cycles;100.00;;\n" "7654321;;instructions;100.00;;\n"

        captured: dict[str, list[str]] = {}

        def fake_run(perf_cmd, **kwargs):
            captured["cmd"] = perf_cmd
            return SimpleNamespace(returncode=0, stdout="", stderr=perf_stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)

        _, counters = run_with_perf_stat(cmd)

        # The wrapped command must be "sched messaging", not "sched pipe".
        assert captured["cmd"][-4:] == ["perf", "bench", "sched", "messaging"][-4:]
        assert counters == parse_perf_csv(perf_stderr)
        assert counters["cycles"] == 1234567
        assert isinstance(counters["cycles"], int)
        assert counters["instructions"] == 7654321
        assert isinstance(counters["instructions"], int)


class TestParsePerfCsvFractionalCounters:
    def test_task_clock_is_parsed_as_float(self):
        # task-clock reports a fractional millisecond value rather than
        # an integer count; it must not be silently dropped, and must
        # keep its float type rather than being truncated to int.
        perf_stderr = "123.456789;msec;task-clock;100.00;;\n"

        counters = parse_perf_csv(perf_stderr)

        assert counters["task-clock"] == 123.456789
        assert isinstance(counters["task-clock"], float)

    def test_whole_counts_stay_int(self):
        # Integer-looking counts must be parsed as int, not float, so
        # that most counters keep exact integer semantics.
        perf_stderr = "1234567;;cycles;100.00;;\n"

        counters = parse_perf_csv(perf_stderr)

        assert counters["cycles"] == 1234567
        assert isinstance(counters["cycles"], int)
