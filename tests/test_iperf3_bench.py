from __future__ import annotations

import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from preemptirq_benchmark.benchmarks.iperf3_bench import Iperf3Benchmark

TCP_JSON = json.dumps(
    {
        "end": {
            "sum_sent": {"bits_per_second": 9.5e9},
            "sum_received": {"bits_per_second": 9.4e9},
        }
    }
)

UDP_JSON_WITH_REVERSE = json.dumps(
    {
        "end": {
            "sum": {
                "bits_per_second": 1.0e9,
                "jitter_ms": 0.05,
                "lost_percent": 0.1,
            },
            "sum_bidir_reverse": {
                "bits_per_second": 0.8e9,
                "jitter_ms": 0.07,
                "lost_percent": 0.2,
            },
        }
    }
)

UDP_JSON_NO_REVERSE = json.dumps(
    {
        "end": {
            "sum": {
                "bits_per_second": 1.0e9,
                "jitter_ms": 0.05,
                "lost_percent": 0.1,
            }
        }
    }
)


def _make_fake_run(tcp_stdout: str, udp_stdout: str):
    def fake_run(cmd, **kwargs):
        if "-u" in cmd:
            return SimpleNamespace(stdout=udp_stdout, stderr="", returncode=0)
        return SimpleNamespace(stdout=tcp_stdout, stderr="", returncode=0)

    return fake_run


class TestRunOnceBidirReverseParsing:
    def test_extracts_both_forward_and_reverse_udp_throughput(self, monkeypatch):
        # Regression test: --bidir requests a simultaneous reverse-direction
        # UDP stream (server -> client), but run_once() previously only
        # ever read end.sum (the forward direction), never
        # end.sum_bidir_reverse -- silently dropping half of the
        # requested bidirectional measurement.
        monkeypatch.setattr(subprocess, "run", _make_fake_run(TCP_JSON, UDP_JSON_WITH_REVERSE))

        bench = Iperf3Benchmark()
        metrics = bench.run_once()

        assert metrics["udp_sender_gbps"] == pytest.approx(1.0)
        assert metrics["udp_receiver_gbps"] == pytest.approx(0.8)
        assert metrics["udp_jitter_ms"] == pytest.approx(0.05)
        assert metrics["udp_lost_pct"] == pytest.approx(0.1)

    def test_raises_clear_error_when_reverse_sum_missing(self, monkeypatch):
        # If iperf3's JSON is missing the reverse-direction summary,
        # the benchmark must fail loudly instead of silently omitting
        # udp_receiver_gbps.
        monkeypatch.setattr(subprocess, "run", _make_fake_run(TCP_JSON, UDP_JSON_NO_REVERSE))

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="cannot find expected keys in iperf3 UDP JSON"):
            bench.run_once()

    def test_get_units_documents_the_reverse_metric(self):
        # udp_receiver_gbps must be a first-class reported metric, not
        # just an internal value that gets computed and then dropped
        # for lack of a unit mapping.
        bench = Iperf3Benchmark()

        assert bench.get_units()["udp_receiver_gbps"] == "Gbps"


class _FakePopen:
    """Minimal stand-in for subprocess.Popen[str] used by setup()."""

    def __init__(self, returncode: int | None):
        self._returncode = returncode

    def poll(self) -> int | None:
        return self._returncode


def _fake_popen_factory(returncode: int | None, stderr_text: str = ""):
    """Build a fake Popen that writes into the real stderr= file setup() passes.

    setup() now captures the server's stderr to a temp file (not a
    pipe) rather than exposing it via a mockable Popen.stderr
    attribute, so the fake here writes directly into that file to
    simulate server output, just as the real subprocess would.
    """

    def fake_popen(*args, **kwargs):
        stderr_file = kwargs.get("stderr")
        if stderr_file is not None:
            stderr_file.write(stderr_text)
            stderr_file.seek(0)
        return _FakePopen(returncode)

    return fake_popen


class TestSetupServerStartupFailure:
    def test_raises_when_server_process_has_already_exited(self, monkeypatch):
        # Regression test: setup() started the iperf3 server via Popen
        # but never checked whether it was still alive after the
        # startup wait, so a server that failed immediately (port in
        # use, missing binary, crash) went unnoticed until the client
        # run failed later with an unrelated, confusing error.
        monkeypatch.setattr(
            subprocess,
            "Popen",
            _fake_popen_factory(
                returncode=1,
                stderr_text="iperf3: error - unable to start listener for connections: Address already in use\n",
            ),
        )
        monkeypatch.setattr(time, "sleep", lambda *_: None)

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="iperf3 server failed to start"):
            bench.setup()

        assert bench.server_proc is None
        assert bench._server_stderr is None

    def test_raises_when_server_exits_with_code_zero(self, monkeypatch):
        # Regression guard: a truthiness check on returncode (`if
        # returncode:`) would be a plausible but wrong alternative
        # implementation, since it silently treats a clean exit (0) as
        # "still running". The check must use `is not None`.
        monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(returncode=0))
        monkeypatch.setattr(time, "sleep", lambda *_: None)

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="iperf3 server failed to start"):
            bench.setup()

        assert bench.server_proc is None
        assert bench._server_stderr is None

    def test_does_not_raise_when_server_is_still_running(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(returncode=None))
        monkeypatch.setattr(time, "sleep", lambda *_: None)

        bench = Iperf3Benchmark()
        bench.setup()

        assert isinstance(bench.server_proc, _FakePopen)
        assert bench._server_stderr is not None


# Note: get_command(), run_once()'s client-side subprocess timing, and
# cleanup()'s real SIGTERM/SIGKILL handling all depend on genuinely
# spawning and waiting on OS processes. That behavior is deliberately
# left untested here rather than forced into a brittle mock, since it
# would mostly re-test subprocess/signal semantics rather than this
# module's logic.
