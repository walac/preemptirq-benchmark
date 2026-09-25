from __future__ import annotations

import json
import socket
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
            "sum_sent_bidir_reverse": {"bits_per_second": 7.2e9},
            "sum_received_bidir_reverse": {"bits_per_second": 7.1e9},
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
            "sum_received_bidir_reverse": {"bits_per_second": 0.6e9},
        }
    }
)

UDP_JSON_NO_REVERSE_RECEIVE = json.dumps(
    {
        "end": {
            "sum": {
                "bits_per_second": 1.0e9,
                "jitter_ms": 0.05,
                "lost_percent": 0.1,
            },
            "sum_bidir_reverse": {"bits_per_second": 0.8e9},
        }
    }
)

TCP_JSON_NO_REVERSE = json.dumps(
    {
        "end": {
            "sum_sent": {"bits_per_second": 9.5e9},
            "sum_received": {"bits_per_second": 9.4e9},
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
        # In the reverse UDP stream, sent and received rates differ when
        # packets are lost. Report the client's received rate, not the
        # server's offered send rate.
        monkeypatch.setattr(subprocess, "run", _make_fake_run(TCP_JSON, UDP_JSON_WITH_REVERSE))

        bench = Iperf3Benchmark()
        metrics = bench.run_once()

        assert metrics["udp_sender_gbps"] == pytest.approx(1.0)
        assert metrics["udp_receiver_gbps"] == pytest.approx(0.6)
        assert metrics["udp_jitter_ms"] == pytest.approx(0.05)
        assert metrics["udp_lost_pct"] == pytest.approx(0.1)

    def test_extracts_both_forward_and_reverse_tcp_throughput(self, monkeypatch):
        # Regression test: --bidir also requests a simultaneous
        # reverse-direction TCP stream (server -> client), but
        # tcp_receiver_gbps previously read end.sum_received, which is
        # the *forward* stream measured at its receiving end (the
        # server) -- not the reverse stream. That silently duplicated
        # tcp_sender_gbps under a different name and never reported the
        # actual reverse-direction throughput.
        monkeypatch.setattr(subprocess, "run", _make_fake_run(TCP_JSON, UDP_JSON_WITH_REVERSE))

        bench = Iperf3Benchmark()
        metrics = bench.run_once()

        assert metrics["tcp_sender_gbps"] == pytest.approx(9.5)
        assert metrics["tcp_receiver_gbps"] == pytest.approx(7.1)

    def test_raises_clear_error_when_reverse_receive_sum_missing(self, monkeypatch):
        # If iperf3's JSON is missing the reverse receive summary,
        # the benchmark must fail loudly instead of silently omitting
        # udp_receiver_gbps or using the server send rate.
        monkeypatch.setattr(
            subprocess, "run", _make_fake_run(TCP_JSON, UDP_JSON_NO_REVERSE_RECEIVE)
        )

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="sum_received_bidir_reverse"):
            bench.run_once()

    def test_raises_clear_error_when_tcp_reverse_sum_missing(self, monkeypatch):
        # Same failure mode as above, but for the TCP path: if iperf3's
        # JSON is missing sum_received_bidir_reverse, the benchmark must
        # fail loudly instead of silently falling back to some other
        # field (e.g. the forward-direction sum_received).
        monkeypatch.setattr(
            subprocess, "run", _make_fake_run(TCP_JSON_NO_REVERSE, UDP_JSON_WITH_REVERSE)
        )

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="cannot find expected keys in iperf3 TCP JSON"):
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

    def kill(self) -> None:
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self._returncode


class _FakeSocket:
    """Minimal stand-in for the socket returned by socket.create_connection."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_create_connection_refused(*args, **kwargs):
    raise OSError("connection refused")


def _fake_create_connection_accepts(*args, **kwargs):
    return _FakeSocket()


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
        monkeypatch.setattr(socket, "create_connection", _fake_create_connection_refused)
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
        monkeypatch.setattr(socket, "create_connection", _fake_create_connection_refused)
        monkeypatch.setattr(time, "sleep", lambda *_: None)

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="iperf3 server failed to start"):
            bench.setup()

        assert bench.server_proc is None
        assert bench._server_stderr is None

    def test_raises_when_server_never_becomes_connectable(self, monkeypatch):
        # Regression test: setup() previously did a single flat sleep(0.5)
        # and never actually verified the server was accepting
        # connections, so a slow-starting server (e.g. under load) would
        # let setup() succeed and the first run_once() connect would fail
        # with a confusing, unrelated error instead.
        monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(returncode=None))
        monkeypatch.setattr(socket, "create_connection", _fake_create_connection_refused)
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        monkeypatch.setattr(Iperf3Benchmark, "_SERVER_STARTUP_TIMEOUT_S", 0.0)

        bench = Iperf3Benchmark()

        with pytest.raises(RuntimeError, match="did not become connectable"):
            bench.setup()

        assert bench.server_proc is None
        assert bench._server_stderr is None

    def test_does_not_raise_once_server_becomes_connectable(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(returncode=None))
        monkeypatch.setattr(socket, "create_connection", _fake_create_connection_accepts)
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
