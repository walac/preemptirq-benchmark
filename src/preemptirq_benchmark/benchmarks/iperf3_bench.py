from __future__ import annotations

import json
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import IO

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class Iperf3Benchmark(BenchmarkBase):
    """Network throughput and jitter benchmark using iperf3 over loopback."""

    name = "iperf3"
    description = "Networking throughput and jitter"
    default_iterations = 10

    _SERVER_PORT = 5201
    _SERVER_STARTUP_TIMEOUT_S = 5.0
    _SERVER_POLL_INTERVAL_S = 0.05

    def __init__(self) -> None:
        self.server_proc: subprocess.Popen[str] | None = None
        self._server_stderr: IO[str] | None = None

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that iperf3 is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("iperf3"):
            return True, ""
        return False, "iperf3 not found (install: dnf install iperf3)"

    def setup(self) -> None:
        """Start the iperf3 server in daemon mode.

        The server's stderr is captured to a temp file rather than a
        pipe: a pipe's OS buffer is bounded (64KB on Linux), and
        nothing reads it once startup succeeds, so a pipe would risk
        the server blocking on a write() call and hanging for the
        rest of the run if it ever logs enough warnings over a long
        benchmark session. A temp file has no such bound.

        Raises:
            RuntimeError: If the server process exits before it starts
                accepting connections (e.g. port already in use, or the
                binary crashed immediately), or if it doesn't become
                connectable within the startup timeout.
        """
        self._server_stderr = tempfile.TemporaryFile(mode="w+")
        self.server_proc = subprocess.Popen(
            ["iperf3", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=self._server_stderr,
            text=True,
        )
        proc = self.server_proc
        deadline = time.monotonic() + self._SERVER_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                with socket.create_connection(
                    ("127.0.0.1", self._SERVER_PORT), timeout=self._SERVER_POLL_INTERVAL_S
                ):
                    return
            except OSError:
                time.sleep(self._SERVER_POLL_INTERVAL_S)

        returncode = proc.poll()
        self._server_stderr.seek(0)
        stderr = self._server_stderr.read()
        self._server_stderr.close()
        self._server_stderr = None
        self.server_proc = None
        if returncode is not None:
            raise RuntimeError(
                f"iperf3 server failed to start (exit code {returncode}): {stderr.strip()}"
            )
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"iperf3 server did not become connectable within "
            f"{self._SERVER_STARTUP_TIMEOUT_S}s: {stderr.strip()}"
        )

    def run_once(self) -> dict[str, float]:
        """Run TCP and UDP bidirectional tests and parse JSON results.

        Returns:
            Dict with tcp_sender_gbps, tcp_receiver_gbps,
            udp_sender_gbps, udp_receiver_gbps, udp_jitter_ms,
            and udp_lost_pct.

        Raises:
            RuntimeError: If iperf3 JSON output cannot be parsed.
        """
        metrics: dict[str, float] = {}

        tcp = subprocess.run(
            ["iperf3", "-c", "127.0.0.1", "--bidir", "-t", "10", "-J"],
            capture_output=True,
            text=True,
            check=True,
        )
        try:
            tcp_data = json.loads(tcp.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cannot parse iperf3 TCP JSON: {tcp.stdout[:200]}") from e
        try:
            tcp_end = tcp_data["end"]
            metrics["tcp_sender_gbps"] = tcp_end["sum_sent"]["bits_per_second"] / 1e9
            metrics["tcp_receiver_gbps"] = (
                tcp_end["sum_received_bidir_reverse"]["bits_per_second"] / 1e9
            )
        except KeyError as e:
            raise RuntimeError(f"cannot find expected keys in iperf3 TCP JSON: {e}") from e

        udp = subprocess.run(
            ["iperf3", "-c", "127.0.0.1", "--bidir", "-t", "10", "-u", "-b", "100G", "-J"],
            capture_output=True,
            text=True,
            check=True,
        )
        try:
            udp_data = json.loads(udp.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cannot parse iperf3 UDP JSON: {udp.stdout[:200]}") from e

        try:
            udp_sum = udp_data["end"]["sum"]
            metrics["udp_sender_gbps"] = udp_sum["bits_per_second"] / 1e9
            metrics["udp_jitter_ms"] = udp_sum["jitter_ms"]
            metrics["udp_lost_pct"] = udp_sum["lost_percent"]
            udp_received_reverse = udp_data["end"]["sum_received_bidir_reverse"]
            metrics["udp_receiver_gbps"] = udp_received_reverse["bits_per_second"] / 1e9
        except KeyError as e:
            raise RuntimeError(f"cannot find expected keys in iperf3 UDP JSON: {e}") from e

        return metrics

    def get_command(self) -> list[str]:
        """Return the TCP client command for perf stat wrapping.

        Returns:
            The iperf3 TCP client command.
        """
        return ["iperf3", "-c", "127.0.0.1", "--bidir", "-t", "10"]

    def cleanup(self) -> None:
        """Kill the iperf3 server process."""
        if self.server_proc:
            self.server_proc.send_signal(signal.SIGTERM)
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                self.server_proc.wait()
            self.server_proc = None
        if self._server_stderr:
            self._server_stderr.close()
            self._server_stderr = None

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for iperf3 metrics.

        Returns:
            Dict mapping each metric to its unit string.
        """
        return {
            "tcp_sender_gbps": "Gbps",
            "tcp_receiver_gbps": "Gbps",
            "udp_sender_gbps": "Gbps",
            "udp_receiver_gbps": "Gbps",
            "udp_jitter_ms": "ms",
            "udp_lost_pct": "%",
        }
