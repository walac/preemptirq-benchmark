from __future__ import annotations

import json
import shutil
import signal
import subprocess
import time

from preemptirq_benchmark.benchmarks import BenchmarkBase, register


@register
class Iperf3Benchmark(BenchmarkBase):
    """Network throughput and jitter benchmark using iperf3 over loopback."""

    name = "iperf3"
    default_iterations = 10

    def __init__(self) -> None:
        self.server_proc: subprocess.Popen[bytes] | None = None

    def check_prerequisites(self) -> tuple[bool, str]:
        """Check that iperf3 is installed.

        Returns:
            (True, "") if found, or (False, install hint) otherwise.
        """
        if shutil.which("iperf3"):
            return True, ""
        return False, "iperf3 not found (install: dnf install iperf3)"

    def setup(self) -> None:
        """Start the iperf3 server in daemon mode."""
        self.server_proc = subprocess.Popen(
            ["iperf3", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)

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
            metrics["tcp_receiver_gbps"] = tcp_end["sum_received"]["bits_per_second"] / 1e9
        except KeyError as e:
            raise RuntimeError(f"cannot find expected keys in iperf3 TCP JSON: {e}") from e

        udp = subprocess.run(
            ["iperf3", "-c", "127.0.0.1", "-t", "10", "-u", "-b", "0", "-J"],
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

    def get_units(self) -> dict[str, str]:
        """Return unit mapping for iperf3 metrics.

        Returns:
            Dict mapping each metric to its unit string.
        """
        return {
            "tcp_sender_gbps": "Gbps",
            "tcp_receiver_gbps": "Gbps",
            "udp_sender_gbps": "Gbps",
            "udp_jitter_ms": "ms",
            "udp_lost_pct": "%",
        }
