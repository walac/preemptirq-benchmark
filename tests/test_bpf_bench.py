from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from preemptirq_benchmark.benchmarks.bpf_bench import BpfLpmTrieLookupBenchmark


class TestLpmTrieLookupParsing:
    """Tests for bpf-lpm-trie-lookup benchmark output parsing."""

    def test_parses_throughput_with_k_unit(self, monkeypatch):
        # Regression test: the regex captured the unit prefix (K/M/G)
        # but the code ignored it, storing only the numeric value.
        # When bench output varied between "1500 K ops/s" and
        # "1.5 M ops/s", both stored as raw numbers but reported as
        # "M ops/s", producing false improvements.
        stdout = (
            "Summary: throughput 1500.00 ± 10.00 K ops/s, "
            "latency 78.05 ns/op\n"
        )

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = BpfLpmTrieLookupBenchmark()
        metrics = bench.run_once()

        # 1500 K ops/s = 1.5 M ops/s
        assert metrics["throughput_ops_per_sec"] == pytest.approx(1.5)
        assert metrics["latency_per_op"] == pytest.approx(78.05)

    def test_parses_throughput_with_m_unit(self, monkeypatch):
        stdout = (
            "Summary: throughput 12.82 ± 0.05 M ops/s, "
            "latency 78.05 ns/op\n"
        )

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = BpfLpmTrieLookupBenchmark()
        metrics = bench.run_once()

        # 12.82 M ops/s stored as-is (M is 1.0x multiplier)
        assert metrics["throughput_ops_per_sec"] == pytest.approx(12.82)
        assert metrics["latency_per_op"] == pytest.approx(78.05)

    def test_parses_throughput_with_g_unit(self, monkeypatch):
        # Edge case: if bench ever outputs in G ops/s
        stdout = (
            "Summary: throughput 2.5 ± 0.01 G ops/s, "
            "latency 0.40 ns/op\n"
        )

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = BpfLpmTrieLookupBenchmark()
        metrics = bench.run_once()

        # 2.5 G ops/s = 2500 M ops/s
        assert metrics["throughput_ops_per_sec"] == pytest.approx(2500.0)
        assert metrics["latency_per_op"] == pytest.approx(0.40)

    def test_raises_on_missing_throughput(self, monkeypatch):
        # Partial parse failure — latency present but throughput missing
        stdout = "Summary: latency 78.05 ns/op\n"

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = BpfLpmTrieLookupBenchmark()

        with pytest.raises(
            RuntimeError,
            match="lpm-trie-lookup partial parse failure, missing: throughput_ops_per_sec",
        ):
            bench.run_once()

    def test_raises_on_unknown_unit_prefix(self, monkeypatch):
        # Unit conversion error path: bench outputs an unknown unit
        # prefix (lowercase, typo, or new unit not in the K/M/G map).
        # The code must fail loudly instead of silently misinterpreting
        # the value.
        stdout = "Summary: throughput 1500.00 ± 10.00 k ops/s, latency 78.05 ns/op\n"

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bench = BpfLpmTrieLookupBenchmark()

        with pytest.raises(RuntimeError, match="unexpected unit prefix in bench output"):
            bench.run_once()

    def test_get_units_documents_both_metrics(self):
        bench = BpfLpmTrieLookupBenchmark()
        units = bench.get_units()

        assert units["throughput_ops_per_sec"] == "M ops/s"
        assert units["latency_per_op"] == "ns/op"
