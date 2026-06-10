from __future__ import annotations

import pytest

from preemptirq_benchmark.benchmarks import (
    ALL_BENCHMARK_NAMES,
    BENCHMARK_DESCRIPTIONS,
    REGISTRY,
    BenchmarkBase,
    BenchmarkResult,
    check_all_prerequisites,
    get_benchmark,
    import_all,
    register,
    resolve_benchmarks,
)


class TestResolveBenchmarks:
    def test_no_flags_returns_all(self):
        result = resolve_benchmarks(None, None, False)
        assert result == ALL_BENCHMARK_NAMES

    def test_all_flag_returns_all(self):
        result = resolve_benchmarks(None, None, True)
        assert result == ALL_BENCHMARK_NAMES

    def test_include_specific(self):
        result = resolve_benchmarks("hackbench,fio", None, False)
        assert result == ["hackbench", "fio"]

    def test_include_single(self):
        result = resolve_benchmarks("hackbench", None, False)
        assert result == ["hackbench"]

    def test_include_with_whitespace(self):
        result = resolve_benchmarks("hackbench , fio", None, False)
        assert result == ["hackbench", "fio"]

    def test_exclude_specific(self):
        result = resolve_benchmarks(None, "hackbench,fio", False)
        assert "hackbench" not in result
        assert "fio" not in result
        assert len(result) == len(ALL_BENCHMARK_NAMES) - 2

    def test_include_and_all_flag_error(self):
        with pytest.raises(SystemExit):
            resolve_benchmarks("hackbench", None, True)

    def test_include_and_exclude_error(self):
        with pytest.raises(SystemExit):
            resolve_benchmarks("hackbench", "fio", False)

    def test_include_unknown_benchmark(self):
        with pytest.raises(SystemExit):
            resolve_benchmarks("hackbench,nonexistent", None, False)

    def test_exclude_unknown_benchmark(self):
        with pytest.raises(SystemExit):
            resolve_benchmarks(None, "nonexistent", False)

    def test_exclude_all_benchmarks(self):
        all_names = ",".join(ALL_BENCHMARK_NAMES)
        with pytest.raises(SystemExit):
            resolve_benchmarks(None, all_names, False)

    def test_preserves_order(self):
        result = resolve_benchmarks("fio,hackbench", None, False)
        assert result == ["fio", "hackbench"]

    def test_exclude_preserves_original_order(self):
        result = resolve_benchmarks(None, "hackbench", False)
        remaining = [n for n in ALL_BENCHMARK_NAMES if n != "hackbench"]
        assert result == remaining


class TestRegisterAndGetBenchmark:
    def test_import_all_populates_registry(self):
        # Note: after the first call, subsequent import_all() calls are
        # no-ops due to Python's module cache. This test verifies the
        # steady-state (registry is populated), not that a fresh import
        # transitions it from empty.
        import_all()
        assert len(REGISTRY) > 0
        assert "hackbench" in REGISTRY

    def test_get_benchmark_returns_instance(self):
        import_all()
        bench = get_benchmark("hackbench")
        assert isinstance(bench, BenchmarkBase)
        assert bench.name == "hackbench"

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(KeyError):
            get_benchmark("nonexistent_benchmark_xyz")

    def test_register_decorator(self):
        @register
        class _FakeBench(BenchmarkBase):
            name = "fake-test-bench"
            default_iterations = 1

            def check_prerequisites(self):
                return True, ""

            def run_once(self):
                return {"val": 1.0}

        assert "fake-test-bench" in REGISTRY
        assert _FakeBench.name == "fake-test-bench"
        instance = get_benchmark("fake-test-bench")
        assert instance.name == "fake-test-bench"

        del REGISTRY["fake-test-bench"]

    def test_descriptions_match_names(self):
        for name in ALL_BENCHMARK_NAMES:
            assert name in BENCHMARK_DESCRIPTIONS, f"Missing description for {name}"


class TestCheckAllPrerequisites:
    def test_all_pass(self, capsys):
        class OkBench(BenchmarkBase):
            name = "ok-bench"
            default_iterations = 1

            def check_prerequisites(self):
                return True, ""

            def run_once(self):
                return {}

        check_all_prerequisites([OkBench()])
        captured = capsys.readouterr()
        assert "[OK]" in captured.out

    def test_one_fails(self, capsys):
        class FailBench(BenchmarkBase):
            name = "fail-bench"
            default_iterations = 1

            def check_prerequisites(self):
                return False, "missing tool"

            def run_once(self):
                return {}

        with pytest.raises(SystemExit):
            check_all_prerequisites([FailBench()])

        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out
        assert "missing tool" in captured.out

    def test_mixed_pass_and_fail(self, capsys):
        class OkBench(BenchmarkBase):
            name = "ok"
            default_iterations = 1

            def check_prerequisites(self):
                return True, ""

            def run_once(self):
                return {}

        class FailBench(BenchmarkBase):
            name = "fail"
            default_iterations = 1

            def check_prerequisites(self):
                return False, "not found"

            def run_once(self):
                return {}

        with pytest.raises(SystemExit):
            check_all_prerequisites([OkBench(), FailBench()])

        captured = capsys.readouterr()
        assert "[OK]" in captured.out
        assert "[FAIL]" in captured.out


class TestBenchmarkResult:
    def test_default_fields(self):
        r = BenchmarkResult(name="test")
        assert r.name == "test"
        assert r.metrics == {}
        assert r.units == {}
        assert r.perf_counters == {}
        assert r.iterations == 0


class TestBenchmarkBase:
    def test_default_methods(self):
        class MinimalBench(BenchmarkBase):
            name = "minimal"
            default_iterations = 1

            def check_prerequisites(self):
                return True, ""

            def run_once(self):
                return {"v": 1.0}

        bench = MinimalBench()
        bench.configure(kernel_src=None)
        bench.setup()
        assert bench.get_command() is None
        assert bench.get_units() == {}
        bench.cleanup()
