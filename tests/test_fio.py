from __future__ import annotations

from types import SimpleNamespace

import preemptirq_benchmark.benchmarks.fio as fio_module
from preemptirq_benchmark.benchmarks.fio import FioBenchmark


class TestFioWorkingSet:
    def test_command_uses_the_whole_device_as_its_working_set(self):
        # Regression test: fio treats an unqualified --size value as bytes.
        # A 16,384-byte range only contains four 4 KiB blocks, so it cannot
        # supply meaningful random I/O at iodepth 64. Omitting --size lets
        # fio use the full device, including an already-loaded null_blk.
        command = FioBenchmark().get_command()

        assert not any(argument.startswith("--size=") for argument in command)

    def test_prerequisites_create_a_device_large_enough_for_the_job(self, monkeypatch):
        # The device created by null_blk defaults to 250 MiB. Its setup must
        # request 4 GiB or the corrected fio job cannot run.
        device_created = False

        def device_exists():
            return device_created

        def run_modprobe(command, **kwargs):
            nonlocal device_created
            device_created = command == ["modprobe", "null_blk", "irqmode=1", "gb=4"]
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(fio_module.shutil, "which", lambda name: "/usr/bin/fio")
        monkeypatch.setattr(fio_module, "NULLB_DEV", SimpleNamespace(exists=device_exists))
        monkeypatch.setattr(fio_module.subprocess, "run", run_modprobe)

        ready, _ = FioBenchmark().check_prerequisites()

        assert ready is True
