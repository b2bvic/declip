"""Portable discovery tests; no accelerator or host configuration required."""

import subprocess
from pathlib import Path

import pytest

from declip import hardware
from declip.contracts import GpuInfo, HardwareInfo


def host(monkeypatch, system, machine, cores=8):
    monkeypatch.setattr(hardware.sys, "platform", system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: machine)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: cores)


def commands(monkeypatch, values):
    seen = []

    def run(argv, **kwargs):
        assert kwargs["timeout"] == 5
        assert kwargs["capture_output"] and kwargs["text"]
        seen.append(tuple(argv))
        output = values.get(tuple(argv))
        if isinstance(output, Exception):
            raise output
        return subprocess.CompletedProcess(
            argv, 0 if output is not None else 1, output or "", ""
        )

    monkeypatch.setattr(hardware.subprocess, "run", run)
    return seen


GPU_COMMAND = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
)


@pytest.mark.contract
def test_apple_silicon_contract(monkeypatch):
    host(monkeypatch, "darwin", "arm64", 12)
    seen = commands(
        monkeypatch,
        {
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple Test Chip\n",
            ("sysctl", "-n", "hw.memsize"): str(24 * 1024**3),
        },
    )
    assert hardware.detect() == HardwareInfo(
        "darwin", "arm64", "Apple Test Chip", 24.0, 12, True, ()
    )
    assert len(seen) == 3


@pytest.mark.contract
def test_linux_proc_and_multiple_gpus(monkeypatch):
    host(monkeypatch, "linux", "AMD64")
    proc = {
        "/proc/cpuinfo": "processor : 0\nmodel name : Example CPU\nmodel name : Second CPU\n",
        "/proc/meminfo": f"MemTotal: {32 * 1024**2} kB\nMemFree: 128 kB\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: proc[str(self)])
    commands(
        monkeypatch,
        {GPU_COMMAND: "Example GPU, 8192, 555.42\nSecond GPU, 4096, 555.42\n"},
    )
    assert hardware.detect() == HardwareInfo(
        "linux",
        "x86_64",
        "Example CPU",
        32.0,
        8,
        False,
        (GpuInfo("Example GPU", 8.0, "555.42"), GpuInfo("Second GPU", 4.0, "555.42")),
    )


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), PermissionError(), subprocess.TimeoutExpired("probe", 5)],
)
def test_failed_commands_leave_fields_empty(monkeypatch, error):
    host(monkeypatch, "darwin", "x86_64", None)
    commands(
        monkeypatch,
        {
            ("sysctl", "-n", "machdep.cpu.brand_string"): error,
            ("sysctl", "-n", "hw.memsize"): error,
            GPU_COMMAND: error,
        },
    )
    assert hardware.detect() == HardwareInfo(
        "darwin", "x86_64", None, None, 1, False, ()
    )


@pytest.mark.parametrize("memory", ["garbage", "nan", "inf", "-1", "0"])
def test_invalid_memory_does_not_raise(monkeypatch, memory):
    host(monkeypatch, "darwin", "arm64")
    commands(monkeypatch, {("sysctl", "-n", "hw.memsize"): memory})
    assert hardware.detect().ram_gb is None


def test_proc_unreadable_and_machine_probe_fails(monkeypatch):
    host(monkeypatch, "linux", "aarch64")

    def unreadable(*args, **kwargs):
        raise PermissionError("probe denied")

    monkeypatch.setattr(Path, "read_text", unreadable)
    monkeypatch.setattr(hardware.platform, "machine", unreadable)
    monkeypatch.setattr(hardware.os, "cpu_count", unreadable)
    commands(monkeypatch, {})
    info = hardware.detect()
    assert info.chip is None and info.ram_gb is None and info.gpus == ()
    assert info.cpu_cores == 1 and info.arch == ""


def test_gpu_partial_unknown_fields_and_malformed_rows(monkeypatch):
    host(monkeypatch, "linux", "aarch64")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "")
    commands(
        monkeypatch,
        {
            GPU_COMMAND: 'bad row\n, 4096, 555\n"GPU, model", N/A, [N/A]\nGPU B, nan, 555\n'
        },
    )
    info = hardware.detect()
    assert info.arch == "arm64" and not info.apple_silicon
    assert info.gpus == (
        GpuInfo("GPU, model", None, None),
        GpuInfo("GPU B", None, "555"),
    )


def test_windows_cpu_fact_is_portable(monkeypatch):
    host(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "Example CPU")
    commands(monkeypatch, {})
    assert hardware.detect() == HardwareInfo(
        "windows", "x86_64", "Example CPU", None, 8, False, ()
    )
