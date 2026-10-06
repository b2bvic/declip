"""Best-effort hardware discovery without model or GPU library imports."""

from __future__ import annotations

import csv
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, TypeVar

from declip.contracts import GpuInfo, HardwareInfo

_T = TypeVar("_T")


def _safe(call: Callable[[], _T], default: _T) -> _T:
    # Discovery must survive missing commands, permissions, and broken drivers.
    try:
        return call()
    except Exception:
        return default


def _command(*argv: str) -> str | None:
    def run() -> str | None:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=5, check=False
        )
        return (result.stdout.strip() or None) if result.returncode == 0 else None

    return _safe(run, None)


def _text(path: str) -> str:
    return _safe(lambda: Path(path).read_text(encoding="utf-8"), "")


def _memory(value: str | None, divisor: float) -> float | None:
    try:
        number = float(value) / divisor
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def _proc_value(text: str, key: str) -> str | None:
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == key:
            return value.strip() or None
    return None


def _gpus() -> tuple[GpuInfo, ...]:
    output = _command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    if not output:
        return ()
    devices = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 3 or not row[0].strip():
            continue
        name, memory, driver = (entry.strip() for entry in row)
        devices.append(
            GpuInfo(
                name,
                _memory(memory, 1024),
                None if driver in {"", "N/A", "[N/A]", "[Not Supported]"} else driver,
            )
        )
    return tuple(devices)


def detect() -> HardwareInfo:
    """Return the available facts; an unavailable probe leaves its field empty."""
    system = {"win32": "windows"}.get(sys.platform, sys.platform)
    machine = _safe(platform.machine, "").lower()
    arch = {"aarch64": "arm64", "amd64": "x86_64", "x64": "x86_64"}.get(
        machine, machine
    )
    cores = _safe(os.cpu_count, None) or 1
    chip = None
    ram = None
    if system == "darwin":
        chip = _command("sysctl", "-n", "machdep.cpu.brand_string")
        ram = _memory(_command("sysctl", "-n", "hw.memsize"), 1024**3)
    elif system == "linux":
        chip = _proc_value(_text("/proc/cpuinfo"), "model name")
        memory = _proc_value(_text("/proc/meminfo"), "MemTotal")
        ram = _memory(memory.split()[0] if memory else None, 1024**2)
    elif system == "windows":
        # Windows support is deferred; preserve the portable CPU fact only.
        chip = _safe(platform.processor, "") or None
    return HardwareInfo(
        system,
        arch,
        chip,
        ram,
        cores,
        system == "darwin" and machine == "arm64",
        _safe(_gpus, ()),
    )
