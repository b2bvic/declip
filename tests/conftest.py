"""User-state isolation and shared media/golden test helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require",
        action="append",
        default=[],
        metavar="NAME",
        help="Fail instead of skipping tests for a required capability",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Write failing outputs to .actual files without replacing goldens",
    )


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DECLIP_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("DECLIP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DECLIP_TRANSCRIBER", raising=False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped:
        return
    requirements = set(item.config.getoption("--require"))
    names = {marker.name for marker in item.iter_markers()}
    for marker in item.iter_markers():
        if marker.name in {"extra", "hwenc"} and marker.args:
            names.add(marker.args[0])
    reason = str(report.longrepr)
    matched = requirements & names
    matched.update(name for name in requirements if name in reason)
    if matched:
        report.outcome = "failed"
        report.longrepr = (
            f"Required capability skipped ({', '.join(sorted(matched))}): {reason}"
        )


@pytest.fixture
def assert_golden(request):
    def compare(actual, golden_path: Path):
        path = Path(golden_path)
        text = (
            actual
            if isinstance(actual, str)
            else json.dumps(actual, ensure_ascii=False, indent=2) + "\n"
        )
        expected = path.read_text(encoding="utf-8")
        if text != expected:
            if request.config.getoption("--update-golden"):
                path.with_name(path.name + ".actual").write_text(text, encoding="utf-8")
            pytest.fail(f"Golden mismatch: {path}")

    return compare


@pytest.fixture
def make_media():
    def generate(tmp_path: Path, spec: dict) -> Path:
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg unavailable")
        output = Path(tmp_path) / spec.get("name", "clip.mp4")
        argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        inputs = spec.get(
            "inputs",
            ["testsrc2=size=320x240:rate=30", "sine=frequency=1000:sample_rate=48000"],
        )
        for source in inputs:
            argv += ["-f", "lavfi", "-i", source]
        argv += ["-t", str(spec.get("duration", 2))]
        argv += list(
            spec.get("args", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
        )
        subprocess.run([*argv, str(output)], check=True, capture_output=True, text=True)
        return output

    return generate
