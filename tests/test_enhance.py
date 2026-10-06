"""Enhancer contract, executable discovery, and generated PCM audio checks."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from declip import enhance
from declip.contracts import EnhancerUnavailable, RenderError


@pytest.fixture
def fake_deepfilter(tmp_path, monkeypatch):
    directory = tmp_path / "fake bin"
    directory.mkdir()
    executable = directory / "deep-filter"
    log = tmp_path / "deepfilter-argv.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n" + '''
import json
import os
import shutil
import sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--help"]:
    print("--atten-lim-db --output-dir --compensate-delay")
    sys.exit(0)
with open(os.environ["DEEPFILTER_TEST_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
mode = os.environ.get("DEEPFILTER_TEST_MODE", "copy")
if mode == "fail":
    print("synthetic decoder failure", file=sys.stderr)
    sys.exit(7)
source = Path(args[-1])
out = Path(args[args.index("--output-dir") + 1])
out.mkdir(parents=True)
if mode == "missing":
    sys.exit(0)
if mode == "invalid":
    (out / source.name).write_text("invalid WAV")
else:
    shutil.copyfile(source, out / source.name)
''', encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("DEEPFILTER_TEST_LOG", str(log))
    # Test the supported platform independently of the CI host architecture.
    monkeypatch.setattr(enhance.sys, "platform", "linux")
    monkeypatch.setattr(enhance.platform, "machine", lambda: "x86_64")
    return executable, log


def _pcm(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<hhhh", 1000, -2000, 3000, -4000) * 100)


def _probe(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)["streams"][0]


@pytest.fixture
def stereo(tmp_path, make_media):
    return make_media(tmp_path, {
        "name": "stereo source.wav", "duration": 1.013,
        "inputs": ["aevalsrc=0.1*sin(2*PI*440*t)|0.2*sin(2*PI*880*t):s=44100:c=stereo"],
        "args": ["-c:a", "pcm_s24le"],
    })


@pytest.mark.contract
def test_registry_contract(monkeypatch):
    monkeypatch.setattr(enhance.AfftdnEnhancer, "available", lambda self: (True, "FFT"))
    monkeypatch.setattr(enhance.DeepFilterEnhancer, "available", lambda self: (False, "missing"))
    assert enhance.list_enhancers() == [
        ("none", True, "exact PCM copy"), ("afftdn", True, "FFT"),
        ("deepfilter", False, "missing"),
    ]
    assert enhance.select_enhancer().name == "afftdn"


@pytest.mark.parametrize("available,expected", [(True, "deepfilter"), (False, "afftdn")])
def test_auto_priority(monkeypatch, available, expected):
    monkeypatch.setattr(enhance.DeepFilterEnhancer, "available", lambda self: (available, "probe"))
    monkeypatch.setattr(enhance.AfftdnEnhancer, "available", lambda self: (True, "FFT"))
    assert enhance.select_enhancer("auto").name == expected


@pytest.mark.parametrize("name", ["deepfilter", "afftdn", "unknown", "auto"])
def test_unavailable_never_selects_none(monkeypatch, name):
    monkeypatch.setattr(enhance.DeepFilterEnhancer, "available", lambda self: (False, "no binary"))
    monkeypatch.setattr(enhance.AfftdnEnhancer, "available", lambda self: (False, "no filter"))
    with pytest.raises(EnhancerUnavailable):
        enhance.select_enhancer(name)


@pytest.mark.contract
def test_none_is_exact_pcm_copy_without_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    source, output = tmp_path / "source.wav", tmp_path / "copy.wav"
    _pcm(source)
    assert enhance.select_enhancer("none").enhance(source, output, 0.5) is None
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("kind", ["same", "hardlink", "symlink"])
def test_input_alias_is_refused(tmp_path, kind):
    source = tmp_path / "source.wav"
    _pcm(source)
    output = tmp_path / "alias.wav"
    if kind == "same":
        output = source
    elif kind == "hardlink":
        os.link(source, output)
    else:
        output.symlink_to(source)
    original = source.read_bytes()
    with pytest.raises(RenderError, match="differ"):
        enhance.NoneEnhancer().enhance(source, output, 0.5)
    assert source.read_bytes() == original


@pytest.mark.parametrize("strength", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
@pytest.mark.parametrize("backend", [enhance.NoneEnhancer, enhance.AfftdnEnhancer, enhance.DeepFilterEnhancer])
def test_invalid_strength_fails_before_writes(tmp_path, strength, backend):
    with pytest.raises(RenderError, match="strength"):
        backend().enhance(tmp_path / "in.wav", tmp_path / "out.wav", strength)
    assert not (tmp_path / "out.wav").exists()


def test_no_release_asset_is_unavailable(monkeypatch):
    monkeypatch.setattr(enhance.platform, "machine", lambda: "unsupported")
    available, reason = enhance.DeepFilterEnhancer().available()
    assert not available
    assert "unavailable" in reason and "release asset" in reason
    assert enhance.DEEPFILTER_RELEASE_URL in reason


def test_missing_tools(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert not enhance.AfftdnEnhancer().available()[0]
    assert not enhance.DeepFilterEnhancer().available()[0]


@pytest.mark.ffmpeg
def test_available_fake_binary(fake_deepfilter):
    available, reason = enhance.DeepFilterEnhancer().available()
    assert available, reason
    assert "--atten-lim-db" in reason
    assert enhance.select_enhancer().name == "deepfilter"


@pytest.mark.ffmpeg
def test_configured_binary_takes_precedence(fake_deepfilter, tmp_path, monkeypatch):
    config = Path(os.environ["DECLIP_CONFIG_DIR"])
    config.mkdir()
    alternate = tmp_path / "configured deep filter"
    shutil.copyfile(fake_deepfilter[0], alternate)
    alternate.chmod(0o755)
    (config / "config.json").write_text(json.dumps({"deepfilter_path": str(alternate)}))
    assert str(alternate) in enhance.DeepFilterEnhancer().available()[1]
    (config / "config.json").write_text(json.dumps({"deepfilter_path": str(tmp_path / "missing")}))
    assert not enhance.DeepFilterEnhancer().available()[0]


@pytest.mark.ffmpeg
def test_wrong_binary_is_unavailable(fake_deepfilter):
    executable = fake_deepfilter[0]
    executable.write_text(f"#!{sys.executable}\nprint('unrelated tool')\n")
    available, reason = enhance.DeepFilterEnhancer().available()
    assert not available
    assert "incompatible" in reason


@pytest.mark.ffmpeg
@pytest.mark.contract
@pytest.mark.parametrize("name", ["none", "afftdn", "deepfilter"])
def test_stereo_rate_layout_and_sample_count_kept(name, stereo, fake_deepfilter, tmp_path):
    output = tmp_path / "enhanced.wav"
    enhance.select_enhancer(name).enhance(stereo, output, 0.5)
    before, after = _probe(stereo), _probe(output)
    for key in ("sample_rate", "channels", "channel_layout", "duration_ts", "time_base"):
        assert after[key] == before[key], (key, before[key], after[key])
    assert after["bits_per_sample"] == 24


@pytest.mark.ffmpeg
@pytest.mark.parametrize("strength,limit", [(0.0, "0"), (0.5, "50"), (1.0, "100")])
def test_deepfilter_channel_order_and_strength(stereo, fake_deepfilter, tmp_path, strength, limit):
    output = tmp_path / "processed.wav"
    enhance.select_enhancer("deepfilter").enhance(stereo, output, strength)
    calls = [json.loads(line) for line in fake_deepfilter[1].read_text().splitlines()]
    assert len(calls) == 2
    for index, call in enumerate(calls):
        assert call[call.index("--atten-lim-db") + 1] == limit
        assert "--compensate-delay" in call
        assert Path(call[-1]).name == f"channel-{index}.wav"
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
        "-f", "f32le", "-c:a", "pcm_f32le", "-",
    ], capture_output=True, check=True).stdout
    samples = struct.unpack(f"<{len(raw) // 4}f", raw)
    # Distinct channel frequencies and levels expose swaps or mono collapse.
    for channel, frequency, amplitude in ((0, 440, 0.1), (1, 880, 0.2)):
        actual = samples[channel::2]
        expected = [amplitude * math.sin(2 * math.pi * frequency * i / 44100)
                    for i in range(len(actual))]
        rms_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)) / len(actual))
        assert rms_error < 0.0001


@pytest.mark.ffmpeg
@pytest.mark.parametrize("mode,match", [
    ("fail", "synthetic decoder failure"), ("missing", "produced no WAV"),
    ("invalid", "ffprobe"),
])
def test_deepfilter_failure_keeps_destination(stereo, fake_deepfilter, tmp_path, monkeypatch, mode, match):
    monkeypatch.setenv("DEEPFILTER_TEST_MODE", mode)
    output = tmp_path / "destination.wav"
    output.write_bytes(b"existing output")
    original = stereo.read_bytes()
    with pytest.raises(RenderError, match=match):
        enhance.select_enhancer("deepfilter").enhance(stereo, output, 0.5)
    assert output.read_bytes() == b"existing output"
    assert stereo.read_bytes() == original
    assert not list(tmp_path.glob("declip-deepfilter-*"))


@pytest.mark.ffmpeg
def test_afftdn_available_and_filter_strength(stereo, tmp_path, monkeypatch):
    backend = enhance.select_enhancer("afftdn")
    assert backend.available()[0]
    calls = []
    original = enhance._run

    def capture(argv, **kwargs):
        calls.append(argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(enhance, "_run", capture)
    backend.enhance(stereo, tmp_path / "output.wav", 1)
    assert any("afftdn=nr=24:nf=-50:tn=1" in argv for argv in calls)


def test_filter_absent_is_unavailable(monkeypatch):
    monkeypatch.setattr(enhance.shutil, "which", lambda name: name)
    monkeypatch.setattr(enhance, "_run", lambda *args, **kwargs: " ... unrelated A->A\n")
    assert not enhance.AfftdnEnhancer().available()[0]


def test_probe_failure_is_reported(monkeypatch):
    monkeypatch.setattr(enhance.shutil, "which", lambda name: name)

    def fail(*args, **kwargs):
        raise RenderError("probe failed")

    monkeypatch.setattr(enhance, "_run", fail)
    available, reason = enhance.AfftdnEnhancer().available()
    assert not available and "probe failed" in reason


def test_missing_copy_input_is_domain_error(tmp_path):
    with pytest.raises(RenderError, match="Cannot copy"):
        enhance.NoneEnhancer().enhance(tmp_path / "missing.wav", tmp_path / "out.wav", 0.5)
