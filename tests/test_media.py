"""Hand-authored probe records and source-binding regression tests."""

import hashlib
import json
import os
import subprocess
import sys
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from declip import media
from declip.contracts import DeclipError, MediaInfo, ToolMissing

pytestmark = pytest.mark.contract
FIXTURES = Path(__file__).parent / "fixtures" / "ffprobe"


@pytest.fixture
def probe_fixture(tmp_path, monkeypatch):
    source = tmp_path / "clip.mov"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(
        media, "require_tools", lambda: (Path("ffmpeg"), Path("ffprobe"))
    )
    calls = []

    def run(name, mutate=None):
        data = json.loads((FIXTURES / name).read_text())
        if mutate:
            mutate(data)

        def invoke(argv, **kwargs):
            calls.append(argv)
            assert kwargs["capture_output"] and kwargs["timeout"] == 60
            response = (
                {"packets": data.get("packets", [])}
                if "-show_packets" in argv
                else data
            )
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

        monkeypatch.setattr(media.subprocess, "run", invoke)
        return media.probe_media(source), calls

    return run


@pytest.mark.parametrize(
    "rate, expected",
    [
        (Fraction(2997, 100), Fraction(30000, 1001)),
        (Fraction(5994, 100), Fraction(60000, 1001)),
        (Fraction(24), Fraction(24)),
        (Fraction(27), Fraction(27)),
        (Fraction(0), None),
        (Fraction(-1), None),
        ("0/0", None),
        ("n/a", None),
        (Fraction(3003, 100), Fraction(30)),
        (Fraction(30031, 1000), Fraction(30031, 1000)),
    ],
)
def test_snap_fps(rate, expected):
    assert media.snap_fps(rate) == expected


def test_cfr_rate_and_time_origin(probe_fixture):
    result, calls = probe_fixture("cfr_2997.json")
    assert result.fps == Fraction(30000, 1001)
    assert not result.vfr
    assert result.start_time == 1.5
    assert result.duration == 10.01
    assert result.audio_index == 1
    assert result.timecode == "01:00:00:00"
    assert MediaInfo.from_dict(result.to_dict()) == result
    assert calls[1][calls[1].index("-read_intervals") + 1] == "%+#300"
    assert calls[1][calls[1].index("-show_entries") + 1] == "packet=pts_time"


def test_vfr_conform_and_display_matrix_rotation(probe_fixture):
    result, _ = probe_fixture("vfr_phone.json")
    assert result.vfr
    assert result.fps == Fraction(30000, 1001)
    assert result.rotation == 270


def test_10bit_color_and_audio_depth(probe_fixture):
    result, _ = probe_fixture("hevc_hlg.json")
    assert result.bit_depth == 10
    assert result.color.trc == "arib-std-b67"
    assert result.color.primaries == "bt2020"
    assert result.color.matrix == "bt2020nc"
    assert result.color.range == "tv"
    assert result.audio_streams[0].bit_depth == 24
    assert result.dropped_streams == (4,)
    assert not result.vfr  # Packets are deliberately out of decode order.


@pytest.mark.parametrize(
    "fmt, raw, expected",
    [
        ("p010le", "8", 10),
        ("yuv422p12le", "8", 12),
        ("unknown", "10", 10),
        ("yuv420p", "0", 8),
    ],
)
def test_pixel_format_precedes_raw_depth(probe_fixture, fmt, raw, expected):
    def mutate(data):
        data["streams"][0].update(pix_fmt=fmt, bits_per_raw_sample=raw)

    result, _ = probe_fixture("hevc_hlg.json", mutate)
    assert result.bit_depth == expected


def test_cover_art_is_audio_only(probe_fixture):
    result, calls = probe_fixture("mp3_cover.json")
    assert not result.has_video
    assert result.video_index is None and result.fps is None
    assert result.audio_index == 0
    assert result.dropped_streams == (1,)
    assert len(calls) == 1


def test_first_real_video_and_default_audio_indices(probe_fixture):
    result, calls = probe_fixture("two_audio.json")
    assert result.video_index == 2 and result.audio_index == 3
    assert [a.index for a in result.audio_streams] == [1, 3]
    assert result.audio_streams[1].language == "fra"
    assert result.dropped_streams == (0, 4, 5, 6)
    assert calls[1][calls[1].index("-select_streams") + 1] == "2"


def test_no_default_audio_uses_first(probe_fixture):
    def mutate(data):
        data["streams"][3]["disposition"]["default"] = 0

    result, _ = probe_fixture("two_audio.json", mutate)
    assert result.audio_index == 1


def test_zero_average_falls_back_to_nominal(probe_fixture):
    result, _ = probe_fixture("zero_avg.json")
    assert result.fps == Fraction(24) and not result.vfr


def test_neither_rate_valid_is_vfr(probe_fixture):
    def mutate(data):
        data["streams"][0]["r_frame_rate"] = "invalid"

    result, _ = probe_fixture("zero_avg.json", mutate)
    assert result.vfr and result.fps == Fraction(30000, 1001)


def test_packet_vfr_even_when_rates_agree(probe_fixture):
    def mutate(data):
        data["streams"][0]["avg_frame_rate"] = "30/1"
        data["streams"][0]["r_frame_rate"] = "30/1"

    result, _ = probe_fixture("vfr_phone.json", mutate)
    assert result.vfr and result.fps == Fraction(30)


def test_packet_vfr_strict_one_percent_boundary():
    # 100 deltas, median 1. One 2-second delta is exactly 1%, two exceed it.
    times = list(range(100)) + [101]
    assert not media._packet_vfr([{"pts_time": str(t)} for t in times])
    times[-2:] = [100, 102]
    assert media._packet_vfr([{"pts_time": str(t)} for t in times])
    assert not media._packet_vfr([{"pts_time": "N/A"}])


@pytest.mark.parametrize(
    "platform, hint", [("darwin", "brew"), ("linux", "apt"), ("win32", "PATH")]
)
def test_missing_tools_install_hint(monkeypatch, platform, hint):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(media.sys, "platform", platform)
    with pytest.raises(ToolMissing, match=hint) as error:
        media.require_tools()
    assert "ffmpeg" in str(error.value) and "ffprobe" in str(error.value)


def test_empty_path_reports_without_traceback():
    env = dict(os.environ, PATH="")
    code = (
        "from declip.media import require_tools\n"
        "from declip.contracts import ToolMissing\n"
        "try: require_tools()\n"
        "except ToolMissing as e: print(str(e)); raise SystemExit(e.exit_code)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "ffmpeg" in result.stdout and "ffprobe" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_require_tools_returns_paths(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/tools/{name}")
    assert media.require_tools() == (Path("/tools/ffmpeg"), Path("/tools/ffprobe"))


def test_hash_reads_middle_of_equal_wavs(tmp_path):
    paths = [tmp_path / name for name in ("a.wav", "b.wav")]
    for path in paths:
        with wave.open(str(path), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x00\x00" * 200000)
    body = bytearray(paths[1].read_bytes())
    body[200000:200002] = b"\x01\x00"
    paths[1].write_bytes(body)
    a, b = (path.read_bytes() for path in paths)
    assert len(a) == len(b) and a[:65536] == b[:65536] and a[-65536:] == b[-65536:]
    assert media.file_hash(paths[0]) == hashlib.sha256(a).hexdigest()
    assert media.file_hash(paths[0]) != media.file_hash(paths[1])


def test_hash_cache_and_refresh_with_restored_stat(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"old")
    old = media.file_hash(path)
    stat = path.stat()
    path.write_bytes(b"new")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert media.file_hash(path) == old
    assert media.file_hash(path, refresh=True) == hashlib.sha256(b"new").hexdigest()
    path.write_bytes(b"a different size")
    assert media.file_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    store = media.cache_dir() / "hashes.json"
    assert str(path.resolve()) in store.read_text()
    store.write_text("broken")
    assert media.file_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_failure_is_domain_error(tmp_path, monkeypatch):
    path = tmp_path / "clip"
    path.touch()
    monkeypatch.setattr(
        media, "require_tools", lambda: (Path("ffmpeg"), Path("ffprobe"))
    )
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="invalid input"
        ),
    )
    with pytest.raises(DeclipError, match="invalid input"):
        media.probe_media(path)


def test_hash_missing_source_is_domain_error(tmp_path):
    with pytest.raises(DeclipError, match="Cannot hash source"):
        media.file_hash(tmp_path / "missing")
