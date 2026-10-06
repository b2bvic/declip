"""Output guards against real, generated 8-bit and 10-bit encoded media."""

from dataclasses import replace
from fractions import Fraction
import shutil
import subprocess

import pytest

from declip import encoders
from declip.contracts import Capabilities, ColorTags, RenderError, VideoTarget

pytestmark = pytest.mark.ffmpeg


@pytest.fixture
def ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg or ffprobe unavailable")
    binary = shutil.which("ffmpeg")
    result = subprocess.run(
        [binary, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    return binary, result.stdout


def target(codec, depth, container="mp4"):
    return VideoTarget(
        codec,
        depth,
        "yuv420p10le" if depth == 10 else "yuv420p",
        ColorTags("bt2020", "arib-std-b67", "bt2020nc", "tv")
        if depth == 10
        else ColorTags(None, None, None, None),
        "match",
        None,
        128,
        128,
        Fraction(30),
        container,
    )


def encode(ffmpeg, tmp_path, video_target, *, force_pix_fmt=None):
    binary, listed = ffmpeg
    needed = "libx265" if video_target.codec == "hevc" else "libx264"
    if needed not in listed:
        pytest.skip(f"{needed} unavailable")
    encoder = encoders.select_encoder(
        video_target,
        Capabilities("test", frozenset({needed}), frozenset(), None),
        prefer="software",
    )
    args = encoder.video_args(video_target)
    if force_pix_fmt:
        args[args.index("-pix_fmt") + 1] = force_pix_fmt
    output = (
        tmp_path
        / f"output-{video_target.codec}-{force_pix_fmt or video_target.pix_fmt}.{video_target.container}"
    )
    source = "testsrc2=size=128x128:rate=30"
    if video_target.bit_depth == 10:
        source += ",setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=limited"
    subprocess.run(
        [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            "0.3",
            "-an",
            *args,
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return output


@pytest.mark.contract
@pytest.mark.parametrize("container", ["mp4", "mov"])
def test_real_libx265_10bit_guard(ffmpeg, tmp_path, container):
    video_target = target("hevc", 10, container)
    output = encode(ffmpeg, tmp_path, video_target)
    encoders.check_output(output, video_target)
    stream = encoders._video_stream(output)
    assert stream["profile"] == "Main 10"
    assert stream["pix_fmt"] == "yuv420p10le"
    assert stream["codec_tag_string"] == "hvc1"
    assert stream["color_primaries"] == "bt2020"
    assert stream["color_transfer"] == "arib-std-b67"
    assert stream["color_space"] == "bt2020nc"
    assert stream["color_range"] == "tv"


@pytest.mark.contract
def test_real_libx264_8bit_guard(ffmpeg, tmp_path):
    video_target = target("h264", 8)
    output = encode(ffmpeg, tmp_path, video_target)
    encoders.check_output(output, video_target)
    with pytest.raises(RenderError, match="codec"):
        encoders.check_output(output, target("hevc", 10))


def test_guard_fails_on_forced_yuv420p(ffmpeg, tmp_path):
    video_target = target("hevc", 10)
    output = encode(ffmpeg, tmp_path, video_target, force_pix_fmt="yuv420p")
    stream = encoders._video_stream(output)
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["profile"] == "Main"
    with pytest.raises(RenderError, match="10-bit pixel format"):
        encoders.check_output(output, video_target)
    encoders.check_output(output, replace(video_target, bit_depth=8, pix_fmt="yuv420p"))


def test_guard_fails_on_invalid_file(ffmpeg, tmp_path):
    output = tmp_path / "invalid.mp4"
    output.write_bytes(b"invalid synthetic media")
    with pytest.raises(RenderError, match="cannot probe"):
        encoders.check_output(output, target("h264", 8))


@pytest.mark.contract
def test_real_capability_probe(ffmpeg):
    capabilities = encoders.probe_capabilities(refresh=True)
    assert capabilities.ffmpeg_version.startswith("ffmpeg version")
    assert "libx264" in capabilities.encoders
    assert "libx265" in capabilities.encoders
    # Every successful hardware test must have a listed encoder; no GPU is required.
    for entry in capabilities.hw_encode_ok:
        encoder, depth = entry.rsplit(":", 1)
        assert encoder in capabilities.encoders
        assert depth in {"8", "10"}
    assert encoders.probe_capabilities() == capabilities
