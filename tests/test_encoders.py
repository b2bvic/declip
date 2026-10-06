"""Encoder policy, argument tables, and probes without requiring a GPU."""

from dataclasses import replace
from fractions import Fraction
import json
import subprocess

import pytest

from declip import encoders
from declip.contracts import (
    AudioStream,
    AudioTarget,
    Capabilities,
    ColorTags,
    EncoderUnavailable,
    MediaInfo,
    RenderError,
    ToolMissing,
)


@pytest.fixture
def media():
    return MediaInfo(
        2.0,
        0.0,
        "mov,mp4",
        True,
        0,
        2,
        Fraction(30000, 1001),
        False,
        1920,
        1080,
        0,
        "h264",
        "yuv420p",
        8,
        24_000_000,
        ColorTags("bt709", "bt709", "bt709", "tv"),
        None,
        (
            AudioStream(1, "aac", 48000, 1, "mono", None, None),
            AudioStream(2, "aac", 44100, 2, "stereo", None, "en"),
        ),
        (),
    )


@pytest.fixture
def target(media):
    return encoders.video_target_for(
        media, codec="match", quality="match", allow_8bit=False, container="mp4"
    )


def caps(*names, passed=()):
    return Capabilities("ffmpeg test", frozenset(names), frozenset(passed), None)


def option(args, flag):
    return args[args.index(flag) + 1]


@pytest.mark.contract
@pytest.mark.parametrize(
    "codec,pix_fmt,depth,expected",
    [
        ("h264", "yuv420p", 8, ("h264", 8)),
        ("hevc", "yuv420p", 8, ("hevc", 8)),
        ("hevc", "yuv420p10le", 10, ("hevc", 10)),
        ("prores", "yuv422p", 8, ("hevc", 10)),
        ("prores", "yuv422p10le", 10, ("hevc", 10)),
        ("hevc", "yuv420p12le", 12, ("hevc", 10)),
        ("h264", "p010le", None, ("hevc", 10)),
    ],
)
def test_match_source(media, codec, pix_fmt, depth, expected):
    result = encoders.video_target_for(
        replace(media, vcodec=codec, pix_fmt=pix_fmt, bit_depth=depth),
        codec="match",
        quality="high",
        allow_8bit=False,
        container="mov",
    )
    assert (result.codec, result.bit_depth) == expected
    assert result.color == media.color
    assert (result.width, result.height, result.fps, result.container) == (
        1920,
        1080,
        media.fps,
        "mov",
    )
    assert (media.vcodec, media.bit_depth) == ("h264", 8)


def test_explicit_h264_requires_allow_8bit(media):
    media = replace(media, vcodec="hevc", pix_fmt="yuv420p10le", bit_depth=10)
    with pytest.raises(EncoderUnavailable, match="--allow-8bit"):
        encoders.video_target_for(
            media, codec="h264", quality="match", allow_8bit=False, container="mp4"
        )
    target = encoders.video_target_for(
        media, codec="h264", quality="match", allow_8bit=True, container="mp4"
    )
    assert (target.codec, target.bit_depth, target.pix_fmt) == ("h264", 8, "yuv420p")
    # Permission alone never downgrades match-source output.
    assert (
        encoders.video_target_for(
            media, codec="match", quality="match", allow_8bit=True, container="mp4"
        ).bit_depth
        == 10
    )


@pytest.mark.parametrize(
    "kwargs", [{"codec": "prores"}, {"quality": "bad"}, {"container": "mkv"}]
)
def test_invalid_video_options(media, kwargs):
    options = dict(codec="match", quality="match", allow_8bit=False, container="mp4")
    options.update(kwargs)
    with pytest.raises(EncoderUnavailable):
        encoders.video_target_for(media, **options)


@pytest.mark.parametrize(
    "changes", [{"has_video": False}, {"fps": None}, {"width": None}]
)
def test_missing_video_information(media, changes):
    with pytest.raises(EncoderUnavailable):
        encoders.video_target_for(
            replace(media, **changes),
            codec="match",
            quality="match",
            allow_8bit=False,
            container="mp4",
        )


@pytest.mark.contract
@pytest.mark.parametrize(
    "container,pcm,codec,rate",
    [
        ("mp4", False, "aac", 256000),
        ("m4a", False, "aac", 256000),
        ("wav", True, "pcm_s24le", None),
        ("wav", False, "pcm_s24le", None),
    ],
)
def test_audio_target_selected_stream(media, container, pcm, codec, rate):
    target = encoders.audio_target_for(
        media, bitrate=256000, container=container, pcm=pcm
    )
    assert target == AudioTarget(codec, rate, 44100, 2, container)


def test_audio_missing_stream_and_bad_container(media):
    for changes in ({"audio_index": None}, {"audio_streams": ()}):
        with pytest.raises(EncoderUnavailable):
            encoders.audio_target_for(
                replace(media, **changes), bitrate=192000, container="mp4"
            )
    with pytest.raises(EncoderUnavailable):
        encoders.audio_target_for(media, bitrate=192000, container="mp4", pcm=True)


@pytest.mark.contract
def test_encoder_selection_requires_probe(target, monkeypatch):
    monkeypatch.setattr(encoders.sys, "platform", "darwin")
    listed = ("h264_videotoolbox", "h264_nvenc", "libx264", "aac_at")
    assert encoders.select_encoder(target, caps(*listed)).name == "software"
    proven = caps(*listed, passed=("h264_nvenc:8",))
    assert encoders.select_encoder(target, proven).name == "nvenc"
    proven = caps(*listed, passed=("h264_nvenc:8", "h264_videotoolbox:8"))
    assert encoders.select_encoder(target, proven).name == "videotoolbox"
    assert encoders.select_encoder(target, proven, prefer="software").name == "software"
    monkeypatch.setattr(encoders.sys, "platform", "linux")
    assert encoders.select_encoder(target, proven).name == "nvenc"


def test_10bit_does_not_select_8bit_encoder(target):
    target = replace(target, codec="hevc", bit_depth=10, pix_fmt="yuv420p10le")
    with pytest.raises(EncoderUnavailable, match="10-bit"):
        encoders.select_encoder(
            target, caps("h264_nvenc", "libx264", passed=("h264_nvenc:8",))
        )
    with pytest.raises(EncoderUnavailable):
        encoders.select_encoder(target, caps("hevc_nvenc"))
    assert encoders.select_encoder(target, caps("libx265")).name == "software"
    assert (
        encoders.select_encoder(
            target, caps("hevc_nvenc", passed=("hevc_nvenc:10",))
        ).name
        == "nvenc"
    )
    with pytest.raises(EncoderUnavailable):
        encoders.select_encoder(replace(target, pix_fmt="yuv420p"), caps("libx265"))


@pytest.mark.parametrize(
    "name,codec,depth,pix_fmt",
    [
        ("videotoolbox", "h264", 8, "yuv420p"),
        ("videotoolbox", "hevc", 10, "p010le"),
        ("nvenc", "h264", 8, "yuv420p"),
        ("nvenc", "hevc", 10, "p010le"),
        ("software", "h264", 8, "yuv420p"),
        ("software", "hevc", 10, "yuv420p10le"),
        ("software", "hevc", 8, "yuv420p"),
    ],
)
@pytest.mark.parametrize("quality", ["match", "high", "small"])
def test_video_argument_tables(target, name, codec, depth, pix_fmt, quality):
    target = replace(
        target, codec=codec, bit_depth=depth, pix_fmt=pix_fmt, quality=quality
    )
    args = encoders._Encoder(name).video_args(target)
    expected_encoder = {
        "videotoolbox": f"{codec}_videotoolbox",
        "nvenc": f"{codec}_nvenc",
        "software": "libx264" if codec == "h264" else "libx265",
    }[name]
    assert option(args, "-c:v") == expected_encoder
    assert option(args, "-pix_fmt") == pix_fmt
    assert option(args, "-color_primaries") == "bt709"
    assert option(args, "-color_trc") == "bt709"
    assert option(args, "-colorspace") == "bt709"
    assert option(args, "-color_range") == "tv"
    if codec == "hevc":
        assert option(args, "-tag:v") == "hvc1"
    else:
        assert "-tag:v" not in args
    if name == "videotoolbox":
        assert option(args, "-b:v") == str(
            {"match": 24_000_000, "high": 50_000_000, "small": 8_000_000}[quality]
        )
    elif name == "nvenc":
        assert option(args, "-preset") == "p5"
        assert option(args, "-rc") == "vbr"
        assert option(args, "-b:v") == "0"
        assert option(args, "-cq") == str(
            {"match": (19, 21), "high": (17, 19), "small": (23, 25)}[quality][
                codec == "hevc"
            ]
        )
    else:
        assert option(args, "-preset") == "medium"
        assert option(args, "-crf") == str(
            {"match": (18, 20), "high": (16, 18), "small": (22, 24)}[quality][
                codec == "hevc"
            ]
        )
    if depth == 10 and name != "software":
        assert option(args, "-profile:v") == "main10"
    assert "-q:v" not in args


@pytest.mark.parametrize(
    "dimensions,source,expected",
    [
        ((1920, 1080), 1, 8_000_000),
        ((1920, 1080), 100_000_000, 50_000_000),
        ((1080, 1920), None, 20_000_000),
        ((3840, 2160), None, 80_000_000),
        ((3840, 2160), 1, 35_000_000),
        ((3840, 2160), 200_000_000, 150_000_000),
    ],
)
def test_videotoolbox_rate_clamping(target, dimensions, source, expected):
    target = replace(
        target,
        width=dimensions[0],
        height=dimensions[1],
        source_bitrate=source,
        color=ColorTags(None, None, None, None),
    )
    args = encoders._Encoder("videotoolbox").video_args(target)
    assert option(args, "-b:v") == str(expected)
    assert "-color_trc" not in args


@pytest.mark.parametrize("listed,expected", [(True, "aac_at"), (False, "aac")])
def test_videotoolbox_audio_fallback(target, monkeypatch, listed, expected):
    monkeypatch.setattr(encoders.sys, "platform", "darwin")
    names = ["h264_videotoolbox"] + (["aac_at"] if listed else [])
    encoder = encoders.select_encoder(
        target, caps(*names, passed=("h264_videotoolbox:8",))
    )
    audio = AudioTarget("aac", 256000, 44100, 6, "mov")
    assert encoder.audio_args(audio) == [
        "-c:a",
        expected,
        "-b:a",
        "256000",
        "-ar",
        "44100",
        "-ac",
        "6",
    ]
    assert encoder.audio_args(
        replace(audio, codec="pcm_s24le", bitrate=None, container="wav")
    ) == ["-c:a", "pcm_s24le", "-ar", "44100", "-ac", "6"]


def test_probe_capabilities_encodes_and_invalidates_cache(monkeypatch, tmp_path):
    calls = []
    state = {"version": "ffmpeg version test", "driver": "1"}
    monkeypatch.setattr(encoders, "_tool", lambda name: name)
    monkeypatch.setattr(encoders, "_driver_version", lambda: state["driver"])

    def run(argv, **kwargs):
        calls.append(argv)
        if "-version" in argv:
            output = state["version"] + "\n"
        elif "-encoders" in argv:
            output = " V....D h264_videotoolbox test\n V....D hevc_videotoolbox test\n V....D h264_nvenc test\n V....D hevc_nvenc test\n V....D libx265 test\n A..... aac test\n"
        else:
            assert option(argv, "-frames:v") == "1"
            assert "-f" in argv and "lavfi" in argv
            return subprocess.CompletedProcess(
                argv, 1 if "h264_nvenc" in argv else 0, "", ""
            )
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(encoders, "_run", run)

    def guard(path, target):
        if "hevc_nvenc" in path.name:
            raise RenderError("silently wrote 8-bit")

    monkeypatch.setattr(encoders, "check_output", guard)
    first = encoders.probe_capabilities()
    assert first.hw_encode_ok == frozenset(
        {"h264_videotoolbox:8", "hevc_videotoolbox:10"}
    )
    assert first.gpu_driver == "1"

    def encodes():
        return sum("-frames:v" in call for call in calls)

    assert encodes() == 4
    assert encoders.probe_capabilities() == first
    assert encodes() == 4
    encoders.probe_capabilities(refresh=True)
    assert encodes() == 8
    state["driver"] = "2"
    encoders.probe_capabilities()
    assert encodes() == 12
    state["version"] = "ffmpeg version updated"
    encoders.probe_capabilities()
    assert encodes() == 16
    (tmp_path / "cache" / "capabilities.json").write_text("bad JSON", encoding="utf-8")
    encoders.probe_capabilities()
    assert encodes() == 20


@pytest.mark.contract
@pytest.mark.parametrize(
    "returncode,stdout,expected",
    [
        (1, "", []),
        (0, "frame=0\n", []),
        (0, "frame=30\n", ["-hwaccel", "videotoolbox"]),
    ],
)
def test_encode_pass_decode_independent(
    media, monkeypatch, tmp_path, returncode, stdout, expected
):
    monkeypatch.setattr(encoders.sys, "platform", "darwin")
    monkeypatch.setattr(encoders, "_tool", lambda name: name)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"head" * 16384 + b"middle-a" + b"tail" * 16384)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert option(argv, "-map") == "0:0"
        assert option(argv, "-t") == "1"
        assert "-hwaccel_output_format" not in argv
        assert "-vf" in argv
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    monkeypatch.setattr(encoders, "_run", run)
    capability = caps("hevc_videotoolbox", passed=("hevc_videotoolbox:10",))
    assert encoders.hwaccel_args(source, media, capability) == expected
    assert encoders.hwaccel_args(source, media, capability) == expected
    assert len(calls) == 1
    source.write_bytes(b"head" * 16384 + b"middle-b" + b"tail" * 16384)
    assert encoders.hwaccel_args(source, media, capability) == expected
    assert len(calls) == 2
    encoders.hwaccel_args(source, media, replace(capability, ffmpeg_version="new"))
    assert len(calls) == 3


def test_decode_software_default(media, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("no hardware capability should run no decode probe")

    monkeypatch.setattr(encoders, "_run", forbidden)
    assert encoders.hwaccel_args(tmp_path / "absent.mp4", media, caps("libx264")) == []
    assert (
        encoders.hwaccel_args(
            tmp_path / "absent.mp4",
            replace(media, has_video=False),
            caps("hevc_nvenc", passed=("hevc_nvenc:10",)),
        )
        == []
    )


def test_tools_missing(monkeypatch):
    monkeypatch.setattr(encoders.shutil, "which", lambda name: None)
    with pytest.raises(ToolMissing, match="ffmpeg is missing"):
        encoders.probe_capabilities()


@pytest.mark.parametrize(
    "stream,match",
    [
        ({"codec_name": "h264", "pix_fmt": "yuv420p", "profile": "High"}, "codec"),
        (
            {"codec_name": "hevc", "pix_fmt": "yuv420p", "profile": "Main"},
            "pixel format",
        ),
        (
            {"codec_name": "hevc", "pix_fmt": "yuv420p10le", "profile": "Main"},
            "Main 10",
        ),
        (
            {
                "codec_name": "hevc",
                "pix_fmt": "unknown",
                "bits_per_raw_sample": "10",
                "profile": "Main 10",
            },
            "pixel format",
        ),
    ],
)
def test_guard_refuses_probe_mismatches(target, monkeypatch, tmp_path, stream, match):
    monkeypatch.setattr(encoders, "_video_stream", lambda path: stream)
    with pytest.raises(RenderError, match=match):
        encoders.check_output(
            tmp_path / "out.mp4",
            replace(target, codec="hevc", bit_depth=10, pix_fmt="yuv420p10le"),
        )


def test_guard_probe_missing_video(target, monkeypatch, tmp_path):
    monkeypatch.setattr(encoders, "_tool", lambda name: name)
    monkeypatch.setattr(
        encoders,
        "_run",
        lambda argv: subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "disposition": {"attached_pic": 1}}
                    ]
                }
            ),
            "",
        ),
    )
    with pytest.raises(RenderError, match="no readable video"):
        encoders.check_output(tmp_path / "out.mp4", target)


def test_nvenc_decode_timeout_is_cached(media, monkeypatch, tmp_path):
    monkeypatch.setattr(encoders, "_tool", lambda name: name)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert option(argv, "-hwaccel") == "cuda"
        assert "-hwaccel_output_format" not in argv
        return None

    monkeypatch.setattr(encoders, "_run", run)
    capability = caps("hevc_nvenc", passed=("hevc_nvenc:10",))
    assert encoders.hwaccel_args(source, media, capability) == []
    assert encoders.hwaccel_args(source, media, capability) == []
    assert len(calls) == 1
