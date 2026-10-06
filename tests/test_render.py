"""Render contracts and content-based verification on small generated media."""

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from declip import editlist as ed, encoders, media, render
from declip.contracts import (
    AudioStream,
    Capabilities,
    ColorTags,
    Cut,
    CutKind,
    CutOrigin,
    CutStatus,
    FilterChainRejected,
    Loudness,
    MediaInfo,
    OutputMode,
    OutputSpec,
    Processing,
    RenderError,
    ResolvedOptions,
    ReviewRequired,
    RigRef,
    SourceMismatch,
    TranscribeOptions,
)


@pytest.fixture
def options():
    return ResolvedOptions(
        None,
        None,
        TranscribeOptions("small"),
        "auto",
        "none",
        0.5,
        "",
        None,
        0,
        "match",
        "match",
        "software",
        False,
        192000,
        OutputMode.RENDER,
        None,
        False,
        120,
        0.5,
        -55,
        300,
        450,
        True,
        {},
    )


@pytest.fixture
def caps():
    return Capabilities(
        "test", frozenset({"libx264", "libx265", "aac", "pcm_s24le"}), frozenset(), None
    )


@pytest.fixture
def synthetic(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    info = MediaInfo(
        10,
        0,
        "mov",
        True,
        0,
        1,
        Fraction(30),
        False,
        160,
        90,
        0,
        "h264",
        "yuv420p",
        8,
        1000000,
        ColorTags(None, None, None, None),
        None,
        (AudioStream(1, "aac", 48000, 2, "stereo", None, "eng"),),
        (),
    )
    return source, info


def reviewed(source, info, *, ranges=(), processing=None):
    edit = ed.new_edit_list(
        source,
        info,
        media.file_hash(source),
        transcript=None,
        processing=processing or Processing("none", 0.5, "", None, 0),
        output=OutputSpec(OutputMode.RENDER, None, False),
        rig=RigRef(None, None, {}),
    )
    cuts = tuple(
        Cut(
            ed.cut_id(CutKind.GAP, a, b, "gap"),
            CutKind.GAP,
            a,
            b,
            "gap",
            1,
            False,
            None,
            CutOrigin.AUTO,
            CutStatus.ACCEPTED,
        )
        for a, b in ranges
    )
    return ed.mark_review_passed(
        replace(edit, cuts=cuts), now=datetime.now(timezone.utc)
    )


def plan_for(
    source, info, options, caps, tmp_path, *, ranges=(), processing=None, output=None
):
    return render.build_render_plan(
        reviewed(source, info, ranges=ranges, processing=processing),
        source,
        output or render.default_output_path(source, info),
        options,
        caps,
        temp_dir=tmp_path / "render-temp",
    )


@pytest.fixture
def ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    return shutil.which("ffmpeg")


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def generate(
    ffmpeg, path, *, duration=3, stereo=True, tenbit=False, multitrack=False, video=True
):
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if video:
        args += ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30000/1001"]
    args += [
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:sample_rate={44100 if stereo else 48000}",
    ]
    if multitrack:
        args += ["-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000"]
    if video:
        args += [
            "-map",
            "0:v",
            "-c:v",
            "libx265" if tenbit else "libx264",
            "-pix_fmt",
            "yuv420p10le" if tenbit else "yuv420p",
        ]
        if tenbit:
            args += [
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "arib-std-b67",
                "-colorspace",
                "bt2020nc",
            ]
    args += [
        "-map",
        f"{int(video)}:a",
        "-c:a",
        "pcm_s24le" if path.suffix == ".wav" else "aac",
    ]
    if stereo:
        args += ["-ac:a:0", "2"]
    if multitrack:
        args += [
            "-map",
            "2:a",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=fra",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
        ]
    args += ["-metadata", "title=Synthetic fixture", "-t", str(duration), str(path)]
    run(args)
    return path, media.probe_media(path)


@pytest.mark.contract
@pytest.mark.parametrize("count", [1, 2, 41, 250])
def test_plan_runs_no_process(synthetic, options, caps, tmp_path, monkeypatch, count):
    source, info = synthetic
    info = replace(info, duration=count * 2)
    ranges = [(2 * i + 1, 2 * i + 2) for i in range(count)]
    edit = reviewed(source, info, ranges=ranges)
    before = set(tmp_path.rglob("*"))

    def forbidden(*args, **kwargs):
        pytest.fail("plan started a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    plan = render.build_render_plan(
        edit, source, tmp_path / "output.mp4", options, caps, temp_dir=tmp_path / "temp"
    )
    assert set(tmp_path.rglob("*")) == before
    batches = [s for s in plan.steps if s.name.startswith("video-batch")]
    assert len(batches) == math.ceil(count / 40)
    assert all(s.argv[s.argv.index("-ss") + 1] for s in batches)
    assert all(s.argv.index("-ss") < s.argv.index("-i") for s in batches)
    assert all(s.argv[s.argv.index("-vf") + 1].count("between(") <= 40 for s in batches)
    assert plan.timeline == ed.effective_timeline(edit)
    assert len([s for s in plan.steps if s.name == "audio-final"]) == 1
    graph = plan.steps[0].argv[plan.steps[0].argv.index("-filter_complex") + 1]
    assert f"concat=n={count}:v=0:a=1" in graph
    assert "[acut]" in graph


def test_render_module_has_no_encoder_literals():
    path = Path(render.__file__)
    assert path.is_file()
    text = path.read_text()
    for forbidden in ("aac_at", "videotoolbox", "nvenc", "libx26"):
        assert forbidden not in text


@pytest.mark.contract
@pytest.mark.parametrize(
    "suffix,video,expected",
    [
        (".mov", True, ".mov"),
        (".MP4", True, ".mp4"),
        (".mkv", True, ".mp4"),
        (".wav", False, ".wav"),
        (".aif", False, ".wav"),
        (".aiff", False, ".wav"),
        (".flac", False, ".wav"),
        (".mp3", False, ".m4a"),
    ],
)
def test_containers(synthetic, suffix, video, expected):
    source, info = synthetic
    assert (
        render.default_output_path(
            source.with_suffix(suffix), replace(info, has_video=video)
        ).name
        == "source_clean" + expected
    )


def test_bad_suffix(synthetic, options, caps, tmp_path):
    source, info = synthetic
    with pytest.raises(RenderError) as error:
        plan_for(source, info, options, caps, tmp_path, output=tmp_path / "out.wav")
    assert error.value.exit_code == 2


def test_overwrite_refusals(synthetic, options, caps, tmp_path):
    source, info = synthetic
    output = tmp_path / "output.mp4"
    output.write_bytes(b"existing")
    edit = reviewed(source, info)
    for path in (source, output):
        with pytest.raises(RenderError):
            render.build_render_plan(
                edit, source, path, options, caps, temp_dir=tmp_path / "temp"
            )
    for path in (source, tmp_path / "link.mp4", tmp_path / "hard.mp4"):
        if path.name == "link.mp4":
            path.symlink_to(source)
        elif path.name == "hard.mp4":
            os.link(source, path)
        with pytest.raises(RenderError, match="source"):
            render.build_render_plan(
                edit,
                source,
                path,
                options,
                caps,
                temp_dir=tmp_path / "temp",
                overwrite=True,
            )
    assert (
        render.build_render_plan(
            edit,
            source,
            output,
            options,
            caps,
            temp_dir=tmp_path / "temp",
            overwrite=True,
        ).output
        == output
    )
    assert source.read_bytes() == b"synthetic source"


def test_render_refuses_pending_review(synthetic, options, caps, tmp_path):
    source, info = synthetic
    edit = reviewed(source, info)
    pending = ed.reset_decisions(edit)
    with pytest.raises(ReviewRequired):
        render.build_render_plan(
            pending,
            source,
            tmp_path / "out.mp4",
            options,
            caps,
            temp_dir=tmp_path / "temp",
        )
    with pytest.raises(ReviewRequired):
        render.render_processed_audio(
            pending, source, tmp_path / "out.wav", options, temp_dir=tmp_path / "temp"
        )


def test_render_refuses_source_mutated_after_review(synthetic, options, caps, tmp_path):
    source, info = synthetic
    edit = reviewed(source, info)
    source.write_bytes(b"mutated source")
    with pytest.raises(SourceMismatch):
        render.build_render_plan(
            edit,
            source,
            tmp_path / "out.mp4",
            options,
            caps,
            temp_dir=tmp_path / "temp",
        )


@pytest.mark.parametrize(
    "chain",
    [
        "movie=x",
        "amovie=x",
        "sendcmd=x",
        "asendcmd=x",
        "ladspa=x",
        "lv2=x",
        "ametadata=mode=print:file=x",
        "metadata=mode=print:file=x",
    ],
)
def test_render_refuses_denied_filter(synthetic, options, caps, tmp_path, chain):
    source, info = synthetic
    with pytest.raises(FilterChainRejected):
        plan_for(
            source,
            info,
            options,
            caps,
            tmp_path,
            processing=Processing("none", 0.5, chain, None, 0),
        )


def test_all_media_removed(synthetic, options, caps, tmp_path):
    source, info = synthetic
    with pytest.raises(RenderError, match="all media removed"):
        plan_for(source, info, options, caps, tmp_path, ranges=[(0, info.duration)])


def test_concat_escape(tmp_path):
    path = tmp_path / "a 'quoted' directory" / "clip.mp4"
    text = render._concat_text((path,))
    assert "'\\''quoted'\\''" in text
    assert text == "file '" + path.resolve().as_posix().replace("'", "'\\''") + "'\n"


def test_vfr_conform_plan(synthetic, options, caps, tmp_path):
    source, info = synthetic
    plan = plan_for(source, replace(info, vfr=True), options, caps, tmp_path)
    step = next(s for s in plan.steps if s.name == "video-conform")
    assert "fps=30" in step.argv
    assert (
        step.outputs[0]
        in next(s for s in plan.steps if s.name.startswith("video-batch")).inputs
    )


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_channels_kept(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "stereo.wav", video=False)
    processing = Processing("afftdn", 0.3, "highpass=f=80", None, 20)
    plan = plan_for(
        source,
        info,
        options,
        caps,
        tmp_path,
        ranges=[(0.7, 1.1)],
        processing=processing,
    )
    events = []
    result = render.run_render_plan(
        plan, progress=lambda step, fraction: events.append((step, fraction))
    )
    out = media.probe_media(result.output)
    assert out.audio_streams[0].channels == 2
    assert out.audio_streams[0].sample_rate == 44100
    assert out.audio_streams[0].bit_depth == 24
    assert abs(result.duration - plan.timeline.duration_out) <= 0.001
    assert not plan.temp_dir.exists()
    assert any(name == "audio-final" and value == 1 for name, value in events)
    assert all(0 <= value <= 1 for _, value in events)


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_multitrack_mapping(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "tracks.mp4", multitrack=True)
    assert info.audio_index == 2
    plan = plan_for(
        source,
        info,
        options,
        caps,
        tmp_path,
        ranges=[(0.6, 1.1)],
        processing=Processing("none", 0.5, "volume=0.5", None, 0),
    )
    extraction = [s for s in plan.steps if re.fullmatch(r"audio-\d+", s.name)]
    assert len(extraction) == 2
    assert (
        "end_sample=26486"
        in extraction[0].argv[extraction[0].argv.index("-filter_complex") + 1]
    )
    assert (
        "end_sample=28829"
        in extraction[1].argv[extraction[1].argv.index("-filter_complex") + 1]
    )
    result = render.run_render_plan(plan)
    raw = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(result.output),
            ]
        ).stdout
    )
    tracks = [s for s in raw["streams"] if s["codec_type"] == "audio"]
    assert len(tracks) == 2
    assert [s["tags"]["language"] for s in tracks] == ["eng", "fra"]
    assert [s["disposition"]["default"] for s in tracks] == [0, 1]
    assert [int(s["sample_rate"]) for s in tracks] == [44100, 48000]
    assert raw["format"]["tags"]["title"] == "Synthetic fixture"


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_loudness_within_1_lu(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "voice.wav", video=False, duration=10)
    processing = Processing(
        "none", 0.5, "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11", None, 0
    )
    plan = plan_for(source, info, options, caps, tmp_path, processing=processing)
    assert plan.loudness == Loudness(-16, -1.5, 11)
    assert len([s for s in plan.steps if s.name == "loudness-measure"]) == 1
    assert all(
        s.argv[s.argv.index("-af") + 1].count("loudnorm=") == 1
        for s in plan.steps
        if s.name in {"loudness-measure", "audio-final"}
    )
    result = render.run_render_plan(plan)
    report = run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(result.output),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]
    ).stderr
    integrated = float(re.findall(r"I:\s*(-?[\d.]+) LUFS", report)[-1])
    assert abs(integrated - (-16)) <= 1, report[-1000:]
    assert result.integrated_lufs is not None


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_processed_audio_sidecar(ffmpeg, tmp_path, options):
    source, info = generate(ffmpeg, tmp_path / "clip.mp4")
    edit = reviewed(source, info, ranges=[(0.6, 1.1)])
    output = tmp_path / "sidecar.wav"
    result = render.render_processed_audio(
        edit, source, output, options, temp_dir=tmp_path / "side-temp"
    )
    assert result == output
    out = media.probe_media(result)
    assert out.audio_streams[0].codec == "pcm_s24le"
    assert out.audio_streams[0].channels == 2
    assert abs(out.duration - ed.effective_timeline(edit).duration_out) <= 0.001


@pytest.mark.ffmpeg
def test_enhance_has_one_whole_source_keep(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "clip.mp4")
    plan = render.build_enhance_plan(
        source,
        tmp_path / "enhanced.mp4",
        info,
        options,
        caps,
        temp_dir=tmp_path / "temp",
    )
    assert len(plan.timeline.keeps) == 1
    assert plan.timeline.keeps[0].start == 0
    assert plan.timeline.keeps[0].end == info.duration
    assert not plan.timeline.removed
    result = render.run_render_plan(plan)
    assert abs(result.duration - info.duration) <= float(1 / info.fps)


@pytest.mark.ffmpeg
@pytest.mark.extra("libx265")
def test_10bit_libx265_end_to_end(ffmpeg, tmp_path, options, caps):
    if "libx265" not in run([ffmpeg, "-hide_banner", "-encoders"]).stdout:
        pytest.skip("libx265 unavailable")
    source, info = generate(ffmpeg, tmp_path / "tenbit.mp4", tenbit=True)
    plan = plan_for(source, info, options, caps, tmp_path, ranges=[(0.7, 1.1)])
    result = render.run_render_plan(plan)
    assert result.bit_depth == 10
    assert result.profile == "Main 10"
    assert result.video_codec == "hevc"
    assert media.probe_media(result.output).color == info.color


@pytest.mark.ffmpeg
@pytest.mark.extra("libx265")
def test_guard_fails_on_forced_yuv420p(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "tenbit.mp4", tenbit=True)
    plan = plan_for(source, info, options, caps, tmp_path)
    steps = []
    for step in plan.steps:
        if step.name.startswith("video-batch"):
            argv = list(step.argv)
            argv[argv.index("-pix_fmt") + 1] = "yuv420p"
            step = replace(step, argv=tuple(argv))
        steps.append(step)
    with pytest.raises(RenderError, match="10-bit pixel format"):
        render.run_render_plan(replace(plan, steps=tuple(steps)))
    assert not plan.output.exists()
    assert not plan.temp_dir.exists()
    assert not list(tmp_path.glob("*.partial-*"))


@pytest.mark.ffmpeg
@pytest.mark.hwenc("videotoolbox")
def test_videotoolbox_main10(ffmpeg, tmp_path, options):
    capabilities = encoders.probe_capabilities(refresh=True)
    if "hevc_videotoolbox:10" not in capabilities.hw_encode_ok:
        pytest.skip("videotoolbox Main 10 unavailable")
    source, info = generate(ffmpeg, tmp_path / "tenbit.mp4", tenbit=True)
    plan = plan_for(
        source, info, replace(options, encoder="auto"), capabilities, tmp_path
    )
    assert plan.encoder == "videotoolbox"
    result = render.run_render_plan(plan)
    assert result.bit_depth == 10
    assert result.profile == "Main 10"


@pytest.mark.ffmpeg
def test_failure_preserves_output_and_cleans_temp(
    ffmpeg, tmp_path, synthetic, options, caps
):
    source, info = synthetic
    output = tmp_path / "out.mp4"
    output.write_bytes(b"keep existing")
    plan = render.build_render_plan(
        reviewed(source, info),
        source,
        output,
        options,
        caps,
        temp_dir=tmp_path / "temp",
        overwrite=True,
    )
    with pytest.raises(RenderError, match="audio-1"):
        render.run_render_plan(plan)
    assert output.read_bytes() == b"keep existing"
    assert not plan.temp_dir.exists()
    assert not list(tmp_path.glob("*.partial-*"))


@pytest.mark.ffmpeg
def test_interrupt_stops_child_and_cleans_temp(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "clip.mp4")
    plan = plan_for(source, info, options, caps, tmp_path)

    def interrupt(name, fraction):
        raise KeyboardInterrupt()

    with pytest.raises(RenderError, match="audio-1.*interrupted"):
        render.run_render_plan(plan, progress=interrupt)
    assert not plan.output.exists()
    assert not plan.temp_dir.exists()


@pytest.mark.ffmpeg
@pytest.mark.slow
def test_av_sync_flash_beep_250_keeps(ffmpeg, tmp_path, options, caps):
    # White frame closest to each integer second; beep uses exact source samples.
    fps = Fraction(30000, 1001)
    source = tmp_path / "flash-beep.mp4"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=160x90:rate=30000/1001,geq=lum='if(lt(abs(T-round(T)),0.016684),235,16)':cb=128:cr=128",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=if(lt(mod(t\\,1)\\,0.02)\\,0.5*sin(2*PI*1000*t)\\,0):s=48000",
            "-t",
            "330",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    info = media.probe_media(source)
    keeps = [(n - 0.213, n + 0.287) for n in range(1, 251)]
    ranges = [
        (0, keeps[0][0]),
        *[(a[1], b[0]) for a, b in zip(keeps, keeps[1:])],
        (keeps[-1][1], info.duration),
    ]
    plan = plan_for(source, info, options, caps, tmp_path, ranges=ranges)
    assert len(plan.timeline.keeps) == 250
    result = render.run_render_plan(plan)
    # signalstats gives per-frame luminance and timestamps; no decoded media on disk.
    report = run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(result.output),
            "-map",
            "0:v:0",
            "-vf",
            "signalstats,metadata=mode=print",
            "-an",
            "-f",
            "null",
            "-",
        ]
    ).stderr
    flashes, current = [], None
    for line in report.splitlines():
        match = re.search(r"pts_time:([\d.]+)", line)
        if match:
            current = float(match[1])
        match = re.search(r"lavfi.signalstats.YAVG=([\d.]+)", line)
        if match and float(match[1]) > 200:
            flashes.append(current)
    report = run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(result.output),
            "-map",
            "0:a:0",
            "-af",
            "silencedetect=noise=-25dB:d=0.05",
            "-vn",
            "-f",
            "null",
            "-",
        ]
    ).stderr
    beeps = [
        float(v)
        for v in re.findall(r"silence_end:\s*([\d.]+)", report)
        if float(v) < result.duration - float(1 / fps)
    ]
    assert len(flashes) == len(beeps) == 250, (len(flashes), len(beeps))
    offsets = [abs(a - b) * float(fps) for a, b in zip(flashes, beeps)]
    assert max(offsets) <= 1, (max(offsets), offsets[39:42], offsets[79:82])
    assert max(offsets[39:42] + offsets[79:82]) <= 1
    assert abs(result.duration - plan.timeline.duration_out) <= float(1 / fps)
    # pytest retains tmp_path. Remove this test's larger generated source and output.
    source.unlink()
    result.output.unlink()


@pytest.mark.ffmpeg
def test_vfr_conforms_end_to_end(ffmpeg, tmp_path, options, caps):
    source = tmp_path / "vfr.mp4"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000",
            "-t",
            "3",
            "-vf",
            "select='not(eq(mod(n,10),0))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    info = media.probe_media(source)
    assert info.vfr
    plan = plan_for(source, info, options, caps, tmp_path, ranges=[(0.51, 1.13)])
    result = render.run_render_plan(plan)
    out = media.probe_media(result.output)
    assert not out.vfr
    assert out.fps == info.fps
    assert abs(out.duration - plan.timeline.duration_out) <= float(1 / info.fps)


@pytest.mark.ffmpeg
def test_audio_only_aac(ffmpeg, tmp_path, options):
    source, info = generate(ffmpeg, tmp_path / "audio.m4a", video=False)
    plan = plan_for(
        source,
        info,
        options,
        Capabilities("", frozenset({"aac"}), frozenset(), None),
        tmp_path,
    )
    assert plan.video is None
    result = render.run_render_plan(plan)
    assert result.audio_channels == 2
    assert result.output.suffix == ".m4a"
    assert abs(result.duration - plan.timeline.duration_out) <= 0.001


@pytest.mark.ffmpeg
def test_video_without_audio(ffmpeg, tmp_path, options, caps):
    source = tmp_path / "mute.mp4"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30",
            "-t",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ]
    )
    info = media.probe_media(source)
    plan = plan_for(source, info, options, caps, tmp_path, ranges=[(0.5, 1)])
    assert not any(step.name.startswith("audio-") for step in plan.steps)
    result = render.run_render_plan(plan)
    assert result.audio_channels == 0
    assert not media.probe_media(result.output).audio_streams


@pytest.mark.ffmpeg
def test_silent_loudness_has_no_nonfinite_filter_values(
    ffmpeg, tmp_path, options, caps
):
    source = tmp_path / "silence.wav"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "3",
            "-c:a",
            "pcm_s24le",
            str(source),
        ]
    )
    info = media.probe_media(source)
    plan = plan_for(
        source,
        info,
        options,
        caps,
        tmp_path,
        processing=Processing("none", 0.5, "", Loudness(-16, -1.5, 11), 0),
    )
    result = render.run_render_plan(plan)
    assert result.integrated_lufs <= -70
    assert result.audio_channels == 2
    decoded = run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(result.output),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "-",
        ]
    ).stdout
    assert decoded and not decoded.strip("\0")


@pytest.mark.ffmpeg
def test_source_changed_after_planning(ffmpeg, synthetic, options, caps, tmp_path):
    source, info = synthetic
    plan = plan_for(source, info, options, caps, tmp_path)
    source.write_bytes(b"changed after planning")
    with pytest.raises(RenderError, match="source changed"):
        render.run_render_plan(plan)
    assert not plan.temp_dir.exists()
    assert not plan.output.exists()


@pytest.mark.ffmpeg
def test_existing_temp_directory_is_preserved(ffmpeg, tmp_path, options, caps):
    source, info = generate(ffmpeg, tmp_path / "clip.mp4")
    plan = plan_for(source, info, options, caps, tmp_path)
    plan.temp_dir.mkdir()
    marker = plan.temp_dir / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(RenderError, match="prepare"):
        render.run_render_plan(plan)
    assert marker.read_text() == "keep"


@pytest.mark.ffmpeg
def test_concat_apostrophe_runs(ffmpeg, tmp_path, options, caps):
    directory = tmp_path / "a 'quoted' directory"
    directory.mkdir()
    source, info = generate(ffmpeg, directory / "clip.mp4")
    plan = plan_for(source, info, options, caps, directory)
    result = render.run_render_plan(plan)
    assert result.output.exists()


def test_short_keep_fades_are_inside_keep(synthetic, options, caps, tmp_path):
    source, info = synthetic
    plan = plan_for(
        source,
        info,
        options,
        caps,
        tmp_path,
        ranges=[(1 / 30, 10)],
        processing=Processing("none", 0.5, "", None, 1000),
    )
    graph = plan.steps[0].argv[plan.steps[0].argv.index("-filter_complex") + 1]
    assert "end_sample=1600" in graph
    assert "afade=t=in:ss=0:ns=800" in graph
    assert "afade=t=out:ss=800:ns=800" in graph


def test_processing_uses_edit_list_snapshot(synthetic, options, caps, tmp_path):
    source, info = synthetic
    plan = plan_for(
        source,
        info,
        replace(options, eq_chain="volume=0.1", loudness=Loudness(-23, -1.5, 11)),
        caps,
        tmp_path,
        processing=Processing("none", 0.5, "volume=0.7", None, 0),
    )
    step = next(s for s in plan.steps if s.name == "audio-final")
    assert "volume=0.7" in step.argv[step.argv.index("-af") + 1]
    assert plan.loudness is None
