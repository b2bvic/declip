"""Hand-authored NLE expectations and real processed-sidecar verification."""

import array
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from declip import editlist as ed, export, media, render
from declip.contracts import (
    AudioStream,
    ColorTags,
    Cut,
    CutKind,
    CutOrigin,
    CutStatus,
    ExportFormat,
    ExportRefused,
    MediaInfo,
    NleFormat,
    OutputMode,
    OutputSpec,
    Processing,
    ReviewRequired,
    RigRef,
    Segment,
    SourceMismatch,
    Transcript,
    Word,
)
from declip.rig import resolve_options

GOLDEN = Path(__file__).parent / "golden"
FPS = Fraction(30000, 1001)


@pytest.fixture
def options():
    return resolve_options({"loudness": None, "crossfade_ms": 0}, None, {})


def info_for(fps=FPS, frames=600):
    return MediaInfo(
        float(frames / fps),
        0,
        "mov",
        True,
        0,
        1,
        fps,
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


def cut(start, end, label="gap", status=CutStatus.ACCEPTED):
    return Cut(
        ed.cut_id(CutKind.GAP, start, end, label),
        CutKind.GAP,
        start,
        end,
        label,
        1,
        False,
        None,
        CutOrigin.AUTO,
        status,
    )


def reviewed(source, info, cuts=(), transcript=None):
    edit = ed.new_edit_list(
        source,
        info,
        media.file_hash(source),
        transcript=transcript,
        processing=Processing("none", 0.5, "", None, 0),
        output=OutputSpec(OutputMode.NLE, NleFormat.EDL, False),
        rig=RigRef(None, None, {}),
    )
    return ed.mark_review_passed(
        replace(edit, cuts=tuple(cuts)), now=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"synthetic source")
    edit = reviewed(
        source,
        info_for(),
        [
            cut(float(60 / FPS), float(120 / FPS)),
            cut(float(300 / FPS), float(360 / FPS)),
        ],
    )
    return source, edit


@pytest.mark.contract
@pytest.mark.parametrize(
    "fps,frames,expected",
    [
        (FPS, 0, "00:00:00;00"),
        (FPS, 1799, "00:00:59;29"),
        (FPS, 1800, "00:01:00;02"),
        (FPS, 17982, "00:10:00;00"),
        (FPS, 107892, "01:00:00;00"),
        (Fraction(60000, 1001), 3599, "00:00:59;59"),
        (Fraction(60000, 1001), 3600, "00:01:00;04"),
        (Fraction(60000, 1001), 35964, "00:10:00;00"),
        (FPS, 2589408, "00:00:00;00"),
    ],
)
def test_drop_frame_anchors(fps, frames, expected):
    assert export.frames_to_timecode(frames, fps, drop_frame=True) == expected


@pytest.mark.parametrize(
    "fps",
    [
        Fraction(24),
        Fraction(24000, 1001),
        Fraction(25),
        FPS,
        Fraction(30),
        Fraction(48),
        Fraction(50),
        Fraction(60000, 1001),
        Fraction(60),
        Fraction(120),
    ],
)
def test_nominal_timecode(fps):
    assert (
        export.frames_to_timecode(round(fps) * 3600, fps, drop_frame=False)
        == "01:00:00:00"
    )


@pytest.mark.parametrize(
    "frames,fps,drop",
    [(-1, FPS, True), (1, Fraction(27), False), (1, Fraction(25), True)],
)
def test_invalid_timecode(frames, fps, drop):
    with pytest.raises(ExportRefused):
        export.frames_to_timecode(frames, fps, drop_frame=drop)


@pytest.mark.contract
def test_paths(sample, tmp_path):
    source, _ = sample
    paths = export.export_paths(source)
    assert {k: p.name for k, p in paths.items()} == {
        "edl": "sample.declip.edl",
        "fcpxml": "sample.declip.fcpxml",
        "srt": "sample.declip.srt",
        "markers": "sample.declip.markers.json",
        "wav": "sample.declip.wav",
    }
    assert all(
        p.parent == tmp_path / "exports"
        for p in export.export_paths(source, out_dir=tmp_path / "exports").values()
    )


@pytest.mark.contract
def test_edl_golden(sample, options, assert_golden):
    source, edit = sample
    plan = export.build_export_plan(edit, source, ExportFormat.EDL, options)
    text = plan.files[0].text.replace(str(source.parent), "/synthetic")
    assert_golden(text, GOLDEN / "edl_2997df.edl")


def test_fcpxml_golden(sample, options, assert_golden):
    source, edit = sample
    plan = export.build_export_plan(edit, source, ExportFormat.FCPXML, options)
    assert_golden(
        plan.files[0].text.replace(source.parent.as_uri(), "file:///synthetic"),
        GOLDEN / "fcpxml_2997.fcpxml",
    )
    assert_frame_times(ET.fromstring(plan.files[0].text), FPS)


def assert_frame_times(root, fps):
    for node in root.iter():
        for key in ("frameDuration", "duration", "start", "offset", "tcStart"):
            if key in node.attrib:
                assert (Fraction(node.attrib[key][:-1]) * fps).denominator == 1


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("01:00:00;00", "01:00:00;00"),
        ("01:00:00:00", "01:00:03;18"),
        ("00:01:00;02", "00:01:00;02"),
    ],
)
def test_source_timecode_origin(sample, options, tag, expected):
    source, edit = sample
    edit = replace(edit, media=replace(edit.media, timecode=tag))
    plan = export.build_export_plan(edit, source, ExportFormat.EDL, options)
    event = next(
        line for line in plan.files[0].text.splitlines() if line.startswith("001")
    )
    assert event.split()[4] == expected
    assert event.split()[6] == "00:00:00;00"


@pytest.mark.parametrize(
    "tag", ["bad", "00:01:00;00", "24:00:00;00", "00:00:60;00", "00:00:00;30"]
)
def test_bad_source_timecode(sample, options, tag):
    source, edit = sample
    with pytest.raises(ExportRefused, match="timecode"):
        export.build_export_plan(
            replace(edit, media=replace(edit.media, timecode=tag)),
            source,
            ExportFormat.EDL,
            options,
        )


@pytest.mark.parametrize("sidecar", [False, True])
def test_fcpxml_5994_escaping_and_source_origin(sample, options, sidecar):
    source, _ = sample
    source = source.with_name('clip & "test".mp4')
    source.write_bytes(b"synthetic escaped source")
    fps = Fraction(60000, 1001)
    info = replace(info_for(fps), timecode="01:00:00;00")
    edit = reviewed(source, info, [cut(0, float(30 / fps))])
    plan = export.build_export_plan(
        edit, source, ExportFormat.FCPXML, replace(options, sidecar_audio=sidecar)
    )
    root = ET.fromstring(plan.files[0].text)
    assert root.attrib["version"] == "1.10"
    assert root.find("resources/format").attrib["frameDuration"] == "1001/60000s"
    asset = root.find("resources/asset")
    assert asset.attrib["name"] == source.name
    assert asset.find("media-rep").attrib["src"] == source.as_uri()
    assert asset.find("format") is None
    assert Fraction(asset.attrib["start"][:-1]) == Fraction(215784, 1) / fps
    clip = root.find(".//asset-clip")
    assert Fraction(clip.attrib["start"][:-1]) == Fraction(215814, 1) / fps
    assert_frame_times(root, fps)
    if sidecar:
        connected = root.find('.//asset-clip[@lane="-1"]')
        assert connected.attrib["offset"] == "0/1s"
        assert (
            connected.attrib["duration"] == root.find(".//sequence").attrib["duration"]
        )
        assert root.find('.//asset[@id="r3"]').attrib["hasVideo"] == "0"
        assert clip.attrib["srcEnable"] == "video"


def test_edl_sidecar(sample, options):
    source, edit = sample
    plan = export.build_export_plan(
        edit, source, ExportFormat.EDL, replace(options, sidecar_audio=True)
    )
    assert (
        "* DECLIP SIDECAR AUDIO: sample.declip.wav STARTS AT RECORD 00:00:00;00"
        in plan.files[0].text
    )
    events = [line for line in plan.files[0].text.splitlines() if line[:3].isdigit()]
    assert all(line.split()[2] == "V" for line in events)
    assert plan.files[1].kind == "wav" and plan.files[1].text is None


def test_fcpxml_video_only(sample, options):
    source, edit = sample
    edit = replace(edit, media=replace(edit.media, audio_index=None, audio_streams=()))
    root = ET.fromstring(
        export.build_export_plan(edit, source, ExportFormat.FCPXML, options)
        .files[0]
        .text
    )
    assert root.find("resources/asset").attrib["hasAudio"] == "0"
    assert (
        " V "
        in export.build_export_plan(edit, source, ExportFormat.EDL, options)
        .files[0]
        .text
    )
    with pytest.raises(ExportRefused, match="audio stream"):
        export.build_export_plan(
            edit, source, ExportFormat.FCPXML, replace(options, sidecar_audio=True)
        )


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_export_refuses_pending_review(sample, options, fmt):
    source, edit = sample
    pending = ed.reset_decisions(edit)
    with pytest.raises(ReviewRequired, match="declip review"):
        export.build_export_plan(pending, source, fmt, options, overwrite=True)


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_export_refuses_source_mutated_after_review(sample, options, fmt):
    source, edit = sample
    original = source.stat()
    source.write_bytes(
        b"Synthetic source"
    )  # Same size and restored timestamp defeat a stale cache.
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(SourceMismatch):
        export.build_export_plan(edit, source, fmt, options)


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_export_refuses_stale_review(sample, options, fmt):
    source, edit = sample
    edit = replace(
        edit, cuts=(replace(edit.cuts[0], status=CutStatus.REJECTED), *edit.cuts[1:])
    )
    with pytest.raises(ReviewRequired):
        export.build_export_plan(edit, source, fmt, options)


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_export_refuses_all_media_removed(sample, options, fmt):
    source, edit = sample
    edit = reviewed(source, edit.media, [cut(0, edit.media.duration)])
    with pytest.raises(ExportRefused, match="all media removed"):
        export.build_export_plan(edit, source, fmt, options)


@pytest.mark.parametrize("fmt", [ExportFormat.EDL, ExportFormat.FCPXML])
@pytest.mark.parametrize("case", ["vfr", "unusual", "audio"])
def test_nle_refusal_and_conform_command(sample, options, fmt, case):
    source, edit = sample
    info = (
        replace(edit.media, vfr=True)
        if case == "vfr"
        else replace(edit.media, fps=Fraction(27))
        if case == "unusual"
        else replace(edit.media, has_video=False, fps=None, video_index=None)
    )
    edit = reviewed(source, info)
    with pytest.raises(ExportRefused, match="new plan and review") as failure:
        export.build_export_plan(edit, source, fmt, options)
    assert "ffmpeg -i" in str(failure.value) and "-vf fps=" in str(failure.value)
    for allowed in (ExportFormat.SRT, ExportFormat.MARKERS):
        export.build_export_plan(edit, source, allowed, options)


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_overwrite_and_no_plan_output(sample, options, fmt, tmp_path, monkeypatch):
    source, edit = sample
    directory = tmp_path / "output"

    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not run a process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    plan = export.build_export_plan(edit, source, fmt, options, out_dir=directory)
    assert not directory.exists()
    result = export.run_export_plan(plan)
    assert result.files == (plan.files[0].path,)
    before = result.files[0].read_bytes()
    with pytest.raises(ExportRefused, match="exists"):
        export.build_export_plan(edit, source, fmt, options, out_dir=directory)
    assert result.files[0].read_bytes() == before
    export.run_export_plan(
        export.build_export_plan(
            edit, source, fmt, options, out_dir=directory, overwrite=True
        )
    )


@pytest.mark.parametrize("kind", ["source", "edit"])
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_export_never_overwrites_source_or_editlist(sample, options, kind, link):
    source, edit = sample
    target = source if kind == "source" else source.with_name("sample.declip.json")
    if kind == "edit":
        ed.save_edit_list(target, edit)
    before = target.read_bytes()
    output = export.export_paths(source)["markers"]
    output.symlink_to(target) if link == "symlink" else os.link(target, output)
    with pytest.raises(ExportRefused):
        export.build_export_plan(
            edit, source, ExportFormat.MARKERS, options, overwrite=True
        )
    assert target.read_bytes() == before


def test_markers_edit_list_bytes_unchanged(sample, options):
    source, edit = sample
    path = source.with_name("sample.declip.json")
    ed.save_edit_list(path, edit)
    original = path.read_bytes()
    plan = export.build_export_plan(
        ed.load_edit_list(path), source, ExportFormat.MARKERS, options
    )
    result = export.run_export_plan(plan)
    assert path.read_bytes() == original
    assert result.files[0].name == "sample.declip.markers.json"
    assert json.loads(result.files[0].read_text()) == {
        "fps": "30000/1001",
        "markers": [
            {
                "time": 2.002,
                "source_time": 2.002,
                "kind": "gap",
                "label": "gap",
                "removed": 2.002,
            },
            {
                "time": 8.008,
                "source_time": 10.01,
                "kind": "gap",
                "label": "gap",
                "removed": 2.002,
            },
        ],
    }


def test_markers_overlapping_and_zero_frame_cuts(sample, options):
    source, edit = sample
    cuts = [
        cut(0.009, 0.01, "tiny"),
        cut(1, 3, "first"),
        cut(2, 4, "overlap"),
        cut(5, 6, "rejected", CutStatus.REJECTED),
    ]
    edit = reviewed(source, edit.media, cuts)
    markers = json.loads(
        export.build_export_plan(edit, source, ExportFormat.MARKERS, options)
        .files[0]
        .text
    )["markers"]
    assert len(markers) == 3 and markers[0]["removed"] == 0
    assert markers[1]["time"] == markers[2]["time"] == 1.001


def captions(source, info):
    rows = [
        (0, 1, "Cut", 0),
        (1, 2, "Hello", 0),
        (2, 3, "um", 0),
        (3, 4, "world.", 0),
        (4, 5, "Cut", 0),
        (5, 6, "Start", 1),
        (6, 7, "Keep", 1),
        (7, 8, "End", 1),
    ]
    words = tuple(
        Word(i, a, b, text, 1, segment) for i, (a, b, text, segment) in enumerate(rows)
    )
    transcript = Transcript(
        "synthetic",
        "fixture",
        "en",
        None,
        words,
        (
            Segment(0, 0, 5, "Cut Hello um world. Cut", (0, 1, 2, 3, 4)),
            Segment(1, 5, 8, "Start Keep End", (5, 6, 7)),
        ),
    )
    return reviewed(
        source, info, [cut(0, 1), cut(2, 3), cut(4, 6), cut(7, 8)], transcript
    )


def test_caption_events_use_post_cut_timeline(sample, options, assert_golden):
    source, _ = sample
    edit = captions(source, info_for(Fraction(30), 240))
    plan = export.build_export_plan(edit, source, ExportFormat.SRT, options)
    assert_golden(plan.files[0].text, GOLDEN / "srt_postcut.srt")


def test_caption_word_overlapping_cut_is_omitted(sample, options):
    source, _ = sample
    edit = captions(source, info_for(Fraction(30), 240))
    edit = reviewed(source, edit.media, [cut(1.5, 2.5)], edit.transcript)
    text = (
        export.build_export_plan(edit, source, ExportFormat.SRT, options).files[0].text
    )
    assert "Hello" not in text and "um" not in text
    assert "world." in text


def test_caption_length_duration_and_segment_limits(sample, options):
    source, _ = sample
    words = tuple(
        Word(i, i * 0.9, i * 0.9 + 0.8, "captionword", 1, i // 15) for i in range(30)
    )
    transcript = Transcript(
        "synthetic",
        "fixture",
        "en",
        None,
        words,
        (
            Segment(0, 0, 13.4, "", tuple(range(15))),
            Segment(1, 13.5, 27, "", tuple(range(15, 30))),
        ),
    )
    edit = reviewed(source, info_for(Fraction(30), 900), transcript=transcript)
    text = (
        export.build_export_plan(edit, source, ExportFormat.SRT, options).files[0].text
    )

    def seconds(value):
        h, m, s = value.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    for cue in text.strip().split("\n\n"):
        lines = cue.splitlines()
        start, end = map(seconds, lines[1].split(" --> "))
        assert end - start <= 6.001 and len(lines[2:]) <= 2
        assert all(len(line) <= 42 for line in lines[2:])
        assert not (start < 13.5 < end)
    assert text.count("captionword") == 30


def test_sidecar_preflight(sample, options):
    source, edit = sample
    options = replace(options, sidecar_audio=True)
    wav = export.export_paths(source)["wav"]
    wav.write_bytes(b"existing sidecar")
    with pytest.raises(ExportRefused, match="exists"):
        export.build_export_plan(edit, source, ExportFormat.EDL, options)
    plan = export.build_export_plan(
        edit, source, ExportFormat.EDL, options, overwrite=True
    )
    with pytest.raises(ExportRefused, match="render_processed_audio"):
        export.run_export_plan(plan)
    assert not plan.files[0].path.exists() and wav.read_bytes() == b"existing sidecar"

    def failed(target):
        target.write_bytes(b"partial")
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        export.run_export_plan(plan, render_audio=failed)
    assert wav.read_bytes() == b"existing sidecar"
    assert not list(source.parent.glob(".declip-audio-*"))


@pytest.mark.ffmpeg
@pytest.mark.contract
@pytest.mark.parametrize("fmt", [ExportFormat.EDL, ExportFormat.FCPXML])
def test_real_processed_sidecar(make_media, tmp_path, options, fmt):
    source = make_media(
        tmp_path,
        {
            "name": "tiny.mp4",
            "duration": 2,
            "inputs": [
                "testsrc2=size=160x90:rate=30",
                "sine=frequency=800:sample_rate=44100",
            ],
            "args": ["-c:v", "libx264", "-c:a", "aac", "-ac", "2"],
        },
    )
    edit = reviewed(source, media.probe_media(source), [cut(0.5, 1)], None)
    edit = replace(edit, processing=Processing("none", 0.5, "volume=0.5", None, 0))
    options = replace(options, sidecar_audio=True)
    plan = export.build_export_plan(edit, source, fmt, options)
    wav = plan.files[1].path
    wav.write_bytes(b"previous wav")
    plan = export.build_export_plan(edit, source, fmt, options, overwrite=True)

    def audio(target):
        return render.render_processed_audio(
            edit, source, target, options, temp_dir=tmp_path / "processing"
        )

    result = export.run_export_plan(plan, render_audio=audio)
    probe = media.probe_media(wav)
    assert probe.audio_streams[0].codec == "pcm_s24le"
    assert probe.audio_streams[0].sample_rate == 44100
    assert probe.audio_streams[0].channels == 2
    assert abs(probe.duration - plan.timeline.duration_out) <= 1 / 44100
    assert result.files == tuple(f.path for f in plan.files)

    def pcm(path):
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-c:a",
                "pcm_f32le",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        values = array.array("f")
        values.frombytes(decoded.stdout)
        return values

    original, processed = pcm(source), pcm(wav)
    expected = [
        sample * 0.5
        for keep in plan.timeline.keeps
        for sample in original[keep.start_sample * 2 : keep.end_sample * 2]
    ]
    assert len(processed) == len(expected)
    assert max(abs(a - b) for a, b in zip(processed, expected)) < 0.000001
    if fmt == ExportFormat.FCPXML and shutil.which("xmllint"):
        subprocess.run(
            ["xmllint", "--noout", str(plan.files[0].path)],
            check=True,
            capture_output=True,
        )
    # Avoid leaving generated media in pytest's retained temporary directories.
    for path in [source, wav]:
        path.unlink()


def test_edl_across_drop_frame_minutes(sample, options, assert_golden):
    source, _ = sample
    edit = reviewed(
        source,
        info_for(FPS, 18000),
        [
            cut(float(1800 / FPS), float(1860 / FPS)),
            cut(float(17922 / FPS), float(17982 / FPS)),
        ],
    )
    plan = export.build_export_plan(edit, source, ExportFormat.EDL, options)
    assert_golden(
        plan.files[0].text.replace(str(source.parent), "/synthetic"),
        GOLDEN / "edl_2997df_long.edl",
    )


def test_audio_only_captions_and_sample_grid_markers(sample, options):
    source, _ = sample
    info = replace(
        info_for(Fraction(30), 240),
        has_video=False,
        video_index=None,
        fps=None,
        audio_index=1,
        audio_streams=(AudioStream(1, "pcm_s24le", 44100, 1, "mono", 24, None),),
    )
    edit = captions(source, info)
    text = (
        export.build_export_plan(edit, source, ExportFormat.SRT, options).files[0].text
    )
    assert "Hello" in text and "world." in text and "Keep" in text
    assert "Cut" not in text and "um" not in text
    edit = reviewed(source, info, [cut(1.00001, 2.00001)])
    payload = json.loads(
        export.build_export_plan(edit, source, ExportFormat.MARKERS, options)
        .files[0]
        .text
    )
    assert payload == {
        "fps": None,
        "markers": [
            {
                "time": 1.0,
                "source_time": 1.0,
                "kind": "gap",
                "label": "gap",
                "removed": 1.0,
            }
        ],
    }


def test_caption_long_token_and_word_duration(sample, options):
    source, _ = sample
    words = (Word(0, 0, 0.5, "a" * 120, 1, 0), Word(1, 1, 14, "held", 1, 0))
    transcript = Transcript(
        "synthetic", "fixture", "en", None, words, (Segment(0, 0, 14, "", (0, 1)),)
    )
    edit = reviewed(source, info_for(Fraction(30), 600), transcript=transcript)
    text = (
        export.build_export_plan(edit, source, ExportFormat.SRT, options).files[0].text
    )
    cues = text.strip().split("\n\n")
    assert len(cues) == 5
    assert text.count("a") == 120
    assert all(len(cue.splitlines()[2:]) <= 2 for cue in cues)
    assert all(len(line) <= 42 for cue in cues for line in cue.splitlines()[2:])
    assert "00:00:01,000 --> 00:00:05,333" in text
    assert "00:00:09,667 --> 00:00:14,000" in text


def test_caption_submillisecond_cues_keep_contiguous_numbers(sample, options):
    source, _ = sample
    words = (Word(0, 0, 0.0001, "brief", 1, 0), Word(1, 1, 2, "visible", 1, 1))
    transcript = Transcript(
        "synthetic",
        "fixture",
        "en",
        None,
        words,
        (Segment(0, 0, 0.0001, "brief", (0,)), Segment(1, 1, 2, "visible", (1,))),
    )
    edit = reviewed(source, info_for(), transcript=transcript)
    text = (
        export.build_export_plan(edit, source, ExportFormat.SRT, options).files[0].text
    )
    assert text.startswith("1\n") and "visible" in text and "brief" not in text


def test_empty_transcript_and_no_optional_sidecar_for_text_formats(sample, options):
    source, edit = sample
    options = replace(options, sidecar_audio=True)
    assert (
        export.build_export_plan(edit, source, ExportFormat.SRT, options).files[0].text
        == ""
    )
    for fmt in (ExportFormat.SRT, ExportFormat.MARKERS):
        assert len(export.build_export_plan(edit, source, fmt, options).files) == 1


def test_invalid_audio_callback_preserves_existing_files(sample, options):
    source, edit = sample
    plan = export.build_export_plan(
        edit, source, ExportFormat.EDL, replace(options, sidecar_audio=True)
    )
    with pytest.raises(ExportRefused, match="requested sidecar"):
        export.run_export_plan(plan, render_audio=lambda target: target)
    assert all(not file.path.exists() for file in plan.files)
    assert not list(source.parent.glob(".declip-audio-*"))
