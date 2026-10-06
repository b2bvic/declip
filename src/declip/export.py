"""Reviewed NLE files, captions, and markers on the effective timeline."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import tempfile
import textwrap
from fractions import Fraction
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import quoteattr

from declip import editlist as ed, fsutil
from declip.contracts import (
    STANDARD_RATES,
    CutStatus,
    EditList,
    EffectiveTimeline,
    ExportFile,
    ExportFormat,
    ExportPlan,
    ExportRefused,
    ExportResult,
    ResolvedOptions,
)

_DROP_RATES = (Fraction(30000, 1001), Fraction(60000, 1001))


def export_paths(source: Path, *, out_dir: Path | None = None) -> dict[str, Path]:
    directory = out_dir if out_dir is not None else source.parent
    return {
        kind: directory / f"{source.stem}.declip.{suffix}"
        for kind, suffix in (
            ("edl", "edl"),
            ("fcpxml", "fcpxml"),
            ("srt", "srt"),
            ("markers", "markers.json"),
            ("wav", "wav"),
        )
    }


def frames_to_timecode(frames: int, fps: Fraction, *, drop_frame: bool) -> str:
    if frames < 0 or fps not in STANDARD_RATES:
        raise ExportRefused(
            "Timecode needs non-negative frames and a standard frame rate"
        )
    nominal = round(fps)
    separator = ";" if drop_frame else ":"
    if drop_frame:
        if fps not in _DROP_RATES:
            raise ExportRefused(
                "Drop-frame timecode requires 30000/1001 or 60000/1001 fps"
            )
        drop = nominal // 15
        ten_minutes = nominal * 600 - drop * 9
        minute = nominal * 60 - drop
        blocks, remainder = divmod(frames, ten_minutes)
        frames += drop * 9 * blocks
        if remainder >= drop:
            frames += drop * ((remainder - drop) // minute)
    seconds, frame = divmod(frames, nominal)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour % 24:02d}:{minute:02d}:{second:02d}{separator}{frame:02d}"


def _source_origin(edit_list: EditList) -> int:
    tag, fps = edit_list.media.timecode, edit_list.media.fps
    if not tag:
        return 0
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})([:;])(\d{2,3})", tag)
    if match is None or fps is None:
        raise ExportRefused(f"Invalid source timecode: {tag}")
    hour, minute, second, separator, frame = match.groups()
    hour, minute, second, frame = map(int, (hour, minute, second, frame))
    nominal = round(fps)
    if hour > 23 or minute > 59 or second > 59 or frame >= nominal:
        raise ExportRefused(f"Invalid source timecode: {tag}")
    total_minutes = hour * 60 + minute
    result = ((total_minutes * 60 + second) * nominal) + frame
    if separator == ";":
        if fps not in _DROP_RATES:
            raise ExportRefused(
                f"Drop-frame source timecode incompatible with {fps} fps"
            )
        drop = nominal // 15
        if minute % 10 and second == 0 and frame < drop:
            raise ExportRefused(f"Invalid dropped source timecode label: {tag}")
        result -= drop * (total_minutes - total_minutes // 10)
    return result


def _single_line(value: str) -> str:
    return " ".join(value.splitlines())


def _edl(
    edit_list: EditList, source: Path, timeline: EffectiveTimeline, wav: Path | None
) -> str:
    fps = timeline.fps
    drop = fps in _DROP_RATES
    origin = _source_origin(edit_list)

    def tc(frames):
        return frames_to_timecode(frames, fps, drop_frame=drop)

    lines = [
        f"TITLE: {_single_line(source.stem)}",
        f"FCM: {'DROP FRAME' if drop else 'NON-DROP FRAME'}",
        "",
    ]
    if wav:
        lines += [
            f"* DECLIP SIDECAR AUDIO: {_single_line(wav.name)} STARTS AT RECORD {tc(0)}",
            "",
        ]
    record = 0
    channel = "V" if wav or not edit_list.media.audio_streams else "AA/V"
    for i, keep in enumerate(timeline.keeps, 1):
        count = keep.end_frame - keep.start_frame
        lines += [
            f"{i:03d}  AX       {channel:<4} C        {tc(origin + keep.start_frame)} {tc(origin + keep.end_frame)} {tc(record)} {tc(record + count)}",
            f"* FROM CLIP NAME: {_single_line(source.name)}",
            f"* SOURCE FILE: {_single_line(str(source))}",
            "",
        ]
        record += count
    return "\n".join(lines)


def _xml_tag(tag: str, **attrs) -> str:
    return (
        "<"
        + tag
        + "".join(
            f" {key}={quoteattr(str(value))}"
            for key, value in attrs.items()
            if value is not None
        )
        + ">"
    )


def _fcpxml(
    edit_list: EditList, source: Path, timeline: EffectiveTimeline, wav: Path | None
) -> str:
    fps = timeline.fps
    origin = _source_origin(edit_list)

    def time(frames):
        value = Fraction(frames, 1) / fps
        return f"{value.numerator}/{value.denominator}s"

    total = sum(k.end_frame - k.start_frame for k in timeline.keeps)
    source_frames = math.floor(
        Fraction(str(edit_list.media.duration)) * fps + Fraction(1, 2)
    )
    audio = next(
        (
            a
            for a in edit_list.media.audio_streams
            if a.index == edit_list.media.audio_index
        ),
        None,
    )
    audio_attrs = (
        {
            "audioSources": "1",
            "audioChannels": audio.channels or 1,
            "audioRate": audio.sample_rate or 48000,
        }
        if audio
        else {}
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        '<fcpxml version="1.10">',
        "  <resources>",
    ]
    lines += [
        "    "
        + _xml_tag(
            "format",
            id="r1",
            frameDuration=time(1),
            width=edit_list.media.width,
            height=edit_list.media.height,
        )[:-1]
        + " />"
    ]
    lines += [
        "    "
        + _xml_tag(
            "asset",
            id="r2",
            name=source.name,
            start=time(origin),
            duration=time(source_frames),
            hasVideo="1",
            hasAudio="1" if audio else "0",
            format="r1",
            **audio_attrs,
        ),
        "      "
        + _xml_tag("media-rep", kind="original-media", src=source.as_uri())[:-1]
        + " />",
        "    </asset>",
    ]
    if wav:
        lines += [
            "    "
            + _xml_tag(
                "asset",
                id="r3",
                name=wav.name,
                start=time(0),
                duration=time(total),
                hasVideo="0",
                hasAudio="1",
                **audio_attrs,
            ),
            "      "
            + _xml_tag("media-rep", kind="original-media", src=wav.resolve().as_uri())[
                :-1
            ]
            + " />",
            "    </asset>",
        ]
    lines += [
        "  </resources>",
        "  <library>",
        '    <event name="declip">',
        "      " + _xml_tag("project", name=source.stem),
        "        "
        + _xml_tag(
            "sequence",
            format="r1",
            duration=time(total),
            tcStart=time(0),
            tcFormat="DF" if fps in _DROP_RATES else "NDF",
        ),
        "          <spine>",
    ]
    # The zero-origin container anchors the complete WAV at record zero even when
    # the first retained source frame or the source timecode is nonzero.
    indent = "            "
    if wav:
        lines += [
            indent
            + _xml_tag(
                "clip",
                name=source.stem,
                offset=time(0),
                start=time(0),
                duration=time(total),
                format="r1",
            ),
            indent + "  <spine>",
        ]
        indent += "    "
    record = 0
    for keep in timeline.keeps:
        count = keep.end_frame - keep.start_frame
        attrs = {
            "ref": "r2",
            "name": source.name,
            "offset": time(record),
            "start": time(origin + keep.start_frame),
            "duration": time(count),
        }
        if wav:
            attrs["srcEnable"] = "video"
        lines.append(indent + _xml_tag("asset-clip", **attrs)[:-1] + " />")
        record += count
    if wav:
        lines += [
            "              </spine>",
            "              "
            + _xml_tag(
                "asset-clip",
                ref="r3",
                name=wav.name,
                lane="-1",
                offset=time(0),
                start=time(0),
                duration=time(total),
                srcEnable="audio",
            )[:-1]
            + " />",
            "            </clip>",
        ]
    lines += [
        "          </spine>",
        "        </sequence>",
        "      </project>",
        "    </event>",
        "  </library>",
        "</fcpxml>",
    ]
    return "\n".join(lines) + "\n"


def _srt_time(seconds: float) -> str:
    ms = math.floor(Fraction(str(seconds)) * 1000 + Fraction(1, 2))
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def _srt(edit_list: EditList, timeline: EffectiveTimeline) -> str:
    if edit_list.transcript is None:
        return ""
    cues = []
    for keep in timeline.keeps:
        words = [
            w
            for w in edit_list.transcript.words
            if keep.start <= w.start < w.end <= keep.end and w.text.strip()
        ]
        pending = []

        def flush():
            if pending:
                start = ed.remap_time(pending[0].start, timeline)
                end = ed.remap_time(pending[-1].end, timeline)
                if end is None:  # A keep's exclusive endpoint may begin a cut.
                    end = ed.remap_time(
                        math.nextafter(pending[-1].end, -math.inf), timeline
                    )
                lines = textwrap.wrap(
                    " ".join(w.text.strip() for w in pending), width=42
                )
                cues.append((start, end, "\n".join(lines)))
                pending.clear()

        for word in words:
            if pending and (
                word.segment != pending[-1].segment
                or word.end - pending[0].start > 6
                or len(
                    textwrap.wrap(
                        " ".join(
                            [*(w.text.strip() for w in pending), word.text.strip()]
                        ),
                        width=42,
                    )
                )
                > 2
            ):
                flush()
            # Unusually long tokens get bounded cues rather than invalid SRT.
            lines = textwrap.wrap(word.text.strip(), width=42)
            if word.end - word.start > 6 or len(lines) > 2:
                flush()
                start = ed.remap_time(word.start, timeline)
                duration = word.end - word.start
                parts = max(math.ceil(duration / 6), math.ceil(len(lines) / 2))
                for i in range(parts):
                    chunk = lines[2 * i : 2 * i + 2] if len(lines) > 2 else lines
                    if chunk:
                        cues.append(
                            (
                                start + duration * i / parts,
                                start + duration * (i + 1) / parts,
                                "\n".join(chunk),
                            )
                        )
            else:
                pending.append(word)
        flush()
    visible = [
        (start, end, text)
        for start, end, text in cues
        if _srt_time(start) != _srt_time(end)
    ]
    return "\n\n".join(
        f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}"
        for i, (start, end, text) in enumerate(visible, 1)
    ) + ("\n" if visible else "")


def _markers(edit_list: EditList, timeline: EffectiveTimeline) -> str:
    grid = timeline.fps or Fraction(timeline.sample_rate)

    def snap(seconds):
        seconds = max(0, min(edit_list.media.duration, seconds))
        return float(math.floor(Fraction(str(seconds)) * grid + Fraction(1, 2)) / grid)

    markers = []
    for cut in sorted(edit_list.cuts, key=lambda c: (c.start, c.end, c.id)):
        if cut.status != CutStatus.ACCEPTED:
            continue
        start, end = snap(cut.start), snap(cut.end)
        output = sum(max(0, min(start, k.end) - k.start) for k in timeline.keeps)
        markers.append(
            {
                "time": round(output, 6),
                "source_time": round(start, 6),
                "kind": cut.kind.value,
                "label": cut.label,
                "removed": round(end - start, 6),
            }
        )
    fps = timeline.fps
    return (
        json.dumps(
            {
                "fps": f"{fps.numerator}/{fps.denominator}" if fps else None,
                "markers": markers,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and os.path.samefile(left, right)
    )


def build_export_plan(
    edit_list: EditList,
    source: Path,
    fmt: ExportFormat,
    options: ResolvedOptions,
    *,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> ExportPlan:
    ed.require_current_review(edit_list, source)
    source = source.resolve()
    if fmt in (ExportFormat.EDL, ExportFormat.FCPXML):
        info = edit_list.media
        if not info.has_video or info.vfr or info.fps not in STANDARD_RATES:
            rate = (
                min(STANDARD_RATES, key=lambda r: abs(r - info.fps))
                if info.fps
                else Fraction(30000, 1001)
            )
            destination = source.with_name(
                f"{source.stem}_cfr{source.suffix if info.has_video else '.mp4'}"
            )
            command = shlex.join(
                [
                    "ffmpeg",
                    "-i",
                    str(source),
                    "-vf",
                    f"fps={rate.numerator}/{rate.denominator}",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(destination),
                ]
            )
            raise ExportRefused(
                f"NLE export requires CFR video at a standard frame rate. Conform: {command}\nThe conformed file needs a new plan and review."
            )
    timeline = ed.effective_timeline(edit_list)
    if not timeline.keeps:
        raise ExportRefused("all media removed")
    paths = export_paths(source, out_dir=out_dir)
    wav = (
        paths["wav"]
        if options.sidecar_audio and fmt in (ExportFormat.EDL, ExportFormat.FCPXML)
        else None
    )
    if wav and not edit_list.media.audio_streams:
        raise ExportRefused("Sidecar audio requires a source audio stream")
    candidates = [paths[fmt.value], *([wav] if wav else [])]
    reserved = source.with_name(f"{source.stem}.declip.json")
    for path in candidates:
        if (
            _same_file(path, source)
            or _same_file(path, reserved)
            or path.resolve().name.endswith(".declip.json")
        ):
            raise ExportRefused(f"Export cannot replace source or edit list: {path}")
        if path.is_symlink():
            raise ExportRefused(f"Export destination is a symlink: {path}")
        if path.exists() and (not overwrite or not path.is_file()):
            raise ExportRefused(f"Export destination exists: {path}; use --overwrite")
    if fmt == ExportFormat.EDL:
        text = _edl(edit_list, source, timeline, wav)
    elif fmt == ExportFormat.FCPXML:
        text = _fcpxml(edit_list, source, timeline, wav)
    elif fmt == ExportFormat.SRT:
        text = _srt(edit_list, timeline)
    elif fmt == ExportFormat.MARKERS:
        text = _markers(edit_list, timeline)
    else:
        raise ExportRefused(f"Unknown export format: {fmt}")
    files = (ExportFile(paths[fmt.value], fmt.value, text),)
    if wav:
        files += (ExportFile(wav, "wav", None),)
    return ExportPlan(fmt, source, timeline, files)


def run_export_plan(
    plan: ExportPlan, *, render_audio: Callable[[Path], Path] | None = None
) -> ExportResult:
    if any(file.kind == "wav" for file in plan.files) and render_audio is None:
        raise ExportRefused(
            "Sidecar export requires render.render_processed_audio via render_audio"
        )
    for file in plan.files:
        if file.kind != "wav" and file.text is None:
            raise ExportRefused(f"Missing export text: {file.path}")
    # Render to a fresh path so P7's overwrite refusal also permits replacement
    # authorized during planning. Failed audio leaves existing exports intact.
    for file in plan.files:
        if file.kind != "wav":
            continue
        file.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".declip-audio-", dir=file.path.parent
        ) as temporary:
            target = Path(temporary) / file.path.name
            rendered = render_audio(target)
            if rendered.resolve() != target.resolve() or not target.is_file():
                raise ExportRefused(
                    "Audio renderer did not produce the requested sidecar WAV"
                )
            os.replace(target, file.path)
    for file in plan.files:
        if file.kind != "wav":
            fsutil.atomic_write_text(file.path, file.text)
    return ExportResult(tuple(file.path for file in plan.files))
