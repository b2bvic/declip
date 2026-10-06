"""Pure render plans and an atomic, progress-reporting media runner."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Callable

from declip import audiochain, editlist, encoders, enhance, media as media_tools
from declip.contracts import (
    AudioStream,
    AudioTarget,
    Capabilities,
    EditList,
    EffectiveTimeline,
    Keep,
    Loudness,
    MediaInfo,
    Processing,
    RenderError,
    RenderPlan,
    RenderResult,
    RenderStep,
    ResolvedOptions,
)


class _SuffixError(RenderError):
    exit_code = 2


class _AudioEncoder:
    """Audio-only targets do not need a usable video encoder."""

    name = "software"

    def audio_args(self, target: AudioTarget) -> list[str]:
        args = [
            "-c:a",
            target.codec,
            "-ar",
            str(target.sample_rate),
            "-ac",
            str(target.channels),
        ]
        if target.bitrate is not None:
            args += ["-b:a", str(target.bitrate)]
        return args


def default_output_path(source: Path, media: MediaInfo) -> Path:
    suffix = source.suffix.lower()
    extension = (
        (".mov" if suffix == ".mov" else ".mp4")
        if media.has_video
        else (".wav" if suffix in {".wav", ".aif", ".aiff", ".flac"} else ".m4a")
    )
    return source.with_name(source.stem + "_clean" + extension)


def _check_output_path(source: Path, output: Path, overwrite: bool) -> None:
    if source.resolve() == output.resolve() or (
        output.exists() and os.path.samefile(source, output)
    ):
        raise RenderError("output must differ from the source")
    if output.exists() and not overwrite:
        raise RenderError(f"output exists; use --overwrite: {output}")


def _number(value: float | Fraction) -> str:
    return f"{float(value):.12f}"


def _round_sample(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _sample_bounds(
    keep: Keep, timeline: EffectiveTimeline, rate: int
) -> tuple[int, int]:
    # Each stream can have a different rate. Derive from absolute frame indices.
    if timeline.fps:
        return tuple(
            _round_sample(Fraction(frame * rate) / timeline.fps)
            for frame in (keep.start_frame, keep.end_frame)
        )
    return tuple(
        _round_sample(Fraction(sample * rate, timeline.sample_rate))
        for sample in (keep.start_sample, keep.end_sample)
    )


def _audio_graph(
    stream: AudioStream, timeline: EffectiveTimeline, fade_ms: float
) -> str:
    count = len(timeline.keeps)
    rate = stream.sample_rate
    pieces = [
        f"[0:{stream.index}]asetpts=PTS-STARTPTS,asplit={count}"
        + "".join(f"[s{i}]" for i in range(count))
    ]
    for i, keep in enumerate(timeline.keeps):
        start, end = _sample_bounds(keep, timeline, rate)
        fade = min(
            _round_sample(Fraction(str(fade_ms)) * rate / 1000), (end - start) // 2
        )
        filters = f"atrim=start_sample={start}:end_sample={end},asetpts=PTS-STARTPTS"
        if fade:
            filters += f",afade=t=in:ss=0:ns={fade},afade=t=out:ss={end - start - fade}:ns={fade}"
        pieces.append(f"[s{i}]{filters}[a{i}]")
    samples = _round_sample(Fraction(str(timeline.duration_out)) * rate)
    pieces.append(
        "".join(f"[a{i}]" for i in range(count))
        + f"concat=n={count}:v=0:a=1,apad,atrim=end_sample={samples}[acut]"
    )
    return ";".join(pieces)


def _loudnorm(target: Loudness, *, measure: bool) -> str:
    base = f"loudnorm=I={target.i}:TP={target.tp}:LRA={target.lra}"
    if measure:
        return base + ":print_format=json"
    return base + (
        ":measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}"
        ":measured_thresh={input_thresh}:offset={target_offset}:linear=true"
    )


def _ffmpeg(
    name: str,
    args: list[str],
    inputs: tuple[Path, ...],
    outputs: tuple[Path, ...],
    duration: float | None,
) -> RenderStep:
    return RenderStep(
        name,
        "ffmpeg",
        tuple(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-progress",
                "pipe:1",
                "-nostats",
                *args,
            ]
        ),
        inputs,
        outputs,
        duration,
    )


def _build(
    source: Path,
    output: Path,
    info: MediaInfo,
    timeline: EffectiveTimeline,
    processing: Processing,
    options: ResolvedOptions,
    caps: Capabilities,
    temp_dir: Path,
    overwrite: bool,
    *,
    audio_only: bool = False,
    source_hash: str = "",
) -> RenderPlan:
    _check_output_path(source, output, overwrite)
    if not timeline.keeps:
        raise RenderError("all media removed")
    if temp_dir.resolve() in {source.resolve(), output.resolve()} or (
        temp_dir.resolve() in source.resolve().parents
        or temp_dir.resolve() in output.resolve().parents
    ):
        raise RenderError("temp_dir must be separate from source and output")
    audiochain.validate_filter_chain(processing.eq_chain)
    chain, trailing = audiochain.split_loudnorm(processing.eq_chain)
    loudness = processing.loudness if processing.loudness is not None else trailing
    if not math.isfinite(processing.crossfade_ms) or processing.crossfade_ms < 0:
        raise RenderError("crossfade_ms must be finite and non-negative")
    if (
        not math.isfinite(processing.enhancer_strength)
        or not 0 <= processing.enhancer_strength <= 1
    ):
        raise RenderError("enhancer strength must be between 0 and 1")
    if loudness and not all(
        math.isfinite(v) for v in (loudness.i, loudness.tp, loudness.lra)
    ):
        raise RenderError("loudness targets must be finite")
    container = output.suffix.lower().lstrip(".")
    video_enabled = info.has_video and not audio_only
    if container not in ({"mp4", "mov"} if video_enabled else {"wav", "m4a"}):
        raise _SuffixError(f"output suffix does not fit the codec: {output.suffix}")
    video = (
        encoders.video_target_for(
            info,
            codec=options.video_codec,
            quality=options.quality,
            allow_8bit=options.allow_8bit,
            container=container,
        )
        if video_enabled
        else None
    )
    encoder = (
        encoders.select_encoder(video, caps, prefer=options.encoder)
        if video
        else _AudioEncoder()
    )
    audio = (
        encoders.audio_target_for(
            info,
            bitrate=options.audio_bitrate,
            container=container,
            pcm=container == "wav",
        )
        if info.audio_streams
        else AudioTarget("aac", options.audio_bitrate, 48000, 0, container)
    )
    if not info.audio_streams and not video:
        raise RenderError("source has no audio")
    duration = timeline.duration_out
    steps, finals, streams = [], [], list(info.audio_streams)
    if audio_only:
        streams = [s for s in streams if s.index == info.audio_index]
    # Extract every track before processing the selected track.
    for stream in streams:
        if not stream.sample_rate or not stream.channels:
            raise RenderError(
                f"audio stream {stream.index} has no rate or channel count"
            )
        wav = temp_dir / f"audio-{stream.index}.wav"
        pcm = AudioTarget("pcm_s24le", None, stream.sample_rate, stream.channels, "wav")
        args = [
            "-i",
            str(source),
            "-filter_complex",
            _audio_graph(stream, timeline, processing.crossfade_ms),
            "-map",
            "[acut]",
            "-map_metadata",
            "-1",
            *encoder.audio_args(pcm),
        ]
        if stream.channel_layout:
            args += ["-channel_layout", stream.channel_layout]
        steps.append(
            _ffmpeg(
                f"audio-{stream.index}", [*args, str(wav)], (source,), (wav,), duration
            )
        )
    for stream in streams:
        current = temp_dir / f"audio-{stream.index}.wav"
        selected = stream.index == info.audio_index
        if selected and processing.enhancer != "none":
            enhanced = temp_dir / "enhanced.wav"
            steps.append(
                RenderStep(
                    "enhance",
                    "enhance",
                    (processing.enhancer, str(processing.enhancer_strength)),
                    (current,),
                    (enhanced,),
                    duration,
                )
            )
            current = enhanced
        filters = [chain] if selected and chain else []
        if selected and loudness:
            steps.append(
                _ffmpeg(
                    "loudness-measure",
                    [
                        "-i",
                        str(current),
                        "-map",
                        "0:a:0",
                        "-af",
                        ",".join([*filters, _loudnorm(loudness, measure=True)]),
                        "-f",
                        "null",
                        "-",
                    ],
                    (current,),
                    (),
                    duration,
                )
            )
            filters.append(_loudnorm(loudness, measure=False))
        filters += [
            f"aresample={stream.sample_rate}",
            "apad",
            f"atrim=end_sample={_round_sample(Fraction(str(duration)) * stream.sample_rate)}",
        ]
        final = (
            temp_dir / f"final-{stream.index}.{'wav' if container == 'wav' else 'm4a'}"
        )
        target = replace(
            audio, sample_rate=stream.sample_rate, channels=stream.channels
        )
        args = [
            "-i",
            str(current),
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-af",
            ",".join(filters),
            *encoder.audio_args(target),
        ]
        if stream.channel_layout:
            args += ["-channel_layout", stream.channel_layout]
        steps.append(
            _ffmpeg(
                "audio-final" if selected else f"audio-final-{stream.index}",
                [*args, str(final)],
                (current,),
                (final,),
                duration,
            )
        )
        finals.append(final)
    joined = None
    if video:
        video_source, index = source, info.video_index
        if info.vfr:
            video_source, index = temp_dir / "conformed.mp4", 0
            steps.append(
                _ffmpeg(
                    "video-conform",
                    [
                        "-i",
                        str(source),
                        "-map",
                        f"0:{info.video_index}",
                        "-vf",
                        f"fps={info.fps}",
                        "-an",
                        "-sn",
                        "-dn",
                        *encoder.video_args(
                            replace(video, quality="high", container="mp4")
                        ),
                        str(video_source),
                    ],
                    (source,),
                    (video_source,),
                    info.duration,
                )
            )
        batches = []
        for n, offset in enumerate(range(0, len(timeline.keeps), 40)):
            keeps = timeline.keeps[offset : offset + 40]
            first = keeps[0].start_frame
            seek = max(Fraction(0), (Fraction(first) - Fraction(1, 2)) / timeline.fps)
            expression = "+".join(
                f"between(n,{k.start_frame - first},{k.end_frame - first - 1})"
                for k in keeps
            )
            batch = temp_dir / f"video-{n:02d}.{container}"
            frames = sum(k.end_frame - k.start_frame for k in keeps)
            args = [
                "-ss",
                _number(seek),
                "-i",
                str(video_source),
                "-map",
                f"0:{index}",
                "-vf",
                f"select='{expression}',setpts=N/({timeline.fps})/TB",
                "-r",
                str(timeline.fps),
                "-fps_mode",
                "cfr",
                "-frames:v",
                str(frames),
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                *encoder.video_args(video),
                str(batch),
            ]
            steps.append(
                _ffmpeg(
                    f"video-batch-{n:02d}",
                    args,
                    (video_source,),
                    (batch,),
                    float(frames / timeline.fps),
                )
            )
            batches.append(batch)
        joined = temp_dir / f"joined.{container}"
        listing = temp_dir / "concat.txt"
        steps.append(
            _ffmpeg(
                "video-join",
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-map",
                    "0:v:0",
                    "-c",
                    "copy",
                    str(joined),
                ],
                tuple(batches),
                (joined,),
                duration,
            )
        )
    partial = output.with_name(
        f"{output.stem}.partial-{uuid.uuid4().hex}{output.suffix}"
    )
    mux_inputs = ([joined] if joined else []) + finals + [source]
    args = []
    for path in mux_inputs:
        args += ["-i", str(path)]
    if joined:
        args += ["-map", "0:v:0"]
    for n, stream in enumerate(streams):
        args += [
            "-map",
            f"{n + bool(joined)}:a:0",
            f"-disposition:a:{n}",
            f"{{disposition_{stream.index}}}",
        ]
        if stream.language:
            args += [f"-metadata:s:a:{n}", f"language={stream.language}"]
    args += [
        "-map_metadata",
        str(len(mux_inputs) - 1),
        "-c",
        "copy",
        "-t",
        _number(duration),
    ]
    if container in {"mp4", "mov", "m4a"}:
        args += ["-movflags", "+faststart", "-write_tmcd", "0"]
    steps.append(
        _ffmpeg("mux", [*args, str(partial)], tuple(mux_inputs), (partial,), duration)
    )
    steps.append(
        RenderStep(
            "guard",
            "guard",
            (source_hash, str(int(overwrite))),
            (partial,),
            (output,),
            duration,
        )
    )
    return RenderPlan(
        source,
        output,
        temp_dir,
        timeline,
        video,
        audio,
        encoder.name,
        loudness,
        tuple(steps),
    )


def build_render_plan(
    edit_list: EditList,
    source: Path,
    output: Path,
    options: ResolvedOptions,
    caps: Capabilities,
    *,
    temp_dir: Path,
    overwrite: bool = False,
) -> RenderPlan:
    editlist.require_current_review(edit_list, source)
    # Stored processing is the edit list's snapshot, not a subsequently changed rig.
    return _build(
        source,
        output,
        edit_list.media,
        editlist.effective_timeline(edit_list),
        edit_list.processing,
        options,
        caps,
        temp_dir,
        overwrite,
        source_hash=edit_list.source.sha256,
    )


def build_enhance_plan(
    source: Path,
    output: Path,
    media: MediaInfo,
    options: ResolvedOptions,
    caps: Capabilities,
    *,
    temp_dir: Path,
    overwrite: bool = False,
) -> RenderPlan:
    stream = next(
        (s for s in media.audio_streams if s.index == media.audio_index), None
    )
    rate = (stream.sample_rate if stream else None) or 48000
    frames = (
        _round_sample(Fraction(str(media.duration)) * media.fps) if media.fps else None
    )
    samples = _round_sample(Fraction(str(media.duration)) * rate)
    timeline = EffectiveTimeline(
        (Keep(0, media.duration, 0 if frames else None, frames, 0, samples, 0),),
        (),
        media.fps,
        rate,
        media.duration,
    )
    processing = Processing(
        options.enhancer,
        options.enhancer_strength,
        options.eq_chain,
        options.loudness,
        options.crossfade_ms,
    )
    return _build(
        source, output, media, timeline, processing, options, caps, temp_dir, overwrite
    )


def _concat_text(paths: tuple[Path, ...]) -> str:
    return "".join(
        "file '" + p.resolve().as_posix().replace("'", "'\\''") + "'\n" for p in paths
    )


def _stop(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def _run_ffmpeg(step: RenderStep, argv: list[str], progress, verbose) -> str:
    tail = deque(maxlen=40)
    measurement = []
    child = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def read_stderr():
        collecting = False
        for line in child.stderr:
            tail.append(line.rstrip())
            if verbose:
                sys.stderr.write(line)
                sys.stderr.flush()
            if step.name == "loudness-measure":
                if line.strip() == "{":
                    collecting = True
                if collecting:
                    measurement.append(line)
                if line.strip() == "}":
                    collecting = False

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    try:
        if progress:
            progress(step.name, 0.0)
        for line in child.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and step.expected_duration and progress:
                try:
                    fraction = min(
                        1.0, max(0.0, int(value) / 1e6 / step.expected_duration)
                    )
                except ValueError:
                    continue
                progress(step.name, fraction)
        child.wait()
        reader.join()
        if child.returncode:
            raise RenderError(
                f"{step.name}: ffmpeg exited {child.returncode}\n" + "\n".join(tail)
            )
        if progress:
            progress(step.name, 1.0)
        return (
            "".join(measurement) if step.name == "loudness-measure" else "\n".join(tail)
        )
    except BaseException as exc:
        _stop(child)
        reader.join(timeout=5)
        if isinstance(exc, KeyboardInterrupt):
            raise RenderError(f"{step.name}: interrupted\n" + "\n".join(tail)) from exc
        raise
    finally:
        child.stdout.close()
        child.stderr.close()


def _stream_details(source: Path) -> list[dict]:
    _, ffprobe = media_tools.require_tools()
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RenderError(f"cannot read stream details: {result.stderr}")
    return json.loads(result.stdout)["streams"]


def _source_dispositions(source: Path) -> dict[str, str]:
    return {
        f"{{disposition_{s['index']}}}": "+".join(
            key for key, enabled in s.get("disposition", {}).items() if enabled
        )
        or "0"
        for s in _stream_details(source)
    }


def run_render_plan(
    plan: RenderPlan,
    *,
    progress: Callable[[str, float], None] | None = None,
    verbose: bool = False,
) -> RenderResult:
    started = time.monotonic()
    partials = [s.outputs[0] for s in plan.steps if s.name == "mux"]
    step_name = "prepare"
    owned_temp = False
    try:
        ffmpeg, _ = media_tools.require_tools()
        guard = plan.steps[-1]
        _check_output_path(plan.source, plan.output, guard.argv[1] == "1")
        if (
            guard.argv[0]
            and media_tools.file_hash(plan.source, refresh=True) != guard.argv[0]
        ):
            raise RenderError("source changed after render plan was built")
        plan.temp_dir.mkdir(parents=True, exist_ok=False)
        owned_temp = True
        plan.output.parent.mkdir(parents=True, exist_ok=True)
        measured = {}
        silent = False
        result_info = None
        profile = None
        integrated_lufs = None
        decoder_args = []
        if plan.video and plan.encoder != "software":
            decoder_args = encoders.hwaccel_args(
                plan.source,
                media_tools.probe_media(plan.source),
                encoders.probe_capabilities(),
            )
        for step in plan.steps:
            step_name = step.name
            if step.kind == "enhance":
                if progress:
                    progress(step.name, 0.0)
                enhance.select_enhancer(step.argv[0]).enhance(
                    step.inputs[0], step.outputs[0], float(step.argv[1])
                )
                if progress:
                    progress(step.name, 1.0)
            elif step.kind == "ffmpeg":
                argv = [str(ffmpeg), *step.argv[1:]]
                if (
                    step.name == "video-conform" or step.name.startswith("video-batch")
                ) and step.inputs == (plan.source,):
                    position = argv.index("-i")
                    argv[position:position] = decoder_args
                if step.name == "video-join":
                    listing = Path(argv[argv.index("-i") + 1])
                    listing.write_text(_concat_text(step.inputs), encoding="utf-8")
                if step.name == "audio-final" and measured:
                    position = argv.index("-af") + 1
                    if silent:
                        argv[position] = re.sub(r"loudnorm=[^,]+,?", "", argv[position])
                    else:
                        for key, value in measured.items():
                            argv[position] = argv[position].replace(
                                "{" + key + "}", value
                            )
                if step.name == "mux":
                    dispositions = _source_dispositions(plan.source)
                    argv = [dispositions.get(value, value) for value in argv]
                report = _run_ffmpeg(step, argv, progress, verbose)
                if step.name == "loudness-measure":
                    try:
                        measured = json.loads(report)
                        required = (
                            "input_i",
                            "input_tp",
                            "input_lra",
                            "input_thresh",
                            "target_offset",
                        )
                        silent = not math.isfinite(float(measured["input_i"]))
                        if not silent and not all(
                            math.isfinite(float(measured[k])) for k in required
                        ):
                            raise ValueError("non-finite loudness measurement")
                    except (ValueError, KeyError, TypeError) as exc:
                        raise RenderError(
                            f"loudness-measure: invalid measurement: {report}"
                        ) from exc
            elif step.kind == "guard":
                partial = step.inputs[0]
                if plan.video:
                    encoders.check_output(partial, plan.video)
                result_info = media_tools.probe_media(partial)
                tolerance = float(1 / plan.timeline.fps) if plan.video else 0.001
                if (
                    abs(result_info.duration - plan.timeline.duration_out)
                    > tolerance + 1e-6
                ):
                    raise RenderError(
                        f"guard: duration {result_info.duration} differs from timeline {plan.timeline.duration_out}"
                    )
                audio_steps = [
                    s for s in plan.steps if s.name.startswith("audio-final")
                ]
                # Extraction/final steps retain source stream order, even when the selected track is later.
                if len(result_info.audio_streams) != len(audio_steps):
                    raise RenderError("guard: audio stream count changed")
                for stream, final in zip(result_info.audio_streams, audio_steps):
                    expected_rate = int(final.argv[final.argv.index("-ar") + 1])
                    expected_channels = int(final.argv[final.argv.index("-ac") + 1])
                    if (stream.sample_rate, stream.channels) != (
                        expected_rate,
                        expected_channels,
                    ):
                        raise RenderError(
                            "guard: audio channels or sample rate changed"
                        )
                if plan.video:
                    profile = next(
                        s.get("profile")
                        for s in _stream_details(partial)
                        if s.get("codec_type") == "video"
                    )
                if plan.loudness and audio_steps:
                    primary = next(
                        i
                        for i, final in enumerate(audio_steps)
                        if final.name == "audio-final"
                    )
                    measurement_step = _ffmpeg(
                        "guard",
                        [
                            "-i",
                            str(partial),
                            "-map",
                            f"0:a:{primary}",
                            "-af",
                            "ebur128=peak=true",
                            "-f",
                            "null",
                            "-",
                        ],
                        (partial,),
                        (),
                        plan.timeline.duration_out,
                    )
                    report = _run_ffmpeg(
                        measurement_step,
                        [str(ffmpeg), *measurement_step.argv[1:]],
                        None,
                        verbose,
                    )
                    matches = re.findall(r"I:\s*(-?[\d.]+|-inf) LUFS", report)
                    if not matches:
                        raise RenderError("guard: missing output loudness measurement")
                    value = float(matches[-1])
                    integrated_lufs = value if math.isfinite(value) else None
                if (
                    step.argv[0]
                    and media_tools.file_hash(plan.source, refresh=True) != step.argv[0]
                ):
                    raise RenderError("guard: source changed during rendering")
                _check_output_path(plan.source, plan.output, step.argv[1] == "1")
                os.replace(partial, plan.output)
                if progress:
                    progress(step.name, 1.0)
        if result_info is None:
            raise RenderError("guard: render plan has no guard")
        return RenderResult(
            plan.output,
            result_info.duration,
            result_info.vcodec,
            result_info.pix_fmt,
            profile,
            result_info.bit_depth,
            plan.audio.channels,
            integrated_lufs,
            plan.encoder,
            time.monotonic() - started,
        )
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, RenderError) and str(exc).startswith(step_name + ":"):
            raise
        raise RenderError(f"{step_name}: {str(exc) or 'interrupted'}") from exc
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
        if owned_temp:
            shutil.rmtree(plan.temp_dir)


def render_processed_audio(
    edit_list: EditList,
    source: Path,
    output_wav: Path,
    options: ResolvedOptions,
    *,
    temp_dir: Path,
) -> Path:
    editlist.require_current_review(edit_list, source)
    plan = _build(
        source,
        output_wav,
        edit_list.media,
        editlist.effective_timeline(edit_list),
        edit_list.processing,
        options,
        Capabilities("", frozenset(), frozenset(), None),
        temp_dir,
        False,
        audio_only=True,
        source_hash=edit_list.source.sha256,
    )
    run_render_plan(plan)
    return output_wav
