"""Command-line integration for the plan, review, render workflow."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import click

from . import __version__, audiochain, config as cfg, detect, editlist, encoders
from . import (
    enhance as enhancement,
    export as exporting,
    fillers,
    hardware,
    measure,
    paths,
    render,
    rig,
    transcribe,
)
from .contracts import (
    CutKind,
    DeclipError,
    ExportFormat,
    OutputSpec,
    Processing,
    ReviewState,
    RigRef,
    SourceMismatch,
)
from .fsutil import atomic_write_json, atomic_write_text
from .review import proxy, server
from .ui import ui_warn

FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
PATH = click.Path(path_type=Path)

# All defaults are unset here. Only explicit flags participate in precedence.
OPTION_SPECS = [
    (("--preset", "-p"), {}),
    (("--eq", "eq_chain"), {}),
    (("--enhance/--no-enhance",), {"default": None}),
    (("--enhancer",), {"type": click.Choice(["none", "auto", "afftdn", "deepfilter"])}),
    (("--enhancer-strength",), {"type": click.FloatRange(0, 1)}),
    (("--margin", "margin_ms"), {"type": click.FloatRange(min=0)}),
    (("--crossfade", "crossfade_ms"), {"type": click.FloatRange(min=0)}),
    (("--model",), {}),
    (("--language",), {}),
    (("--device",), {"type": click.Choice(["auto", "metal", "cuda", "cpu"])}),
    (("--backend",), {"type": click.Choice(["auto", "mlx", "faster"])}),
    (
        ("--compute-type",),
        {"type": click.Choice(["auto", "float16", "int8_float16", "int8"])},
    ),
    (
        ("--prompt-mode",),
        {"type": click.Choice(["auto", "initial", "hotwords", "chunked"])},
    ),
    (("--output", "-o"), {"type": PATH}),
    (("--min-confidence",), {"type": click.FloatRange(0, 1)}),
    (("--verbose", "-v"), {"is_flag": True, "default": None}),
    (("--json", "--json-output", "json_out"), {"is_flag": True, "default": None}),
    (("--max-gap", "max_gap_ms"), {"type": click.FloatRange(min=0)}),
    (("--gap-noise-db",), {"type": float}),
    (("--min-silence-ms",), {"type": click.FloatRange(min=0)}),
    (("--retakes/--no-retakes",), {"default": None}),
    (("--cut-range",), {"multiple": True}),
    (("--keep-transcript",), {"is_flag": True, "default": None}),
    (("--cpu",), {"is_flag": True, "default": None}),
    (("--execute",), {"is_flag": True, "default": None}),
    (
        ("--export", "export_fmt"),
        {"type": click.Choice(["edl", "srt", "markers", "fcpxml"])},
    ),
    (("--rig", "rig_name"), {}),
    (("--edit-list",), {"type": PATH}),
    (("--stages",), {}),
    (("--reset",), {"is_flag": True, "default": None}),
    (("--overwrite",), {"is_flag": True, "default": None}),
    (("--allow-8bit/--no-allow-8bit",), {"default": None}),
    (("--codec", "video_codec"), {"type": click.Choice(["match", "h264", "hevc"])}),
    (("--quality",), {"type": click.Choice(["match", "high", "small"])}),
    (("--encoder",), {"type": click.Choice(["auto", "software"])}),
    (("--loudness",), {"type": click.Choice(["-14", "-16", "-23", "none"])}),
    (("--output-mode",), {"type": click.Choice(["render", "nle"])}),
    (("--nle-format",), {"type": click.Choice(["edl", "fcpxml"])}),
]


def options(*, hidden=False):
    def decorate(function):
        for flags, settings in reversed(OPTION_SPECS):
            function = click.option(*flags, hidden=hidden, **settings)(function)
        return function

    return decorate


@click.group()
@click.version_option(__version__)
@options(hidden=True)
@click.pass_context
def cli(ctx, **values):
    """Plan edits, review every cut, then render or export."""
    ctx.ensure_object(dict)
    ctx.obj.update(values)


def values_for(ctx, values):
    result = dict(ctx.find_root().obj or {})
    for key, value in values.items():
        if value is not None and value != ():
            result[key] = value
    if result.get("cpu"):
        ui_warn("--cpu is deprecated and has no effect; use --device cpu.")
    if result.get("enhance") is not None:
        result.setdefault("enhancer", "auto" if result["enhance"] else "none")
        if result.get("enhancer") is None:
            result["enhancer"] = "auto" if result["enhance"] else "none"
    return result


def emit(payload, *, json_out=False, text=None):
    if json_out or text is None:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        click.echo(text)


def resolved(values, *, stored=None):
    config = cfg.load_config()
    defaults = cfg.run_defaults(config)
    profile = None
    if stored is not None:
        # Freeze the resolved rig values for an existing edit list.
        if stored.rig.snapshot:
            from .contracts import RigProfile

            profile = RigProfile.from_dict(stored.rig.snapshot)
        defaults.update(
            mode=stored.output.mode.value,
            nle_format=stored.output.nle_format,
            sidecar_audio=stored.output.sidecar_audio,
        )
    else:
        name = values.get("rig_name") or config.get("default_rig")
        if name:
            profile, warnings = rig.load_rig(rig.rig_path(name, paths.config_dir()))
            for warning in warnings:
                ui_warn(warning)
        else:
            ui_warn("No default rig; run declip setup to create one.")
    cli_values = dict(values)
    if cli_values.get("preset") is not None:
        cli_values["preset"] = audiochain.resolve_preset(
            cli_values["preset"], paths.config_dir()
        )
    # Setup records hardware. Runtime selection always probes the current machine.
    hardware.detect()
    return rig.resolve_options(cli_values, profile, defaults), profile


def edit_path(source, values):
    return values.get("edit_list") or source.with_name(source.stem + ".declip.json")


def input_edit(file, values):
    if file.suffix == ".json" and not values.get("edit_list"):
        path = file
        document = editlist.load_edit_list(path)
        source = Path(document.source.path)
    else:
        source = file.resolve()
        path = edit_path(source, values)
        document = editlist.load_edit_list(path)
    document = editlist.load_edit_list(path, source=source)
    return source, path, document


def stages_for(value):
    stages = (value or "fillers,gaps,retakes").split(",")
    if not stages or any(
        stage not in {"fillers", "gaps", "retakes"} for stage in stages
    ):
        raise click.BadParameter("use fillers,gaps,retakes", param_hint="--stages")
    return list(dict.fromkeys(stages))


def manual_ranges(values, info):
    result = []
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    tolerance = float(1 / info.fps) if info.fps else 0
    for value in values.get("cut_range", ()):
        match = re.fullmatch(rf"({number})-({number})", value)
        if match is None:
            raise click.BadParameter(
                "expected finite start-end seconds", param_hint="--cut-range"
            )
        start, end = map(float, match.groups())
        if (
            not all(map(math.isfinite, (start, end)))
            or start < 0
            or start >= end
            or end > info.duration + tolerance
        ):
            raise click.BadParameter(
                "require 0 <= start < end <= duration (plus one frame)",
                param_hint="--cut-range",
            )
        end = min(end, info.duration)
        minimum = float(1 / info.fps) if info.fps else 0.01
        if end - start < minimum:
            raise click.BadParameter(
                "cut must span at least one frame (10 ms for audio)",
                param_hint="--cut-range",
            )
        result.append((start, end, "manual"))
    return result


def transcript_for(source, info, opts):
    if info.audio_index is None:
        raise DeclipError("Transcription requires an audio stream")
    selection = transcribe.select(opts.transcribe.device, backend=opts.backend)
    return transcribe.transcribe_file(
        source,
        opts.transcribe,
        selection=selection,
        cache_dir=paths.cache_dir(),
        audio_index=info.audio_index,
        fillers_for=lambda language: fillers.load_fillers(
            language, config_dir=paths.config_dir()
        ),
    )


def make_plan(source, values):
    source = source.resolve()
    from . import media

    info = media.probe_media(source)
    ranges = manual_ranges(values, info)
    stages = stages_for(values.get("stages"))
    opts, profile = resolved(values)
    path = edit_path(source, values)
    digest = media.file_hash(source)
    processing = Processing(
        opts.enhancer,
        opts.enhancer_strength,
        opts.eq_chain,
        opts.loudness,
        opts.crossfade_ms,
    )
    output = OutputSpec(opts.output_mode, opts.nle_format, opts.sidecar_audio)
    ref = RigRef(opts.rig_name, opts.tier, profile.to_dict() if profile else {})
    previous = None
    revision = None
    if path.exists():
        revision = editlist.revision_of(path)
        previous = editlist.load_edit_list(path)
        if (
            previous.source.sha256 != digest
            or Path(previous.source.path).resolve() != source
        ):
            if not values.get("reset"):
                raise SourceMismatch(
                    "Edit list source changed; use plan --reset to create a new plan"
                )
            previous = None
    elif source.with_name(source.stem + ".edit.json").exists():
        ui_warn(
            f"Legacy edit list exists; run declip migrate {source.with_name(source.stem + '.edit.json')}"
        )
    transcript = (
        transcript_for(source, info, opts)
        if any(s in stages for s in ("fillers", "retakes"))
        else (previous.transcript if previous else None)
    )
    if previous:
        document = replace(
            previous,
            media=info,
            transcript=transcript,
            processing=processing,
            output=output,
            rig=ref,
        )
        if values.get("reset"):
            document = editlist.reset_decisions(document)
    else:
        document = editlist.new_edit_list(
            source,
            info,
            digest,
            transcript=transcript,
            processing=processing,
            output=output,
            rig=ref,
        )
    for stage in stages:
        proposals = []
        if stage == "fillers":
            language = transcript.language
            lexicon = fillers.load_fillers(language, config_dir=paths.config_dir())
            if lexicon:
                proposals = detect.detect_fillers(
                    transcript.words,
                    lexicon,
                    min_confidence=opts.min_confidence,
                    margin_ms=opts.margin_ms,
                    duration=info.duration,
                )
            else:
                ui_warn(
                    f"No filler file for language {language}; proposing gaps and retakes only."
                )
        elif stage == "retakes" and opts.retakes:
            proposals = detect.detect_retakes(
                transcript.words, min_confidence=opts.min_confidence
            )
        elif stage == "gaps" and opts.max_gap_ms > 0 and info.audio_index is not None:
            proposals = detect.detect_waveform_gaps(
                source,
                noise_db=opts.gap_noise_db,
                max_gap_ms=opts.max_gap_ms,
                min_silence_ms=opts.min_silence_ms,
                duration=info.duration,
                audio_index=info.audio_index,
            )
            proposals = [
                replace(p, low_confidence=p.confidence < opts.min_confidence)
                for p in proposals
            ]
        document = editlist.replace_stage_cuts(
            document, CutKind(stage[:-1] if stage != "retakes" else "retake"), proposals
        )
    if ranges:
        existing = {cut.id for cut in document.cuts}
        ranges = [
            row
            for row in ranges
            if editlist.cut_id(CutKind.MANUAL, *row) not in existing
        ]
        document = editlist.apply_decisions(
            document, decisions={}, add_manual=ranges, remove_manual=[]
        )
    # Multiple stage replacements can temporarily invalidate the hash. Restore only
    # when the final decisions are identical to the original passed review.
    if (
        previous
        and not values.get("reset")
        and previous.review.state == ReviewState.PASSED
        and editlist.review_content_hash(document) == previous.review.content_sha256
    ):
        document = replace(document, review=previous.review)
    return path, document, revision


def report(document):
    timeline = editlist.effective_timeline(document)
    return {
        "edit_list": document.to_dict(),
        "summary": {
            "counts": {
                state: sum(c.status.value == state for c in document.cuts)
                for state in ("proposed", "accepted", "rejected")
            },
            "removed_seconds": document.media.duration - timeline.duration_out,
            "output_duration": timeline.duration_out,
        },
    }


def plan_command(ctx, file, values, *, write):
    path, document, revision = make_plan(file, values)
    if write:
        editlist.save_edit_list(path, document, expected_revision=revision)
        if values.get("keep_transcript") and document.transcript:
            atomic_write_json(
                file.with_name(file.stem + ".transcript.json"),
                document.transcript.to_dict(),
            )
    next_stage = (
        "render"
        if document.review.state == ReviewState.PASSED and not values.get("export_fmt")
        else "review"
    )
    next_command = f"declip {next_stage} {shlex_path(file)}"
    edit_option = f" --edit-list {shlex_path(path)}" if values.get("edit_list") else ""
    next_command += edit_option
    message = f"{len(document.cuts)} cuts; output duration {editlist.effective_timeline(document).duration_out:.6f}s"
    if write:
        message += f"\nEdit list: {path}\n{next_command}"
        if values.get("export_fmt"):
            message += f"\ndeclip export {shlex_path(file)} --format {values['export_fmt']}{edit_option}"
    payload = report(document)
    if write:
        payload.update(path=str(path), next_command=next_command)
    emit(payload, json_out=values.get("json_out"), text=message)


def shlex_path(path):
    import shlex

    return shlex.quote(str(path))


@cli.command()
@click.argument("file", type=FILE)
@options()
@click.pass_context
def plan(ctx, file, **values):
    """Write a plan with proposed cuts. Review is required."""
    plan_command(ctx, file, values_for(ctx, values), write=True)


for _name in ("process", "clean", "detect"):

    def register(name):
        @cli.command(name)
        @click.argument("file", type=FILE)
        @options()
        @click.pass_context
        def command(ctx, file, **values):
            """Report proposed cuts; --execute writes the edit list for review."""
            values = values_for(ctx, values)
            if name == "clean":
                values.update(stages="fillers", enhancer="none")
            write = name != "detect" and bool(
                values.get("execute") or values.get("export_fmt")
            )
            plan_command(ctx, file, values, write=write)

        return command

    register(_name)


@cli.command("transcribe")
@click.argument("file", type=FILE)
@options()
@click.pass_context
def transcribe_command(ctx, file, **values):
    """Print one transcript JSON document."""
    from .media import probe_media

    values = values_for(ctx, values)
    opts, _ = resolved(values)
    emit(transcript_for(file.resolve(), probe_media(file), opts).to_dict())


@cli.command()
@click.argument("file", type=FILE)
@click.option("--no-browser", is_flag=True)
@click.option("--port", type=click.IntRange(0, 65535), default=0)
@options()
@click.pass_context
def review(ctx, file, no_browser, port, **values):
    """Open the local review page and save each human decision."""
    values = values_for(ctx, values)
    source, path, document = input_edit(file, values)
    if values.get("verbose"):
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        logging.getLogger("declip.review.server").setLevel(logging.DEBUG)
    preview = proxy.proxy_path(
        paths.cache_dir(), document.source.sha256, document.media
    )
    proxy.build_proxy(source, preview, proxy.proxy_args(document.media))
    result = server.serve(
        path,
        preview,
        open_browser=not no_browser,
        port=port,
        on_ready=lambda url: click.echo(url, err=bool(values.get("json_out"))),
    )
    if values.get("json_out"):
        emit(asdict(result))
    if not result.passed:
        raise click.exceptions.Exit(1)


@cli.command("render")
@click.argument("file", type=FILE)
@options()
@click.pass_context
def render_command(ctx, file, **values):
    """Render accepted cuts after a current passed review."""
    values = values_for(ctx, values)
    source, _, document = input_edit(file, values)
    editlist.require_current_review(document, source)
    opts, _ = resolved(values, stored=document)
    output = values.get("output") or render.default_output_path(source, document.media)
    with tempfile.TemporaryDirectory(prefix="declip-render-") as tmp:
        render_plan = render.build_render_plan(
            document,
            source,
            output,
            opts,
            encoders.probe_capabilities(),
            temp_dir=Path(tmp) / "work",
            overwrite=bool(values.get("overwrite")),
        )
        result = render.run_render_plan(
            render_plan, verbose=bool(values.get("verbose"))
        )
    emit(asdict(result), json_out=values.get("json_out"), text=str(result.output))


@cli.command("export")
@click.argument("file", type=FILE)
@click.option(
    "--format", "fmt", required=True, type=click.Choice([x.value for x in ExportFormat])
)
@click.option("--out-dir", type=PATH)
@options()
@click.pass_context
def export_command(ctx, file, fmt, out_dir, **values):
    """Export accepted cuts after a current passed review."""
    values = values_for(ctx, values)
    source, _, document = input_edit(file, values)
    editlist.require_current_review(document, source)
    opts, _ = resolved(values, stored=document)
    export_plan = exporting.build_export_plan(
        document,
        source,
        ExportFormat(fmt),
        opts,
        out_dir=out_dir,
        overwrite=bool(values.get("overwrite")),
    )
    with tempfile.TemporaryDirectory(prefix="declip-export-") as tmp:
        result = exporting.run_export_plan(
            export_plan,
            render_audio=lambda output: render.render_processed_audio(
                document, source, output, opts, temp_dir=Path(tmp) / "work"
            ),
        )
    emit(
        {"files": [str(path) for path in result.files]},
        json_out=values.get("json_out"),
        text="\n".join(map(str, result.files)),
    )


@cli.command("enhance")
@click.argument("file", type=FILE)
@options()
@click.pass_context
def enhance_command(ctx, file, **values):
    """Process whole-file audio without removing time."""
    from .media import probe_media

    values = values_for(ctx, values)
    opts, _ = resolved(values)
    info = probe_media(file)
    output = values.get("output") or render.default_output_path(file, info)
    with tempfile.TemporaryDirectory(prefix="declip-enhance-") as tmp:
        render_plan = render.build_enhance_plan(
            file.resolve(),
            output,
            info,
            opts,
            encoders.probe_capabilities(),
            temp_dir=Path(tmp) / "work",
            overwrite=bool(values.get("overwrite")),
        )
        if values.get("execute"):
            result = render.run_render_plan(
                render_plan, verbose=bool(values.get("verbose"))
            )
            emit(
                asdict(result), json_out=values.get("json_out"), text=str(result.output)
            )
        else:
            emit(
                asdict(render_plan),
                json_out=values.get("json_out"),
                text=f"Whole-file audio plan: {output}; pass --execute to render.",
            )


@cli.command()
@click.argument("old", type=FILE)
@click.option("--out", type=PATH)
@click.option("--source", type=FILE)
@click.option("--json", "json_out", is_flag=True, default=None)
@click.pass_context
def migrate(ctx, old, out, source, json_out):
    """Convert schema 2 without changing the original file."""
    json_out = values_for(ctx, {"json_out": json_out}).get("json_out")
    destination = out or editlist.migration_destination(old)
    if destination.exists() or destination.resolve() == old.resolve():
        raise DeclipError(f"Migration destination exists: {destination}")
    document = editlist.migrate_schema2(
        old,
        source=source,
        resolve_preset=lambda name: audiochain.resolve_preset(name, paths.config_dir()),
    )
    editlist.save_edit_list(destination, document)
    emit({"path": str(destination)}, json_out=json_out, text=str(destination))


@cli.command()
@click.option("--json", "json_out", is_flag=True)
@click.option(
    "--device", type=click.Choice(["auto", "metal", "cuda", "cpu"]), default=None
)
@click.option("--backend", type=click.Choice(["auto", "mlx", "faster"]), default=None)
@click.pass_context
def doctor(ctx, json_out, device, backend):
    """Report tools, backend availability, hardware, and paths."""
    from .media import require_tools

    values = values_for(
        ctx, {"json_out": json_out or None, "device": device, "backend": backend}
    )
    status = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "config_dir": str(paths.config_dir()),
        "cache_dir": str(paths.cache_dir()),
        "gpu": asdict(hardware.detect()),
        "decoder": {"default": "software", "source_probe_required": True},
        "deepfilter_release_url": "https://github.com/Rikorose/DeepFilterNet/releases",
    }
    errors = []
    try:
        require_tools()
        status["encoder"] = asdict(encoders.probe_capabilities())
    except DeclipError as exc:
        errors.append(str(exc))
        status["encoder"] = {"error": str(exc)}
    status["enhancer"] = [
        {"name": name, "available": ok, "reason": reason}
        for name, ok, reason in enhancement.list_enhancers()
    ]
    status["transcribers"] = [
        {
            "name": name,
            "available": transcribe.get(name).available()[0],
            "reason": transcribe.get(name).available()[1],
        }
        for name in ("mlx", "faster")
    ]
    try:
        selection = transcribe.select(
            values.get("device") or "auto", backend=values.get("backend") or "auto"
        )
        status["transcriber"] = {
            "backend": selection.transcriber.name,
            "device": selection.device,
            "compute_type": selection.compute_type,
        }
    except DeclipError as exc:
        errors.append(str(exc))
        status["transcriber"] = {"available": False, "error": str(exc)}
    status["errors"] = errors
    # Convert capability sets to arrays, not opaque strings.
    if "encoders" in status["encoder"]:
        for key in ("encoders", "hw_encode_ok"):
            status["encoder"][key] = sorted(status["encoder"][key])
    emit(status, json_out=True)
    if errors:
        raise click.exceptions.Exit(1)


@cli.command()
@click.option("--name")
@click.option(
    "--mic",
    type=click.Choice(["built-in", "usb", "lavalier", "shotgun", "xlr-interface"]),
)
@click.option("--room", type=click.Choice(["treated", "normal", "echo", "outdoor"]))
@click.option("--loudness", type=click.Choice(["-14", "-16", "-23", "none"]))
@click.option("--destination", type=click.Choice(["publish", "nle"]))
@click.option("--nle-format", type=click.Choice(["edl", "fcpxml"]))
@click.option("--sample", type=FILE)
@click.option("--log-profile/--no-log-profile", default=None)
@click.option("--preset")
@click.option("--yes", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--make-default", is_flag=True)
@click.option("--json", "json_out", is_flag=True, default=None)
@click.pass_context
def setup(
    ctx,
    name,
    mic,
    room,
    loudness,
    destination,
    nle_format,
    sample,
    log_profile,
    preset,
    yes,
    overwrite,
    make_default,
    json_out,
):
    """Measure a sample and save a named rig profile."""
    json_out = values_for(ctx, {"json_out": json_out}).get("json_out")
    answers = dict(
        mic=mic,
        room=room,
        loudness=loudness,
        destination=destination,
        nle_format=nle_format,
        log_profile=log_profile,
    )
    defaults = dict(
        mic="usb",
        room="normal",
        loudness="-16",
        destination="publish",
        nle_format="edl",
    )
    missing = [
        key
        for key in ("mic", "room", "loudness", "destination")
        if answers[key] is None
    ]
    terminal = sys.stdin.isatty()
    if not terminal and not yes and missing:
        raise click.UsageError(
            "Missing flags: "
            + ", ".join("--" + key.replace("_", "-") for key in missing)
        )
    if terminal and not yes:
        name = name or click.prompt("Profile name", default="default", err=True)
        for key in missing:
            answers[key] = click.prompt(
                key.capitalize(), default=defaults[key], err=True
            )
        if answers["destination"] == "nle" and nle_format is None:
            answers["nle_format"] = click.prompt(
                "NLE format",
                default="edl",
                type=click.Choice(["edl", "fcpxml"]),
                err=True,
            )
        if sample is None:
            sample_text = click.prompt(
                "Sample clip (empty to skip)", default="", show_default=False, err=True
            )
            sample = Path(sample_text) if sample_text else None
    name = name or "default"
    path = rig.rig_path(name, paths.config_dir())
    if path.exists() and not overwrite:
        if not terminal or not click.confirm(f"Replace rig {name}?", err=True):
            raise DeclipError(f"Rig exists: {path}; use --overwrite")
    measured = measure.measure_clip(sample) if sample else None
    if (
        measured
        and measured.video
        and (measured.video.bit_depth or 0) >= 10
        and log_profile is None
        and not yes
    ):
        if not terminal:
            raise click.UsageError(
                "Missing --log-profile or --no-log-profile for a 10-bit sample"
            )
        answers["log_profile"] = click.confirm(
            "Does this camera record in a log profile?", err=True
        )
    profile = rig.compute_profile(
        name,
        rig.setup_answers(answers),
        measured,
        hardware.detect(),
        preset=audiochain.resolve_preset(preset, paths.config_dir())
        if preset
        else None,
        now=datetime.now(timezone.utc),
    )
    for warning in (
        ["Clipping found in the sample; lower the input gain"]
        if measured and measured.clipping
        else []
    ):
        ui_warn(warning)
    rig.save_rig(path, profile)
    config = cfg.load_config()
    if not config.get("default_rig") or make_default:
        config["default_rig"] = name
        cfg.save_config(config)
    emit(
        profile.to_dict(),
        json_out=json_out,
        text=f"Rig {name}: {profile.tier.value}\n"
        + "\n".join(profile.tier_reasons)
        + f"\nSaved: {path}",
    )


@cli.group("rig")
def rig_command():
    """List, inspect, or remove rig profiles."""


@rig_command.command("list")
def rig_list():
    directory = paths.config_dir() / "rigs"
    emit({"rigs": sorted(path.stem for path in directory.glob("*.json"))})


@rig_command.command("show")
@click.argument("name")
def rig_show(name):
    profile, warnings = rig.load_rig(rig.rig_path(name, paths.config_dir()))
    for warning in warnings:
        ui_warn(warning)
    emit(profile.to_dict())


@rig_command.command("rm")
@click.argument("name")
@click.option("--force", is_flag=True)
def rig_rm(name, force):
    path = rig.rig_path(name, paths.config_dir())
    if not path.is_file():
        raise DeclipError(f"Rig does not exist: {path}")
    if not force:
        if not sys.stdin.isatty():
            raise click.UsageError("rig rm requires --force without a terminal")
        if not click.confirm(f"Remove rig {name}?", err=True):
            raise click.Abort()
    path.unlink()
    config = cfg.load_config()
    if config.get("default_rig") == name:
        config.pop("default_rig")
        cfg.save_config(config)
    emit({"removed": name})


@cli.group("config")
def config_command():
    """Manage user defaults, presets, and filler words."""


@config_command.command("show")
def config_show():
    config = cfg.load_config()
    cfg.run_defaults(config)
    emit(config)


@config_command.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    config = cfg.load_config()
    key = cfg.CONFIG_KEY_ALIASES.get(key, key)
    if key in {"enhance", "remove_retakes", "cpu"}:
        if value.lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise click.BadParameter("expected true or false")
        enabled = value.lower() in {"true", "1", "yes"}
        if key == "cpu":
            ui_warn("cpu is deprecated and has no effect; use device=cpu.")
            emit(config)
            return
        value = (
            ("auto" if enabled else "none")
            if key == "enhance"
            else str(enabled).lower()
        )
        key = "enhancer" if key == "enhance" else "retakes"
    if key == "default_rig":
        rig.load_rig(rig.rig_path(value, paths.config_dir()))
        config[key] = value
    else:
        if key not in cfg.DEFAULT_CONFIG["defaults"]:
            raise click.BadParameter(f"Unknown setting: {key}")
        default = cfg.DEFAULT_CONFIG["defaults"][key]
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        if isinstance(default, bool) and type(parsed) is not bool:
            raise click.BadParameter("expected true or false")
        config["defaults"][key] = parsed
        rig.resolve_options({}, None, cfg.run_defaults(config))
    cfg.save_config(config)
    emit(config)


def user_presets():
    path = paths.config_dir() / "presets.json"
    if not path.exists():
        return path, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        return path, data
    except (OSError, ValueError) as exc:
        raise DeclipError(f"Cannot read {path}: {exc}") from exc


@config_command.command("presets")
def config_presets():
    emit(
        {
            name: asdict(preset)
            for name, preset in audiochain.load_presets(paths.config_dir()).items()
        }
    )


@config_command.command("preset-add")
@click.argument("name")
@click.option("--chain", required=True)
@click.option("--description", default="")
def preset_add(name, chain, description):
    audiochain.validate_filter_chain(chain)
    audiochain.split_loudnorm(chain)
    path, presets = user_presets()
    presets[name] = {"chain": chain, "description": description}
    atomic_write_json(path, presets)
    emit({"saved": name})


@config_command.command("preset-rm")
@click.argument("name")
def preset_rm(name):
    path, presets = user_presets()
    if name not in presets:
        raise DeclipError(f"User preset not found: {name}")
    del presets[name]
    atomic_write_json(path, presets)
    emit({"removed": name})


def english_fillers():
    lexicon = fillers.load_fillers("en", config_dir=paths.config_dir())
    return lexicon, set(lexicon.single | lexicon.double)


@config_command.command("fillers")
def config_fillers():
    _, words = english_fillers()
    emit({"fillers": sorted(words)})


for _action in ("add", "rm"):

    def register_filler(action):
        @config_command.command("filler-" + action)
        @click.argument("word")
        def command(word):
            lexicon, words = english_fillers()
            word = word.lower().strip()
            if action == "add":
                if (
                    not 1 <= len(word.split()) <= 2
                    or "\n" in word
                    or word.startswith("#")
                ):
                    raise click.BadParameter("use a one- or two-word filler")
                words.add(word)
            elif word not in words:
                raise DeclipError(f"Filler not found: {word}")
            else:
                words.remove(word)
            path = paths.config_dir() / "fillers" / "en.txt"
            atomic_write_text(
                path,
                (f"# prompt: {lexicon.prompt}\n" if lexicon.prompt else "")
                + "\n".join(sorted(words))
                + "\n",
            )
            emit({"fillers": sorted(words)})

        return command

    register_filler(_action)


def main() -> None:
    try:
        cli(standalone_mode=False)
    except DeclipError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(exc.exit_code) from exc
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        click.echo("Aborted!", err=True)
        raise SystemExit(1) from exc
    except OSError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
