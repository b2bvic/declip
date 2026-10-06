"""Frozen Appendix A assertions. Do not derive expectations from implementation."""

import pytest

pytestmark = pytest.mark.contract

EXPECTED_FIELDS = {
    "Loudness": [("i", "float"), ("tp", "float"), ("lra", "float")],
    "Preset": [
        ("name", "str"),
        ("description", "str"),
        ("chain", "str"),
        ("loudness", "Loudness | None"),
    ],
    "FillerFile": [
        ("language", "str"),
        ("prompt", "str | None"),
        ("single", "frozenset[str]"),
        ("double", "frozenset[str]"),
    ],
    "Word": [
        ("i", "int"),
        ("start", "float"),
        ("end", "float"),
        ("text", "str"),
        ("p", "float"),
        ("segment", "int"),
    ],
    "Segment": [
        ("i", "int"),
        ("start", "float"),
        ("end", "float"),
        ("text", "str"),
        ("word_indices", "tuple[int, ...]"),
    ],
    "Transcript": [
        ("backend", "str"),
        ("model", "str"),
        ("language", "str"),
        ("prompt", "str | None"),
        ("words", "tuple[Word, ...]"),
        ("segments", "tuple[Segment, ...]"),
    ],
    "TranscribeOptions": [
        ("model", "str"),
        ("language", "str | None"),
        ("initial_prompt", "str | None"),
        ("prompt_mode", "str"),
        ("condition_on_previous_text", "bool"),
        ("device", "str"),
        ("compute_type", "str"),
        ("quiet", "bool"),
    ],
    "CutProposal": [
        ("kind", "CutKind"),
        ("start", "float"),
        ("end", "float"),
        ("label", "str"),
        ("confidence", "float"),
        ("word_indices", "tuple[int, int] | None"),
        ("low_confidence", "bool"),
    ],
    "Cut": [
        ("id", "str"),
        ("kind", "CutKind"),
        ("start", "float"),
        ("end", "float"),
        ("label", "str"),
        ("confidence", "float"),
        ("low_confidence", "bool"),
        ("words", "tuple[int, int] | None"),
        ("origin", "CutOrigin"),
        ("status", "CutStatus"),
    ],
    "SourceRef": [("path", "str"), ("name", "str"), ("size", "int"), ("sha256", "str")],
    "ColorTags": [
        ("primaries", "str | None"),
        ("trc", "str | None"),
        ("matrix", "str | None"),
        ("range", "str | None"),
    ],
    "AudioStream": [
        ("index", "int"),
        ("codec", "str | None"),
        ("sample_rate", "int | None"),
        ("channels", "int | None"),
        ("channel_layout", "str | None"),
        ("bit_depth", "int | None"),
        ("language", "str | None"),
    ],
    "MediaInfo": [
        ("duration", "float"),
        ("start_time", "float"),
        ("container", "str"),
        ("has_video", "bool"),
        ("video_index", "int | None"),
        ("audio_index", "int | None"),
        ("fps", "Fraction | None"),
        ("vfr", "bool"),
        ("width", "int | None"),
        ("height", "int | None"),
        ("rotation", "int"),
        ("vcodec", "str | None"),
        ("pix_fmt", "str | None"),
        ("bit_depth", "int | None"),
        ("video_bitrate", "int | None"),
        ("color", "ColorTags"),
        ("timecode", "str | None"),
        ("audio_streams", "tuple[AudioStream, ...]"),
        ("dropped_streams", "tuple[int, ...]"),
    ],
    "Processing": [
        ("enhancer", "str"),
        ("enhancer_strength", "float"),
        ("eq_chain", "str"),
        ("loudness", "Loudness | None"),
        ("crossfade_ms", "float"),
    ],
    "OutputSpec": [
        ("mode", "OutputMode"),
        ("nle_format", "NleFormat | None"),
        ("sidecar_audio", "bool"),
    ],
    "Review": [
        ("state", "ReviewState"),
        ("mode", "str | None"),
        ("passed_at", "str | None"),
        ("content_sha256", "str | None"),
    ],
    "RigRef": [
        ("name", "str | None"),
        ("tier", "Tier | None"),
        ("snapshot", "Mapping[str, Any]"),
    ],
    "EditList": [
        ("schema_version", "int"),
        ("declip_version", "str"),
        ("source", "SourceRef"),
        ("media", "MediaInfo"),
        ("rig", "RigRef"),
        ("transcript", "Transcript | None"),
        ("cuts", "tuple[Cut, ...]"),
        ("processing", "Processing"),
        ("output", "OutputSpec"),
        ("review", "Review"),
        ("history", "tuple[Mapping[str, Any], ...]"),
    ],
    "Keep": [
        ("start", "float"),
        ("end", "float"),
        ("start_frame", "int | None"),
        ("end_frame", "int | None"),
        ("start_sample", "int"),
        ("end_sample", "int"),
        ("out_start", "float"),
    ],
    "EffectiveTimeline": [
        ("keeps", "tuple[Keep, ...]"),
        ("removed", "tuple[tuple[float, float], ...]"),
        ("fps", "Fraction | None"),
        ("sample_rate", "int"),
        ("duration_out", "float"),
    ],
    "VideoTarget": [
        ("codec", "str"),
        ("bit_depth", "int"),
        ("pix_fmt", "str"),
        ("color", "ColorTags"),
        ("quality", "str"),
        ("source_bitrate", "int | None"),
        ("width", "int"),
        ("height", "int"),
        ("fps", "Fraction"),
        ("container", "str"),
    ],
    "AudioTarget": [
        ("codec", "str"),
        ("bitrate", "int | None"),
        ("sample_rate", "int"),
        ("channels", "int"),
        ("container", "str"),
    ],
    "Capabilities": [
        ("ffmpeg_version", "str"),
        ("encoders", "frozenset[str]"),
        ("hw_encode_ok", "frozenset[str]"),
        ("gpu_driver", "str | None"),
    ],
    "GpuInfo": [("name", "str"), ("vram_gb", "float | None"), ("driver", "str | None")],
    "HardwareInfo": [
        ("os", "str"),
        ("arch", "str"),
        ("chip", "str | None"),
        ("ram_gb", "float | None"),
        ("cpu_cores", "int"),
        ("apple_silicon", "bool"),
        ("gpus", "tuple[GpuInfo, ...]"),
    ],
    "VideoSummary": [
        ("codec", "str | None"),
        ("pix_fmt", "str | None"),
        ("bit_depth", "int | None"),
        ("trc", "str | None"),
        ("fps", "str | None"),
        ("vfr", "bool"),
    ],
    "ClipMeasurement": [
        ("sample", "str"),
        ("window_start", "float"),
        ("window_seconds", "float"),
        ("has_audio", "bool"),
        ("noise_floor_db", "float | None"),
        ("integrated_lufs", "float | None"),
        ("true_peak_db", "float | None"),
        ("lra", "float | None"),
        ("clipping", "bool"),
        ("peak_count", "int"),
        ("audio", "AudioStream | None"),
        ("video", "VideoSummary | None"),
    ],
    "RigAnswers": [
        ("mic", "str"),
        ("room", "str"),
        ("loudness", "int | None"),
        ("destination", "str"),
        ("nle_format", "str | None"),
        ("log_profile", "bool"),
    ],
    "RigProfile": [
        ("schema_version", "int"),
        ("name", "str"),
        ("created", "str"),
        ("answers", "RigAnswers"),
        ("measured", "ClipMeasurement | None"),
        ("hardware", "HardwareInfo"),
        ("tier", "Tier"),
        ("tier_reasons", "tuple[str, ...]"),
        ("transcribe", "Mapping[str, Any]"),
        ("audio", "Mapping[str, Any]"),
        ("video", "Mapping[str, Any]"),
        ("output", "Mapping[str, Any]"),
        ("detect", "Mapping[str, Any]"),
        ("review", "Mapping[str, Any]"),
    ],
    "ResolvedOptions": [
        ("rig_name", "str | None"),
        ("tier", "Tier | None"),
        ("transcribe", "TranscribeOptions"),
        ("backend", "str"),
        ("enhancer", "str"),
        ("enhancer_strength", "float"),
        ("eq_chain", "str"),
        ("loudness", "Loudness | None"),
        ("crossfade_ms", "float"),
        ("video_codec", "str"),
        ("quality", "str"),
        ("encoder", "str"),
        ("allow_8bit", "bool"),
        ("audio_bitrate", "int"),
        ("output_mode", "OutputMode"),
        ("nle_format", "NleFormat | None"),
        ("sidecar_audio", "bool"),
        ("margin_ms", "float"),
        ("min_confidence", "float"),
        ("gap_noise_db", "float"),
        ("max_gap_ms", "float"),
        ("min_silence_ms", "float"),
        ("retakes", "bool"),
        ("origins", "Mapping[str, str]"),
    ],
    "RenderStep": [
        ("name", "str"),
        ("kind", "str"),
        ("argv", "tuple[str, ...]"),
        ("inputs", "tuple[Path, ...]"),
        ("outputs", "tuple[Path, ...]"),
        ("expected_duration", "float | None"),
    ],
    "RenderPlan": [
        ("source", "Path"),
        ("output", "Path"),
        ("temp_dir", "Path"),
        ("timeline", "EffectiveTimeline"),
        ("video", "VideoTarget | None"),
        ("audio", "AudioTarget"),
        ("encoder", "str"),
        ("loudness", "Loudness | None"),
        ("steps", "tuple[RenderStep, ...]"),
    ],
    "RenderResult": [
        ("output", "Path"),
        ("duration", "float"),
        ("video_codec", "str | None"),
        ("pix_fmt", "str | None"),
        ("profile", "str | None"),
        ("bit_depth", "int | None"),
        ("audio_channels", "int"),
        ("integrated_lufs", "float | None"),
        ("encoder", "str"),
        ("elapsed", "float"),
    ],
    "ExportFile": [("path", "Path"), ("kind", "str"), ("text", "str | None")],
    "ExportPlan": [
        ("format", "ExportFormat"),
        ("source", "Path"),
        ("timeline", "EffectiveTimeline"),
        ("files", "tuple[ExportFile, ...]"),
    ],
    "ExportResult": [("files", "tuple[Path, ...]")],
    "ReviewResult": [
        ("passed", "bool"),
        ("edit_list_path", "Path"),
        ("content_sha256", "str | None"),
        ("unresolved_count", "int"),
        ("url", "str"),
    ],
    "BackendSelection": [
        ("transcriber", "Transcriber"),
        ("device", "str"),
        ("compute_type", "str"),
    ],
}

EXPECTED_ENUMS = {
    "CutKind": {
        "FILLER": "filler",
        "GAP": "gap",
        "RETAKE": "retake",
        "MANUAL": "manual",
    },
    "CutStatus": {
        "PROPOSED": "proposed",
        "ACCEPTED": "accepted",
        "REJECTED": "rejected",
    },
    "CutOrigin": {"AUTO": "auto", "MANUAL": "manual"},
    "ReviewState": {"PENDING": "pending", "PASSED": "passed"},
    "Tier": {"BASIC": "basic", "CREATOR": "creator", "PRO": "pro"},
    "OutputMode": {"RENDER": "render", "NLE": "nle"},
    "NleFormat": {"EDL": "edl", "FCPXML": "fcpxml"},
    "ExportFormat": {
        "EDL": "edl",
        "FCPXML": "fcpxml",
        "SRT": "srt",
        "MARKERS": "markers",
    },
}

EXPECTED_SIGNATURES = {
    "fsutil": [
        "atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None",
        "atomic_write_text(path: Path, text: str) -> None",
    ],
    "paths": ["config_dir() -> Path"],
    "audiochain": [
        "split_loudnorm(chain: str) -> tuple[str, Loudness | None]",
        "validate_filter_chain(chain: str) -> None",
        "load_presets(config_dir: Path) -> dict[str, Preset]",
        "resolve_preset(name: str, config_dir: Path) -> tuple[str, Loudness | None]",
        "loudness_for_target(target: int | None) -> Loudness | None",
    ],
    "fillers": [
        "parse_filler_file(text: str, *, language: str) -> FillerFile",
        "load_fillers(language: str, *, config_dir: Path) -> FillerFile | None",
    ],
    "media": [
        "require_tools() -> tuple[Path, Path]",
        "probe_media(path: Path) -> MediaInfo",
        "file_hash(path: Path, *, refresh: bool = False) -> str",
        "snap_fps(rate: Fraction) -> Fraction | None",
    ],
    "editlist": [
        "cut_id(kind: CutKind, start: float, end: float, label: str) -> str",
        "new_edit_list(source: Path, media: MediaInfo, sha256: str, *, transcript: Transcript | None, processing: Processing, output: OutputSpec, rig: RigRef) -> EditList",
        "load_edit_list(path: Path, *, source: Path | None = None) -> EditList",
        "save_edit_list(path: Path, edit_list: EditList, *, expected_revision: str | None = None) -> str",
        "revision_of(path: Path) -> str",
        "migrate_schema2(path: Path, *, source: Path | None, resolve_preset: Callable[[str], tuple[str, Loudness | None]]) -> EditList",
        "migration_destination(path: Path) -> Path",
        "replace_stage_cuts(edit_list: EditList, kind: CutKind, proposals: Sequence[CutProposal]) -> EditList",
        "reset_decisions(edit_list: EditList) -> EditList",
        "apply_decisions(edit_list: EditList, *, decisions: Mapping[str, CutStatus], add_manual: Sequence[tuple[float, float, str]], remove_manual: Sequence[str]) -> EditList",
        "review_content_hash(edit_list: EditList) -> str",
        "mark_review_passed(edit_list: EditList, *, now: datetime) -> EditList",
        "render_allowed(edit_list: EditList, source_sha256: str) -> bool",
        "require_current_review(edit_list: EditList, source: Path) -> None",
        "accepted_intervals(edit_list: EditList) -> list[tuple[float, float]]",
        "effective_timeline(edit_list: EditList) -> EffectiveTimeline",
        "keep_intervals(edit_list: EditList) -> list[tuple[float, float]]",
        "remap_time(seconds: float, timeline: EffectiveTimeline) -> float | None",
    ],
    "detect": [
        "detect_fillers(words: Sequence[Word], fillers: FillerFile, *, min_confidence: float, margin_ms: float, duration: float) -> list[CutProposal]",
        "detect_retakes(words: Sequence[Word], *, min_confidence: float, similarity_threshold: float = 0.6, window_s: float = 15.0) -> list[CutProposal]",
        "detect_waveform_gaps(audio: Path, *, noise_db: float, max_gap_ms: float, min_silence_ms: float, duration: float, audio_index: int | None = None) -> list[CutProposal]",
        "legacy_words(transcript: Mapping[str, Any]) -> list[Word]",
    ],
    "transcribe": [
        "get(name: str) -> Transcriber",
        'select(device: str = "auto", *, backend: str = "auto") -> BackendSelection',
        "extract_transcription_audio(source: Path, out_wav: Path, *, audio_index: int | None) -> Path",
        "cache_key(wav_sha256: str, backend: str, opts: TranscribeOptions) -> str",
        "transcribe_file(source: Path, opts: TranscribeOptions, *, selection: BackendSelection, cache_dir: Path, audio_index: int | None, fillers_for: Callable[[str], FillerFile | None]) -> Transcript",
    ],
    "transcribe.cuda_libs": ["preload() -> tuple[bool, str]"],
    "transcribe.download": ["ensure_model(name: str, *, backend: str) -> str"],
    "encoders": [
        "probe_capabilities(*, refresh: bool = False) -> Capabilities",
        "video_target_for(media: MediaInfo, *, codec: str, quality: str, allow_8bit: bool, container: str) -> VideoTarget",
        "audio_target_for(media: MediaInfo, *, bitrate: int, container: str, pcm: bool = False) -> AudioTarget",
        'select_encoder(target: VideoTarget, caps: Capabilities, *, prefer: str = "auto") -> Encoder',
        "hwaccel_args(source: Path, media: MediaInfo, caps: Capabilities) -> list[str]",
        "check_output(path: Path, target: VideoTarget) -> None",
    ],
    "enhance": [
        'select_enhancer(name: str = "auto") -> Enhancer',
        "list_enhancers() -> list[tuple[str, bool, str]]",
    ],
    "hardware": ["detect() -> HardwareInfo"],
    "measure": [
        "measure_clip(path: Path, *, seconds: float = 60.0) -> ClipMeasurement"
    ],
    "rig": [
        "rig_path(name: str, config_dir: Path) -> Path",
        "load_rig(path: Path) -> tuple[RigProfile, list[str]]",
        "save_rig(path: Path, profile: RigProfile) -> None",
        "default_model_for(hardware: HardwareInfo) -> tuple[str, str, str]",
        "compute_profile(name: str, answers: RigAnswers, measured: ClipMeasurement | None, hardware: HardwareInfo, *, preset: tuple[str, Loudness | None] | None = None, now: datetime) -> RigProfile",
        "resolve_options(cli_values: Mapping[str, Any], profile: RigProfile | None, config_defaults: Mapping[str, Any]) -> ResolvedOptions",
    ],
    "render": [
        "default_output_path(source: Path, media: MediaInfo) -> Path",
        "build_render_plan(edit_list: EditList, source: Path, output: Path, options: ResolvedOptions, caps: Capabilities, *, temp_dir: Path, overwrite: bool = False) -> RenderPlan",
        "build_enhance_plan(source: Path, output: Path, media: MediaInfo, options: ResolvedOptions, caps: Capabilities, *, temp_dir: Path, overwrite: bool = False) -> RenderPlan",
        "run_render_plan(plan: RenderPlan, *, progress: Callable[[str, float], None] | None = None, verbose: bool = False) -> RenderResult",
        "render_processed_audio(edit_list: EditList, source: Path, output_wav: Path, options: ResolvedOptions, *, temp_dir: Path) -> Path",
    ],
    "export": [
        "export_paths(source: Path, *, out_dir: Path | None = None) -> dict[str, Path]",
        "frames_to_timecode(frames: int, fps: Fraction, *, drop_frame: bool) -> str",
        "build_export_plan(edit_list: EditList, source: Path, fmt: ExportFormat, options: ResolvedOptions, *, out_dir: Path | None = None, overwrite: bool = False) -> ExportPlan",
        "run_export_plan(plan: ExportPlan, *, render_audio: Callable[[Path], Path] | None = None) -> ExportResult",
    ],
    "review.proxy": [
        "proxy_args(media: MediaInfo) -> list[str]",
        "proxy_path(cache_dir: Path, source_sha256: str, media: MediaInfo) -> Path",
        "build_proxy(source: Path, proxy_path: Path, ffmpeg_args: Sequence[str]) -> Path",
    ],
    "review.server": [
        "serve(edit_list_path: Path, proxy_path: Path, *, open_browser: bool = True, port: int = 0, on_ready: Callable[[str], None] | None = None) -> ReviewResult"
    ],
    "cli": ["main() -> None"],
}


def _namespace():
    import datetime
    import fractions
    import pathlib
    import typing

    from declip import contracts

    return {
        **vars(typing),
        **vars(contracts),
        "datetime": datetime.datetime,
        "Fraction": fractions.Fraction,
        "Path": pathlib.Path,
    }


def _example(annotation):
    import dataclasses
    import enum
    import types
    import typing
    from collections.abc import Mapping
    from fractions import Fraction
    from typing import Any, get_args, get_origin, get_type_hints

    if annotation is Any:
        return "value"
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (types.UnionType, typing.Union):
        return _example(next(choice for choice in args if choice is not type(None)))
    if origin is tuple:
        if args[-1] is Ellipsis:
            return (_example(args[0]),)
        return tuple(_example(arg) for arg in args)
    if origin in (dict, Mapping):
        return {_example(args[0]): _example(args[1])}
    if dataclasses.is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(
            **{
                field.name: _example(hints[field.name])
                for field in dataclasses.fields(annotation)
            }
        )
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return next(iter(annotation))
    if annotation is Fraction:
        return Fraction(30000, 1001)
    if annotation is str:
        return "synthetic"
    if annotation is int:
        return 1
    if annotation is float:
        return 0.123456
    if annotation is bool:
        return True
    raise AssertionError(f"No contract example for {annotation}")


def test_all_fields_types_order_and_frozen_dataclasses():
    import dataclasses
    from typing import get_type_hints

    from declip import contracts

    namespace = _namespace()
    for name, expected in EXPECTED_FIELDS.items():
        cls = getattr(contracts, name)
        actual = dataclasses.fields(cls)
        assert [field.name for field in actual] == [name for name, _ in expected], name
        assert cls.__dataclass_params__.frozen, name
        hints = get_type_hints(cls)
        for field, annotation in expected:
            assert hints[field] == eval(annotation, namespace), (name, field)
        for field in actual:
            if name == "TranscribeOptions" and field.name != "model":
                continue
            assert (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            )
    assert contracts.TranscribeOptions("small") == contracts.TranscribeOptions(
        "small", None, None, "auto", False, "auto", "auto"
    )


def test_all_enum_names_and_values():
    from declip import contracts

    for name, expected in EXPECTED_ENUMS.items():
        assert {
            entry.name: entry.value for entry in getattr(contracts, name)
        } == expected


def test_constants_and_errors():
    from fractions import Fraction

    from declip import contracts as c

    assert (c.SCHEMA_VERSION, c.RIG_SCHEMA_VERSION, c.PROXY_SETTINGS_VERSION) == (
        3,
        1,
        "1",
    )
    assert c.STANDARD_RATES == tuple(
        Fraction(rate)
        for rate in (
            "24000/1001",
            "24",
            "25",
            "30000/1001",
            "30",
            "48",
            "50",
            "60000/1001",
            "60",
            "120",
        )
    )
    assert c.BUILTIN_DEFAULTS == {
        "preset": "raw",
        "enhancer": "none",
        "enhancer_strength": 0.5,
        "crossfade_ms": 20,
        "margin_ms": 120,
        "min_confidence": 0.5,
        "gap_noise_db": -55.0,
        "max_gap_ms": 300,
        "min_silence_ms": 450,
        "retakes": True,
        "video_codec": "match",
        "quality": "match",
        "encoder": "auto",
        "allow_8bit": False,
        "audio_bitrate": 192000,
        "output_mode": "render",
        "nle_format": None,
        "sidecar_audio": False,
        "language": None,
        "device": "auto",
        "backend": "auto",
    }
    for name in (
        "ToolMissing",
        "EditListError",
        "RevisionConflict",
        "SourceMismatch",
        "ReviewRequired",
        "TranscriberUnavailable",
        "EncoderUnavailable",
        "EnhancerUnavailable",
        "FilterChainRejected",
        "RenderError",
        "ExportRefused",
    ):
        cls = getattr(c, name)
        assert issubclass(cls, c.DeclipError)
        assert cls.exit_code == (3 if name == "ReviewRequired" else 1)
    assert issubclass(c.RevisionConflict, c.EditListError)


def _assert_signature(actual, expected_text):
    import inspect
    from typing import get_type_hints

    namespace = _namespace()
    exec(
        "from __future__ import annotations\ndef " + expected_text + ":\n    pass\n",
        namespace,
    )
    expected = namespace[expected_text.split("(", 1)[0]]
    actual_sig, expected_sig = inspect.signature(actual), inspect.signature(expected)
    assert list(actual_sig.parameters) == list(expected_sig.parameters)
    for name, parameter in expected_sig.parameters.items():
        other = actual_sig.parameters[name]
        assert other.kind == parameter.kind and other.default == parameter.default, name
    assert get_type_hints(actual) == get_type_hints(expected, namespace)


def test_every_cross_packet_callable_signature():
    import importlib

    for module, signatures in EXPECTED_SIGNATURES.items():
        target = importlib.import_module("declip." + module)
        for signature in signatures:
            _assert_signature(getattr(target, signature.split("(", 1)[0]), signature)


def test_protocol_method_signatures():
    from declip import contracts as c

    signatures = {
        c.Transcriber: (
            "available(self) -> tuple[bool, str]",
            "transcribe(self, wav: Path, opts: TranscribeOptions) -> Transcript",
        ),
        c.Encoder: (
            "supports(self, target: VideoTarget, caps: Capabilities) -> bool",
            "video_args(self, target: VideoTarget) -> list[str]",
            "audio_args(self, target: AudioTarget) -> list[str]",
        ),
        c.Enhancer: (
            "available(self) -> tuple[bool, str]",
            "enhance(self, wav_in: Path, wav_out: Path, strength: float) -> None",
        ),
    }
    for protocol, expected in signatures.items():
        assert protocol._is_protocol
        assert protocol.__annotations__["name"] == "str"
        for signature in expected:
            _assert_signature(getattr(protocol, signature.split("(", 1)[0]), signature)


def test_round_trips_and_field_order_for_every_serializable_type():
    import dataclasses
    import json

    from declip import contracts as c

    for cls in (c.Transcript, c.MediaInfo, c.EditList, c.RigProfile):
        value = _example(cls)
        data = value.to_dict()
        assert list(data) == [field.name for field in dataclasses.fields(cls)]
        assert cls.from_dict(json.loads(json.dumps(data))) == value
    media = _example(c.MediaInfo)
    assert media.to_dict()["fps"] == "30000/1001"
    assert isinstance(media.to_dict()["audio_streams"], list)
    edit = _example(c.EditList)
    assert edit.to_dict()["review"]["state"] == "pending"
    assert isinstance(edit.to_dict()["history"], list)
    assert edit.to_dict()["cuts"][0]["words"] == [1, 1]


def test_optional_fields_and_time_precision():
    from dataclasses import replace

    from declip import contracts as c

    transcript = c.Transcript(
        "faster",
        "small",
        "en",
        None,
        (c.Word(0, 1.123456789, 2.987654321, "hi", 0.98, 0),),
        (c.Segment(0, 1.123456789, 2.987654321, "hi", (0,)),),
    )
    data = transcript.to_dict()
    assert data["words"][0]["start"] == 1.123457
    assert data["words"][0]["end"] == 2.987654
    assert data["words"][0]["p"] == 0.98
    edit = replace(
        _example(c.EditList),
        transcript=None,
        rig=c.RigRef(None, None, {}),
        review=c.Review(c.ReviewState.PENDING, None, None, None),
        processing=c.Processing("none", 0.5, "", None, 20),
        output=c.OutputSpec(c.OutputMode.RENDER, None, False),
    )
    assert c.EditList.from_dict(edit.to_dict()) == edit
    rig = replace(_example(c.RigProfile), measured=None)
    assert c.RigProfile.from_dict(rig.to_dict()) == rig


def test_invalid_edit_list_data_raises_contract_error():

    from declip import contracts as c

    for data in ({}, {"schema_version": "three"}):
        with pytest.raises(c.EditListError):
            c.EditList.from_dict(data)
    invalid = _example(c.EditList).to_dict()
    invalid["cuts"][0]["status"] = "unknown"
    with pytest.raises(c.EditListError):
        c.EditList.from_dict(invalid)


# Literal JSON examples from the approved spec, independent of serializer output.
EDIT_LIST_EXAMPLE = {
    "schema_version": 3,
    "declip_version": "0.5.1",
    "source": {
        "path": "/abs/clip.mp4",
        "name": "clip.mp4",
        "size": 123,
        "sha256": "...",
    },
    "media": {
        "duration": 879.168,
        "start_time": 0.0,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "has_video": True,
        "video_index": 0,
        "audio_index": 1,
        "fps": "30000/1001",
        "vfr": False,
        "width": 3840,
        "height": 2160,
        "rotation": 0,
        "vcodec": "hevc",
        "pix_fmt": "yuv420p10le",
        "bit_depth": 10,
        "video_bitrate": 45000000,
        "color": {
            "primaries": "bt2020",
            "trc": "arib-std-b67",
            "matrix": "bt2020nc",
            "range": "tv",
        },
        "timecode": None,
        "audio_streams": [
            {
                "index": 1,
                "codec": "aac",
                "sample_rate": 48000,
                "channels": 2,
                "channel_layout": "stereo",
                "bit_depth": None,
                "language": "und",
            }
        ],
        "dropped_streams": [2],
    },
    "rig": {"name": "desk", "tier": "creator", "snapshot": {}},
    "transcript": {
        "backend": "mlx",
        "model": "...",
        "language": "en",
        "prompt": "...",
        "words": [
            {"i": 0, "start": 0.42, "end": 0.61, "text": "So", "p": 0.93, "segment": 0}
        ],
        "segments": [
            {
                "i": 0,
                "start": 0.42,
                "end": 3.1,
                "text": "So today...",
                "word_indices": [0, 1, 2],
            }
        ],
    },
    "cuts": [
        {
            "id": "9f2c0a1b2c3d4e5f",
            "kind": "filler",
            "start": 12.31,
            "end": 12.74,
            "label": "um",
            "confidence": 0.88,
            "low_confidence": False,
            "words": [41, 41],
            "origin": "auto",
            "status": "proposed",
        }
    ],
    "processing": {
        "enhancer": "none",
        "enhancer_strength": 0.5,
        "eq_chain": "highpass=f=70:poles=2",
        "loudness": {"i": -16, "tp": -1.5, "lra": 11},
        "crossfade_ms": 20,
    },
    "output": {"mode": "render", "nle_format": None, "sidecar_audio": False},
    "review": {
        "state": "pending",
        "mode": None,
        "passed_at": None,
        "content_sha256": None,
    },
    "history": [{"stage": "plan", "at": "2026-10-06T16:00:00Z"}],
}

RIG_PROFILE_EXAMPLE = {
    "schema_version": 1,
    "name": "desk",
    "created": "2026-10-06T16:00:00Z",
    "answers": {
        "mic": "lavalier",
        "room": "normal",
        "loudness": -16,
        "destination": "nle",
        "nle_format": "edl",
        "log_profile": False,
    },
    "measured": {
        "sample": "clip.mp4",
        "window_start": 409.6,
        "window_seconds": 60.0,
        "has_audio": True,
        "noise_floor_db": -58.2,
        "integrated_lufs": -21.4,
        "true_peak_db": -3.1,
        "lra": 7.9,
        "clipping": False,
        "peak_count": 0,
        "audio": {
            "index": 1,
            "codec": "aac",
            "sample_rate": 48000,
            "channels": 2,
            "channel_layout": "stereo",
            "bit_depth": 16,
            "language": None,
        },
        "video": {
            "codec": "hevc",
            "pix_fmt": "yuv420p10le",
            "bit_depth": 10,
            "trc": "bt709",
            "fps": "30000/1001",
            "vfr": False,
        },
    },
    "hardware": {
        "os": "darwin",
        "arch": "arm64",
        "chip": "<chip name>",
        "ram_gb": 24,
        "cpu_cores": 12,
        "apple_silicon": True,
        "gpus": [],
    },
    "tier": "pro",
    "tier_reasons": ["video bit depth is 10"],
    "transcribe": {
        "backend": "auto",
        "model": "mlx-community/whisper-large-v3-turbo",
        "language": None,
        "compute_type": "auto",
        "device": "auto",
    },
    "audio": {
        "enhancer": "none",
        "enhancer_strength": 0.5,
        "eq_chain": "",
        "loudness": None,
    },
    "video": {"encoder": "auto", "codec": "match", "quality": "match"},
    "output": {"mode": "nle", "nle_format": "edl", "sidecar_audio": False},
    "detect": {
        "margin_ms": 120,
        "min_confidence": 0.5,
        "gap_noise_db": -52.2,
        "max_gap_ms": 300,
        "min_silence_ms": 450,
        "retakes": True,
    },
    "review": {"required": True},
}


def test_serialization_matches_spec_examples():
    from declip.contracts import EditList, RigProfile

    for cls, expected in (
        (EditList, EDIT_LIST_EXAMPLE),
        (RigProfile, RIG_PROFILE_EXAMPLE),
    ):
        value = cls.from_dict(expected)
        actual = value.to_dict()
        assert actual == expected
        assert cls.from_dict(actual) == value
