"""src/declip/contracts.py: cross-packet contract. P0 owns. Frozen before wave 1."""

from __future__ import annotations

import enum
import types
import typing
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Protocol, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 3
RIG_SCHEMA_VERSION = 1
PROXY_SETTINGS_VERSION = "1"
STANDARD_RATES: tuple[Fraction, ...] = (
    Fraction(24000, 1001),
    Fraction(24),
    Fraction(25),
    Fraction(30000, 1001),
    Fraction(30),
    Fraction(48),
    Fraction(50),
    Fraction(60000, 1001),
    Fraction(60),
    Fraction(120),
)


class CutKind(str, enum.Enum):
    FILLER = "filler"
    GAP = "gap"
    RETAKE = "retake"
    MANUAL = "manual"


class CutStatus(str, enum.Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class CutOrigin(str, enum.Enum):
    AUTO = "auto"
    MANUAL = "manual"


class ReviewState(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"


class Tier(str, enum.Enum):
    BASIC = "basic"
    CREATOR = "creator"
    PRO = "pro"


class OutputMode(str, enum.Enum):
    RENDER = "render"
    NLE = "nle"


class NleFormat(str, enum.Enum):
    EDL = "edl"
    FCPXML = "fcpxml"


class ExportFormat(str, enum.Enum):
    EDL = "edl"
    FCPXML = "fcpxml"
    SRT = "srt"
    MARKERS = "markers"


class DeclipError(Exception):
    exit_code = 1


class ToolMissing(DeclipError): ...


class EditListError(DeclipError): ...


class RevisionConflict(EditListError): ...


class SourceMismatch(DeclipError): ...


class ReviewRequired(DeclipError):
    exit_code = 3


class TranscriberUnavailable(DeclipError): ...


class EncoderUnavailable(DeclipError): ...


class EnhancerUnavailable(DeclipError): ...


class FilterChainRejected(DeclipError): ...


class RenderError(DeclipError): ...


class ExportRefused(DeclipError): ...


@dataclass(frozen=True)
class Loudness:
    i: float
    tp: float
    lra: float


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    chain: str  # loudnorm already split out
    loudness: Loudness | None


@dataclass(frozen=True)
class FillerFile:
    language: str
    prompt: str | None
    single: frozenset[str]
    double: frozenset[str]  # two-word fillers, space-joined, lowercase


@dataclass(frozen=True)
class Word:
    i: int  # 0-based, contiguous index in Transcript.words
    start: float
    end: float
    text: str  # whitespace stripped
    p: float  # 1.0 when the backend gives none
    segment: int  # Segment.i


@dataclass(frozen=True)
class Segment:
    i: int
    start: float
    end: float
    text: str
    word_indices: tuple[int, ...]


@dataclass(frozen=True)
class Transcript:
    backend: str
    model: str
    language: str
    prompt: str | None
    words: tuple[Word, ...]
    segments: tuple[Segment, ...]

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Transcript:
        return _from_dict(cls, data)


@dataclass(frozen=True)
class TranscribeOptions:
    model: str
    language: str | None = None
    initial_prompt: str | None = None
    prompt_mode: str = "auto"  # auto | initial | hotwords | chunked
    condition_on_previous_text: bool = False
    device: str = "auto"  # auto | metal | cuda | cpu
    compute_type: str = "auto"  # auto | float16 | int8_float16 | int8


@dataclass(frozen=True)
class CutProposal:
    kind: CutKind
    start: float
    end: float
    label: str
    confidence: float
    word_indices: tuple[int, int] | None  # inclusive; None for gap
    low_confidence: bool


@dataclass(frozen=True)
class Cut:
    id: str
    kind: CutKind
    start: float
    end: float
    label: str
    confidence: float
    low_confidence: bool
    words: tuple[int, int] | None
    origin: CutOrigin
    status: CutStatus


@dataclass(frozen=True)
class SourceRef:
    path: str
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ColorTags:
    primaries: str | None
    trc: str | None
    matrix: str | None
    range: str | None


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None
    bit_depth: int | None
    language: str | None


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    start_time: float
    container: str
    has_video: bool
    video_index: int | None
    audio_index: int | None
    fps: Fraction | None  # snapped; conform rate for VFR
    vfr: bool
    width: int | None
    height: int | None
    rotation: int  # 0, 90, 180, 270
    vcodec: str | None
    pix_fmt: str | None
    bit_depth: int | None
    video_bitrate: int | None
    color: ColorTags
    timecode: str | None
    audio_streams: tuple[AudioStream, ...]
    dropped_streams: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MediaInfo:
        return _from_dict(cls, data)


@dataclass(frozen=True)
class Processing:
    enhancer: str  # none | auto | afftdn | deepfilter
    enhancer_strength: float
    eq_chain: str
    loudness: Loudness | None
    crossfade_ms: float


@dataclass(frozen=True)
class OutputSpec:
    mode: OutputMode
    nle_format: NleFormat | None
    sidecar_audio: bool


@dataclass(frozen=True)
class Review:
    state: ReviewState
    mode: str | None  # None or "interactive"
    passed_at: str | None
    content_sha256: str | None


@dataclass(frozen=True)
class RigRef:
    name: str | None
    tier: Tier | None
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class EditList:
    schema_version: int
    declip_version: str
    source: SourceRef
    media: MediaInfo
    rig: RigRef
    transcript: Transcript | None
    cuts: tuple[Cut, ...]
    processing: Processing
    output: OutputSpec
    review: Review
    history: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EditList:
        return _from_dict(cls, data)


@dataclass(frozen=True)
class Keep:
    start: float  # snapped source seconds
    end: float
    start_frame: int | None  # None with no video
    end_frame: int | None  # exclusive
    start_sample: int  # at the processed audio stream rate
    end_sample: int  # exclusive
    out_start: float


@dataclass(frozen=True)
class EffectiveTimeline:
    keeps: tuple[Keep, ...]
    removed: tuple[tuple[float, float], ...]
    fps: Fraction | None
    sample_rate: int
    duration_out: float


@dataclass(frozen=True)
class VideoTarget:
    codec: str  # h264 | hevc
    bit_depth: int  # 8 | 10
    pix_fmt: str
    color: ColorTags
    quality: str  # match | high | small
    source_bitrate: int | None
    width: int
    height: int
    fps: Fraction
    container: str  # mp4 | mov


@dataclass(frozen=True)
class AudioTarget:
    codec: str  # aac | pcm_s24le
    bitrate: int | None  # bit/s; None for PCM
    sample_rate: int
    channels: int
    container: str  # mp4 | mov | m4a | wav


@dataclass(frozen=True)
class Capabilities:
    ffmpeg_version: str
    encoders: frozenset[str]
    hw_encode_ok: frozenset[str]  # "<encoder>:<bit depth>", e.g. "hevc_videotoolbox:10"
    gpu_driver: str | None


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_gb: float | None
    driver: str | None


@dataclass(frozen=True)
class HardwareInfo:
    os: str  # darwin | linux | windows
    arch: str  # arm64 | x86_64
    chip: str | None
    ram_gb: float | None
    cpu_cores: int
    apple_silicon: bool
    gpus: tuple[GpuInfo, ...]


@dataclass(frozen=True)
class VideoSummary:
    codec: str | None
    pix_fmt: str | None
    bit_depth: int | None
    trc: str | None
    fps: str | None
    vfr: bool


@dataclass(frozen=True)
class ClipMeasurement:
    sample: str  # file name only
    window_start: float
    window_seconds: float
    has_audio: bool
    noise_floor_db: float | None
    integrated_lufs: float | None
    true_peak_db: float | None
    lra: float | None
    clipping: bool
    peak_count: int
    audio: AudioStream | None
    video: VideoSummary | None


@dataclass(frozen=True)
class RigAnswers:
    mic: str  # built-in | usb | lavalier | shotgun | xlr-interface
    room: str  # treated | normal | echo | outdoor
    loudness: int | None  # -14 | -16 | -23 | None
    destination: str  # publish | nle
    nle_format: str | None  # edl | fcpxml | None
    log_profile: bool


@dataclass(frozen=True)
class RigProfile:
    schema_version: int
    name: str
    created: str
    answers: RigAnswers
    measured: ClipMeasurement | None
    hardware: HardwareInfo
    tier: Tier
    tier_reasons: tuple[str, ...]
    transcribe: Mapping[str, Any]  # keys per the rig schema in this spec
    audio: Mapping[str, Any]
    video: Mapping[str, Any]
    output: Mapping[str, Any]
    detect: Mapping[str, Any]
    review: Mapping[str, Any]  # {"required": True} after normalization

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RigProfile:
        return _from_dict(cls, data)


@dataclass(frozen=True)
class ResolvedOptions:
    rig_name: str | None
    tier: Tier | None
    transcribe: TranscribeOptions
    backend: str  # auto | mlx | faster | fake
    enhancer: str  # none | auto | afftdn | deepfilter
    enhancer_strength: float
    eq_chain: str
    loudness: Loudness | None
    crossfade_ms: float
    video_codec: str  # match | h264 | hevc
    quality: str  # match | high | small
    encoder: str  # auto | software
    allow_8bit: bool
    audio_bitrate: int
    output_mode: OutputMode
    nle_format: NleFormat | None
    sidecar_audio: bool
    margin_ms: float
    min_confidence: float
    gap_noise_db: float
    max_gap_ms: float
    min_silence_ms: float
    retakes: bool
    origins: Mapping[str, str]  # field -> "cli" | "rig" | "config" | "builtin"


BUILTIN_DEFAULTS: Mapping[str, Any] = {
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


@dataclass(frozen=True)
class RenderStep:
    name: str  # audio-<n> | enhance | loudness-measure | audio-final |
    # video-conform | video-batch-NN | video-join | mux | guard
    kind: str  # ffmpeg | enhance | guard
    argv: tuple[str, ...]  # full argv for ffmpeg; (name, strength) for enhance
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    expected_duration: float | None


@dataclass(frozen=True)
class RenderPlan:
    source: Path
    output: Path
    temp_dir: Path
    timeline: EffectiveTimeline
    video: VideoTarget | None
    audio: AudioTarget
    encoder: str
    loudness: Loudness | None
    steps: tuple[RenderStep, ...]


@dataclass(frozen=True)
class RenderResult:
    output: Path
    duration: float
    video_codec: str | None
    pix_fmt: str | None
    profile: str | None
    bit_depth: int | None
    audio_channels: int
    integrated_lufs: float | None
    encoder: str
    elapsed: float


@dataclass(frozen=True)
class ExportFile:
    path: Path
    kind: str  # edl | fcpxml | srt | markers | wav
    text: str | None  # rendered at plan time; None for wav


@dataclass(frozen=True)
class ExportPlan:
    format: ExportFormat
    source: Path
    timeline: EffectiveTimeline
    files: tuple[ExportFile, ...]


@dataclass(frozen=True)
class ExportResult:
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ReviewResult:
    passed: bool
    edit_list_path: Path
    content_sha256: str | None
    unresolved_count: int
    url: str


class Transcriber(Protocol):
    name: str  # mlx | faster | fake

    def available(self) -> tuple[bool, str]: ...
    def transcribe(self, wav: Path, opts: TranscribeOptions) -> Transcript: ...


@dataclass(frozen=True)
class BackendSelection:
    transcriber: Transcriber
    device: str
    compute_type: str


class Encoder(Protocol):
    name: str  # videotoolbox | nvenc | software

    def supports(self, target: VideoTarget, caps: Capabilities) -> bool: ...
    def video_args(self, target: VideoTarget) -> list[str]: ...
    def audio_args(self, target: AudioTarget) -> list[str]: ...


class Enhancer(Protocol):
    name: str  # none | afftdn | deepfilter

    def available(self) -> tuple[bool, str]: ...
    def enhance(self, wav_in: Path, wav_out: Path, strength: float) -> None: ...


_TIME_FIELDS = frozenset(
    {
        "start",
        "end",
        "start_time",
        "duration",
        "window_start",
        "window_seconds",
        "out_start",
        "duration_out",
        "elapsed",
    }
)


def _serialize(value: Any, field_name: str = "") -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            f.name: _serialize(getattr(value, f.name), f.name) for f in fields(value)
        }
    if isinstance(value, MappingABC):
        return {key: _serialize(item, key) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item, field_name) for item in value]
    if field_name in _TIME_FIELDS and isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def _to_dict(value: Any) -> dict[str, Any]:
    return _serialize(value)


def _deserialize(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in args:
            return None
        for choice in args:
            if choice is type(None):
                continue
            try:
                return _deserialize(choice, value)
            except (ValueError, TypeError, KeyError):
                pass
        raise ValueError(f"Invalid value for {annotation}: {value!r}")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list")
        if args[-1] is Ellipsis:
            return tuple(_deserialize(args[0], item) for item in value)
        if len(args) != len(value):
            raise ValueError("incorrect tuple length")
        return tuple(_deserialize(t, item) for t, item in zip(args, value))
    if origin in (dict, MappingABC):
        if not isinstance(value, MappingABC):
            raise TypeError("expected an object")
        return {
            _deserialize(args[0], key): _deserialize(args[1], item)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation(value)
    if annotation is Fraction:
        if not isinstance(value, str):
            raise TypeError("expected a rational string")
        return Fraction(value)
    if annotation is Path:
        return Path(value)
    if is_dataclass(annotation):
        if not isinstance(value, MappingABC):
            raise TypeError(f"expected an object for {annotation.__name__}")
        hints = get_type_hints(annotation)
        return annotation(
            **{
                f.name: _deserialize(hints[f.name], value[f.name])
                for f in fields(annotation)
            }
        )
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected a number")
        import math

        if not math.isfinite(value):
            raise ValueError("expected a finite number")
        return float(value)
    if annotation in (str, int, bool):
        if type(value) is not annotation:
            raise TypeError(f"expected {annotation.__name__}")
        return value
    return value


def _from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    try:
        return _deserialize(cls, data)
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        if cls is EditList:
            raise EditListError(f"Invalid edit list: {exc}") from exc
        raise ValueError(f"Invalid {cls.__name__}: {exc}") from exc
