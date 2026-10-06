"""Non-destructive edit lists, human review gates, and one effective timeline."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

from declip import __version__
from declip.contracts import (
    SCHEMA_VERSION,
    Cut,
    CutKind,
    CutOrigin,
    CutProposal,
    CutStatus,
    DeclipError,
    EditList,
    EditListError,
    EffectiveTimeline,
    FilterChainRejected,
    Keep,
    Loudness,
    MediaInfo,
    OutputMode,
    OutputSpec,
    Processing,
    Review,
    ReviewRequired,
    ReviewState,
    RevisionConflict,
    RigRef,
    Segment,
    SourceMismatch,
    SourceRef,
    Transcript,
    Word,
)
from declip.fsutil import atomic_write_json
from declip.media import file_hash, probe_media

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_PENDING = Review(ReviewState.PENDING, None, None, None)


def _timestamp(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate(edit_list: EditList) -> None:
    if edit_list.schema_version != SCHEMA_VERSION:
        raise EditListError(
            "Unsupported edit-list schema; use declip migrate for schema 2"
        )
    media = edit_list.media
    if not _finite(media.duration) or media.duration <= 0:
        raise EditListError("media.duration must be finite and positive")
    if media.has_video and (media.fps is None or media.fps <= 0):
        raise EditListError("Video needs a positive frame rate")
    if not media.has_video and media.fps is not None:
        raise EditListError("Audio-only media must have fps=null")
    if edit_list.review.mode not in (None, "interactive"):
        raise EditListError(
            "review.mode must be null or interactive; review has no bypass"
        )
    if edit_list.review.state not in (ReviewState.PENDING, ReviewState.PASSED):
        raise EditListError("Invalid review state")
    seen = set()
    for cut in edit_list.cuts:
        if not cut.id or cut.id in seen:
            raise EditListError(f"Duplicate or empty cut id: {cut.id}")
        seen.add(cut.id)
        if not (_finite(cut.start) and _finite(cut.end) and cut.start < cut.end):
            raise EditListError(f"Invalid cut times: {cut.id}")
        if not _finite(cut.confidence) or not 0 <= cut.confidence <= 1:
            raise EditListError(f"Invalid cut confidence: {cut.id}")
        if not isinstance(cut.low_confidence, bool):
            raise EditListError(f"low_confidence must be a Boolean: {cut.id}")
        if (
            cut.kind not in CutKind
            or cut.origin not in CutOrigin
            or cut.status not in CutStatus
        ):
            raise EditListError(f"Invalid cut enum: {cut.id}")
        if (cut.kind == CutKind.MANUAL) != (cut.origin == CutOrigin.MANUAL):
            raise EditListError(f"Manual kind and origin must match: {cut.id}")
        if cut.words is not None:
            first, last = cut.words
            count = len(edit_list.transcript.words) if edit_list.transcript else 0
            if not 0 <= first <= last < count:
                raise EditListError(f"Invalid cut word range: {cut.id}")
    if edit_list.transcript:
        transcript = edit_list.transcript
        for i, word in enumerate(transcript.words):
            if word.i != i or not (
                _finite(word.start)
                and _finite(word.end)
                and 0 <= word.start <= word.end
            ):
                raise EditListError(
                    "Transcript words need contiguous indices and finite times"
                )
            if not _finite(word.p) or not 0 <= word.p <= 1:
                raise EditListError("Invalid word probability")
            if not 0 <= word.segment < len(transcript.segments):
                raise EditListError("Invalid word segment")
        for i, segment in enumerate(transcript.segments):
            if segment.i != i or not (
                _finite(segment.start)
                and _finite(segment.end)
                and 0 <= segment.start <= segment.end
            ):
                raise EditListError("Invalid transcript segment")
            if any(
                index < 0
                or index >= len(transcript.words)
                or transcript.words[index].segment != i
                for index in segment.word_indices
            ):
                raise EditListError("Invalid segment word indices")


def cut_id(kind: CutKind, start: float, end: float, label: str) -> str:
    raw = f"{CutKind(kind).value}:{start:.6f}:{end:.6f}:{label}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def new_edit_list(
    source: Path,
    media: MediaInfo,
    sha256: str,
    *,
    transcript: Transcript | None,
    processing: Processing,
    output: OutputSpec,
    rig: RigRef,
) -> EditList:
    source = source.resolve()
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise EditListError(f"Cannot read source {source}: {exc}") from exc
    result = EditList(
        SCHEMA_VERSION,
        __version__,
        SourceRef(str(source), source.name, size, sha256),
        media,
        rig,
        transcript,
        (),
        processing,
        output,
        _PENDING,
        ({"stage": "plan", "at": _timestamp()},),
    )
    _validate(result)
    return result


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, ValueError) as exc:
        raise EditListError(f"Cannot read edit list {path}: {exc}") from exc


def _check_source(edit_list: EditList, source: Path) -> None:
    if source.resolve() != Path(edit_list.source.path).resolve():
        raise SourceMismatch("Edit-list source path does not match input")
    try:
        sha256 = file_hash(source, refresh=True)
    except DeclipError as exc:
        raise SourceMismatch(f"Cannot verify edit-list source: {exc}") from exc
    if sha256 != edit_list.source.sha256:
        raise SourceMismatch(
            "Source hash changed; refusing stale edit list. Run declip plan --reset."
        )


def load_edit_list(path: Path, *, source: Path | None = None) -> EditList:
    data = _read_json(path)
    if data.get("schema_version") == 2:
        raise EditListError(
            f"Schema 2 requires explicit migration: declip migrate {path}"
        )
    try:
        result = EditList.from_dict(data)
        _validate(result)
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        raise EditListError(f"Invalid edit list {path}: {exc}") from exc
    if source is not None:
        _check_source(result, source)
    return result


def save_edit_list(
    path: Path,
    edit_list: EditList,
    *,
    expected_revision: str | None = None,
) -> str:
    _validate(edit_list)
    path = path.resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(path, threading.Lock())
    with lock:
        if expected_revision is not None:
            current = revision_of(path) if path.exists() else None
            if current != expected_revision:
                raise RevisionConflict(f"Edit list changed on disk: {path}")
        atomic_write_json(path, edit_list.to_dict())
        return revision_of(path)


def revision_of(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EditListError(f"Cannot read revision of {path}: {exc}") from exc


def migrate_schema2(
    path: Path,
    *,
    source: Path | None,
    resolve_preset: Callable[[str], tuple[str, Loudness | None]],
) -> EditList:
    data = _read_json(path)
    if data.get("schema_version") != 2:
        raise EditListError("Migration requires a schema-2 edit list")
    try:
        old_source = data["source"]
        source = (source or Path(old_source["path"])).resolve()
        if not source.is_file():
            raise SourceMismatch(f"Migration source not found: {source}")
        sha256 = file_hash(source, refresh=True)
        if sha256 != old_source["sha256"]:
            raise SourceMismatch("Migration source hash differs from schema-2 source")
        media = probe_media(source)
        old_duration = old_source.get("duration_seconds")
        tolerance = float(1 / media.fps) if media.fps else 0.001
        if _finite(old_duration) and abs(old_duration - media.duration) > tolerance:
            warnings.warn(
                "Schema-2 duration differs from the fresh media probe by more than one frame",
                UserWarning,
                stacklevel=2,
            )
        words, segments = [], []
        old_transcript = data.get("transcript", {})
        for i, row in enumerate(old_transcript.get("segments", [])):
            indices = []
            for word in row.get("words", []):
                indices.append(len(words))
                words.append(
                    Word(
                        len(words),
                        float(word["start"]),
                        float(word["end"]),
                        str(word["word"]).strip(),
                        float(word.get("probability", 1)),
                        i,
                    )
                )
            segments.append(
                Segment(
                    i,
                    float(row["start"]),
                    float(row["end"]),
                    row.get("text", ""),
                    tuple(indices),
                )
            )
        transcript = Transcript(
            "mlx",
            old_transcript.get("model", ""),
            "en",
            None,
            tuple(words),
            tuple(segments),
        )
        cuts = []
        for row in data.get("cuts", []):
            kind = CutKind(row["kind"])
            inside = [
                w.i for w in words if w.start >= row["start"] and w.end <= row["end"]
            ]
            cuts.append(
                Cut(
                    row["id"],
                    kind,
                    float(row["start"]),
                    float(row["end"]),
                    row["label"],
                    float(row["confidence"]),
                    row["confidence"] < 0.5,
                    (inside[0], inside[-1])
                    if inside and kind not in (CutKind.GAP, CutKind.MANUAL)
                    else None,
                    CutOrigin.MANUAL if kind == CutKind.MANUAL else CutOrigin.AUTO,
                    CutStatus(
                        "rejected" if row["status"] == "vetoed" else row["status"]
                    ),
                )
            )
        old_processing = data.get("processing", {})
        eq = old_processing.get("eq", {})
        preset_name = eq.get("preset", "raw")
        try:
            chain, loudness = resolve_preset(preset_name)
        except FilterChainRejected:
            raise
        except DeclipError:
            warnings.warn(
                f"Unknown migrated preset {preset_name!r}; using raw loudness",
                UserWarning,
                stacklevel=2,
            )
            chain, loudness = "", Loudness(-16, -1.5, 11)
        sound = old_processing.get("sound", {})
        enhancer = (
            "deepfilter"
            if sound.get("enabled")
            and str(sound.get("engine", "")).startswith("DeepFilterNet")
            else "none"
        )
        history = tuple(dict(stage) for stage in data.get("stages", [])) + (
            {
                "stage": "migrate-dropped",
                "lut": eq.get("lut"),
                "captions": old_processing.get("captions"),
            },
            {
                "stage": "migrate",
                "at": _timestamp(),
                "from_schema": 2,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            },
        )
        result = EditList(
            SCHEMA_VERSION,
            __version__,
            SourceRef(str(source), source.name, source.stat().st_size, sha256),
            media,
            RigRef(None, None, {}),
            transcript,
            tuple(cuts),
            Processing(enhancer, 0.5, chain, loudness, 20),
            OutputSpec(OutputMode.RENDER, None, False),
            _PENDING,
            history,
        )
        _validate(result)
        return result
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise EditListError(f"Invalid schema-2 edit list {path}: {exc}") from exc


def migration_destination(path: Path) -> Path:
    name = path.name
    return path.with_name(
        name[:-10] + ".declip.json"
        if name.endswith(".edit.json")
        else name + ".schema3.json"
    )


def replace_stage_cuts(
    edit_list: EditList,
    kind: CutKind,
    proposals: Sequence[CutProposal],
) -> EditList:
    kind = CutKind(kind)
    if kind == CutKind.MANUAL:
        raise EditListError("Plan cannot replace manual cuts")
    retained = [
        c for c in edit_list.cuts if c.origin == CutOrigin.MANUAL or c.kind != kind
    ]
    previous = {
        c.id: c.status
        for c in edit_list.cuts
        if c.origin == CutOrigin.AUTO and c.kind == kind
    }
    rows = []
    for proposal in proposals:
        if proposal.kind != kind:
            raise EditListError("Proposal kind differs from stage")
        start, end = round(proposal.start, 6), round(proposal.end, 6)
        identifier = cut_id(kind, start, end, proposal.label)
        rows.append(
            Cut(
                identifier,
                kind,
                start,
                end,
                proposal.label,
                proposal.confidence,
                proposal.low_confidence,
                proposal.word_indices,
                CutOrigin.AUTO,
                previous.get(identifier, CutStatus.PROPOSED),
            )
        )
    result = replace(
        edit_list,
        cuts=tuple(sorted(retained + rows, key=lambda c: (c.start, c.end))),
        history=edit_list.history
        + ({"stage": kind.value, "at": _timestamp(), "proposed": len(rows)},),
    )
    _validate(result)
    if (
        result.review.state != ReviewState.PASSED
        or result.review.content_sha256 != review_content_hash(result)
    ):
        result = replace(result, review=_PENDING)
    return result


def reset_decisions(edit_list: EditList) -> EditList:
    return replace(
        edit_list,
        cuts=tuple(
            replace(c, status=CutStatus.PROPOSED)
            for c in edit_list.cuts
            if c.origin == CutOrigin.AUTO
        ),
        review=_PENDING,
    )


def apply_decisions(
    edit_list: EditList,
    *,
    decisions: Mapping[str, CutStatus],
    add_manual: Sequence[tuple[float, float, str]],
    remove_manual: Sequence[str],
) -> EditList:
    _validate(edit_list)
    lookup = {c.id: c for c in edit_list.cuts}
    statuses = {}
    for identifier, status in decisions.items():
        if identifier not in lookup or lookup[identifier].origin != CutOrigin.AUTO:
            raise EditListError(f"Decisions require an existing auto cut: {identifier}")
        try:
            statuses[identifier] = CutStatus(status)
        except ValueError as exc:
            raise EditListError(f"Invalid cut status: {status}") from exc
    for identifier in remove_manual:
        if identifier not in lookup or lookup[identifier].origin != CutOrigin.MANUAL:
            raise EditListError(
                f"Removal requires an existing manual cut: {identifier}"
            )
    rows = [
        replace(c, status=statuses.get(c.id, c.status))
        for c in edit_list.cuts
        if c.id not in remove_manual
    ]
    minimum = 1 / edit_list.media.fps if edit_list.media.has_video else Fraction(1, 100)
    for start, end, label in add_manual:
        if not (
            _finite(start)
            and _finite(end)
            and 0 <= start < end <= edit_list.media.duration
        ):
            raise EditListError(
                "Manual times must be finite and satisfy 0 <= start < end <= duration"
            )
        length = float(Fraction(str(end)) - Fraction(str(start)))
        if length < float(minimum) and not math.isclose(
            length, float(minimum), abs_tol=1e-12, rel_tol=0
        ):
            raise EditListError(
                "Manual cut needs at least one frame (or 10 ms for audio)"
            )
        if not isinstance(label, str) or len(label) > 200:
            raise EditListError("Manual label must be text of at most 200 characters")
        start, end = round(start, 6), round(end, 6)
        identifier = cut_id(CutKind.MANUAL, start, end, label)
        if identifier in lookup:
            raise EditListError(f"Duplicate cut id: {identifier}")
        row = Cut(
            identifier,
            CutKind.MANUAL,
            start,
            end,
            label,
            1.0,
            False,
            None,
            CutOrigin.MANUAL,
            CutStatus.ACCEPTED,
        )
        lookup[identifier] = row
        rows.append(row)
    result = replace(edit_list, cuts=tuple(rows))
    if result.cuts != edit_list.cuts:
        result = replace(result, review=_PENDING)
    _validate(result)
    return result


def review_content_hash(edit_list: EditList) -> str:
    payload = {
        "source_sha256": edit_list.source.sha256,
        "cuts": [
            {key: row[key] for key in ("id", "kind", "start", "end", "status")}
            for row in edit_list.to_dict()["cuts"]
        ],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mark_review_passed(edit_list: EditList, *, now: datetime) -> EditList:
    _validate(edit_list)
    if any(c.status == CutStatus.PROPOSED for c in edit_list.cuts):
        raise EditListError("Review has unresolved proposed cuts")
    return replace(
        edit_list,
        review=Review(
            ReviewState.PASSED,
            "interactive",
            _timestamp(now),
            review_content_hash(edit_list),
        ),
    )


def render_allowed(edit_list: EditList, source_sha256: str) -> bool:
    return (
        edit_list.review.state == ReviewState.PASSED
        and edit_list.review.content_sha256 == review_content_hash(edit_list)
        and edit_list.source.sha256 == source_sha256
    )


def require_current_review(edit_list: EditList, source: Path) -> None:
    _check_source(edit_list, source)
    if not render_allowed(edit_list, edit_list.source.sha256):
        raise ReviewRequired(f"Current review required: declip review {source}")


def _merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def accepted_intervals(edit_list: EditList) -> list[tuple[float, float]]:
    duration = edit_list.media.duration
    return _merge(
        (max(0.0, min(duration, c.start)), max(0.0, min(duration, c.end)))
        for c in edit_list.cuts
        if c.status == CutStatus.ACCEPTED
    )


def _round_grid(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def effective_timeline(edit_list: EditList) -> EffectiveTimeline:
    media = edit_list.media
    audio = next((a for a in media.audio_streams if a.index == media.audio_index), None)
    sample_rate = (audio.sample_rate if audio else None) or 48000
    fps = media.fps if media.has_video else None
    if media.has_video and (fps is None or fps <= 0):
        raise EditListError("Video needs a positive frame rate")
    if sample_rate <= 0:
        raise EditListError("Audio needs a positive sample rate")
    grid = fps or Fraction(sample_rate)
    total = _round_grid(Fraction(str(media.duration)) * grid)
    cuts = _merge(
        (
            _round_grid(Fraction(str(start)) * grid),
            _round_grid(Fraction(str(end)) * grid),
        )
        for start, end in accepted_intervals(edit_list)
    )
    keeps, cursor, output_units = [], 0, 0
    for start, end in [*cuts, (total, total)]:
        if start > cursor:
            start_sample = _round_grid(Fraction(cursor * sample_rate, 1) / grid)
            end_sample = _round_grid(Fraction(start * sample_rate, 1) / grid)
            keeps.append(
                Keep(
                    float(cursor / grid),
                    float(start / grid),
                    cursor if fps else None,
                    start if fps else None,
                    start_sample,
                    end_sample,
                    float(output_units / grid),
                )
            )
            output_units += start - cursor
        cursor = max(cursor, end)
    return EffectiveTimeline(
        tuple(keeps),
        tuple((float(a / grid), float(b / grid)) for a, b in cuts),
        fps,
        sample_rate,
        float(output_units / grid),
    )


def keep_intervals(edit_list: EditList) -> list[tuple[float, float]]:
    return [(k.start, k.end) for k in effective_timeline(edit_list).keeps]


def remap_time(seconds: float, timeline: EffectiveTimeline) -> float | None:
    if not _finite(seconds) or seconds < 0:
        return None
    if any(start <= seconds < end for start, end in timeline.removed):
        return None
    for keep in timeline.keeps:
        if keep.start <= seconds < keep.end:
            return keep.out_start + seconds - keep.start
    # The terminal boundary maps only when the last keep reaches it.
    if timeline.keeps and seconds == timeline.keeps[-1].end:
        return timeline.duration_out
    return None
