"""Edit-list transactions, migration, human decisions, and frame/sample grids."""

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from declip import editlist as ed
from declip.contracts import (
    AudioStream,
    ColorTags,
    Cut,
    CutKind,
    CutOrigin,
    CutProposal,
    CutStatus,
    DeclipError,
    EditList,
    EditListError,
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
    SourceMismatch,
)
from declip.media import file_hash

pytestmark = pytest.mark.contract
NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    media = MediaInfo(
        10.0,
        0.0,
        "mov",
        True,
        0,
        1,
        Fraction(30),
        False,
        320,
        240,
        0,
        "h264",
        "yuv420p",
        8,
        2000000,
        ColorTags(None, None, None, None),
        None,
        (AudioStream(1, "aac", 48000, 2, "stereo", None, None),),
        (),
    )
    return ed.new_edit_list(
        source,
        media,
        file_hash(source),
        transcript=None,
        processing=Processing("none", 0.5, "", Loudness(-16, -1.5, 11), 20),
        output=OutputSpec(OutputMode.RENDER, None, False),
        rig=RigRef(None, None, {}),
    )


def proposal(start=1.0, end=2.0, kind=CutKind.FILLER, label="um"):
    return CutProposal(kind, start, end, label, 0.9, None, False)


def accepted(plan, ranges):
    rows = tuple(
        Cut(
            ed.cut_id(CutKind.GAP, a, b, "silence"),
            CutKind.GAP,
            a,
            b,
            "silence",
            1.0,
            False,
            None,
            CutOrigin.AUTO,
            CutStatus.ACCEPTED,
        )
        for a, b in ranges
    )
    return replace(plan, cuts=rows)


def test_cut_id_uses_enum_value_and_six_decimals():
    expected = hashlib.sha256(b"filler:1.000000:2.000000:um").hexdigest()[:16]
    assert ed.cut_id(CutKind.FILLER, 1, 2, "um") == expected
    assert ed.cut_id(CutKind.FILLER, 1.0000001, 2, "um") == expected


def test_schema3_round_trip_new_plan_and_revision(plan, tmp_path):
    assert plan.schema_version == 3 and plan.declip_version == "0.5.0"
    assert plan.source.name == "source.mp4" and plan.source.size == 16
    assert plan.review == Review(ReviewState.PENDING, None, None, None)
    assert plan.history[0]["stage"] == "plan"
    assert EditList.from_dict(plan.to_dict()) == plan
    path = tmp_path / "source.declip.json"
    revision = ed.save_edit_list(path, plan)
    assert (
        revision
        == hashlib.sha256(path.read_bytes()).hexdigest()
        == ed.revision_of(path)
    )
    assert ed.load_edit_list(path, source=Path(plan.source.path)) == plan
    ed.save_edit_list(path, plan, expected_revision=revision)
    path.write_text(path.read_text() + " ")
    with pytest.raises(RevisionConflict):
        ed.save_edit_list(path, plan, expected_revision=revision)


def test_concurrent_compare_and_save_allows_one_winner(plan, tmp_path):
    path = tmp_path / "edit.json"
    revision = ed.save_edit_list(path, plan)

    def attempt(i):
        try:
            ed.save_edit_list(
                path,
                replace(plan, history=({"stage": str(i)},)),
                expected_revision=revision,
            )
            return "saved"
        except RevisionConflict:
            return "stale"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert outcomes.count("saved") == 1 and outcomes.count("stale") == 7
    ed.load_edit_list(path)


@pytest.mark.parametrize(
    "bad",
    [
        {"schema_version": 2},
        [],
        {"schema_version": 9},
    ],
)
def test_load_rejects_invalid_or_old_schema(tmp_path, bad):
    path = tmp_path / "edit.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(EditListError) as error:
        ed.load_edit_list(path)
    if isinstance(bad, dict) and bad.get("schema_version") == 2:
        assert "declip migrate" in str(error.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("start", float("nan")),
        ("end", 0.5),
        ("status", "vetoed"),
        ("low_confidence", "false"),
    ],
)
def test_load_validates_cut_data(plan, tmp_path, field, value):
    data = accepted(plan, [(1, 2)]).to_dict()
    data["cuts"][0][field] = value
    path = tmp_path / "edit.json"
    path.write_text(json.dumps(data))
    with pytest.raises(EditListError):
        ed.load_edit_list(path)


def test_duplicate_cut_id_rejected(plan, tmp_path):
    data = accepted(plan, [(1, 2)]).to_dict()
    data["cuts"] *= 2
    path = tmp_path / "edit.json"
    path.write_text(json.dumps(data))
    with pytest.raises(EditListError, match="Duplicate"):
        ed.load_edit_list(path)


def test_stage_replacement_is_idempotent_and_carries_status(plan):
    proposals = [proposal()]
    result = ed.replace_stage_cuts(plan, CutKind.FILLER, proposals)
    assert result.cuts[0].status == CutStatus.PROPOSED
    decided = ed.apply_decisions(
        result,
        decisions={result.cuts[0].id: CutStatus.REJECTED},
        add_manual=[(3, 4, "manual")],
        remove_manual=[],
    )
    passed = ed.mark_review_passed(decided, now=NOW)
    repeated = ed.replace_stage_cuts(passed, CutKind.FILLER, proposals)
    assert repeated.cuts == passed.cuts
    assert repeated.review == passed.review
    assert result.cuts[0].status == CutStatus.PROPOSED  # Pure input preserved.
    changed = ed.replace_stage_cuts(passed, CutKind.FILLER, [proposal(1, 2.1)])
    assert changed.review.state == ReviewState.PENDING
    assert changed.cuts[0].status == CutStatus.PROPOSED
    assert changed.cuts[1] == passed.cuts[1]


def test_stage_scope_and_reset(plan):
    filled = ed.replace_stage_cuts(plan, CutKind.FILLER, [proposal()])
    gaps = ed.replace_stage_cuts(filled, CutKind.GAP, [proposal(4, 5, CutKind.GAP)])
    decided = ed.apply_decisions(
        gaps,
        decisions={c.id: CutStatus.ACCEPTED for c in gaps.cuts},
        add_manual=[(6, 7, "manual")],
        remove_manual=[],
    )
    empty_stage = ed.replace_stage_cuts(decided, CutKind.FILLER, [])
    assert [c.kind for c in empty_stage.cuts] == [CutKind.GAP, CutKind.MANUAL]
    reset = ed.reset_decisions(ed.mark_review_passed(decided, now=NOW))
    assert all(
        c.status == CutStatus.PROPOSED and c.origin == CutOrigin.AUTO
        for c in reset.cuts
    )
    assert len(reset.cuts) == 2 and reset.review.state == ReviewState.PENDING


def test_review_hash_invalidates_after_cut_change(plan):
    original = accepted(plan, [(1, 2)])
    payload = {
        "source_sha256": original.source.sha256,
        "cuts": [
            {
                "id": original.cuts[0].id,
                "kind": "gap",
                "start": 1.0,
                "end": 2.0,
                "status": "accepted",
            }
        ],
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert ed.review_content_hash(original) == expected
    passed = ed.mark_review_passed(original, now=NOW)
    changed = ed.apply_decisions(
        passed,
        decisions={original.cuts[0].id: CutStatus.REJECTED},
        add_manual=[],
        remove_manual=[],
    )
    assert ed.review_content_hash(changed) != expected
    assert changed.review == Review(ReviewState.PENDING, None, None, None)
    assert passed.review.passed_at == "2026-10-06T12:00:00Z"


@pytest.mark.parametrize("state", [ReviewState.PENDING, ReviewState.PASSED])
@pytest.mark.parametrize("current_hash", [False, True])
@pytest.mark.parametrize("current_source", [False, True])
def test_render_allowed_only_when_passed_and_current(
    plan, state, current_hash, current_source
):
    content_hash = ed.review_content_hash(plan) if current_hash else "stale"
    result = replace(plan, review=Review(state, "interactive", "now", content_hash))
    source_hash = plan.source.sha256 if current_source else "other-source"
    assert ed.render_allowed(result, source_hash) == (
        state == ReviewState.PASSED and current_hash and current_source
    )


def test_require_current_review_refreshes_hash(plan, tmp_path):
    source = Path(plan.source.path)
    with pytest.raises(ReviewRequired, match="declip review"):
        ed.require_current_review(plan, source)
    passed = ed.mark_review_passed(plan, now=NOW)
    ed.require_current_review(passed, source)
    stat = source.stat()
    source.write_bytes(b"changed source!!")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert (
        file_hash(source) == plan.source.sha256
    )  # Cache alone would miss this change.
    with pytest.raises(SourceMismatch, match="hash changed"):
        ed.require_current_review(passed, source)
    path = tmp_path / "edit.json"
    ed.save_edit_list(path, passed)
    with pytest.raises(SourceMismatch):
        ed.load_edit_list(path, source=source)


def test_source_path_and_missing_source_are_refused(plan, tmp_path):
    other = tmp_path / "other.mp4"
    other.write_bytes(Path(plan.source.path).read_bytes())
    with pytest.raises(SourceMismatch, match="path"):
        ed.require_current_review(plan, other)
    Path(plan.source.path).unlink()
    with pytest.raises(SourceMismatch, match="verify"):
        ed.require_current_review(plan, Path(plan.source.path))


def test_finish_requires_all_decisions_but_zero_cuts_can_pass(plan):
    proposed = ed.replace_stage_cuts(plan, CutKind.FILLER, [proposal()])
    with pytest.raises(EditListError, match="unresolved"):
        ed.mark_review_passed(proposed, now=NOW)
    assert ed.render_allowed(ed.mark_review_passed(plan, now=NOW), plan.source.sha256)


def test_manual_add_remove_and_decision_validation(plan):
    added = ed.apply_decisions(
        plan, decisions={}, add_manual=[(1, 2, "trim")], remove_manual=[]
    )
    cut = added.cuts[0]
    assert (cut.kind, cut.origin, cut.status) == (
        CutKind.MANUAL,
        CutOrigin.MANUAL,
        CutStatus.ACCEPTED,
    )
    assert cut.confidence == 1 and not cut.low_confidence and cut.words is None
    passed = ed.mark_review_passed(added, now=NOW)
    unchanged = ed.apply_decisions(
        passed, decisions={}, add_manual=[], remove_manual=[]
    )
    assert unchanged.review == passed.review
    for identifier in (cut.id, "unknown"):
        with pytest.raises(EditListError):
            ed.apply_decisions(
                added,
                decisions={identifier: CutStatus.REJECTED},
                add_manual=[],
                remove_manual=[],
            )
    with pytest.raises(EditListError, match="Duplicate"):
        ed.apply_decisions(
            added, decisions={}, add_manual=[(1, 2, "trim")], remove_manual=[]
        )
    removed = ed.apply_decisions(
        passed, decisions={}, add_manual=[], remove_manual=[cut.id]
    )
    assert not removed.cuts and removed.review.state == ReviewState.PENDING
    auto = accepted(plan, [(1, 2)])
    with pytest.raises(EditListError):
        ed.apply_decisions(
            auto, decisions={}, add_manual=[], remove_manual=[auto.cuts[0].id]
        )


@pytest.mark.parametrize(
    "start,end,label",
    [
        (-1, 1, "bad"),
        (2, 1, "bad"),
        (1, 1, "bad"),
        (0, 11, "bad"),
        (float("inf"), 2, "bad"),
        (0, float("nan"), "bad"),
        (1, 1.01, "short"),
        (1, 2, "x" * 201),
        (True, 2, "bad"),
    ],
)
def test_manual_cut_constraints(plan, start, end, label):
    with pytest.raises(EditListError):
        ed.apply_decisions(
            plan, decisions={}, add_manual=[(start, end, label)], remove_manual=[]
        )


def test_keep_intervals_merge_accepted_cuts_and_remap(plan):
    result = accepted(plan, [(1, 2), (1.5, 3), (3, 4), (-1, 0.2), (9.9, 11)])
    assert ed.accepted_intervals(result) == [(0, 0.2), (1, 4), (9.9, 10)]
    timeline = ed.effective_timeline(result)
    assert ed.keep_intervals(result) == [(0.2, 1), (4, 9.9)]
    assert ed.remap_time(0.1, timeline) is None
    assert ed.remap_time(1, timeline) is None
    assert ed.remap_time(4, timeline) == pytest.approx(0.8)
    assert ed.remap_time(5, timeline) == pytest.approx(1.8)
    assert ed.remap_time(9.9, timeline) is None
    assert (
        ed.remap_time(10, timeline) is None
    )  # Trailing cut, terminal time is removed.
    assert ed.remap_time(-1, timeline) is None
    assert ed.remap_time(float("nan"), timeline) is None


def test_sub_40ms_keep_stays_and_zero_frame_cut_disappears(plan):
    result = accepted(plan, [(0, 1), (1 + 1 / 30, 10)])
    timeline = ed.effective_timeline(result)
    assert len(timeline.keeps) == 1
    assert timeline.keeps[0].end_frame - timeline.keeps[0].start_frame == 1
    assert timeline.duration_out == pytest.approx(1 / 30)
    tiny = ed.effective_timeline(accepted(plan, [(1.001, 1.002)]))
    assert not tiny.removed and tiny.duration_out == 10


def test_non_frame_aligned_250_keep_timeline_uses_absolute_samples(plan):
    fps = Fraction(30000, 1001)
    plan = replace(
        plan, media=replace(plan.media, fps=fps, duration=float(10000 / fps))
    )
    # Keep exactly 20 frames in each 40-frame block. Cuts land 0.1 frame from the grid.
    ranges = [
        (
            float((40 * i + 20 + Fraction(1, 10)) / fps),
            float((40 * i + 40 - Fraction(1, 10)) / fps),
        )
        for i in range(250)
    ]
    timeline = ed.effective_timeline(accepted(plan, ranges))
    assert len(timeline.keeps) == 250
    for i, keep in enumerate(timeline.keeps):
        assert (keep.start_frame, keep.end_frame) == (40 * i, 40 * i + 20)
        expected_start = math.floor(Fraction(40 * i * 48000, 1) / fps + Fraction(1, 2))
        expected_end = math.floor(
            Fraction((40 * i + 20) * 48000, 1) / fps + Fraction(1, 2)
        )
        assert (keep.start_sample, keep.end_sample) == (expected_start, expected_end)
        assert keep.out_start == pytest.approx(float(20 * i / fps))
        assert ed.remap_time(keep.start, timeline) == pytest.approx(keep.out_start)
    assert timeline.duration_out == float(5000 / fps)
    assert timeline.keeps[40].out_start == pytest.approx(float(800 / fps))
    assert timeline.keeps[80].out_start == pytest.approx(float(1600 / fps))


def test_entire_source_removal_is_valid_review(plan):
    result = accepted(plan, [(0, 10)])
    timeline = ed.effective_timeline(result)
    assert timeline.keeps == () and timeline.duration_out == 0
    assert ed.mark_review_passed(result, now=NOW).review.state == ReviewState.PASSED
    assert ed.remap_time(0, timeline) is None


def test_audio_sample_grid_uses_selected_track_and_one_sample_keeps(plan):
    plan = replace(
        plan,
        media=replace(
            plan.media,
            has_video=False,
            video_index=None,
            fps=None,
            audio_index=3,
            audio_streams=(
                AudioStream(1, "pcm", 48000, 1, "mono", 16, None),
                AudioStream(3, "pcm", 44100, 2, "stereo", 24, None),
            ),
        ),
    )
    timeline = ed.effective_timeline(accepted(plan, [(0, 1), (1 + 1 / 44100, 10)]))
    assert timeline.sample_rate == 44100 and timeline.fps is None
    keep = timeline.keeps[0]
    assert (keep.start_sample, keep.end_sample) == (44100, 44101)
    assert keep.start_frame is None and keep.end_frame is None
    assert timeline.duration_out == float(Fraction(1, 44100))
    with pytest.raises(EditListError, match="10 ms"):
        ed.apply_decisions(
            plan, decisions={}, add_manual=[(1, 1.009, "tiny")], remove_manual=[]
        )
    ed.apply_decisions(
        plan, decisions={}, add_manual=[(1, 1.01, "minimum")], remove_manual=[]
    )


def test_empty_plan_terminal_boundary_and_duration_snapping(plan):
    timeline = ed.effective_timeline(plan)
    assert ed.remap_time(10, timeline) == 10
    assert ed.remap_time(10.1, timeline) is None
    half = replace(plan, media=replace(plan.media, fps=Fraction(2), duration=1.25))
    assert (
        ed.effective_timeline(half).keeps[0].end_frame == 3
    )  # Half-up, never banker's rounding.


@pytest.fixture
def schema2(plan, tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures/editlists/schema2_sample.json"
    data = json.loads(fixture.read_text())
    data["source"].update(path=plan.source.path, sha256=plan.source.sha256)
    path = tmp_path / "old.edit.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(ed, "probe_media", lambda source: plan.media)
    return path, data


def test_migration_field_by_field_and_original_unchanged(schema2):
    path, data = schema2
    before = path.read_bytes()
    calls = []

    def resolve(name):
        calls.append(name)
        return "highpass=f=60:poles=2", Loudness(-19, -1.5, 14)

    result = ed.migrate_schema2(path, source=None, resolve_preset=resolve)
    assert path.read_bytes() == before
    assert not ed.migration_destination(path).exists()
    assert calls == ["natural"]
    assert result.source.path == data["source"]["path"]
    assert result.source.size == 16 and result.source.name == "source.mp4"
    assert result.schema_version == 3 and result.declip_version == "0.5.0"
    assert result.transcript.backend == "mlx" and result.transcript.language == "en"
    assert (
        result.transcript.prompt is None and result.transcript.model == "fixture-model"
    )
    assert [w.i for w in result.transcript.words] == [0, 1, 2]
    assert [w.text for w in result.transcript.words] == ["Um", "hello", "Again"]
    assert [w.p for w in result.transcript.words] == [0.2, 1.0, 0.9]
    assert result.transcript.segments[0].word_indices == (0, 1)
    assert result.transcript.words[2].segment == 1
    filler, gap, manual = result.cuts
    assert filler.id == "legacy-filler" and filler.status == CutStatus.REJECTED
    assert filler.low_confidence and filler.words == (0, 0)
    assert gap.words is None and gap.status == CutStatus.ACCEPTED
    assert manual.words is None and manual.origin == CutOrigin.MANUAL
    assert result.processing == Processing(
        "deepfilter", 0.5, "highpass=f=60:poles=2", Loudness(-19, -1.5, 14), 20
    )
    assert result.output == OutputSpec(OutputMode.RENDER, None, False)
    assert result.rig == RigRef(None, None, {})
    assert result.review == Review(ReviewState.PENDING, None, None, None)
    assert result.history[0] == data["stages"][0]
    assert result.history[1] == {
        "stage": "migrate-dropped",
        "lut": "legacy.cube",
        "captions": {"enabled": True, "ass_path": "legacy.ass"},
    }
    assert result.history[-1]["from_schema"] == 2
    assert result.history[-1]["created_at"] == data["created_at"]
    assert EditList.from_dict(result.to_dict()) == result


def test_migration_override_source_hash_and_unknown_preset(schema2, tmp_path):
    path, data = schema2
    data["source"]["path"] = "/missing/source"
    path.write_text(json.dumps(data))
    replacement = tmp_path / "override.mp4"
    replacement.write_bytes(b"synthetic source")

    def unknown(name):
        raise DeclipError("Unknown preset")

    with pytest.warns(UserWarning, match="Unknown migrated preset"):
        result = ed.migrate_schema2(path, source=replacement, resolve_preset=unknown)
    assert result.source.path == str(replacement.resolve())
    assert result.processing.eq_chain == "" and result.processing.loudness == Loudness(
        -16, -1.5, 11
    )
    with pytest.raises(SourceMismatch, match="not found"):
        ed.migrate_schema2(path, source=None, resolve_preset=unknown)
    replacement.write_bytes(b"different")
    with pytest.raises(SourceMismatch, match="hash differs"):
        ed.migrate_schema2(path, source=replacement, resolve_preset=unknown)


def test_migration_duration_warning(schema2):
    path, data = schema2
    data["source"]["duration_seconds"] = 9
    path.write_text(json.dumps(data))
    with pytest.warns(UserWarning, match="duration differs"):
        ed.migrate_schema2(path, source=None, resolve_preset=lambda name: ("", None))


@pytest.mark.parametrize(
    "name,destination",
    [
        ("clip.edit.json", "clip.declip.json"),
        ("clip.json", "clip.json.schema3.json"),
        ("x.edit.json.bak", "x.edit.json.bak.schema3.json"),
    ],
)
def test_migration_destination(name, destination):
    assert (
        ed.migration_destination(Path("folder") / name) == Path("folder") / destination
    )


def test_manual_exact_one_frame_is_accepted(plan):
    result = ed.apply_decisions(
        plan, decisions={}, add_manual=[(0, 1 / 30, "one frame")], remove_manual=[]
    )
    assert result.cuts[0].end == 0.033333


def test_migration_invalid_filter_is_not_unknown_preset_fallback(schema2):
    from declip.contracts import FilterChainRejected

    path, _ = schema2

    def rejected(name):
        raise FilterChainRejected("denied filter")

    with pytest.raises(FilterChainRejected, match="denied"):
        ed.migrate_schema2(path, source=None, resolve_preset=rejected)
