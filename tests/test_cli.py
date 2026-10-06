"""CLI workflow and compatibility checks with synthetic source text."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from declip import cli as module, config, editlist, paths
from declip.contracts import CutStatus


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def clip(tmp_path, make_media, monkeypatch):
    source = make_media(
        tmp_path,
        {
            "name": "clip.wav",
            "duration": 12,
            "inputs": ["sine=sample_rate=48000"],
            "args": ["-c:a", "pcm_s16le"],
        },
    )
    (tmp_path / "clip.json").write_text(
        json.dumps(
            {
                "language": "en",
                "segments": [
                    {
                        "start": 0,
                        "end": 12,
                        "text": "um hello uh world um end",
                        "words": [
                            {
                                "word": text,
                                "start": start,
                                "end": start + 0.2,
                                "probability": 0.9,
                            }
                            for start, text in [
                                (1, "um"),
                                (2, "hello"),
                                (3, "uh"),
                                (4, "world"),
                                (5, "um"),
                                (6, "end"),
                            ]
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DECLIP_TRANSCRIBER", "fake")
    monkeypatch.setenv("DECLIP_FAKE_TRANSCRIPT_DIR", str(tmp_path))
    return source


def invoke(runner, args):
    result = runner.invoke(module.cli, list(map(str, args)))
    assert result.exit_code == 0, (result.output, result.exception)
    return result


def planned(runner, clip):
    invoke(runner, ["plan", clip, "--stages", "fillers"])
    return clip.with_suffix(".declip.json")


@pytest.mark.parametrize(
    "command",
    [
        "setup",
        "doctor",
        "plan",
        "review",
        "render",
        "export",
        "migrate",
        "rig",
        "process",
        "clean",
        "detect",
        "transcribe",
        "enhance",
        "config",
    ],
)
def test_command_help(runner, command):
    assert "Usage:" in invoke(runner, [command, "--help"]).output


@pytest.mark.ffmpeg
@pytest.mark.parametrize("args", [["--execute", "process"], ["process"]])
def test_process_execute_stops_after_plan(runner, clip, args):
    args = [*args, clip] + (["--execute"] if args == ["process"] else [])
    before = set(clip.parent.iterdir())
    result = invoke(runner, args)
    document = editlist.load_edit_list(clip.with_suffix(".declip.json"))
    assert document.review.state == "pending"
    assert "declip review" in result.stdout
    assert {p.suffix for p in set(clip.parent.iterdir()) - before if p.is_file()} == {
        ".json"
    }


@pytest.mark.parametrize("command", ["process", "clean"])
def test_process_rejects_yes(runner, tmp_path, command):
    source = tmp_path / "clip.wav"
    source.touch()
    assert runner.invoke(module.cli, [command, str(source), "--yes"]).exit_code == 2


@pytest.mark.ffmpeg
@pytest.mark.parametrize("command", ["render", "export"])
def test_render_export_refuses_pending_review(runner, clip, command):
    planned(runner, clip)
    args = [command, str(clip)] + (
        ["--format", "markers"] if command == "export" else []
    )
    result = runner.invoke(module.cli, args)
    assert result.exception.exit_code == 3


@pytest.mark.ffmpeg
def test_plan_json_stdout_is_pure_json(runner, clip):
    result = invoke(runner, ["plan", clip, "--json"])
    assert json.loads(result.stdout)["edit_list"]["schema_version"] == 3
    assert "setup" in result.stderr
    result = invoke(runner, ["--json-output", "transcribe", clip])
    assert json.loads(result.stdout)["backend"] == "fake"


@pytest.mark.ffmpeg
@pytest.mark.parametrize(
    "value", ["2-1", "-1-2", "nan-2", "1-inf", "1-99", "1e999-2", "broken"]
)
def test_bad_cut_range(runner, clip, value):
    result = runner.invoke(module.cli, ["plan", str(clip), "--cut-range", value])
    assert result.exit_code == 2
    assert not clip.with_suffix(".declip.json").exists()


@pytest.mark.ffmpeg
def test_subcommand_wins_over_group_and_config(runner, clip):
    invoke(
        runner,
        [
            "--preset",
            "voice",
            "--margin",
            "80",
            "plan",
            clip,
            "--preset",
            "none",
            "--margin",
            "0",
            "--stages",
            "fillers",
        ],
    )
    document = editlist.load_edit_list(clip.with_suffix(".declip.json"))
    assert document.processing.eq_chain == ""
    assert document.processing.loudness is None
    assert document.cuts[0].start == 1


@pytest.mark.ffmpeg
def test_plan_rerun_carries_statuses_and_review_then_reset(runner, clip):
    path = planned(runner, clip)
    document = editlist.load_edit_list(path)
    document = editlist.apply_decisions(
        document,
        decisions={c.id: CutStatus.ACCEPTED for c in document.cuts},
        add_manual=[(8, 9, "manual")],
        remove_manual=[],
    )
    document = editlist.mark_review_passed(document, now=datetime.now(timezone.utc))
    editlist.save_edit_list(path, document)
    planned(runner, clip)
    assert editlist.load_edit_list(path).review == document.review
    invoke(runner, ["plan", clip, "--stages", "fillers", "--reset"])
    document = editlist.load_edit_list(path)
    assert all(c.status == "proposed" for c in document.cuts)
    assert all(c.kind != "manual" for c in document.cuts)
    assert document.review.state == "pending"


@pytest.mark.ffmpeg
def test_plan_source_change_requires_reset(runner, clip):
    path = planned(runner, clip)
    document = editlist.load_edit_list(path)
    editlist.save_edit_list(
        path, replace(document, source=replace(document.source, sha256="0" * 64))
    )
    result = runner.invoke(module.cli, ["plan", str(clip)])
    assert isinstance(result.exception, module.SourceMismatch)
    invoke(runner, ["plan", clip, "--reset"])


@pytest.mark.ffmpeg
def test_dry_runs_write_no_edit_list(runner, clip):
    for command in ("process", "clean", "detect"):
        invoke(runner, [command, clip])
        assert not clip.with_suffix(".declip.json").exists()


def test_unknown_saved_preset_warns_once_and_preserves_config(runner):
    path = paths.config_dir() / "config.json"
    path.write_text('{"defaults":{"preset":"missing"}}', encoding="utf-8")
    before = path.read_bytes()
    result = invoke(runner, ["config", "show"])
    assert result.stderr.count("using raw") == 1
    assert str(path) in result.stderr
    assert path.read_bytes() == before


def test_preset_add_writes_user_entries_only(runner):
    invoke(runner, ["config", "preset-add", "desk", "--chain", "highpass=f=70"])
    assert set(json.loads((paths.config_dir() / "presets.json").read_text())) == {
        "desk"
    }


@pytest.mark.ffmpeg
def test_setup_sample_yes_and_preset(runner, clip):
    result = invoke(
        runner,
        [
            "setup",
            "--name",
            "desk",
            "--sample",
            clip,
            "--preset",
            "podcast",
            "--yes",
            "--json",
        ],
    )
    profile = json.loads(result.stdout)
    assert profile["measured"]["window_seconds"] == 12
    assert profile["audio"]["loudness"] == {"i": -16, "tp": -2, "lra": 8}
    assert profile["audio"]["eq_chain"]
    assert config.load_config()["default_rig"] == "desk"
    assert (
        runner.invoke(
            module.cli, ["setup", "--name", "desk", "--yes"]
        ).exception.exit_code
        == 1
    )
    invoke(runner, ["setup", "--name", "desk", "--yes", "--overwrite"])


def test_setup_requires_missing_answers_and_rig_rm_force(runner):
    assert runner.invoke(module.cli, ["setup"]).exit_code == 2
    invoke(runner, ["setup", "--yes"])
    assert runner.invoke(module.cli, ["rig", "rm", "default"]).exit_code == 2
    invoke(runner, ["rig", "rm", "default", "--force"])
    assert not config.load_config().get("default_rig")


@pytest.mark.ffmpeg
def test_migrate_is_non_destructive_and_refuses_existing_destination(runner, clip):
    from declip.media import file_hash

    old = clip.with_suffix(".edit.json")
    fixture = Path(__file__).parent / "fixtures/editlists/schema2_sample.json"
    data = json.loads(fixture.read_text())
    data["source"].update(path=str(clip), sha256=file_hash(clip), duration_seconds=12)
    old.write_text(json.dumps(data), encoding="utf-8")
    before = old.read_bytes()
    invoke(runner, ["migrate", old])
    assert old.read_bytes() == before
    assert (
        editlist.load_edit_list(clip.with_suffix(".declip.json")).review.state
        == "pending"
    )
    result = runner.invoke(module.cli, ["migrate", str(old)])
    assert result.exception.exit_code == 1


def test_doctor_json_when_tools_missing(runner, monkeypatch):
    monkeypatch.setenv("PATH", "")
    result = runner.invoke(module.cli, ["doctor", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["ffmpeg"] is None and data["errors"]
    assert "Traceback" not in result.output


@pytest.mark.ffmpeg
def test_v1_export_stops_after_plan(runner, clip):
    result = invoke(runner, ["process", clip, "--export", "edl"])
    assert "declip review" in result.stdout and "declip export" in result.stdout
    assert clip.with_suffix(".declip.json").exists()
    assert not list(clip.parent.glob("*.edl"))


@pytest.mark.ffmpeg
def test_manual_range_needs_review_and_is_idempotent(runner, clip):
    args = ["plan", clip, "--stages", "fillers", "--cut-range", "8-9"]
    invoke(runner, args)
    invoke(runner, args)
    document = editlist.load_edit_list(clip.with_suffix(".declip.json"))
    manual = [cut for cut in document.cuts if cut.kind == "manual"]
    assert len(manual) == 1 and manual[0].status == "accepted"
    assert document.review.state == "pending"


@pytest.mark.ffmpeg
def test_review_required_false_is_ignored(runner, clip):
    invoke(runner, ["setup", "--yes", "--preset", "none"])
    path = paths.config_dir() / "rigs/default.json"
    data = json.loads(path.read_text())
    data["review"]["required"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    result = invoke(runner, ["process", clip, "--execute"])
    assert "review.required=false" in result.stderr
    result = runner.invoke(module.cli, ["render", str(clip)])
    assert result.exception.exit_code == 3


@pytest.mark.ffmpeg
def test_enhance_execute_has_no_cuts_and_needs_no_review(runner, clip):
    output = clip.with_name("processed.wav")
    invoke(
        runner, ["enhance", clip, "--execute", "--output", output, "--preset", "none"]
    )
    from declip.media import probe_media

    assert abs(probe_media(output).duration - probe_media(clip).duration) < 0.001
    assert not clip.with_suffix(".declip.json").exists()


@pytest.mark.ffmpeg
def test_explicit_device_does_not_fall_back(runner, clip):
    result = runner.invoke(module.cli, ["plan", str(clip), "--device", "cuda"])
    assert isinstance(result.exception, module.DeclipError)
    assert result.exception.exit_code == 1
    assert "cuda" in str(result.exception)


@pytest.mark.ffmpeg
def test_unsupported_language_skips_fillers(runner, clip):
    path = clip.with_suffix(".json")
    data = json.loads(path.read_text())
    data["language"] = "de"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = invoke(runner, ["plan", clip, "--stages", "fillers", "--json"])
    assert "No filler file" in result.stderr
    assert not json.loads(result.stdout)["edit_list"]["cuts"]


def test_config_aliases_do_not_mask_explicit_values():
    defaults = config.run_defaults(
        config.normalize_config(
            {
                "defaults": {
                    "margin_ms": 12,
                    "max_gap": 15,
                    "enhance": True,
                    "remove_retakes": False,
                }
            }
        )
    )
    assert defaults["margin_ms"] == 12 and defaults["max_gap_ms"] == 15
    assert defaults["enhancer"] == "auto" and defaults["retakes"] is False


def test_setup_honors_group_json_output(runner):
    result = invoke(runner, ["--json-output", "setup", "--yes"])
    assert json.loads(result.stdout)["name"] == "default"


def test_config_set_legacy_aliases(runner):
    invoke(runner, ["config", "set", "enhance", "true"])
    invoke(runner, ["config", "set", "remove_retakes", "false"])
    result = invoke(runner, ["config", "set", "cpu", "true"])
    assert "deprecated" in result.stderr
    defaults = config.run_defaults(config.load_config())
    assert defaults["enhancer"] == "auto" and defaults["retakes"] is False


@pytest.mark.ffmpeg
def test_v1_export_names_review_even_when_review_is_current(runner, clip):
    path = planned(runner, clip)
    document = editlist.load_edit_list(path)
    document = editlist.apply_decisions(
        document,
        decisions={cut.id: CutStatus.REJECTED for cut in document.cuts},
        add_manual=[],
        remove_manual=[],
    )
    editlist.save_edit_list(
        path, editlist.mark_review_passed(document, now=datetime.now(timezone.utc))
    )
    result = invoke(
        runner, ["process", clip, "--stages", "fillers", "--export", "markers"]
    )
    assert "declip review" in result.stdout and "declip export" in result.stdout
    assert not clip.with_name(clip.stem + ".declip.markers.json").exists()
