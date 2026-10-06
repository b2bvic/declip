import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from declip.contracts import DeclipError, ReviewRequired
from declip.ui import Spinner, progress_tick, ui_done, ui_warn


def test_legacy_copy_uses_filesystem(tool, tmp_path, monkeypatch):
    def no_cp(argv, **kwargs):
        if argv[0] == "cp":
            pytest.fail("legacy copy attempted an external cp command")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", no_cp)
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_bytes(b"media\x00")
    tool.copy_media(source, output)
    assert output.read_bytes() == source.read_bytes()
    assert '["cp"' not in Path(tool.__file__).read_text(encoding="utf-8")


def test_enhancement_is_explicitly_unavailable(tool, tmp_path):
    with pytest.raises(DeclipError, match="enhancement is unavailable in this build"):
        tool.enhance_audio(tmp_path / "input", tmp_path / "output")
    source = tmp_path / "clip.mp4"
    source.touch()
    for argv in (["enhance", str(source)], ["--enhance", "process", str(source)]):
        result = CliRunner().invoke(tool.cli, argv)
        assert isinstance(result.exception, DeclipError)


def test_pending_integration_cannot_bypass_review(tool, tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    for argv in (
        ["--execute", "process", str(source)],
        ["--execute", "clean", str(source)],
        ["--export", "edl", "detect", str(source)],
    ):
        result = CliRunner().invoke(tool.cli, argv)
        assert isinstance(result.exception, ReviewRequired)
        assert "declip review" in str(result.exception)


def test_legacy_json_stdout_has_no_diagnostics(tool, tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.touch()
    monkeypatch.setattr(
        tool,
        "probe_media",
        lambda path: {
            "format": {"duration": "5"},
            "streams": [{"codec_type": "audio"}],
        },
    )
    monkeypatch.setattr(tool, "extract_audio", lambda path, tmp: path)

    def transcript(*args):
        tool.echo("transcription progress")
        return {"segments": []}

    monkeypatch.setattr(tool, "transcribe", transcript)
    runner = CliRunner()
    for argv in (
        ["--json-output", "process", str(source)],
        ["--json-output", "detect", str(source)],
        ["transcribe", str(source)],
    ):
        result = runner.invoke(tool.cli, argv)
        assert result.exit_code == 0, result.output
        assert isinstance(json.loads(result.stdout), dict)
        assert "transcription progress" in result.stderr
        assert "\x1b" not in result.stdout + result.stderr


def test_missing_saved_preset_falls_back_without_changing_file(tool):
    path = tool.CONFIG_FILE
    path.write_text('{"defaults":{"preset":"missing"}}', encoding="utf-8")
    result = CliRunner().invoke(tool.cli, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "using raw" in result.stderr and str(path) in result.stderr
    assert (
        json.loads(path.read_text(encoding="utf-8"))["defaults"]["preset"] == "missing"
    )


def test_preset_add_writes_user_entries_only(tool):
    result = CliRunner().invoke(
        tool.cli,
        [
            "config",
            "preset-add",
            "desk",
            "--chain",
            "highpass=f=70",
            "--description",
            "Desk mic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert set(json.loads(tool.PRESETS_FILE.read_text(encoding="utf-8"))) == {"desk"}


def test_ui_plain_stderr_and_no_color(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    with Spinner("processing"):
        ui_done("done")
        ui_warn("check")
        progress_tick(0, 1, "work", {100})
    captured = capsys.readouterr()
    assert not captured.out
    assert "processing" in captured.err and "Warning: check" in captured.err
    assert "\x1b" not in captured.err
