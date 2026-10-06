from pathlib import Path

from click.testing import CliRunner

from declip import cli, detect, editlist
from declip.contracts import FillerFile, Word


def test_cut_merge_and_inverse_preserve_unedited_time(tmp_path):
    from declip.contracts import (
        AudioStream,
        ColorTags,
        MediaInfo,
        OutputMode,
        OutputSpec,
        Processing,
        RigRef,
    )
    from dataclasses import replace

    info = MediaInfo(
        5,
        0,
        "wav",
        False,
        None,
        0,
        None,
        False,
        None,
        None,
        0,
        None,
        None,
        None,
        None,
        ColorTags(None, None, None, None),
        None,
        (AudioStream(0, "pcm", 48000, 1, "mono", 16, None),),
        (),
    )
    source = tmp_path / "clip.wav"
    source.touch()
    document = editlist.new_edit_list(
        source,
        info,
        "0" * 64,
        transcript=None,
        processing=Processing("none", 0.5, "", None, 0),
        output=OutputSpec(OutputMode.RENDER, None, False),
        rig=RigRef(None, None, {}),
    )
    from declip.contracts import Cut, CutKind, CutOrigin, CutStatus

    cuts = tuple(
        Cut(
            str(i),
            CutKind.MANUAL,
            start,
            end,
            "manual",
            1,
            False,
            None,
            CutOrigin.MANUAL,
            CutStatus.ACCEPTED,
        )
        for i, (start, end) in enumerate([(1, 2), (1.5, 3)])
    )
    assert editlist.keep_intervals(replace(document, cuts=cuts)) == [(0, 1), (3, 5)]


def test_synthetic_filler_detection():
    words = (Word(0, 1, 1.2, "um", 1, 0), Word(1, 1.3, 2, "hello", 1, 0))
    cuts = detect.detect_fillers(
        words,
        FillerFile("en", None, frozenset({"um"}), frozenset()),
        min_confidence=0.5,
        margin_ms=0,
        duration=5,
    )
    assert len(cuts) == 1 and cuts[0].label == "um"


def test_cli_help():
    result = CliRunner().invoke(cli.cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def _run_harness(tmp_path, source, *args):
    import shutil
    import subprocess
    import sys

    shutil.copyfile(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    test_file = tmp_path / "test_harness.py"
    test_file.write_text(source, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(Path(__file__).resolve().parents[1] / "pyproject.toml"),
            str(test_file),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_require_turns_a_marked_skip_into_failure(tmp_path):
    result = _run_harness(
        tmp_path,
        'import pytest\n@pytest.mark.extra("cpu")\ndef test_missing():\n    pytest.skip("backend unavailable")\n',
        "--require",
        "cpu",
    )
    assert result.returncode == 1
    assert "Required capability skipped (cpu)" in result.stdout
    assert "1 failed" in result.stdout


def test_unrequired_skip_stays_a_skip(tmp_path):
    result = _run_harness(
        tmp_path,
        'import pytest\n@pytest.mark.extra("cpu")\ndef test_missing():\n    pytest.skip("backend unavailable")\n',
    )
    assert result.returncode == 0
    assert "1 skipped" in result.stdout


def test_update_golden_writes_actual_and_keeps_golden(tmp_path):
    golden = tmp_path / "expected.txt"
    golden.write_text("expected", encoding="utf-8")
    result = _run_harness(
        tmp_path,
        'from pathlib import Path\ndef test_mismatch(assert_golden):\n    assert_golden("observed", Path(__file__).with_name("expected.txt"))\n',
        "--update-golden",
    )
    assert result.returncode == 1
    assert golden.read_text(encoding="utf-8") == "expected"
    assert (
        golden.with_name("expected.txt.actual").read_text(encoding="utf-8")
        == "observed"
    )
