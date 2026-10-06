from click.testing import CliRunner


def test_cut_merge_and_inverse_preserve_unedited_time(tool):
    cuts = tool.merge_cuts(
        [tool.CutRegion(1, 2, "um", 1), tool.CutRegion(1.5, 3, "uh", 1)], 0
    )
    assert [(cut.start, cut.end) for cut in cuts] == [(1, 3)]
    keeps = tool.invert_cuts(cuts, 5)
    assert [(keep.start, keep.end) for keep in keeps] == [(0, 1), (3, 5)]


def test_synthetic_filler_detection(tool):
    transcript = {
        "segments": [
            {
                "words": [
                    {"word": "um", "start": 1, "end": 1.2, "probability": 1},
                    {"word": "hello", "start": 1.3, "end": 2},
                ]
            }
        ]
    }
    cuts = tool.detect_fillers(transcript, {"um"}, margin_ms=0)
    assert len(cuts) == 1
    assert cuts[0].word == "um"


def test_cli_help(tool):
    result = CliRunner().invoke(tool.cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def _run_harness(tmp_path, source, *args):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

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
