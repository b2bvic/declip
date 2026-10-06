import os
import subprocess
import tomllib
import zipfile
from pathlib import Path

from declip import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_project_version_matches_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__ == "0.5.1"
    assert project["project"]["scripts"] == {"declip": "declip.cli:main"}


def test_wheel_contains_package_resources(tmp_path):
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert {
            "declip/contracts.py",
            "declip/data/presets.json",
            "declip/data/fillers/en.txt",
            "declip/review/static/index.html",
            "declip/review/static/app.js",
            "declip/review/static/style.css",
        } <= set(wheel.namelist())
        for name in ("index.html", "app.js", "style.css"):
            member = f"declip/review/static/{name}"
            assert wheel.read(member) == (ROOT / "src" / member).read_bytes()
            assert len(wheel.read(member)) > 100
        entry_points = wheel.read("declip-0.5.1.dist-info/entry_points.txt").decode()
        assert "declip = declip.cli:main" in entry_points
