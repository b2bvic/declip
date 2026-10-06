"""Check CI boundaries and dependency-audit failure handling."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
SHA_USE = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
PUBLISH = re.compile(
    r"pypa/gh-action-pypi-publish|(?:\btwine|\buv|\bpoetry|\bnpm)\s+publish|"
    r"\btwine\s+upload|\bgh\s+release\s+create|softprops/action-gh-release|"
    r"ncipollo/release-action|\bgit\s+push",
    re.IGNORECASE,
)


def load(path):
    # BaseLoader preserves GitHub's 'on' key instead of YAML 1.1 Boolean coercion.
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def values(node, key):
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                yield value
            yield from values(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from values(value, key)


def test_workflows_exist():
    assert WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_permissions_actions_and_no_publication(path):
    workflow = load(path)
    assert isinstance(workflow, dict) and workflow.get("on")
    assert workflow["permissions"] == {"contents": "read"}
    assert all(
        permission == {"contents": "read"}
        for permission in values(workflow, "permissions")
    )
    uses = list(values(workflow, "uses"))
    assert uses and all(SHA_USE.fullmatch(action) for action in uses)
    assert not PUBLISH.search(path.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        assert any(
            step.get("uses", "").startswith("astral-sh/setup-uv@")
            for step in job["steps"]
        )
        setup = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("astral-sh/setup-uv@")
        )
        assert setup["with"]["python-version"] == "3.11"


def test_ci_matrix_and_required_checks():
    workflow = load(ROOT / ".github" / "workflows" / "ci.yml")
    jobs = workflow["jobs"]
    assert set(jobs) == {"test", "install-smoke", "audit"}
    for name in ("test", "install-smoke"):
        assert jobs[name]["strategy"]["matrix"]["os"] == [
            "ubuntu-latest",
            "macos-latest",
        ]
    matrix = jobs["audit"]["strategy"]["matrix"]["include"]
    assert {(row["extra"], row["platform"], row["os"]) for row in matrix} == {
        ("cpu", "linux", "ubuntu-latest"),
        ("cuda", "linux", "ubuntu-latest"),
        ("cpu", "macos", "macos-latest"),
        ("mac", "macos", "macos-latest"),
    }
    commands = "\n".join(values(jobs["test"], "run"))
    assert "uv sync --locked --extra dev" in commands
    assert "pytest -q --require ffmpeg --require libx265" in commands
    assert '-m "not hwenc"' in commands
    assert "ruff check src tests" in commands and "--select" not in commands
    assert "ffmpeg" in commands and "apt-get" in commands and "brew" in commands
    smoke = "\n".join(values(jobs["install-smoke"], "run"))
    assert "uv tool install ." in smoke and "declip --help" in smoke
    assert "declip doctor" in smoke and "Traceback" in smoke
    assert workflow["defaults"]["run"]["shell"] == "bash"
    assert workflow["env"]["HF_HUB_OFFLINE"] == "1"
    assert {"DECLIP_CONFIG_DIR", "DECLIP_CACHE_DIR"} <= workflow["env"].keys()


def test_dependabot_tracks_uv_and_actions():
    config = load(ROOT / ".github" / "dependabot.yml")
    assert {
        (entry["package-ecosystem"], entry["directory"]) for entry in config["updates"]
    } == {("uv", "/"), ("github-actions", "/")}


@pytest.fixture
def audit_module():
    spec = importlib.util.spec_from_file_location(
        "declip_audit", ROOT / "scripts" / "audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# comments only",
        "thing>=1",
        "thing @ https://example.invalid",
        "thing==1\nthing==1",
    ],
)
def test_audit_rejects_empty_or_unpinned_resolution(audit_module, text):
    with pytest.raises(ValueError):
        audit_module.requirements(text)


@pytest.mark.parametrize(
    "payload,code",
    [
        ({}, 0),
        ({"dependencies": []}, 0),
        (
            {
                "dependencies": [
                    {"name": "thing", "version": "1", "skip_reason": "unavailable"}
                ]
            },
            0,
        ),
        ({"dependencies": [{"name": "thing", "version": "2", "vulns": []}]}, 0),
        ({"dependencies": [{"name": "thing", "version": "1", "vulns": []}]}, 1),
        (
            {
                "dependencies": [
                    {"name": "thing", "version": "1", "vulns": [{"id": "CVE-test"}]}
                ]
            },
            0,
        ),
    ],
)
def test_audit_refuses_partial_or_inconsistent_scan(audit_module, payload, code):
    with pytest.raises(ValueError):
        audit_module.validate_scan(payload, [{"name": "thing", "version": "1"}], code)


@pytest.mark.parametrize("failure", ["compile", "empty", "json", "skip", "finding"])
def test_audit_records_failures_and_holds_index(
    audit_module, tmp_path, monkeypatch, failure
):
    (tmp_path / "pyproject.toml").write_text("synthetic project")
    (tmp_path / "uv.lock").write_text("synthetic lock")
    monkeypatch.setattr(audit_module, "ROOT", tmp_path)

    def run(argv, **kwargs):
        if argv == ["pip-audit", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "pip-audit test\n", "")
        if argv[:3] == ["uv", "pip", "compile"]:
            assert kwargs["check"] is True
            if failure == "compile":
                raise subprocess.CalledProcessError(2, argv, "", "resolution failed")
            Path(argv[-1]).write_text("" if failure == "empty" else "thing==1\n")
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert "--no-deps" in argv and "--disable-pip" in argv
        package = {"name": "thing", "version": "1", "vulns": []}
        if failure == "skip":
            package["skip_reason"] = "unavailable"
        if failure == "finding":
            package["vulns"] = [{"id": "CVE-test"}]
        return subprocess.CompletedProcess(
            argv,
            int(failure == "finding"),
            "bad json"
            if failure == "json"
            else json.dumps({"dependencies": [package]}),
            "",
        )

    monkeypatch.setattr(audit_module.subprocess, "run", run)
    assert audit_module.audit("cpu", "macos") == 1
    receipt = json.loads((tmp_path / "security/audit/cpu-macos.json").read_text())
    assert receipt["release_verdict"] == "held"
    assert (receipt["scan_error"] is None) == (failure == "finding")
    assert receipt["advisory_records"] == int(failure == "finding")
    index = json.loads((tmp_path / "security/dependency-audit.json").read_text())
    assert index["release_verdict"] == "held" and len(index["receipts"]) == 4


def test_audit_index_requires_every_current_receipt(audit_module, tmp_path):
    (tmp_path / "audit").mkdir()
    for extra, platform in audit_module.COMBINATIONS:
        (tmp_path / f"audit/{extra}-{platform}.json").write_text(
            json.dumps(
                {
                    "extra": extra,
                    "platform": platform,
                    "input_sha256": "current",
                    "release_verdict": "clean",
                    "exit_code": 0,
                    "scan_error": None,
                    "macos_deployment_target": audit_module.MACOS_MINIMUM[extra]
                    if platform == "macos"
                    else None,
                    "advisory_records": 0,
                    "resolved": [{"name": "thing", "version": "1"}],
                }
            )
        )
    audit_module.write_index(tmp_path, "current")
    assert (
        json.loads((tmp_path / "dependency-audit.json").read_text())["release_verdict"]
        == "clean"
    )
    audit_module.write_index(tmp_path, "changed")
    assert (
        json.loads((tmp_path / "dependency-audit.json").read_text())["release_verdict"]
        == "held"
    )
