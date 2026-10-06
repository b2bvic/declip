"""Resolve and audit one advertised runtime extra without installing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from declip import __version__
from declip.fsutil import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"linux": "x86_64-unknown-linux-gnu", "macos": "aarch64-apple-darwin"}
COMBINATIONS = (("cpu", "macos"), ("cpu", "linux"), ("cuda", "linux"), ("mac", "macos"))
MACOS_MINIMUM = {"cpu": "13.0", "mac": "14.0"}
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")


def requirements(text: str) -> list[dict[str, str]]:
    resolved = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"Unpinned or unsupported requirement: {line}")
        resolved.append(
            {"name": re.sub(r"[-_.]+", "-", match[1]).lower(), "version": match[2]}
        )
    if not resolved or len({p["name"] for p in resolved}) != len(resolved):
        raise ValueError("Resolution is empty or contains duplicate packages")
    return sorted(resolved, key=lambda package: package["name"])


def validate_scan(
    payload: object, resolved: list[dict[str, str]], exit_code: int
) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("dependencies"), list
    ):
        raise ValueError("Audit output has no dependency list")
    packages = payload["dependencies"]
    actual = []
    for package in packages:
        if not isinstance(package, dict) or package.get("skip_reason"):
            raise ValueError(f"Audit skipped or returned an invalid package: {package}")
        if not isinstance(package.get("vulns"), list):
            raise ValueError("Audit package has no vulnerability list")
        for vuln in package["vulns"]:
            if not isinstance(vuln, dict) or not isinstance(vuln.get("id"), str):
                raise ValueError("Audit returned an invalid advisory")
        actual.append(
            {
                "name": re.sub(r"[-_.]+", "-", package["name"]).lower(),
                "version": package["version"],
            }
        )
    if sorted(actual, key=lambda p: p["name"]) != resolved:
        raise ValueError("Scanned packages differ from the resolved requirements")
    findings = sum(len(package["vulns"]) for package in packages)
    if exit_code not in (0, 1) or (exit_code == 0) != (findings == 0):
        raise ValueError(f"Audit exit {exit_code} disagrees with its findings")
    return packages


def input_hash() -> str:
    return hashlib.sha256(
        (ROOT / "pyproject.toml").read_bytes() + (ROOT / "uv.lock").read_bytes()
    ).hexdigest()


def write_index(directory: Path, fingerprint: str) -> None:
    receipts = []
    for extra, platform in COMBINATIONS:
        filename = f"audit/{extra}-{platform}.json"
        entry = {
            "extra": extra,
            "platform": platform,
            "path": filename,
            "release_verdict": "held",
        }
        try:
            data = json.loads((directory / filename).read_text(encoding="utf-8"))
            valid = (
                data.get("extra") == extra
                and data.get("platform") == platform
                and data.get("input_sha256") == fingerprint
                and data.get("macos_deployment_target")
                == (MACOS_MINIMUM[extra] if platform == "macos" else None)
                and data.get("release_verdict") == "clean"
                and data.get("exit_code") == 0
                and data.get("scan_error") is None
                and data.get("advisory_records") == 0
                and bool(data.get("resolved"))
            )
            entry.update(
                checked_at=data.get("checked_at"),
                advisory_records=data.get("advisory_records"),
                scan_error=data.get("scan_error"),
            )
            entry["release_verdict"] = "clean" if valid else "held"
        except (OSError, ValueError, AttributeError):
            entry["scan_error"] = "Receipt missing or invalid"
        receipts.append(entry)
    atomic_write_json(
        directory / "dependency-audit.json",
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Python 3.11 runtime extras for macOS arm64 and Linux x86_64",
            "input_sha256": fingerprint,
            "macos_deployment_targets": MACOS_MINIMUM,
            "receipts": receipts,
            "release_verdict": "clean"
            if all(r["release_verdict"] == "clean" for r in receipts)
            else "held",
            "limits": [
                f"Windows is deferred and has no audit in {__version__}.",
                "Requirements are resolved from pyproject.toml, not exported from uv.lock.",
                "The mac extra targets macOS 14.0, matching the minimum of the locked MLX and Torch wheels.",
                "The cpu extra targets macOS 13.0; Linux targets x86_64 glibc.",
                "Cross-platform resolution does not prove native installation or hardware support.",
                "A clean scan means no known advisory in the scanned versions at the recorded time.",
            ],
        },
    )


def audit(extra: str, platform: str) -> int:
    fingerprint = input_hash()
    receipt = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "tool": "pip-audit",
        "scope": "Resolved runtime dependencies from pyproject.toml",
        "command": None,
        "exit_code": None,
        "advisory_records": 0,
        "unique_ids": 0,
        "packages": [],
        "extra": extra,
        "platform": platform,
        "python": "3.11",
        "target": TARGETS[platform],
        "resolved": [],
        "scan_error": None,
        "release_verdict": "held",
        "input_sha256": fingerprint,
        "macos_deployment_target": MACOS_MINIMUM[extra]
        if platform == "macos"
        else None,
    }
    try:
        version = subprocess.run(
            ["pip-audit", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        receipt["tool"] = version.stdout.strip()
        with tempfile.TemporaryDirectory(prefix="declip-audit-") as raw:
            requirements_path = Path(raw) / "requirements.txt"
            compile_command = [
                "uv",
                "pip",
                "compile",
                "pyproject.toml",
                "--extra",
                extra,
                "--python-version",
                "3.11",
                "--python-platform",
                TARGETS[platform],
                "-o",
                str(requirements_path),
            ]
            receipt["resolve_command"] = shlex.join(compile_command)
            compile_env = os.environ.copy()
            if platform == "macos":
                compile_env["MACOSX_DEPLOYMENT_TARGET"] = MACOS_MINIMUM[extra]
            compilation = subprocess.run(
                compile_command,
                cwd=ROOT,
                env=compile_env,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            receipt["resolve_stderr"] = compilation.stderr
            receipt["resolved"] = requirements(
                requirements_path.read_text(encoding="utf-8")
            )
            scan_command = [
                "pip-audit",
                "--no-deps",
                "--disable-pip",
                "-r",
                str(requirements_path),
                "--format",
                "json",
            ]
            receipt["command"] = shlex.join(scan_command)
            scan = subprocess.run(
                scan_command, cwd=ROOT, capture_output=True, text=True, timeout=300
            )
            receipt.update(
                exit_code=scan.returncode, stdout=scan.stdout, stderr=scan.stderr
            )
            receipt["packages"] = validate_scan(
                json.loads(scan.stdout), receipt["resolved"], scan.returncode
            )
            vulns = [v for p in receipt["packages"] for v in p["vulns"]]
            receipt["advisory_records"] = len(vulns)
            receipt["unique_ids"] = len({v["id"] for v in vulns})
            if scan.returncode == 0:
                receipt["release_verdict"] = "clean"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        receipt["scan_error"] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            receipt.update(
                exit_code=exc.returncode, stdout=exc.stdout, stderr=exc.stderr
            )
    directory = ROOT / "security"
    destination = directory / "audit" / f"{extra}-{platform}.json"
    atomic_write_json(destination, receipt)
    write_index(directory, fingerprint)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "extra",
                    "platform",
                    "exit_code",
                    "advisory_records",
                    "scan_error",
                    "release_verdict",
                )
            },
            indent=2,
        )
    )
    print(f"Receipt: {destination.relative_to(ROOT)}")
    return 0 if receipt["release_verdict"] == "clean" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", choices=("cpu", "cuda", "mac"), required=True)
    parser.add_argument("--platform", choices=tuple(TARGETS), required=True)
    args = parser.parse_args()
    if (args.extra, args.platform) not in COMBINATIONS:
        parser.error("Use cpu on either platform, cuda on Linux, or mac on macOS")
    return audit(args.extra, args.platform)


if __name__ == "__main__":
    sys.exit(main())
