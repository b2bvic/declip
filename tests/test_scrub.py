"""Check source, tests, documentation, scripts, and workflows for private settings."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bvictor\b",
        "scalewithsearch",
        r"\bsws\b",
        r"\bdescript\b",
        "m4 pro",
        "studio sound",
        "victorvalentineromo",
    )
)
SOURCE_EXTRA = re.compile(r"\bresemble\b", re.IGNORECASE)
ATTRIBUTION_LINES = (
    "[Project page](https://scalewithsearch.com/code/declip)",
    "- [owned-record](https://github.com/b2bvic/owned-record): agent context and session history in files you control, including `web2md`.",
    "- [ops-scripts](https://github.com/b2bvic/ops-scripts): operator scripts, including the X bookmark import.",
    "- [seo-checks](https://github.com/b2bvic/seo-checks): SEO page checks from one command.",
)


def test_personal_settings_are_absent():
    paths = [
        ROOT / name
        for name in (
            "pyproject.toml",
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
        )
    ]
    for directory in ("src", "tests", "docs", ".github", "scripts"):
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    failures = []
    for path in sorted(paths):
        if path.resolve() == Path(__file__).resolve():
            continue
        patterns = PATTERNS + (
            (SOURCE_EXTRA,) if path.is_relative_to(ROOT / "src") else ()
        )
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if path == ROOT / "README.md" and line.rstrip() in ATTRIBUTION_LINES:
                continue
            active = tuple(
                pattern
                for pattern in patterns
                if not (
                    path.is_relative_to(ROOT / "docs" / "receipts")
                    and pattern.pattern == "m4 pro"
                )
            )
            if any(pattern.search(line) for pattern in active):
                failures.append(f"{path.relative_to(ROOT)}:{number}: {line}")
    assert not failures, "\n".join(failures)


def test_attribution_lines_remain_exact():
    lines = {
        line.rstrip()
        for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    }
    assert set(ATTRIBUTION_LINES) <= lines
