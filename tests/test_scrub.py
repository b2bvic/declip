"""P0 scrub scope. P11 adds documentation and workflow scope."""

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
    "- [web2md](https://github.com/b2bvic/web2md)",
    "- [twitter-bookmarks](https://github.com/b2bvic/twitter-bookmarks)",
    "- [sws-skills](https://github.com/b2bvic/sws-skills)",
)


def test_personal_settings_are_absent():
    paths = [ROOT / "pyproject.toml"]
    for directory in ("src", "tests"):
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
            if any(pattern.search(line) for pattern in patterns):
                failures.append(f"{path.relative_to(ROOT)}:{number}: {line}")
    assert not failures, "\n".join(failures)


def test_attribution_lines_remain_exact():
    lines = {
        line.rstrip()
        for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    }
    assert set(ATTRIBUTION_LINES) <= lines
