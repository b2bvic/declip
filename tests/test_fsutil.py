import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from declip import fsutil


def test_atomic_json_round_trip(tmp_path):
    path = tmp_path / "new" / "data.json"
    payload = {"words": "café 日本語", "nested": [1, True, None]}
    fsutil.atomic_write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert "café 日本語" in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob("*.tmp"))


def test_atomic_write_text_utf8(tmp_path):
    path = tmp_path / "message.txt"
    fsutil.atomic_write_text(path, "naïve\n")
    assert path.read_bytes() == "naïve\n".encode()


def test_eight_threads_use_unique_same_directory_temps(tmp_path, monkeypatch):
    names = []
    replace = fsutil.os.replace

    def record(source, destination):
        source = Path(source)
        assert source.parent == destination.parent
        names.append(source.name)
        replace(source, destination)

    monkeypatch.setattr(fsutil.os, "replace", record)
    path = tmp_path / "shared.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: fsutil.atomic_write_json(path, {"i": i, "text": "ø" * 1000}),
                range(8),
            )
        )
    assert len(names) == len(set(names)) == 8
    assert json.loads(path.read_text(encoding="utf-8"))["i"] in range(8)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["replace", "fsync"])
def test_cleanup_and_original_preserved_on_failure(tmp_path, monkeypatch, failure):
    path = tmp_path / "data.json"
    path.write_text("original", encoding="utf-8")

    def fail(*args):
        raise OSError("injected failure")

    monkeypatch.setattr(fsutil.os, failure, fail)
    with pytest.raises(OSError, match="injected failure"):
        fsutil.atomic_write_json(path, {"new": True})
    assert path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [path]
