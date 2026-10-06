"""Subprocess CLI round trip on generated media and a fixture transcriber."""

import json
import os
import subprocess
import sys
import threading
import urllib.request
from urllib.parse import urlsplit, parse_qs

import pytest

from declip import editlist
from declip.media import probe_media
from declip.review.server import serve

pytestmark = pytest.mark.ffmpeg


def run_cli(args, env):
    return subprocess.run(
        [sys.executable, "-m", "declip", *map(str, args)],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )


def test_cli_round_trip(tmp_path, make_media):
    source = make_media(
        tmp_path,
        {
            "duration": 20,
            "inputs": [
                "testsrc2=size=640x360:rate=30000/1001",
                "sine=sample_rate=48000",
            ],
        },
    )
    words = [(1, "um"), (2, "hello"), (3, "uh"), (4, "world"), (5, "um"), (6, "end")]
    fixture = {
        "language": "en",
        "segments": [
            {
                "start": 0,
                "end": 20,
                "text": "synthetic",
                "words": [
                    {
                        "word": text,
                        "start": start,
                        "end": start + 0.2,
                        "probability": 0.9,
                    }
                    for start, text in words
                ],
            }
        ],
    }
    (tmp_path / "clip.json").write_text(json.dumps(fixture), encoding="utf-8")
    env = dict(
        os.environ, DECLIP_TRANSCRIBER="fake", DECLIP_FAKE_TRANSCRIPT_DIR=str(tmp_path)
    )
    before = set(tmp_path.iterdir())
    result = run_cli(
        ["process", source, "--execute", "--preset", "none", "--encoder", "software"],
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "declip review" in result.stdout
    path = source.with_suffix(".declip.json")
    document = editlist.load_edit_list(path)
    assert document.review.state == "pending"
    assert len(document.cuts) == 3
    assert all(
        p.suffix == ".json" for p in set(tmp_path.iterdir()) - before if p.is_file()
    )
    for args in (["render", source], ["export", source, "--format", "edl"]):
        result = run_cli(args, env)
        assert result.returncode == 3, result.stderr
    ready = threading.Event()
    urls, results, failures = [], [], []

    def start():
        try:
            results.append(
                serve(
                    path,
                    source,
                    open_browser=False,
                    on_ready=lambda url: (urls.append(url), ready.set()),
                )
            )
        except BaseException as exc:
            failures.append(exc)
            ready.set()

    worker = threading.Thread(target=start, daemon=True)
    worker.start()
    assert ready.wait(10), failures
    url = urlsplit(urls[0])
    token = parse_qs(url.query)["token"][0]

    def request(route, payload=None, method="GET"):
        req = urllib.request.Request(
            f"http://{url.netloc}{route}",
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={"X-Declip-Token": token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    current = request("/api/edit")
    current = request(
        "/api/edit",
        {
            "revision": current["revision"],
            "decisions": [
                {"id": cut.id, "status": "rejected" if i == 2 else "accepted"}
                for i, cut in enumerate(document.cuts)
            ],
            "manual_cuts": {
                "add": [{"start": 8, "end": 9, "label": "manual"}],
                "remove": [],
            },
        },
        "PUT",
    )
    assert request("/api/finish", {"revision": current["revision"]}, "POST")["passed"]
    worker.join(10)
    assert not worker.is_alive() and not failures and results[0].passed
    result = run_cli(["render", source, "--encoder", "software"], env)
    assert result.returncode == 0, result.stderr
    output = source.with_name("clip_clean.mp4")
    expected = editlist.effective_timeline(editlist.load_edit_list(path)).duration_out
    assert abs(probe_media(output).duration - expected) <= float(1 / document.media.fps)
    passed_bytes = path.read_bytes()
    data = json.loads(passed_bytes)
    data["cuts"][0]["status"] = "rejected"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = run_cli(["render", source, "--overwrite"], env)
    assert result.returncode == 3, result.stderr
    # Restore the reviewed decisions before checking non-destructive export.
    path.write_bytes(passed_bytes)
    result = run_cli(["export", source, "--format", "markers"], env)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == passed_bytes
    assert source.with_name("clip.declip.markers.json").exists()
