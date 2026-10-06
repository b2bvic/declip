"""Real HTTP transactions, review security, and bounded preview generation."""

import hashlib
import http.client
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from urllib.parse import urlsplit

import pytest

from declip import editlist as ed
from declip.contracts import (
    AudioStream,
    ColorTags,
    CutKind,
    CutProposal,
    DeclipError,
    MediaInfo,
    OutputMode,
    OutputSpec,
    Processing,
    ReviewState,
    RigRef,
    SourceMismatch,
)
from declip.media import file_hash, probe_media
from declip.review import proxy, server

pytestmark = pytest.mark.contract


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source")
    media = MediaInfo(
        10,
        0,
        "mov",
        True,
        0,
        1,
        Fraction(30),
        False,
        320,
        240,
        0,
        "h264",
        "yuv420p",
        8,
        1000000,
        ColorTags(None, None, None, None),
        None,
        (AudioStream(1, "aac", 48000, 2, "stereo", None, None),),
        (),
    )
    value = ed.new_edit_list(
        source,
        media,
        file_hash(source),
        transcript=None,
        processing=Processing("none", 0.5, "", None, 20),
        output=OutputSpec(OutputMode.RENDER, None, False),
        rig=RigRef(None, None, {}),
    )
    return ed.replace_stage_cuts(
        value,
        CutKind.FILLER,
        [CutProposal(CutKind.FILLER, 1, 2, "um", 0.2, None, True)],
    )


class Session:
    def __init__(self, tmp_path, plan):
        self.path = tmp_path / "clip.declip.json"
        ed.save_edit_list(self.path, plan)
        self.proxy = tmp_path / "preview.mp4"
        self.proxy.write_bytes(b"0123456789")
        self.server = server._ReviewServer(self.path, self.proxy, 0)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()

    def request(
        self,
        route="/api/edit",
        method="GET",
        data=None,
        *,
        token=True,
        headers=None,
        raw=None,
    ):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        hdr = {}
        if token:
            if route.startswith("/api/"):
                hdr["X-Declip-Token"] = self.server.token
            else:
                route += ("&" if "?" in route else "?") + "token=" + self.server.token
        body = (
            raw if raw is not None else json.dumps(data) if data is not None else None
        )
        if body is not None:
            hdr["Content-Type"] = "application/json"
        hdr.update(headers or {})
        conn.request(method, route, body=body, headers=hdr)
        response = conn.getresponse()
        content = response.read()
        status, response_headers = response.status, dict(response.getheaders())
        conn.close()
        if response_headers.get("Content-Type", "").startswith("application/json"):
            content = json.loads(content)
        return status, content, response_headers

    def get(self):
        status, data, _ = self.request()
        assert status == 200
        return data

    def close(self):
        if not self.server.finished:
            self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        assert not self.thread.is_alive()


@pytest.fixture
def session(tmp_path, plan):
    value = Session(tmp_path, plan)
    try:
        yield value
    finally:
        value.close()


def test_media_requires_token(session):
    assert session.request("/media", token=False)[0] == 403


def test_static_requires_token(session):
    for name in ("index.html", "app.js", "style.css"):
        assert session.request("/static/" + name, token=False)[0] == 403


def test_root_requires_token(session):
    assert session.request("/", token=False)[0] == 403


@pytest.mark.parametrize(
    "route,method",
    [
        ("/api/edit", "GET"),
        ("/api/edit", "PUT"),
        ("/api/finish", "POST"),
        ("/unknown", "GET"),
    ],
)
@pytest.mark.parametrize("token", [None, "wrong", "é"])
def test_all_routes_require_valid_token(session, route, method, token):
    headers = {"X-Declip-Token": token} if token is not None else {}
    assert session.request(route, method, {}, token=False, headers=headers)[0] == 403


def test_bad_host_403(session):
    for host in ("attacker.example", "127.0.0.1", "localhost:1", "127.0.0.1:80"):
        assert session.request(headers={"Host": host})[0] == 403
    assert (
        session.request(headers={"Host": f"localhost:{session.server.server_port}"})[0]
        == 200
    )


@pytest.mark.parametrize(
    "method,route", [("PUT", "/api/edit"), ("POST", "/api/finish")]
)
def test_foreign_origin_put_403(session, method, route):
    data = {"revision": session.get()["revision"]}
    for origin in ("https://attacker.example", "null", "http://127.0.0.1:1"):
        assert (
            session.request(route, method, data, headers={"Origin": origin})[0] == 403
        )
    assert (
        session.request(
            "/api/edit",
            "PUT",
            data,
            headers={"Origin": f"http://localhost:{session.server.server_port}"},
        )[0]
        == 200
    )


@pytest.mark.parametrize(
    "route",
    [
        "/../server.py",
        "/static/%2e%2e/server.py",
        "/static/../server.py",
        "/static/app.js%2f..%2fserver.py",
        "/static%2fapp.js",
        "/static/%5capp.js",
        "/media/other",
        "/api/edit/other",
    ],
)
def test_traversal_404(session, route):
    assert session.request(route)[0] == 404


def test_bind_is_loopback(session):
    assert session.server.server_address[0] == "127.0.0.1"


def test_url_uses_127(session):
    assert urlsplit(session.server.url).hostname == "127.0.0.1"
    assert len(session.server.token) >= 43


def test_page_embeds_token_and_no_cors(session):
    status, body, headers = session.request("/")
    assert status == 200 and session.server.token.encode() in body
    assert b"__DECLIP_TOKEN__" not in body
    assert not any(key.lower().startswith("access-control-") for key in headers)
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert (
        session.request("/api/edit?token=" + session.server.token, token=False)[0]
        == 200
    )


@pytest.mark.parametrize(
    "method,route", [("PUT", "/api/edit"), ("POST", "/api/finish")]
)
def test_content_type_required(session, method, route):
    assert (
        session.request(route, method, {}, headers={"Content-Type": "text/plain"})[0]
        == 415
    )
    assert (
        session.request(
            route,
            method,
            {},
            headers={"Content-Type": "application/json; charset=utf-8"},
        )[0]
        == 400
    )


def test_get_summary_uses_effective_timeline(session):
    value = session.get()
    assert value["revision"] == hashlib.sha256(session.path.read_bytes()).hexdigest()
    assert value["summary"]["counts"] == {"proposed": 1, "accepted": 0, "rejected": 0}
    assert value["summary"]["fps"] == "30/1"
    value = session.request(
        method="PUT",
        data={
            "revision": value["revision"],
            "decisions": [
                {"id": value["edit_list"]["cuts"][0]["id"], "status": "accepted"}
            ],
            "manual_cuts": {"add": [{"start": 1.91, "end": 2.07, "label": "tail"}]},
        },
    )[1]
    timeline = ed.effective_timeline(ed.load_edit_list(session.path))
    assert value["summary"]["output_duration"] == timeline.duration_out
    assert value["summary"]["removed_intervals"] == [
        list(pair) for pair in timeline.removed
    ]


def test_stale_revision_409(session):
    revision = session.get()["revision"]
    assert session.request(method="PUT", data={"revision": "old"})[:2] == (
        409,
        {"error": "stale", "revision": revision},
    )
    assert session.request("/api/finish", "POST", {"revision": "old"})[0] == 409


def test_external_change_409(session):
    revision = session.get()["revision"]
    session.path.write_bytes(session.path.read_bytes() + b" ")
    status, value, _ = session.request(method="PUT", data={"revision": revision})
    assert status == 409 and value["revision"] == ed.revision_of(session.path)


def test_finish_unresolved_422(session):
    revision = session.get()["revision"]
    before = session.path.read_bytes()
    status, value, _ = session.request("/api/finish", "POST", {"revision": revision})
    assert status == 422 and value == {"error": "unresolved", "unresolved_count": 1}
    assert not session.server.finished and session.path.read_bytes() == before


def test_zero_cut_finish(tmp_path, plan):
    session = Session(tmp_path, replace(plan, cuts=()))
    try:
        status, data, _ = session.request(
            "/api/finish", "POST", {"revision": session.get()["revision"]}
        )
        assert status == 200 and data["passed"] is True
        session.thread.join(timeout=5)
        assert not session.thread.is_alive()
        saved = ed.load_edit_list(session.path)
        assert saved.review.state == ReviewState.PASSED
        assert saved.review.mode == "interactive" and saved.review.passed_at
        assert data["content_sha256"] == ed.review_content_hash(saved)
    finally:
        session.close()


def test_manual_add_remove_and_finish(session):
    data = session.get()
    identifier = data["edit_list"]["cuts"][0]["id"]
    status, added, _ = session.request(
        method="PUT",
        data={
            "revision": data["revision"],
            "decisions": [{"id": identifier, "status": "rejected"}],
            "manual_cuts": {"add": [{"start": 3, "end": 4, "label": "retake"}]},
        },
    )
    assert status == 200
    manual = added["edit_list"]["cuts"][-1]
    assert (manual["origin"], manual["kind"], manual["status"]) == (
        "manual",
        "manual",
        "accepted",
    )
    assert manual["confidence"] == 1 and manual["low_confidence"] is False
    status, removed, _ = session.request(
        method="PUT",
        data={"revision": added["revision"], "manual_cuts": {"remove": [manual["id"]]}},
    )
    assert status == 200 and len(removed["edit_list"]["cuts"]) == 1
    assert (
        session.request("/api/finish", "POST", {"revision": removed["revision"]})[0]
        == 200
    )
    assert ed.render_allowed(
        ed.load_edit_list(session.path), file_hash(session.server.source, refresh=True)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"decisions": [{"id": "unknown", "status": "accepted"}]},
        {"decisions": [{"id": "AUTO", "status": "bad"}]},
        {"decisions": [{"id": "AUTO", "status": "accepted"}] * 2},
        {"decisions": "bad"},
        {"manual_cuts": []},
        {"manual_cuts": {"add": ["bad"]}},
        {"manual_cuts": {"remove": ["AUTO"]}},
        {"manual_cuts": {"remove": [5]}},
        {"manual_cuts": {"add": [{"start": -1, "end": 2, "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": 2, "end": 1, "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": 0, "end": 11, "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": 1, "end": 1.01, "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": float("nan"), "end": 2, "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": 0, "end": float("inf"), "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": True, "end": 2, "label": "bad"}]}},
        {"manual_cuts": {"add": [{"start": 1, "end": 2, "label": "x" * 201}]}},
        {"manual_cuts": {"add": [{"start": 1, "end": 2}]}},
        {"manual_cuts": {"add": [{"start": 1, "end": 2, "label": "same"}] * 2}},
    ],
)
def test_invalid_changes_leave_file_unchanged(session, change):
    value = session.get()
    change = json.loads(
        json.dumps(change).replace(
            '"AUTO"', json.dumps(value["edit_list"]["cuts"][0]["id"])
        )
    )
    before = session.path.read_bytes()
    assert (
        session.request(method="PUT", data={"revision": value["revision"], **change})[0]
        == 400
    )
    assert session.path.read_bytes() == before


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "{",
        "{}",
        '{"revision": 5}',
        '{"revision": "x", "decisions": [null]}',
    ],
)
def test_invalid_json_request(session, raw):
    raw = raw.replace('"x"', json.dumps(session.get()["revision"]))
    assert session.request(method="PUT", raw=raw)[0] == 400


def test_manual_decision_and_duplicate_rejected(session):
    initial = session.get()
    change = {"manual_cuts": {"add": [{"start": 3, "end": 4, "label": "same"}]}}
    added = session.request(
        method="PUT", data={"revision": initial["revision"], **change}
    )[1]
    manual = added["edit_list"]["cuts"][-1]["id"]
    before = session.path.read_bytes()
    for patch in (change, {"decisions": [{"id": manual, "status": "rejected"}]}):
        assert (
            session.request(
                method="PUT", data={"revision": added["revision"], **patch}
            )[0]
            == 400
        )
        assert session.path.read_bytes() == before


@pytest.mark.parametrize(
    "range_header,expected,body",
    [
        ("bytes=2-5", 206, b"2345"),
        ("bytes=7-", 206, b"789"),
        ("bytes=-3", 206, b"789"),
        ("bytes=0-99", 206, b"0123456789"),
        ("bytes=-99", 206, b"0123456789"),
        ("bytes=10-", 416, None),
        ("bytes=4-2", 416, None),
        ("bytes=-0", 416, None),
        ("bytes=", 416, None),
        ("bytes=0-1,4-5", 416, None),
        ("words=0-1", 416, None),
    ],
)
def test_range_206_and_416(session, range_header, expected, body):
    status, data, headers = session.request("/media", headers={"Range": range_header})
    assert status == expected
    if body:
        assert data == body and headers["Accept-Ranges"] == "bytes"
        assert int(headers["Content-Length"]) == len(data)
        assert headers["Content-Range"].startswith("bytes ")
    else:
        assert headers["Content-Range"] == "bytes */10"
    assert session.request("/media")[1] == b"0123456789"


def test_concurrent_puts_serialize(session):
    initial = session.get()
    barrier = threading.Barrier(8)

    def write(index):
        barrier.wait()
        return session.request(
            method="PUT",
            data={
                "revision": initial["revision"],
                "manual_cuts": {
                    "add": [{"start": 3, "end": 4, "label": f"manual {index}"}]
                },
            },
        )[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(write, range(8)))
    assert outcomes.count(200) == 1 and outcomes.count(409) == 7
    assert len(ed.load_edit_list(session.path).cuts) == 2


def test_source_change_refused_before_start_and_finish(session):
    session.server.source.write_bytes(b"changed")
    before = session.path.read_bytes()
    assert (
        session.request(
            "/api/finish", "POST", {"revision": ed.revision_of(session.path)}
        )[0]
        == 400
    )
    assert session.path.read_bytes() == before
    with pytest.raises(SourceMismatch):
        server._ReviewServer(session.path, session.proxy, 0)


def test_serve_returns_result_and_ready_callback(tmp_path, plan):
    path, preview = tmp_path / "edit.json", tmp_path / "preview.mp4"
    ed.save_edit_list(path, replace(plan, cuts=()))
    preview.write_bytes(b"preview")
    ready, output, error = threading.Event(), [], []
    url = []

    def run():
        try:
            output.append(
                server.serve(
                    path,
                    preview,
                    open_browser=False,
                    on_ready=lambda value: (url.append(value), ready.set()),
                )
            )
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5)
    parsed = urlsplit(url[0])
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    conn.request("GET", "/api/edit?" + parsed.query)
    revision = json.loads(conn.getresponse().read())["revision"]
    conn.close()
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    conn.request(
        "POST",
        "/api/finish?" + parsed.query,
        json.dumps({"revision": revision}),
        {"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    assert response.status == 200
    response.read()
    conn.close()
    thread.join(5)
    assert not thread.is_alive() and not error
    assert output[0].passed and output[0].unresolved_count == 0
    assert output[0].edit_list_path == path and output[0].url == url[0]


def test_ctrl_c_returns_failed_result(tmp_path, plan, monkeypatch):
    path, preview = tmp_path / "edit.json", tmp_path / "preview.mp4"
    ed.save_edit_list(path, plan)
    preview.write_bytes(b"preview")

    def interrupt(self, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(server._ReviewServer, "serve_forever", interrupt)
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    result = server.serve(path, preview)
    assert not result.passed and result.unresolved_count == 1 and opened == [result.url]


def test_logging_omits_tokens(session, caplog):
    with caplog.at_level(logging.DEBUG, logger=server.__name__):
        session.request("/")
        session.request("/media")
    assert session.server.token not in caplog.text
    assert "GET /media" in caplog.text


def test_static_has_no_external_urls():
    for path in server._STATIC.iterdir():
        text = path.read_text()
        assert "http://" not in text and "https://" not in text
        assert "innerHTML" not in text


@pytest.mark.parametrize(
    "width,height,rotation,expected",
    [
        (1081, 1921, 0, (540, 958)),
        (1921, 1081, 90, (540, 958)),
        (3840, 2160, 0, (960, 540)),
        (320, 241, 0, (320, 240)),
        (240, 320, 0, (240, 320)),
        (1920, 1080, 270, (540, 960)),
    ],
)
def test_proxy_args_portrait_even_dims(plan, width, height, rotation, expected):
    args = proxy.proxy_args(
        replace(plan.media, width=width, height=height, rotation=rotation)
    )
    assert f"scale={expected[0]}:{expected[1]}" in args[args.index("-vf") + 1]
    assert "-ss" not in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-b:v") + 1] == "1000000"
    assert args[args.index("-ac") + 1] == "2"
    assert "0:0" in args and "0:1" in args


def test_proxy_audio_only_and_no_audio(plan, tmp_path):
    media = replace(plan.media, has_video=False, fps=None, video_index=None)
    args = proxy.proxy_args(media)
    assert "-vn" in args and "-c:v" not in args and "-ac" not in args
    assert args[args.index("-b:a") + 1] == "128000"
    assert proxy.proxy_path(tmp_path, "source-hash", media).suffix == ".m4a"
    assert "-an" in proxy.proxy_args(replace(plan.media, audio_index=None))
    with pytest.raises(DeclipError):
        proxy.proxy_args(replace(media, audio_index=None))


def test_proxy_cache_key(plan, tmp_path):
    path = proxy.proxy_path(tmp_path, "source-hash", plan.media)
    expected = hashlib.sha256(b"source-hash:1").hexdigest()
    assert path == tmp_path / "proxies" / (expected + ".mp4")
    assert path != proxy.proxy_path(tmp_path, "other-hash", plan.media)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("audio_only", [False, True])
def test_build_proxy_real_media_and_cache(
    tmp_path, make_media, audio_only, monkeypatch
):
    spec = {"duration": 1}
    if audio_only:
        spec.update(
            name="source.wav",
            inputs=["sine=sample_rate=44100"],
            args=["-c:a", "pcm_s16le"],
        )
    source = make_media(tmp_path, spec)
    media = probe_media(source)
    path = proxy.proxy_path(tmp_path / "cache", file_hash(source), media)
    assert proxy.build_proxy(source, path, proxy.proxy_args(media)) == path
    result = probe_media(path)
    assert result.duration == pytest.approx(media.duration, abs=0.05)
    if not audio_only:
        assert (
            result.vcodec == "h264"
            and result.pix_fmt == "yuv420p"
            and result.audio_streams[0].channels == 2
        )
    else:
        assert not result.has_video and result.audio_streams[0].codec == "aac"
    assert not list(path.parent.glob(".preview-*"))
    monkeypatch.setattr(
        proxy.subprocess, "run", lambda *a, **kw: pytest.fail("cache must skip ffmpeg")
    )
    proxy.build_proxy(source, path, proxy.proxy_args(media))
    with pytest.raises(DeclipError, match="source"):
        proxy.build_proxy(source, source, [])


@pytest.mark.ffmpeg
def test_failed_proxy_cleans_partial_file(tmp_path):
    source = tmp_path / "invalid.mp4"
    source.write_bytes(b"invalid")
    target = tmp_path / "cache" / "proxy.mp4"
    with pytest.raises(DeclipError, match="Preview failed"):
        proxy.build_proxy(source, target, [])
    assert not target.exists() and not list(target.parent.iterdir())


@pytest.mark.parametrize("contents", [b"not json", b"{}"])
def test_external_invalid_edit_still_returns_stale(session, contents):
    revision = session.get()["revision"]
    session.path.write_bytes(contents)
    for method, route in (("PUT", "/api/edit"), ("POST", "/api/finish")):
        assert session.request(route, method, {"revision": revision})[:2] == (
            409,
            {"error": "stale", "revision": ed.revision_of(session.path)},
        )


@pytest.mark.ffmpeg
def test_proxy_preserves_stream_offset_and_container_origin(tmp_path, make_media):
    import subprocess

    source = make_media(
        tmp_path,
        {
            "duration": 2,
            "inputs": ["testsrc2=size=320x240:rate=30", "sine=sample_rate=48000"],
            "args": [
                "-filter:a",
                "adelay=200|200",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-output_ts_offset",
                "5",
            ],
        },
    )
    media = probe_media(source)
    target = proxy.proxy_path(tmp_path, file_hash(source), media)
    proxy.build_proxy(source, target, proxy.proxy_args(media))
    measured = probe_media(target)
    assert media.start_time > 4 and abs(measured.start_time) < 0.05
    # The audible onset remains 200 ms after frame zero, despite a nonzero container origin.
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(target),
            "-af",
            "silencedetect=noise=-40dB:d=0.1",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    import re

    end = float(re.search(r"silence_end: ([\d.]+)", result.stderr)[1])
    assert end == pytest.approx(0.2, abs=0.05)


def test_edit_cannot_switch_session_source(session):
    current = session.get()
    payload = current["edit_list"]
    payload["source"]["sha256"] = "other-hash"
    session.path.write_text(json.dumps(payload))
    assert session.request()[0] == 400
    assert (
        session.request(method="PUT", data={"revision": ed.revision_of(session.path)})[
            0
        ]
        == 400
    )


def test_finish_rechecks_source_after_fast_edit_requests(session, monkeypatch):
    import declip.media

    original = declip.media.file_hash
    calls = []

    def track(path, *, refresh=False):
        calls.append(refresh)
        return original(path, refresh=refresh)

    monkeypatch.setattr(ed, "file_hash", track)
    value = session.get()
    identifier = value["edit_list"]["cuts"][0]["id"]
    updated = session.request(
        method="PUT",
        data={
            "revision": value["revision"],
            "decisions": [{"id": identifier, "status": "rejected"}],
        },
    )[1]
    assert not calls  # Reading and deciding do not rehash multi-gigabyte media.
    assert (
        session.request("/api/finish", "POST", {"revision": updated["revision"]})[0]
        == 200
    )
    assert calls == [True]
