"""Token-protected, loopback-only review of one edit list and preview."""

from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from declip import editlist
from declip.contracts import (
    CutStatus,
    DeclipError,
    EditListError,
    ReviewResult,
    RevisionConflict,
    SourceMismatch,
)

_LOG = logging.getLogger(__name__)
_STATIC = Path(__file__).parent / "static"
_FILES = {
    "/static/" + name: (name, mime)
    for name, mime in (
        ("index.html", "text/html; charset=utf-8"),
        ("app.js", "text/javascript; charset=utf-8"),
        ("style.css", "text/css; charset=utf-8"),
    )
}
_MAX_BODY = 1024 * 1024


class _ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, path: Path, proxy: Path, port: int):
        initial = editlist.load_edit_list(path)
        self.source = Path(initial.source.path)
        self.source_sha256 = initial.source.sha256
        editlist.load_edit_list(path, source=self.source)
        if not proxy.is_file():
            raise DeclipError(f"Preview not found: {proxy}")
        self.edit_path, self.proxy = path, proxy
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.finished = False
        self.content_sha256 = None
        super().__init__(("127.0.0.1", port), _Handler)
        self.hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}
        self.origins = {"http://" + host for host in self.hosts}
        self.url = f"http://127.0.0.1:{self.server_port}/?token={self.token}"

    def current(self, *, validate_source=False):
        revision = editlist.revision_of(self.edit_path)
        value = editlist.load_edit_list(
            self.edit_path, source=self.source if validate_source else None
        )
        if (
            Path(value.source.path).resolve() != self.source.resolve()
            or value.source.sha256 != self.source_sha256
        ):
            raise SourceMismatch("Edit list no longer matches this session's preview")
        if editlist.revision_of(self.edit_path) != revision:
            raise RevisionConflict("Edit list changed while reading")
        return revision, value

    def payload(self, revision, value):
        timeline = editlist.effective_timeline(value)
        return {
            "revision": revision,
            "edit_list": value.to_dict(),
            "summary": {
                "counts": {
                    status.value: sum(c.status == status for c in value.cuts)
                    for status in CutStatus
                },
                "removed_seconds": sum(b - a for a, b in timeline.removed),
                "output_duration": timeline.duration_out,
                "fps": f"{timeline.fps.numerator}/{timeline.fps.denominator}"
                if timeline.fps
                else None,
                # The page uses the same snapped boundaries as render for playback.
                "removed_intervals": timeline.removed,
            },
        }


class _Handler(BaseHTTPRequestHandler):
    server: _ReviewServer

    def log_message(self, format, *args):
        # Debug logging is opt-in; omit query tokens from all request logs.
        _LOG.debug("%s %s", self.command, urlsplit(self.path).path)

    def reply(self, status, data, mime="application/json; charset=utf-8", extra=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "media-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def authorized(self):
        if self.headers.get("Host") not in self.server.hosts:
            self.reply(403, {"error": "host"})
            return False
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query).get("token", [""])[0]
        token = (
            self.headers.get("X-Declip-Token") or query
            if parsed.path.startswith("/api/")
            else query
        )
        if not hmac.compare_digest(
            token.encode("utf-8"), self.server.token.encode("ascii")
        ):
            self.reply(403, {"error": "token"})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        route = urlsplit(self.path).path
        try:
            if route == "/" or route in _FILES:
                name, mime = _FILES.get(
                    route, ("index.html", "text/html; charset=utf-8")
                )
                data = (_STATIC / name).read_text(encoding="utf-8")
                if name == "index.html":
                    data = data.replace("__DECLIP_TOKEN__", self.server.token)
                self.reply(200, data.encode("utf-8"), mime)
            elif route == "/media":
                self.media()
            elif route == "/api/edit":
                with self.server.lock:
                    revision, value = self.server.current()
                    self.reply(200, self.server.payload(revision, value))
            else:
                self.reply(404, {"error": "not found"})
        except RevisionConflict:
            self.stale()
        except (DeclipError, OSError) as exc:
            self.reply(400, {"error": str(exc)})

    def media(self):
        # Stream bounded chunks so a preview never has to fit in memory.
        with self.server.proxy.open("rb") as stream:
            size = self.server.proxy.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                valid = match and any(match.groups())
                if valid:
                    first, last = match.groups()
                    if first:
                        start, end = (
                            int(first),
                            min(int(last), size - 1) if last else size - 1,
                        )
                    else:
                        length = int(last)
                        start, end = max(0, size - length), size - 1
                        valid = length > 0
                    valid = valid and 0 <= start <= end < size
                if not valid:
                    self.reply(
                        416,
                        {"error": "range"},
                        extra={"Content-Range": f"bytes */{size}"},
                    )
                    return
                status = 206
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "video/mp4" if self.server.proxy.suffix == ".mp4" else "audio/mp4",
            )
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def stale(self):
        self.reply(
            409,
            {"error": "stale", "revision": editlist.revision_of(self.server.edit_path)},
        )

    def body(self):
        if self.headers.get("Origin") not in (None, *self.server.origins):
            self.reply(403, {"error": "origin"})
            return None
        if self.headers.get_content_type() != "application/json":
            self.reply(415, {"error": "application/json required"})
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= _MAX_BODY:
                raise ValueError("Invalid request length")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or not isinstance(data.get("revision"), str):
                raise ValueError("Request needs a revision string")
            return data
        except (ValueError, UnicodeError) as exc:
            self.reply(400, {"error": str(exc)})
            return None

    @staticmethod
    def changes(data):
        decisions, manual = data.get("decisions", []), data.get("manual_cuts", {})
        if not isinstance(decisions, list) or not isinstance(manual, dict):
            raise EditListError("Invalid decisions or manual_cuts")
        statuses = {}
        for row in decisions:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise EditListError("Decision needs an id and status")
            if row["id"] in statuses:
                raise EditListError("Duplicate decision id")
            statuses[row["id"]] = CutStatus(row["status"])
        adds, removes = manual.get("add", []), manual.get("remove", [])
        if not isinstance(adds, list) or not isinstance(removes, list):
            raise EditListError("Manual add and remove must be arrays")
        if any(not isinstance(row, dict) for row in adds):
            raise EditListError("Manual add must contain objects")
        if any(not isinstance(row, str) for row in removes) or len(set(removes)) != len(
            removes
        ):
            raise EditListError("Manual remove needs unique ids")
        return statuses, [(r["start"], r["end"], r["label"]) for r in adds], removes

    def write(self, finish):
        if not self.authorized():
            return
        if urlsplit(self.path).path != ("/api/finish" if finish else "/api/edit"):
            self.reply(404, {"error": "not found"})
            return
        data = self.body()
        if data is None:
            return
        try:
            with self.server.lock:
                # Check bytes before parsing. An external malformed edit is stale too.
                disk_revision = editlist.revision_of(self.server.edit_path)
                if self.server.finished or disk_revision != data["revision"]:
                    self.stale()
                    return
                revision, value = self.server.current(validate_source=finish)
                if revision != data["revision"]:
                    self.stale()
                    return
                if finish:
                    count = sum(c.status == CutStatus.PROPOSED for c in value.cuts)
                    if count:
                        self.reply(
                            422, {"error": "unresolved", "unresolved_count": count}
                        )
                        return
                    changed = editlist.mark_review_passed(
                        value, now=datetime.now(timezone.utc)
                    )
                else:
                    statuses, adds, removes = self.changes(data)
                    changed = editlist.apply_decisions(
                        value,
                        decisions=statuses,
                        add_manual=adds,
                        remove_manual=removes,
                    )
                revision = editlist.save_edit_list(
                    self.server.edit_path, changed, expected_revision=revision
                )
                if finish:
                    self.server.finished = True
                    self.server.content_sha256 = changed.review.content_sha256
                    self.reply(
                        200,
                        {"passed": True, "content_sha256": self.server.content_sha256},
                    )
                else:
                    self.reply(200, self.server.payload(revision, changed))
            if finish:
                self.server.shutdown()
        except RevisionConflict:
            self.stale()
        except (DeclipError, ValueError, TypeError, KeyError, OSError) as exc:
            self.reply(400, {"error": str(exc)})

    def do_PUT(self):
        self.write(False)

    def do_POST(self):
        self.write(True)


def serve(
    edit_list_path: Path,
    proxy_path: Path,
    *,
    open_browser: bool = True,
    port: int = 0,
    on_ready: Callable[[str], None] | None = None,
) -> ReviewResult:
    server = _ReviewServer(edit_list_path, proxy_path, port)
    try:
        if on_ready:
            on_ready(server.url)
        if open_browser:
            webbrowser.open(server.url)
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    value = editlist.load_edit_list(edit_list_path)
    return ReviewResult(
        server.finished,
        edit_list_path,
        server.content_sha256,
        sum(c.status == CutStatus.PROPOSED for c in value.cuts),
        server.url,
    )
