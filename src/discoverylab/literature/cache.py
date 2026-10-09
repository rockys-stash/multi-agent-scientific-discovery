"""A content-addressed cache of raw scholarly-API responses.

Every request is keyed by its method, URL and sorted parameters. The stored entry keeps the raw
response body, the HTTP status, the URL and the retrieval time, so a run can be replayed exactly
and any record can be traced back to what the source returned.

Modes:
- ``live``: always fetch, store the response;
- ``record``: use the cache, fetch on a miss;
- ``replay``: cache only; a miss raises ``CacheMissError`` (offline, reproducible).

A response that is replaced (record mode refetching an earlier "try later") is archived, never
overwritten, and a run records which version of each response it consumed (``begin_trace``), so
replaying that run serves exactly those versions (``plan_replay``) even after the cache has moved on.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from discoverylab.canonical import canonical, sha256

Mode = Literal["live", "record", "replay"]
USER_AGENT = "discoverylab/0.1 (research prototype; https://github.com/rockys-stash/multi-agent-scientific-discovery)"


class CacheMissError(LookupError):
    pass


# Statuses that say "try later", not "no such record". They are retried with backoff and, if they
# persist, surface as SourceUnavailable so a rate limit is never read as a missing paper.
TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})
MAX_TRIES = 4
MAX_WAIT_S = 60.0
# Hosts whose terms ask for more than the default spacing (arXiv API: one request every 3 s).
HOST_INTERVAL_S = {"export.arxiv.org": 3.0}


@dataclass(frozen=True)
class CachedResponse:
    key: str
    url: str
    params: dict[str, str]
    status: int
    body: str
    retrieved_at: str

    def json(self) -> Any:
        return json.loads(self.body)


def request_key(url: str, params: dict[str, str]) -> str:
    return sha256(canonical({"method": "GET", "url": url, "params": dict(sorted(params.items()))}))


class ResponseCache:
    def __init__(
        self,
        root: Path,
        mode: Mode = "record",
        min_interval_s: float = 1.0,
        fetch: Callable[[str, dict[str, str]], tuple[int, str]] | None = None,
    ) -> None:
        self.root, self.mode = root, mode
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self._fetch = fetch or self._http_get
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self._trace: list[dict[str, Any]] | None = None
        self._plan: dict[str, deque[str]] = {}

    def begin_trace(self) -> None:
        """Start recording which response version each request consumed."""
        self._trace = []

    def end_trace(self) -> list[dict[str, Any]]:
        trace, self._trace = self._trace or [], None
        return trace

    def plan_replay(self, trace: list[dict[str, Any]]) -> None:
        """Serve each key's recorded versions in order; keys not in the plan get the current version."""
        self._plan = {}
        for t in trace:
            self._plan.setdefault(t["key"], deque()).append(t["retrieved_at"])

    def _version_path(self, key: str, retrieved_at: str) -> Path:
        return self.root / key[:2] / f"{key}@{retrieved_at.replace(':', '')}.json"

    def _load(self, path: Path) -> CachedResponse:
        return CachedResponse(**json.loads(path.read_text(encoding="utf-8")))

    def _record(self, resp: CachedResponse) -> CachedResponse:
        if self._trace is not None:
            self._trace.append({"key": resp.key, "retrieved_at": resp.retrieved_at, "status": resp.status})
        return resp

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _throttle(self, host: str) -> None:
        with self._lock:  # polite: at most one request per host per interval
            interval = max(self.min_interval_s, HOST_INTERVAL_S.get(host, 0.0))
            wait = self._last.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def _http_get(self, url: str, params: dict[str, str]) -> tuple[int, str]:
        host = httpx.URL(url).host
        status, body = 0, ""
        for attempt in range(MAX_TRIES):
            self._throttle(host)
            try:
                r = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30.0,
                              follow_redirects=True)  # fmt: skip
                status, body = r.status_code, r.text
                retry_after = r.headers.get("retry-after", "")
            except httpx.TransportError as exc:  # recorded as a gateway failure, not a missing record
                status, body, retry_after = 504, f"transport error: {exc}", ""
            if status not in TRANSIENT or attempt == MAX_TRIES - 1:
                break
            backoff = 2.0 * 2**attempt
            if retry_after.isdigit():
                backoff = max(backoff, float(retry_after))
            if backoff > MAX_WAIT_S:  # e.g. a daily quota: waiting will not help this run
                break
            time.sleep(backoff)
        return status, body

    def get(self, url: str, params: dict[str, str] | None = None) -> CachedResponse:
        params = {k: str(v) for k, v in (params or {}).items()}
        key = request_key(url, params)
        path = self._path(key)
        planned = self._plan.get(key)
        if planned:
            at = planned.popleft()
            current = self._load(path) if path.exists() else None
            if current is not None and current.retrieved_at == at:
                self.hits += 1
                return self._record(current)
            archived = self._version_path(key, at)
            if archived.exists():
                self.hits += 1
                return self._record(self._load(archived))
            raise CacheMissError(f"{url} {params} retrieved at {at} is not in the cache")
        if self.mode != "live" and path.exists():
            cached = self._load(path)
            # replay returns exactly what was recorded; record retries an earlier "try later"
            if self.mode == "replay" or cached.status not in TRANSIENT:
                self.hits += 1
                return self._record(cached)
        if self.mode == "replay":
            raise CacheMissError(f"{url} {params} is not in the cache")
        self.misses += 1
        status, body = self._fetch(url, params)
        resp = CachedResponse(key, url, params, status, body,
                              datetime.now(UTC).isoformat(timespec="microseconds"))  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():  # keep the version earlier runs consumed
            previous = self._load(path)
            path.replace(self._version_path(key, previous.retrieved_at))
        path.write_text(json.dumps(resp.__dict__, ensure_ascii=False), encoding="utf-8")
        return self._record(resp)
