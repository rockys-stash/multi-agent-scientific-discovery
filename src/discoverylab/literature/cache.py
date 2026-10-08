"""A content-addressed cache of raw scholarly-API responses.

Every request is keyed by its method, URL and sorted parameters. The stored entry keeps the raw
response body, the HTTP status, the URL and the retrieval time, so a run can be replayed exactly
and any record can be traced back to what the source returned.

Modes:
- ``live``: always fetch, store the response;
- ``record``: use the cache, fetch on a miss;
- ``replay``: cache only; a miss raises ``CacheMissError`` (offline, reproducible).
"""

from __future__ import annotations

import json
import threading
import time
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

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _http_get(self, url: str, params: dict[str, str]) -> tuple[int, str]:
        host = httpx.URL(url).host
        with self._lock:  # polite: at most one request per host per interval
            wait = self._last.get(host, 0.0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()
        r = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30.0,
                      follow_redirects=True)  # fmt: skip
        return r.status_code, r.text

    def get(self, url: str, params: dict[str, str] | None = None) -> CachedResponse:
        params = {k: str(v) for k, v in (params or {}).items()}
        key = request_key(url, params)
        path = self._path(key)
        if self.mode != "live" and path.exists():
            self.hits += 1
            return CachedResponse(**json.loads(path.read_text(encoding="utf-8")))
        if self.mode == "replay":
            raise CacheMissError(f"{url} {params} is not in the cache")
        self.misses += 1
        status, body = self._fetch(url, params)
        resp = CachedResponse(key, url, params, status, body,
                              datetime.now(UTC).isoformat(timespec="seconds"))  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resp.__dict__, ensure_ascii=False), encoding="utf-8")
        return resp
