"""HTTP API for the Lattice console.

- Reads are open on localhost unless ``DISCOVERYLAB_API_TOKEN`` is set; then every ``/api`` route
  except ``/api/health`` needs ``Authorization: Bearer <token>``.
- The only write is a reviewer's corrections for a paused run. It always needs the token, and it
  only stores the corrections; the run is resumed from the command line.
- Every client is rate limited.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from discoverylab.evaluate import run_metrics
from discoverylab.graph import build_graph
from discoverylab.human import CHECKPOINTS
from discoverylab.log import ProcessLog, verify
from discoverylab.models import Correction, RunState

EXPERIMENTS = ("e1_runs", "e2_verifier", "e3_critic", "e4_fluency", "e5_reproducibility")
_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,120}$")


class RateLimiter:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate, self.burst = rate, burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            if len(self._buckets) > 10_000:
                self._buckets = dict(list(self._buckets.items())[-5_000:])
            return True


class GuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, token: str | None, limiter: RateLimiter) -> None:
        super().__init__(app)
        self.token, self.limiter = token, limiter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path.startswith("/api"):
            client = request.client.host if request.client else "unknown"
            if not self.limiter.allow(client):
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "1"})
            needs_token = request.method != "GET" or (self.token and path != "/api/health")
            if needs_token:
                supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                if not self.token or not hmac.compare_digest(supplied.encode(), self.token.encode()):
                    return JSONResponse({"detail": "unauthorized"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response


class CorrectionsIn(BaseModel):
    stage: str
    actor: str = Field(min_length=1, max_length=60)
    corrections: list[Correction] = Field(default_factory=list, max_length=200)


def _safe(segment: str) -> str:
    if not _ID.match(segment) or segment in {".", ".."}:
        raise HTTPException(400, "invalid identifier")
    return segment


def create_app(runs: Path, results: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    runs = runs.resolve()
    root = Path(__file__).resolve().parents[2]
    results = (results or root / "results").resolve()
    static = (static_dir or root / "web" / "dist").resolve()
    app = FastAPI(title="discoverylab", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        GuardMiddleware,
        token=os.environ.get("DISCOVERYLAB_API_TOKEN") or None,
        limiter=RateLimiter(rate=float(os.environ.get("DISCOVERYLAB_RATE_PER_SEC", "30")), burst=120),
    )

    def run_dir(run_id: str) -> Path:
        d = runs / _safe(run_id)
        if not (d / "state.json").is_file():
            raise HTTPException(404, "run not found")
        return d

    def load(d: Path) -> tuple[RunState, dict[str, Any] | None]:
        state = RunState.model_validate_json((d / "state.json").read_text(encoding="utf-8"))
        vf = d / "verification.json"
        return state, json.loads(vf.read_text(encoding="utf-8")) if vf.is_file() else None

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        items = []
        for d in sorted(runs.glob("*/state.json"), reverse=True) if runs.exists() else []:
            state, ver = load(d.parent)
            m = run_metrics(state, ver)
            items.append({"run_id": state.run_id, "question": state.question.text, "question_id": state.question.id,
                          "reasoner": state.reasoner, "status": state.status, "stage": state.stage.value,
                          "citation_accuracy": m["citation_accuracy"], "evidence_correctness": m["evidence_correctness"],
                          "corrections": m["human_corrections"], "human_reviewed": m["human_reviewed"],
                          "verdicts": m["verdicts"]})  # fmt: skip
        return {"items": items}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        d = run_dir(run_id)
        state, ver = load(d)
        rep = verify(d / "log.jsonl")
        return {"state": state.model_dump(mode="json"), "verification": ver, "metrics": run_metrics(state, ver),
                "graph": build_graph(state, ver), "log_integrity": rep.__dict__,
                "checkpoints": [s.value for s in CHECKPOINTS]}  # fmt: skip

    @app.get("/api/runs/{run_id}/log")
    def get_log(run_id: str) -> dict[str, Any]:
        d = run_dir(run_id)
        events = [e.__dict__ for e in ProcessLog(d / "log.jsonl").events()]
        return {"events": events, "integrity": verify(d / "log.jsonl").__dict__}

    @app.post("/api/runs/{run_id}/corrections")
    def post_corrections(run_id: str, body: CorrectionsIn) -> dict[str, Any]:
        d = run_dir(run_id)
        state, _ = load(d)
        if state.status != "awaiting_review" or state.stage.value != body.stage:
            raise HTTPException(409, f"run is not waiting for review of {body.stage!r}")
        path = d / "corrections.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data[body.stage] = [
            {**c.model_dump(mode="json", exclude={"stage"}), "actor": body.actor} for c in body.corrections
        ]
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return {
            "saved": len(body.corrections),
            "resume": f"discoverylab resume runs/{state.run_id} --human runs/{state.run_id}/corrections.json",
        }

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        out = []
        for exp in EXPERIMENTS:
            latest = results / exp / "LATEST"
            if latest.is_file():
                prov = json.loads((results / exp / latest.read_text().strip() / "provenance.json").read_text())
                out.append({"experiment": exp, "status": "complete", **prov})
            else:
                out.append({"experiment": exp, "status": "pending"})
        return {"items": out}

    @app.get("/api/experiments/{exp}")
    def experiment(exp: str) -> dict[str, Any]:
        latest = results / _safe(exp) / "LATEST"
        if not latest.is_file():
            raise HTTPException(404, "experiment has not been run")
        run = results / exp / latest.read_text().strip()
        return {
            **json.loads((run / "summary.json").read_text()),
            "provenance": json.loads((run / "provenance.json").read_text()),
        }

    @app.get("/api/{rest:path}")
    def api_404(rest: str) -> Response:
        return JSONResponse({"detail": "not found"}, status_code=404)

    @app.get("/{rest:path}")
    def spa(rest: str) -> Response:
        if not static.exists():
            return JSONResponse({"detail": "console not built (npm --prefix web run build)"}, status_code=404)
        candidate = (static / rest).resolve()
        if rest and candidate.is_file() and candidate.is_relative_to(static):
            return FileResponse(candidate)
        return FileResponse(static / "index.html")

    return app
