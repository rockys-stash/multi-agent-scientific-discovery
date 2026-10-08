"""Command line: ``discoverylab run | resume | verify-log | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="discoverylab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the research workflow on a question")
    r.add_argument("question", type=Path)
    r.add_argument("--reasoner", choices=["rule", "claude"], default="rule")
    r.add_argument("--model-critic", action="store_true", help="add the language-model critic to the rule critic")
    r.add_argument("--cache-mode", choices=["live", "record", "replay"], default="record")
    r.add_argument("--sources", default="openalex,crossref,arxiv,semanticscholar")
    r.add_argument("--human", default="none", help="'none' or a corrections JSON file written by the reviewer")
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--cache", type=Path, default=Path("cache"))

    rs = sub.add_parser("resume", help="continue a run paused at a human checkpoint")
    rs.add_argument("run_dir", type=Path)
    rs.add_argument("--human", required=True)
    rs.add_argument("--cache", type=Path, default=Path("cache"))
    rs.add_argument("--cache-mode", choices=["live", "record", "replay"], default="record")

    v = sub.add_parser("verify-log", help="check a run's process log has not been edited")
    v.add_argument("run_dir", type=Path)

    s = sub.add_parser("serve", help="serve the console and its API")
    s.add_argument("--host", default=os.environ.get("DISCOVERYLAB_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("DISCOVERYLAB_PORT", "8000")))
    s.add_argument("--runs", type=Path, default=Path("runs"))

    args = ap.parse_args(argv)

    if args.cmd == "run":
        from discoverylab.models import RunState
        from discoverylab.run import build_registry, execute, human_gate, load_question, make_reasoner, new_run_id

        q, op = load_question(args.question)
        reasoner = make_reasoner(args.reasoner)
        critic = make_reasoner("claude") if args.model_critic else None
        registry = build_registry(args.cache, args.cache_mode, args.sources.split(","))
        state = RunState(
            run_id=new_run_id(q.id, reasoner.name + ("+mc" if critic else "")), question=q, reasoner=reasoner.name
        )
        state = execute(args.runs / state.run_id, state, registry, reasoner, op, human_gate(args.human), critic)
        print(json.dumps({"run_id": state.run_id, "status": state.status, "stage": state.stage.value}))
        return 0
    if args.cmd == "resume":
        from discoverylab.models import RunState
        from discoverylab.run import build_registry, execute, human_gate, load_question, make_reasoner

        state = RunState.model_validate_json((args.run_dir / "state.json").read_text(encoding="utf-8"))
        state.status = "running"
        qfile = next(Path("configs/questions").glob(f"{state.question.id}.yaml"), None)
        op = load_question(qfile)[1] if qfile else None
        registry = build_registry(args.cache, args.cache_mode)
        state = execute(args.run_dir, state, registry, make_reasoner(state.reasoner), op, human_gate(args.human))
        print(json.dumps({"run_id": state.run_id, "status": state.status, "stage": state.stage.value}))
        return 0
    if args.cmd == "verify-log":
        from discoverylab.log import verify

        rep = verify(args.run_dir / "log.jsonl")
        print(json.dumps(rep.__dict__))
        return 0 if rep.valid else 1
    if args.cmd == "serve":
        if args.host not in LOOPBACK and not os.environ.get("DISCOVERYLAB_API_TOKEN"):
            print(f"refusing to serve on {args.host} without DISCOVERYLAB_API_TOKEN", file=sys.stderr)
            return 2
        import uvicorn

        from discoverylab.api import create_app

        uvicorn.run(create_app(args.runs), host=args.host, port=args.port, log_level="info")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
