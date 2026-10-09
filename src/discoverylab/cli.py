"""Command line: ``discoverylab run | resume | verify-log | experiment | serve``."""

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
    r.add_argument("--reasoner", choices=["rule", "claude", "local"], default="rule")
    r.add_argument("--model-critic", choices=["claude", "local"], help="add a language-model critic to the rule critic")
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

    e = sub.add_parser("experiment", help="run one of the experiments E1-E5 from its config")
    e.add_argument(
        "name", choices=["e1_runs", "e1_compare", "e2_verifier", "e3_critic", "e4_fluency", "e5_reproducibility"]
    )
    e.add_argument("--config", type=Path, help="defaults to configs/experiments/<name>.yaml")
    e.add_argument("--runs", type=Path, default=Path("runs"))
    e.add_argument("--cache", type=Path, default=Path("cache"))
    e.add_argument("--results", type=Path, default=Path("results"))

    rm = sub.add_parser("results", help="render the latest result of every experiment as Markdown")
    rm.add_argument("--results", type=Path, default=Path("results"))
    rm.add_argument("--out", type=Path, default=Path("docs/generated/results.md"))

    s = sub.add_parser("serve", help="serve the console and its API")
    s.add_argument("--host", default=os.environ.get("DISCOVERYLAB_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("DISCOVERYLAB_PORT", "8000")))
    s.add_argument("--runs", type=Path, default=Path("runs"))

    args = ap.parse_args(argv)

    if args.cmd == "results":
        from discoverylab.experiments.results_md import write

        print(write(args.results, args.out))
        return 0

    if args.cmd == "run":
        from discoverylab.models import RunState
        from discoverylab.run import build_registry, execute, human_gate, load_question, make_reasoner, new_run_id

        q, op = load_question(args.question)
        reasoner = make_reasoner(args.reasoner)
        critic = make_reasoner(args.model_critic) if args.model_critic else None
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
        base = state.reasoner.removesuffix("+mc")
        critic = make_reasoner(base) if state.reasoner.endswith("+mc") else None
        state = execute(args.run_dir, state, registry, make_reasoner(base), op, human_gate(args.human), critic)
        print(json.dumps({"run_id": state.run_id, "status": state.status, "stage": state.stage.value}))
        return 0
    if args.cmd == "verify-log":
        from discoverylab.log import verify

        rep = verify(args.run_dir / "log.jsonl")
        print(json.dumps(rep.__dict__))
        return 0 if rep.valid else 1
    if args.cmd == "experiment":
        return _experiment(args)
    if args.cmd == "serve":
        if args.host not in LOOPBACK and not os.environ.get("DISCOVERYLAB_API_TOKEN"):
            print(f"refusing to serve on {args.host} without DISCOVERYLAB_API_TOKEN", file=sys.stderr)
            return 2
        import uvicorn

        from discoverylab.api import create_app

        uvicorn.run(create_app(args.runs), host=args.host, port=args.port, log_level="info")
        return 0
    return 1


def _experiment(args: argparse.Namespace) -> int:
    from discoverylab.experiments import e1_compare, e1_runs, e2_verifier, e3_critic, e4_fluency, e5_reproducibility
    from discoverylab.experiments.common import load_config
    from discoverylab.run import build_registry

    cfg = load_config(args.config or Path("configs/experiments") / f"{args.name}.yaml")
    registry = build_registry(args.cache, cfg.get("cache_mode", "replay"), cfg.get("sources"))
    if args.name == "e1_runs":
        out = e1_runs.run(cfg, args.runs, registry, results=args.results)
    elif args.name == "e1_compare":
        out = e1_compare.run(cfg, args.runs, results=args.results)
    elif args.name == "e2_verifier":
        out = e2_verifier.run(cfg, args.runs, registry, results=args.results)
    elif args.name == "e3_critic":
        out = e3_critic.run(cfg, args.runs, results=args.results)
    elif args.name == "e4_fluency":
        out = e4_fluency.run(cfg, args.runs, registry, results=args.results)
    else:
        out = e5_reproducibility.run(cfg, args.runs, registry, results=args.results)
    print(json.dumps({"experiment": args.name, "results": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
