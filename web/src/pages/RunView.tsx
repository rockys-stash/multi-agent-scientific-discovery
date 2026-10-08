import { useEffect, useMemo, useState } from "react";
import { Link, NavLink, useParams } from "react-router-dom";
import { Graph, nodeName } from "../components/Graph";
import { CritiqueList, Inspector } from "../components/Inspector";
import {
  Empty,
  ErrorPanel,
  Hash,
  Loading,
  Mark,
  PageHead,
  Segmented,
} from "../components/ui";
import { api, useApi, type LogEvent, type RunDetail } from "../lib/api";
import {
  ACTOR_LABEL,
  KIND_LABEL,
  OWNER,
  REASONER,
  statusOf,
  STAGES,
  STAGE_LABEL,
  frac,
  pct,
  verificationMark,
} from "../lib/format";

type Tab = "network" | "process" | "citations" | "corrections";

function useNarrow(): boolean {
  const q = "(max-width: 760px)";
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.matchMedia(q).matches,
  );
  useEffect(() => {
    const m = window.matchMedia(q);
    const on = () => setNarrow(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return narrow;
}

function Summary({ run }: { run: RunDetail }) {
  const m = run.metrics;
  const corr = Object.entries(m.human_corrections);
  return (
    <div className="stat-row">
      <div className="stat">
        <div className="stat-label">Citations verified</div>
        <div className="stat-value num">{pct(m.citation_accuracy)}</div>
        <div className="muted small">
          {frac(m.citation_accuracy)} distinct sources
        </div>
      </div>
      <div className="stat">
        <div className="stat-label">Evidence correct</div>
        <div className="stat-value num">{pct(m.evidence_correctness)}</div>
        <div className="muted small">
          {frac(m.evidence_correctness)} quotes found in a verified source
        </div>
      </div>
      <div className="stat">
        <div className="stat-label">Critiques</div>
        <div className="stat-value num">{run.state.critiques.length}</div>
        <div className="muted small">
          {Object.entries(m.critique_resolution)
            .map(([k, v]) => `${v} ${k}`)
            .join(", ") || "none raised"}
        </div>
      </div>
      <div className="stat">
        <div className="stat-label">Human corrections</div>
        <div className="stat-value num">
          {m.human_reviewed ? corr.reduce((a, [, v]) => a + (v ?? 0), 0) : "–"}
        </div>
        <div className="muted small">
          {m.human_reviewed
            ? `across ${corr.length} checkpoints`
            : "no human reviewed this run"}
        </div>
      </div>
      <div className="stat">
        <div className="stat-label">Process log</div>
        <div className="stat-value">
          {run.log_integrity.valid ? (
            <Mark tone="ok" glyph="✓">
              Intact
            </Mark>
          ) : (
            <Mark tone="bad" glyph="✕">
              Broken at {run.log_integrity.first_bad_seq}
            </Mark>
          )}
        </div>
        <div className="muted small">
          {run.log_integrity.events} chained events
        </div>
      </div>
    </div>
  );
}

function NetworkTab({ run }: { run: RunDetail }) {
  const narrow = useNarrow();
  const [view, setView] = useState<"graph" | "list">(narrow ? "list" : "graph");
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => setView(narrow ? "list" : "graph"), [narrow]);
  const nodes = run.graph.nodes.filter((n) => n.kind !== "critique");
  const critiqueCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of run.state.critiques)
      m.set(c.target_id, (m.get(c.target_id) ?? 0) + 1);
    return m;
  }, [run.state.critiques]);
  const node = nodes.find((n) => n.id === selected) ?? null;
  return (
    <div className="stack">
      <div className="toolbar">
        <Segmented
          label="Network view"
          options={[
            { id: "graph", label: "Graph" },
            { id: "list", label: "List" },
          ]}
          value={view}
          onChange={setView}
        />
        <span className="legend small">
          <span className="key">
            <svg width="12" height="12" aria-hidden="true">
              <circle cx="6" cy="6" r="4.5" className="lg-shape" />
            </svg>
            paper
          </span>
          <span className="key">
            <svg width="12" height="12" aria-hidden="true">
              <rect x="2" y="2" width="8" height="8" className="lg-shape" />
            </svg>
            evidence
          </span>
          <span className="key">
            <svg width="12" height="12" aria-hidden="true">
              <path d="M6 1 11 6 6 11 1 6z" className="lg-shape" />
            </svg>
            gap
          </span>
          <span className="key">
            <svg width="12" height="12" aria-hidden="true">
              <path
                d="M1 3.5 6 1 11 3.5 11 8.5 6 11 1 8.5z"
                className="lg-shape"
              />
            </svg>
            hypothesis
          </span>
          <span className="key">
            <svg width="22" height="12" aria-hidden="true">
              <path d="M1 6h20" className="edge verified" />
            </svg>
            verified quote
          </span>
          <span className="key">
            <svg width="22" height="12" aria-hidden="true">
              <path d="M1 6h20" className="edge fail" />
            </svg>
            failed check
          </span>
        </span>
      </div>
      <div className={node ? "network with-inspector" : "network"}>
        {view === "graph" ? (
          <div className="panel">
            <Graph
              nodes={nodes}
              edges={run.graph.edges}
              selected={selected}
              onSelect={setSelected}
              critiqueCount={critiqueCount}
            />
          </div>
        ) : (
          <div className="panel">
            <div className="panel-body stack">
              {(
                [
                  "question",
                  "paper",
                  "evidence",
                  "gap",
                  "hypothesis",
                  "design",
                  "result",
                  "conclusion",
                ] as const
              ).map((k) => {
                const items = nodes.filter((n) => n.kind === k);
                if (!items.length) return null;
                return (
                  <section key={k} aria-label={KIND_LABEL[k]}>
                    <h3>
                      {KIND_LABEL[k]}{" "}
                      <span className="muted">{items.length}</span>
                    </h3>
                    <ul className="node-list">
                      {items.map((n) => (
                        <li key={n.id}>
                          <button
                            type="button"
                            className="link-btn"
                            aria-pressed={n.id === selected}
                            onClick={() =>
                              setSelected(n.id === selected ? null : n.id)
                            }
                          >
                            {nodeName(n)}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </section>
                );
              })}
            </div>
          </div>
        )}
        {node && (
          <Inspector
            node={node}
            run={run}
            onSelect={setSelected}
            onClose={() => setSelected(null)}
          />
        )}
      </div>
    </div>
  );
}

function ProcessTab({ run, runId }: { run: RunDetail; runId: string }) {
  const log = useApi<{ events: LogEvent[] }>(api.log(runId));
  const [stage, setStage] = useState<string>("all");
  const s = run.state;
  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-head">
          <h2>Workflow</h2>
          <span className="muted small">
            Director hands each stage to its agent; the Critic reviews it before
            the next starts.
          </span>
        </div>
        <ol className="stages">
          {STAGES.map((st) => {
            const done = s.completed.includes(st.id);
            const crit = s.critiques.filter(
              (c) => c.stage === st.id && c.round !== 99,
            );
            const final = s.critiques.filter(
              (c) => c.stage === st.id && c.round === 99,
            );
            const rev = s.reviewed[st.id];
            return (
              <li
                key={st.id}
                className={done ? "done" : s.stage === st.id ? "current" : ""}
              >
                <div className="stage-name">
                  <Mark
                    tone={
                      done ? "ok" : s.stage === st.id ? "attest" : "neutral"
                    }
                    glyph={done ? "✓" : s.stage === st.id ? "◐" : "○"}
                  >
                    {st.label}
                  </Mark>
                  <span className="muted small">{OWNER[st.id]}</span>
                </div>
                <div className="small secondary">
                  {crit.length} critique{crit.length === 1 ? "" : "s"} in review
                  rounds
                  {final.length ? `, ${final.length} in the final review` : ""}
                  {rev
                    ? ` · checkpoint: ${rev === "none" ? "no human reviewer" : "reviewed by a person"}`
                    : ""}
                </div>
              </li>
            );
          })}
        </ol>
      </div>
      <div className="panel">
        <div className="panel-head">
          <h2>Critique cycle</h2>
        </div>
        <div className="panel-body">
          {s.critiques.length ? (
            <CritiqueList items={s.critiques} />
          ) : (
            <Empty title="No critiques raised">
              Every stage passed the critic without a finding.
            </Empty>
          )}
        </div>
      </div>
      <div className="panel">
        <div className="panel-head">
          <h2>Process log</h2>
          <label className="field">
            Stage
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              <option value="all">All stages</option>
              {["question", ...STAGES.map((x) => x.id), "complete"].map((x) => (
                <option key={x} value={x}>
                  {STAGE_LABEL[x] ?? x}
                </option>
              ))}
            </select>
          </label>
        </div>
        {log.error ? (
          <ErrorPanel
            error={log.error}
            retry={log.retry}
            what="the process log"
          />
        ) : !log.data ? (
          <Loading rows={6} label="Loading log" />
        ) : (
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Process log"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col" className="r">
                    #
                  </th>
                  <th scope="col">Actor</th>
                  <th scope="col">Kind</th>
                  <th scope="col">Stage</th>
                  <th scope="col">Content</th>
                  <th scope="col">Hash</th>
                </tr>
              </thead>
              <tbody>
                {log.data.events
                  .filter((e) => stage === "all" || e.stage === stage)
                  .map((e) => (
                    <tr key={e.seq}>
                      <td className="r num">{e.seq}</td>
                      <td>{ACTOR_LABEL(e.actor)}</td>
                      <td>{e.kind.replace("_", " ")}</td>
                      <td>{STAGE_LABEL[e.stage] ?? e.stage}</td>
                      <td className="small mono log-data">
                        {JSON.stringify(e.data).slice(0, 220)}
                      </td>
                      <td>
                        <Hash value={e.hash} n={8} />
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function CitationsTab({ run }: { run: RunDetail }) {
  const checks = run.verification?.evidence ?? [];
  if (!run.verification)
    return (
      <Empty title="Not verified yet">
        Verification runs when the run completes.
      </Empty>
    );
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Every emitted citation</h2>
        <span className="muted small">
          Re-resolved through the indexes, not taken from the agent's own
          records.
        </span>
      </div>
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label="Citations"
      >
        <table>
          <thead>
            <tr>
              <th scope="col">Evidence</th>
              <th scope="col">Cited as</th>
              <th scope="col">Source says</th>
              <th scope="col">Quote</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {checks.map((c) => {
              const e = run.state.evidence.find((x) => x.id === c.evidence_id);
              const v = verificationMark(
                c.correct
                  ? "verified"
                  : c.citation.status !== "verified"
                    ? c.citation.status
                    : `quote_${c.quote}`,
              );
              return (
                <tr key={c.evidence_id}>
                  <td className="mono">{c.evidence_id}</td>
                  <td className="small">
                    <div className="mono">{c.citation.identifier}</div>
                    {e && (
                      <div className="secondary">
                        {e.citation.first_author} {e.citation.year ?? ""} ·{" "}
                        {e.citation.title}
                      </div>
                    )}
                  </td>
                  <td className="small">
                    {c.citation.resolved_title ? (
                      <>
                        <div>{c.citation.resolved_title}</div>
                        <div className="secondary">
                          {c.citation.resolved_first_author}{" "}
                          {c.citation.resolved_year ?? ""} · via{" "}
                          {c.citation.resolved_by}
                        </div>
                      </>
                    ) : (
                      <span className="muted">
                        no record (
                        {c.citation.tried.join(", ") || "not an identifier"})
                      </span>
                    )}
                  </td>
                  <td className="small">{c.quote.replace(/_/g, " ")}</td>
                  <td>
                    <Mark tone={v.tone} glyph={v.glyph} title={v.help}>
                      {v.word}
                    </Mark>
                    {c.citation.problems.length > 0 && (
                      <div className="small secondary">
                        {c.citation.problems.join("; ")}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CorrectionsTab({ run }: { run: RunDetail }) {
  const s = run.state;
  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-head">
          <h2>Checkpoints</h2>
        </div>
        <div className="panel-body">
          <ul className="checklist">
            {run.checkpoints.map((c) => {
              const r = s.reviewed[c];
              return (
                <li key={c}>
                  <Mark
                    tone={r ? (r === "none" ? "neutral" : "ok") : "attest"}
                    glyph={r ? (r === "none" ? "–" : "✓") : "◐"}
                  >
                    {STAGE_LABEL[c] ?? c}
                  </Mark>{" "}
                  <span className="secondary small">
                    {!r
                      ? "not reached or waiting for review"
                      : r === "none"
                        ? "no human reviewer configured"
                        : "reviewed by a person"}
                  </span>
                </li>
              );
            })}
          </ul>
          {s.status === "awaiting_review" && (
            <p className="notice">
              Paused at {STAGE_LABEL[s.stage]}. Write the reviewer's decisions
              to <code>runs/{s.run_id}/corrections.json</code> and run{" "}
              <code>
                discoverylab resume runs/{s.run_id} --human runs/{s.run_id}
                /corrections.json
              </code>
              .
            </p>
          )}
        </div>
      </div>
      <div className="panel">
        <div className="panel-head">
          <h2>Correction log</h2>
        </div>
        {s.corrections.length === 0 ? (
          <Empty
            title={
              run.metrics.human_reviewed ? "No corrections" : "No human review"
            }
          >
            {run.metrics.human_reviewed
              ? "The reviewer approved every checkpoint without changes."
              : "Nobody reviewed this run, so there is no correction count to report. Zero corrections would mean a person approved everything."}
          </Empty>
        ) : (
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Corrections"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Stage</th>
                  <th scope="col">Target</th>
                  <th scope="col">Action</th>
                  <th scope="col">Reason</th>
                  <th scope="col">By</th>
                </tr>
              </thead>
              <tbody>
                {s.corrections.map((c, i) => (
                  <tr key={i}>
                    <td>{STAGE_LABEL[c.stage] ?? c.stage}</td>
                    <td className="mono">{c.target_id}</td>
                    <td>{c.action}</td>
                    <td>{c.reason || "–"}</td>
                    <td>{c.actor}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export function RunView({ tab }: { tab: Tab }) {
  const { runId = "" } = useParams();
  const run = useApi<RunDetail>(api.run(runId));
  const base = `/runs/${encodeURIComponent(runId)}`;
  if (run.error) {
    if (run.error.status === 404)
      return (
        <div className="panel">
          <Empty title="Run not found">
            There is no run <code>{runId}</code>.{" "}
            <Link to="/runs">Back to runs</Link>.
          </Empty>
        </div>
      );
    return <ErrorPanel error={run.error} retry={run.retry} what="the run" />;
  }
  if (!run.data) return <Loading rows={8} label="Loading run" />;
  const s = run.data.state;
  const st = statusOf(s.status);
  return (
    <div className="stack">
      <Link to="/runs" className="small">
        All runs
      </Link>
      <PageHead
        title={s.question.text}
        actions={
          <Mark tone={st.tone} glyph={st.glyph}>
            {st.word}
          </Mark>
        }
      >
        <span className="mono small">{s.run_id}</span> ·{" "}
        {REASONER[s.reasoner] ?? s.reasoner}
        {s.error && <span className="notice"> {s.error}</span>}
      </PageHead>
      <Summary run={run.data} />
      <nav className="tabs" aria-label="Run views">
        <NavLink end to={base}>
          Network
        </NavLink>
        <NavLink to={`${base}/process`}>Process</NavLink>
        <NavLink to={`${base}/citations`}>Citations</NavLink>
        <NavLink to={`${base}/corrections`}>Corrections</NavLink>
      </nav>
      {tab === "network" && <NetworkTab run={run.data} />}
      {tab === "process" && <ProcessTab run={run.data} runId={runId} />}
      {tab === "citations" && <CitationsTab run={run.data} />}
      {tab === "corrections" && <CorrectionsTab run={run.data} />}
    </div>
  );
}
