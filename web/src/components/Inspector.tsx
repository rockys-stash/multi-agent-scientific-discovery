import type { ReactNode } from "react";
import type { Critique, EvidenceCheck, GraphNode, RunDetail } from "../lib/api";
import {
  KIND_LABEL,
  verdictOf,
  num,
  pval,
  signed,
  verificationMark,
} from "../lib/format";
import { Mark } from "./ui";

function sourceLink(id: string): string | null {
  if (id.startsWith("doi:")) return `https://doi.org/${id.slice(4)}`;
  if (id.startsWith("arxiv:")) return `https://arxiv.org/abs/${id.slice(6)}`;
  if (id.startsWith("openalex:")) return `https://openalex.org/${id.slice(9)}`;
  return null;
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{children}</dd>
    </>
  );
}

function Ref({ id, onSelect }: { id: string; onSelect: (id: string) => void }) {
  return (
    <button
      type="button"
      className="link-btn mono"
      onClick={() => onSelect(id)}
    >
      {id}
    </button>
  );
}

export function CritiqueList({ items }: { items: Critique[] }) {
  if (!items.length) return null;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <h3>Critiques</h3>
      {items.map((c) => (
        <div key={c.id} className={`finding sev-${c.severity}`}>
          <div className="small">
            <strong>{c.issue.replace(/_/g, " ")}</strong> · {c.severity} ·{" "}
            {c.reviewer === "rule"
              ? "rule critic"
              : c.reviewer === "model"
                ? "model critic"
                : "researcher"}{" "}
            · {c.round === 99 ? "final review" : `round ${c.round}`}
          </div>
          <div>{c.message}</div>
          <div className="muted small">
            {c.resolution}
            {c.resolution_note ? `: ${c.resolution_note}` : ""}
          </div>
        </div>
      ))}
    </div>
  );
}

function VerificationDetail({ chk }: { chk: EvidenceCheck }) {
  const c = chk.citation;
  const tick = (b: boolean | null) =>
    b === null ? "not claimed" : b ? "matches" : "does not match";
  return (
    <dl className="kv">
      <Row k="Resolved by">
        {c.resolved_by || `none of ${c.tried.join(", ") || "the indexes"}`}
      </Row>
      <Row k="Source title">{c.resolved_title || "–"}</Row>
      <Row k="Title">
        {tick(c.title_similarity === null ? null : c.title_similarity >= 0.85)}
        {c.title_similarity !== null && (
          <span className="muted">
            {" "}
            (similarity {c.title_similarity.toFixed(2)})
          </span>
        )}
      </Row>
      <Row k="First author">
        {tick(c.author_match)}{" "}
        <span className="muted">({c.resolved_first_author || "–"})</span>
      </Row>
      <Row k="Year">
        {tick(c.year_match)}{" "}
        <span className="muted">({c.resolved_year ?? "–"})</span>
      </Row>
      <Row k="Quote">
        {chk.quote === "found"
          ? `found in ${chk.found_in.join(", ")}`
          : chk.quote === "not_found"
            ? "not found in the source text"
            : "no source text to check"}
      </Row>
    </dl>
  );
}

export function Inspector({
  node,
  run,
  onSelect,
  onClose,
}: {
  node: GraphNode;
  run: RunDetail;
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  const s = run.state;
  const crit = s.critiques.filter((c) => c.target_id === node.id);
  const missing = (
    <p className="notice">
      This node has no matching artefact in the run state.
    </p>
  );
  let body: ReactNode = null;

  if (node.kind === "question") {
    body = (
      <>
        <p>{s.question.text}</p>
        <dl className="kv">
          <Row k="Concepts">{s.question.concepts.join(", ")}</Row>
          <Row k="Reasoner">{s.reasoner}</Row>
        </dl>
      </>
    );
  } else if (node.kind === "paper") {
    const p = s.papers.find((x) => x.id === node.id);
    const link = sourceLink(node.id);
    const quoting = run.graph.edges
      .filter((e) => e.rel === "quotes" && e.target === node.id)
      .map((e) => e.source);
    body = p ? (
      <>
        <p>
          <strong>{p.title}</strong>
        </p>
        <dl className="kv">
          <Row k="Authors">{p.authors.slice(0, 6).join(", ") || "–"}</Row>
          <Row k="Year">{p.year ?? "–"}</Row>
          <Row k="Venue">{p.venue || "–"}</Row>
          <Row k="Identifier">
            {link ? (
              <a href={link} rel="noreferrer noopener" target="_blank">
                {p.id}
              </a>
            ) : (
              p.id
            )}
          </Row>
          <Row k="Found by">
            {p.source} · query <q>{p.query}</q>
          </Row>
          <Row k="Quoted by">
            {quoting.length
              ? quoting.map((id) => (
                  <Ref key={id} id={id} onSelect={onSelect} />
                ))
              : "not cited"}
          </Row>
        </dl>
        {p.abstract ? (
          <details>
            <summary>Abstract as retrieved</summary>
            <p className="secondary">{p.abstract}</p>
          </details>
        ) : (
          <p className="muted">
            The index returned no abstract, so no evidence can be quoted from
            it.
          </p>
        )}
      </>
    ) : (
      <p className="notice">
        Cited but never retrieved by the literature search.
      </p>
    );
  } else if (node.kind === "evidence") {
    const e = s.evidence.find((x) => x.id === node.id);
    const chk = run.verification?.evidence.find(
      (c) => c.evidence_id === node.id,
    );
    const v = verificationMark(node.meta?.verification as string | undefined);
    body = !e ? (
      missing
    ) : (
      <>
        <p>{e.claim}</p>
        <blockquote className="quote">{e.quote}</blockquote>
        <dl className="kv">
          <Row k="Cites">
            <Ref id={e.citation.identifier} onSelect={onSelect} />{" "}
            {e.citation.first_author} {e.citation.year ?? ""}
          </Row>
          <Row k="Stance">{e.stance}</Row>
          <Row k="Verification">
            <Mark tone={v.tone} glyph={v.glyph} title={v.help}>
              {v.word}
            </Mark>
          </Row>
        </dl>
        {chk && <VerificationDetail chk={chk} />}
      </>
    );
  } else if (node.kind === "gap") {
    const g = s.gaps.find((x) => x.id === node.id);
    body = !g ? (
      missing
    ) : (
      <>
        <p>{g.description}</p>
        <dl className="kv">
          <Row k="Kind">{g.kind.replace("_", " ")}</Row>
          <Row k="Concepts">{g.concepts.join(", ")}</Row>
          <Row k="Rests on">
            {g.evidence_ids.map((i) => (
              <Ref key={i} id={i} onSelect={onSelect} />
            ))}
          </Row>
        </dl>
      </>
    );
  } else if (node.kind === "hypothesis") {
    const h = s.hypotheses.find((x) => x.id === node.id);
    const nov = run.metrics.novelty_max_tfidf_similarity[h?.id ?? ""];
    body = !h ? (
      missing
    ) : (
      <>
        <p>{h.statement}</p>
        <dl className="kv">
          <Row k="Independent">{h.independent_variable}</Row>
          <Row k="Dependent">{h.dependent_variable}</Row>
          <Row k="Expected">{h.expected_direction.replace("_", " ")}</Row>
          <Row k="Addresses">
            {h.gap_ids.map((i) => (
              <Ref key={i} id={i} onSelect={onSelect} />
            ))}
          </Row>
          <Row k="Closest prior">
            {num(nov ?? null, 2)}{" "}
            <span className="muted">
              TF-IDF similarity to the nearest retrieved abstract (an indicator,
              not proof of novelty)
            </span>
          </Row>
        </dl>
        {h.rationale && <p className="secondary small">{h.rationale}</p>}
      </>
    );
  } else if (node.kind === "design") {
    const d = s.designs.find((x) => x.id === node.id);
    const val = run.metrics.design_validity[d?.id ?? ""];
    body = !d ? (
      missing
    ) : (
      <>
        <dl className="kv">
          <Row k="Tests">
            <Ref id={d.hypothesis_id} onSelect={onSelect} />
          </Row>
          <Row k="Control">
            <code>{d.control}</code>
          </Row>
          <Row k="Treatments">
            {d.treatments.map((t) => (
              <code key={t}>{t} </code>
            ))}
          </Row>
          <Row k="Datasets">{d.datasets.join(", ")}</Row>
          <Row k="Metrics">{d.metrics.join(", ")} (primary first)</Row>
          <Row k="Protocol">
            {d.folds}-fold × {d.seeds.length} seeds, {d.test.replace("_", " ")}
          </Row>
        </dl>
        {val && (
          <>
            <h3>
              Validity checklist {val.passed}/{val.of}
            </h3>
            <ul className="checklist">
              {Object.entries(val.checks).map(([k, ok]) => (
                <li key={k}>
                  <Mark tone={ok ? "ok" : "bad"} glyph={ok ? "✓" : "✕"}>
                    {k.replace(/_/g, " ")}
                  </Mark>
                </li>
              ))}
            </ul>
          </>
        )}
      </>
    );
  } else if (node.kind === "result") {
    const r = s.results.find((x) => x.id === node.id);
    body = !r ? (
      missing
    ) : (
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label="Comparisons"
      >
        <table>
          <thead>
            <tr>
              <th scope="col">Dataset</th>
              <th scope="col">Metric</th>
              <th scope="col">Treatment</th>
              <th scope="col" className="r">
                Difference [95% CI]
              </th>
              <th scope="col" className="r">
                Holm p
              </th>
            </tr>
          </thead>
          <tbody>
            {r.comparisons.map((c, i) => (
              <tr key={i}>
                <td>{c.dataset}</td>
                <td>{c.metric}</td>
                <td className="mono small">{c.treatment}</td>
                <td className="r num">
                  {signed(c.mean_diff)}{" "}
                  <span className="muted">
                    [{signed(c.lo)}, {signed(c.hi)}]
                  </span>
                </td>
                <td className="r num">{pval(c.p_holm)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  } else if (node.kind === "conclusion") {
    const k = s.conclusions.find((x) => x.id === node.id);
    const v = verdictOf(k?.verdict ?? "");
    body = !k ? (
      missing
    ) : (
      <>
        <Mark tone={v.tone} glyph={v.glyph}>
          {v.word}
        </Mark>
        <p style={{ marginTop: 8 }}>{k.statement}</p>
        {k.limitations.length > 0 && (
          <ul className="small secondary">
            {k.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        )}
      </>
    );
  }

  return (
    <aside
      className="panel inspector"
      aria-label={`${KIND_LABEL[node.kind]} ${node.id}`}
    >
      <div className="panel-head">
        <h2>
          {KIND_LABEL[node.kind]}{" "}
          <span className="mono muted">
            {node.kind === "paper" ? "" : node.id}
          </span>
        </h2>
        <button
          type="button"
          className="btn ghost"
          onClick={onClose}
          aria-label="Close inspector"
        >
          Close
        </button>
      </div>
      <div className="panel-body stack">
        {body}
        <CritiqueList items={crit} />
      </div>
    </aside>
  );
}
