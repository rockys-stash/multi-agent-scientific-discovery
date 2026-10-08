import { Link } from "react-router-dom";
import { Empty, ErrorPanel, Loading, Mark, PageHead } from "../components/ui";
import { api, useApi, type RunSummary } from "../lib/api";
import {
  REASONER,
  statusOf,
  STAGE_LABEL,
  verdictOf,
  frac,
  pct,
} from "../lib/format";

function corrections(r: RunSummary): string {
  if (!r.human_reviewed) return "No human review";
  const n = Object.values(r.corrections).reduce<number>(
    (a, b) => a + (b ?? 0),
    0,
  );
  return `${n} correction${n === 1 ? "" : "s"}`;
}

export function Runs() {
  const runs = useApi<{ items: RunSummary[] }>(api.runs);
  return (
    <>
      <PageHead title="Runs">
        Each run takes one research question through literature search,
        evidence, gaps, hypotheses, an experiment and its critique. Citation
        accuracy is measured by re-resolving every cited identifier,
        independently of the agent that cited it.
      </PageHead>
      {runs.error ? (
        <ErrorPanel error={runs.error} retry={runs.retry} what="runs" />
      ) : !runs.data ? (
        <Loading label="Loading runs" />
      ) : runs.data.items.length === 0 ? (
        <div className="panel">
          <Empty title="No runs yet">
            Start one with{" "}
            <code>discoverylab run configs/questions/q1_calibration.yaml</code>,
            then reload.
          </Empty>
        </div>
      ) : (
        <div className="panel">
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Runs"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Question</th>
                  <th scope="col">Reasoner</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="r">
                    Citations verified
                  </th>
                  <th scope="col" className="r">
                    Evidence correct
                  </th>
                  <th scope="col">Human review</th>
                  <th scope="col">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.items.map((r) => {
                  const st = statusOf(r.status);
                  const verdicts = Object.values(r.verdicts);
                  return (
                    <tr key={r.run_id}>
                      <th scope="row" style={{ maxWidth: 420 }}>
                        <Link to={`/runs/${encodeURIComponent(r.run_id)}`}>
                          {r.question}
                        </Link>
                        <div className="muted small mono">{r.run_id}</div>
                      </th>
                      <td>{REASONER[r.reasoner] ?? r.reasoner}</td>
                      <td>
                        <Mark tone={st.tone} glyph={st.glyph}>
                          {st.word}
                        </Mark>
                        {r.status !== "complete" && (
                          <div className="muted small">
                            at {STAGE_LABEL[r.stage] ?? r.stage}
                          </div>
                        )}
                      </td>
                      <td className="r num">
                        {pct(r.citation_accuracy)}{" "}
                        <span className="muted small">
                          ({frac(r.citation_accuracy)})
                        </span>
                      </td>
                      <td className="r num">
                        {pct(r.evidence_correctness)}{" "}
                        <span className="muted small">
                          ({frac(r.evidence_correctness)})
                        </span>
                      </td>
                      <td>{corrections(r)}</td>
                      <td>
                        {verdicts.length === 0
                          ? "–"
                          : verdicts.map((v, i) => (
                              <Mark
                                key={i}
                                tone={verdictOf(v).tone}
                                glyph={verdictOf(v).glyph}
                              >
                                {verdictOf(v).word}
                              </Mark>
                            ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
