import { useSearchParams } from "react-router-dom";
import {
  Empty,
  ErrorPanel,
  Loading,
  Mark,
  PageHead,
  ProvenanceLine,
} from "../components/ui";
import {
  api,
  useApi,
  type ExperimentInfo,
  type ExperimentSummary,
  type SummaryTable,
} from "../lib/api";

const LABEL: Record<string, { title: string; asks: string }> = {
  e1_runs: {
    title: "E1 End-to-end runs",
    asks: "Do runs on real questions cite real sources, quote them correctly and design valid experiments?",
  },
  e2_verifier: {
    title: "E2 Citation verifier",
    asks: "Does the verifier catch fabricated, altered and misattributed citations without rejecting real ones?",
  },
  e3_critic: {
    title: "E3 Critic",
    asks: "Which injected process faults does the critic catch, at which stage?",
  },
  e4_fluency: {
    title: "E4 Fluency versus process",
    asks: "Does a fluent report predict a sound process?",
  },
  e5_reproducibility: {
    title: "E5 Reproducibility",
    asks: "Does replaying a recorded run reproduce every artefact and number?",
  },
};

function cell(v: string | number | null): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number")
    return Number.isInteger(v) ? v.toLocaleString("en-GB") : v.toFixed(3);
  return v;
}

function Table({ t }: { t: SummaryTable }) {
  return (
    <div className="stack" style={{ gap: 6 }}>
      <h3>{t.caption}</h3>
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label={t.caption}
      >
        <table>
          <thead>
            <tr>
              {t.columns.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  className={c.align === "r" ? "r" : undefined}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {t.rows.map((r, i) => (
              <tr key={i}>
                {t.columns.map((c) => (
                  <td
                    key={c.key}
                    className={c.align === "r" ? "r num" : undefined}
                  >
                    {cell(r[c.key] ?? null)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Detail({ exp }: { exp: string }) {
  const s = useApi<ExperimentSummary>(api.experiment(exp));
  if (s.error)
    return <ErrorPanel error={s.error} retry={s.retry} what="the experiment" />;
  if (!s.data) return <Loading rows={6} label="Loading experiment" />;
  const d = s.data;
  return (
    <section className="panel" aria-label={d.title}>
      <div className="panel-head">
        <h2>{d.title}</h2>
        <ProvenanceLine p={d.provenance} />
      </div>
      <div className="panel-body stack">
        <p className="secondary" style={{ margin: 0 }}>
          {d.question}
        </p>
        {d.findings.length > 0 && (
          <ul className="findings">
            {d.findings.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        )}
        {d.tables.map((t) => (
          <Table key={t.id} t={t} />
        ))}
        {d.notes.length > 0 && (
          <ul className="small secondary">
            {d.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

export function Evaluation() {
  const list = useApi<{ items: ExperimentInfo[] }>(api.experiments);
  const [params, setParams] = useSearchParams();
  const head = (
    <PageHead title="Evaluation">
      The process is judged, not the prose: citations against their sources,
      designs against a validity checklist, and the critic against faults
      planted on purpose. An experiment that has not been run is shown as
      pending, with no numbers.
    </PageHead>
  );
  if (list.error)
    return (
      <>
        {head}
        <ErrorPanel error={list.error} retry={list.retry} what="experiments" />
      </>
    );
  if (!list.data)
    return (
      <>
        {head}
        <Loading label="Loading experiments" />
      </>
    );
  const items = list.data.items;
  const done = items.filter((i) => i.status === "complete");
  const chosen = params.get("exp") ?? done[0]?.experiment ?? null;
  return (
    <>
      {head}
      <div className="stack">
        <div className="panel">
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Experiments"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Experiment</th>
                  <th scope="col">Question it answers</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {items.map((i) => {
                  const l = LABEL[i.experiment] ?? {
                    title: i.experiment,
                    asks: "",
                  };
                  return (
                    <tr
                      key={i.experiment}
                      aria-current={
                        i.experiment === chosen ? "true" : undefined
                      }
                    >
                      <th scope="row">
                        {i.status === "complete" ? (
                          <button
                            type="button"
                            className="link-btn"
                            aria-pressed={i.experiment === chosen}
                            onClick={() => setParams({ exp: i.experiment })}
                          >
                            {l.title}
                          </button>
                        ) : (
                          l.title
                        )}
                      </th>
                      <td className="secondary">{l.asks}</td>
                      <td>
                        {i.status === "complete" ? (
                          <Mark tone="ok" glyph="✓">
                            Complete
                          </Mark>
                        ) : (
                          <Mark tone="neutral" glyph="○">
                            Pending
                          </Mark>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
        {chosen && done.some((d) => d.experiment === chosen) ? (
          <Detail exp={chosen} />
        ) : (
          <div className="panel">
            <Empty title="No experiment has been run">
              Run one with <code>make experiments</code>. Results appear here
              with the commit and configuration that produced them.
            </Empty>
          </div>
        )}
      </div>
    </>
  );
}
