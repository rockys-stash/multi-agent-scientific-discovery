import { PageHead } from "../components/ui";
import { OWNER, STAGES } from "../lib/format";

const ISSUES: [string, string][] = [
  [
    "citation not retrieved",
    "Evidence cites a source the literature search never returned.",
  ],
  [
    "quote not in source",
    "The quoted text does not occur in the retrieved abstract.",
  ],
  [
    "ungrounded hypothesis",
    "A hypothesis is not linked to any identified gap.",
  ],
  [
    "no control / no treatment",
    "The design cannot isolate the effect it claims to test.",
  ],
  [
    "insufficient repeats",
    "Fewer than five paired measurements, too few for the paired test.",
  ],
  [
    "metric mismatch",
    "The primary metric is not the hypothesis's dependent variable.",
  ],
  [
    "arithmetic error",
    "A reported difference or interval disagrees with the per-fold numbers.",
  ],
  [
    "overreach / verdict mismatch",
    "The conclusion's verdict disagrees with the Holm-adjusted test on the primary metric.",
  ],
  ["unknown reference", "An artefact refers to an id that does not exist."],
];

export function Method() {
  return (
    <>
      <PageHead title="Method and limits">
        How a run is carried out, how it is checked, and what the checks cannot
        tell you.
      </PageHead>
      <div className="prose">
        <h2>The workflow</h2>
        <p>
          The Research Director takes a question through fixed stages and hands
          each one to a specialist agent. After every stage the Critic reviews
          what was produced; blocking findings send the stage back for another
          round, up to a limit. At four checkpoints (evidence, hypotheses,
          design and conclusion) the run can pause for a person, whose
          approvals, edits and rejections are logged and counted.
        </p>
        <ol>
          {STAGES.map((s) => (
            <li key={s.id}>
              <strong>{s.label}</strong>, by the {OWNER[s.id]?.toLowerCase()}.
            </li>
          ))}
        </ol>
        <h2>Citations are checked independently</h2>
        <p>
          Whatever an agent cites is resolved again, by identifier, through the
          scholarly indexes, without using the agent&apos;s own copy. A citation
          is verified only when the identifier resolves, the title matches
          (token similarity at least 0.85), the first author&apos;s surname
          matches, the year is within one of the source&apos;s, and the quoted
          sentence occurs in the source text after normalising case, whitespace
          and punctuation. A source whose index holds no abstract is reported as
          having no text to check, never as verified.
        </p>
        <h2>The critic</h2>
        <p>
          The rule critic uses no language model, so it cannot be persuaded by
          fluent writing. It raises these issues; a model critic can add
          judgements on top, which are logged separately.
        </p>
        <ul>
          {ISSUES.map(([k, v]) => (
            <li key={k}>
              <strong>{k}.</strong> {v}
            </li>
          ))}
        </ul>
        <h2>Experiments are actually run</h2>
        <p>
          Designs are executed by a small, fixed toolbox of scikit-learn models,
          datasets and modifiers with stratified cross-validation over several
          seeds. Comparisons against the control use paired tests, bootstrap
          intervals and a Holm correction across treatments. The toolbox limits
          what can be tested; a hypothesis outside it is rejected at design time
          rather than approximated.
        </p>
        <h2>The process log</h2>
        <p>
          Every agent message, critique, human decision and artefact is appended
          to a log in which each event&apos;s hash covers the previous one, so a
          change, deletion or reordering anywhere breaks the chain. The console
          shows whether the chain is intact for each run.
        </p>
        <h2>Limits</h2>
        <ul>
          <li>
            A verified citation shows the quote exists in the source, not that
            it supports the claim; support is judged separately and reported
            only when judged.
          </li>
          <li>
            Abstracts are the only source text, so a correct quote from a
            paper&apos;s body is reported as not found.
          </li>
          <li>
            Novelty is the TF-IDF similarity of a hypothesis to the nearest
            retrieved abstract: an indicator, not proof.
          </li>
          <li>
            The rule reasoner is a baseline with a hand-written
            operationalisation per question; it is not a research agent.
          </li>
        </ul>
      </div>
    </>
  );
}
