# Metrics

Every metric is computed from a run's artefacts (`state.json`), its process log and the
independent verification (`verification.json`). Code: `src/discoverylab/evaluate.py`,
`src/discoverylab/judges.py`, `src/discoverylab/experiments/`.

| Metric | Definition | Needs |
|---|---|---|
| Citation accuracy | Distinct cited identifiers whose record resolves through the indexes **and** whose stated title (token similarity ≥ 0.85), first-author surname and year (±1) match the record, divided by distinct cited identifiers. Metadata the agent did not state is not checked. | Indexes |
| Evidence correctness | Evidence items whose citation is verified **and** whose quote occurs in the source text (title + abstract) after normalising case, Unicode, whitespace and punctuation, divided by evidence items. A source with no abstract gives "no source text", which is not correct. | Indexes |
| Quote supports claim | A judge's verdict per evidence item: supports / partly / does not support. | Judge |
| Hypothesis quality | Rubric of five criteria (specificity, testability, grounding, falsifiability, consistency), each 0–2, total out of 10 (`judges.RUBRIC`). | Judge |
| Experimental validity | Share of nine checks passed by each design: runnable by the toolbox, has a control, has a treatment, held-out evaluation, primary metric equals the hypothesis's dependent variable, paired test, at least 5 paired measurements, more than one seed, more than one dataset. | None |
| Novelty indicator | Maximum TF-IDF cosine similarity of a hypothesis to any retrieved abstract. High values mean the hypothesis restates prior work. An indicator, not proof. | None |
| Critique counts | Critiques by stage, reviewer (rule / model / human) and severity, and how each was resolved (fixed / accepted / dismissed / open). | None |
| Human-correction count | Edits, rejections and additions per checkpoint stage. Reported as "no review" (`null`), never 0, when nobody reviewed the stage. | A person |
| Log integrity | Whether the hash chain of `log.jsonl` verifies, and the first bad event if not. | None |

## Experiment metrics

| Experiment | Metric |
|---|---|
| E2 | Share of corrupted variants rejected (target 100%) and share of benign variants accepted (target 100%), by variant. |
| E3 | Per fault: share of injections caught as an expected issue on the faulted artefact, and share where the artefact was flagged at all. Critiques raised on unmodified final states (false alarms). |
| E4 | Mean readability (Flesch reading ease), judged convincingness (1–10) and process score per variant; Spearman correlation of each fluency score with the process score. Process score = mean of evidence correctness and the share of conclusions whose verdict matches the statistics. |
| E5 | Per run: artefact groups (papers, evidence, gaps, hypotheses, designs, results without runtimes, conclusions, critiques) identical under replay, and identical verification. |

## Expected verdict

A conclusion's verdict is checked against a fixed rule on the primary metric (`critic.expected_verdict`): supported when every dataset shows a Holm-adjusted p < 0.05 difference in the hypothesised direction; not supported when any Holm-significant difference goes against it; inconclusive otherwise. For "no difference" hypotheses, supported means no unadjusted test is significant, a weak criterion that the report states.
