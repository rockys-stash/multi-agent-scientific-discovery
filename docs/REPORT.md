# Judging a research agent by its process: a first study on real literature

**Scope.** Every number below comes from executed code and is copied from
[generated/results.md](generated/results.md), which `discoverylab results` regenerates from the
versioned result directories in `results/`. The literature is real: arXiv, Crossref, Semantic
Scholar and OpenAlex as of 2026-10-08. **Only the rule baseline has been run.** The Claude
reasoner, the Claude reasoner with a model critic, the hypothesis-rubric judge and the
convincingness judge need `ANTHROPIC_API_KEY`, which was not available; they are
*Status: pending* and appear nowhere below as results. No person reviewed any run.

## 1. Question

A Research Director agent takes a research question through literature search, evidence
extraction, gap identification, hypothesis generation, experiment design, analysis, critique and
conclusion. Reports written by such systems can read well whatever happened underneath, so we
judge the **process**: is every cited source real and correctly described, is every quote in it,
is every experiment valid and actually run, does every conclusion follow from the statistics, and
can the run be replayed?

| | Question | Experiment |
|---|---|---|
| RQ1 | Do runs on real questions cite real sources, quote them correctly and design valid experiments? | E1 |
| RQ2 | Does the independent verifier reject fabricated, altered and misattributed citations without rejecting real ones? | E2 |
| RQ3 | Which process faults does the rule critic catch, and which can it not see? | E3 |
| RQ4 | Does a fluent report predict a sound process? | E4 |
| RQ5 | Does replaying a recorded run reproduce every artefact? | E5 |

**Hypotheses stated before running.** H-RQ2: the verifier rejects every corruption type and no
benign variant. H-RQ3: the rule critic catches structural faults and misses semantic ones (it does
not read for meaning, DECISIONS D6). H-RQ4: readability is not a proxy for process quality.
H-RQ5: replay from the recorded cache reproduces every artefact.

## 2. Setup

**Questions.** Three machine-learning questions whose answers can be tested with the bundled
toolbox (configs/questions/): post-hoc calibration of random forests (q1), feature
standardisation for k-nearest neighbours (q2), and oversampling versus class weighting for
imbalanced logistic regression (q3).

**Reasoner.** The rule baseline (DECISIONS D4): BM25 evidence selection, gap rules over concept
co-occurrence, and a hypothesis and design filled from a person-written operationalisation. It is
deterministic, so one run per question is the run; variance across runs is not defined for it.
Its designs are therefore not its own reasoning, which the comparison with Claude was meant to
expose (pending).

**Retrieval.** Up to 10 results per source per query, at most 25 papers per run. Every response
is cached by request hash with its status and retrieval time; each run records which version of
each response it consumed (`retrievals.json`, D13).

**Experiments run by the agents.** Stratified 5-fold cross-validation × 3 seeds on bundled
scikit-learn datasets, bootstrap 95% intervals, paired Wilcoxon tests, Holm correction across
comparisons. The verdict rule is fixed (critic.py): *supported* only if every dataset shows a
Holm-significant difference in the hypothesised direction; *not supported* if any shows one
against it; otherwise *inconclusive*.

**Verification.** Every cited identifier is re-resolved through the indexes, independently of the
record the agent worked from; title (token similarity ≥ 0.85), first author (surname), year
(± 1) and the verbatim quote are checked against what the authoritative source returns.

## 3. Results

### RQ1 · E1 end-to-end runs

| Question | Papers | Evidence | Citations verified | Evidence correct | Design checks | Verdict |
|---|---:|---:|---:|---:|---:|---|
| q1 calibration | 25 | 24 | 24/24 | 24/24 | 1.000 | inconclusive |
| q2 kNN scaling | 25 | 22 | 21/22 | 21/22 | 1.000 | not supported |
| q3 imbalance | 25 | 25 | 24/25 | 24/25 | 1.000 | inconclusive |

All three runs completed with an intact hash-chained log, every design passed the validity
checks, and the rule critic raised no critique on the final states. Papers came from arXiv (35),
Semantic Scholar (23) and Crossref (17); OpenAlex contributed none because its daily anonymous
credit was exhausted (33 requests answered "try later"; Semantic Scholar 34, D12). No citation
ended as "could not check".

The agents' own experiments:

- **q1.** Platt scaling and isotonic regression lowered the Brier score of a random forest on all
  three datasets, but only on digits was the difference Holm-significant
  (Platt −0.0166 [−0.0182, −0.0151], isotonic −0.0162 [−0.0179, −0.0148], Holm p = 0.00037). On
  breast cancer and wine the intervals were within ±0.005 and Holm p ≥ 0.19. Verdict:
  inconclusive.
- **q2.** Standardisation raised kNN accuracy on breast cancer (+0.032 [+0.023, +0.042]) and wine
  (+0.081 [+0.058, +0.105]) and lowered it on digits (−0.0039 [−0.0059, −0.0020]), all
  Holm-significant. Digits' pixel features already share one scale; a plausible but untested explanation is
  that rescaling amplifies near-constant pixels. Under the fixed rule one significant reversal makes the verdict
  *not supported*; a reader of the same numbers would say "helps when scales differ", which is
  the effect the question was about. The rule is conservative by design; the gap between the
  verdict and the more useful reading is a finding about verdict rules, not about kNN.
- **q3.** With a 10:1 imbalance, random oversampling did not recover more minority recall than
  class weighting (−0.0095 [−0.029, 0.000] and −0.0074 [−0.019, 0.000], Holm p = 0.36).
  Verdict: inconclusive.

**Error analysis.** The two citations that failed are real papers, so both are verifier false
rejections in the conservative direction:

1. *Online-first versus issue year* (q3, DOI 10.2174/0126662558347788241127051934): OpenAlex and
   Semantic Scholar give 2024, Crossref, the first authoritative resolver for DOIs, gives 2026.
   The ± 1-year tolerance covers preprint/print gaps but not a two-year issue backlog (D14).
2. *Missing author at the source* (q2, DOI 10.52465/joscex.v1i1.7): Crossref returns no author;
   the verifier treats that as a mismatch (D15).

Neither was tuned away after the fact. Verification establishes that a source exists, matches its
metadata and contains the quote; it does not establish relevance. The citation table for q3
(docs/screenshots/run-citations-desktop.png) shows verified evidence drawn from neonatal
mortality, credit-card fraud and HCV detection papers: lexical selection finds papers that
*mention* class imbalance and logistic regression, not papers that *test* oversampling against
class weighting. Gap identification has the same character: most gaps are "no retrieved evidence
addresses A together with B" over concept pairs, which says more about the 25 retrieved papers
than about the literature. Whether a language-model reasoner selects more relevant evidence is
the main pending comparison.

**Live variability.** The three runs made before the replay fix (13:56–14:12 UTC, kept in
`results/e1_runs/20261008T135637Z-bf1cf81`) selected different evidence for q2 (21 items, all
verified) than the runs at 14:39–14:51 (22 items, one rejected), because Semantic Scholar answered
requests in the second batch that it had rate-limited in the first. Retrieval from live indexes is
not stationary even within an hour; only replay from the recorded responses is exactly
reproducible (RQ5).

### RQ2 · E2 citation verifier

69 verified evidence items from E1 were varied 755 times. The verifier rejected **482/482**
corruptions (fabricated identifier, wrong title, wrong first author, wrong year, swapped
identifier, misattributed quote, altered quote: each 68–69 trials, all rejected) and wrongly
rejected **0/273** benign variants (unchanged, reformatted title, preprint year, shorter quote).
H-RQ2 holds on this sample. One harness defect was found and fixed before this result: the
"wrong first author" corruption could pick another paper by the same author, which made the
"corruption" correct (D15). The two real false rejections from E1 are not in E2's sample, because
E2 varies only items that verified; E2 therefore bounds false rejections of *perturbations*, not
of real-world metadata disagreements between indexes.

### RQ3 · E3 critic fault injection

90 faults were injected into the 3 completed runs. The rule critic caught **72/72** structural
faults (12 types: citation to an unretrieved source, altered or missing quote, ungrounded
hypothesis, dangling reference, control also used as treatment, single measurement, primary metric
mismatch, unknown dataset, misreported mean, interval excluding the mean, verdict not supported by
the statistics) and **0/18** semantic faults (claim contradicting its quote, hypothesis direction
flipped). It raised no critique on the unmodified runs. H-RQ3 holds. The "stance flipped" fault
had 0 trials: the rule reasoner emits no stance to flip, so this fault can only be measured on
model runs. Whether the model critic catches the semantic faults is pending.

### RQ4 · E4 fluency versus process

For each run, the template report (D11) was corrupted three ways and all three combined:

| Variant | Readability (Flesch) | Process score |
|---|---:|---:|
| original | 2.88 | 0.986 |
| swapped citations | 2.88 | 0.500 |
| fabricated citations | 4.63 | 0.500 |
| inflated conclusions | 0.42 | 0.486 |
| all three | 2.21 | 0.000 |

Means over 3 reports. The Spearman correlation of readability with process score over the 15
reports is **0.04**. Fabricating citations made the report slightly *easier* to read; destroying
every part of the process left readability within the range of the original. H-RQ4 holds for
readability. Readability is a weak fluency proxy (all scores are near the bottom of the Flesch
scale because the reports are dense and technical); the stronger test, a model judging
convincingness without access to the sources, is pending.

### RQ5 · E5 reproducibility

All **3/3** runs replayed offline from the recorded responses with every artefact group (8/8) and
the verification identical. The first E5 run failed this (0/3): a later record-mode run had
refetched requests the E1 runs had seen rate-limited, and replay served the newer answers. The
cache now archives replaced responses and replays each run from its own retrieval manifest (D13).
That failure is the reason E5 exists: "replayable" was assumed by the design and turned out to be
false under real rate limits until measured. Live re-run overlap and replay of model transcripts
are pending.

### Reproduction from a fresh clone

On 2026-10-09 a fresh clone at commit e3319b4 ran `make setup`, `make test` and E1–E5 from an
empty cache against the live indexes (that clone's results are not committed; this section
summarises them).

| | Committed (2026-10-08) | Fresh clone (2026-10-09) |
|---|---|---|
| E1 citations verified | 24/24, 21/22, 24/25 | 23/23, 15/15 (+6 could not check), 23/25 |
| E1 verdicts | inconclusive, not supported, inconclusive | same |
| E2 corruptions rejected / benign rejected | 482/482, 0/273 | 412/412, 0/232 |
| E3 structural / semantic caught | 72/72, 0/18 | 72/72, 0/18 |
| E4 Spearman (readability, process) | 0.04 | 0.19 |
| E5 replay identical | 3/3 | 3/3 |

Every qualitative finding reproduced: the verdicts, perfect separation by the verifier, the
critic's structural/semantic split, a weak readability–process correlation, and exact replay.
The literature did not: evidence counts and which papers were cited changed with what the
indexes answered on the day. The fresh run found a third real metadata disagreement (DOI
10.47738/jdmdc.v2i2.34: Semantic Scholar gives Shuang Li as first author, Crossref Agung Budi
Prasetio), and exposed a metric defect: quotes from a rate-limited arXiv were counted as wrong
(D16, fixed; it did not affect the committed results).

## 4. Threats to validity

- **One reasoner, three questions.** All findings about agent behaviour are about a deterministic
  baseline whose designs come from a person-written operationalisation. They say nothing yet about
  language-model agents.
- **Index availability shaped the evidence.** OpenAlex contributed nothing and Semantic Scholar
  was rate-limited on the day; a different day would retrieve different papers.
- **Abstracts only.** Quotes are checked against titles and abstracts, so a correct quote from a
  paper's body would be reported as not found; the rule baseline only quotes abstracts.
- **E2 and E3 test the faults we thought of.** Perfect detection of designed corruptions is an
  upper bound on detection of real errors, and E1 already found two real disagreements outside
  E2's design.
- **Small toolbox.** Experiments run on three or two bundled datasets; the q1–q3 effects are
  measurements on those datasets, not general claims about the methods.

## 5. Reproduce

```bash
make setup
make e1 e2 e3 e5 e4    # E1 needs network access to the indexes; the Claude arms need ANTHROPIC_API_KEY
make results           # regenerates docs/generated/results.md
make screenshots       # regenerates docs/screenshots from the latest E1 runs
```

E2, E3 and E5 are offline once E1's cache exists. A fresh clone reproduced E1–E5 (§3,
"Reproduction from a fresh clone").

## 6. Pending

- E1 Claude and Claude-with-model-critic arms (3 repeats each, for run-to-run variance).
- Hypothesis rubric and quote-supports-claim judgements (E1), model critic on semantic faults
  (E3), convincingness judge (E4), transcript replay of model runs and live re-run overlap (E5).
- A run with a human reviewer at the four checkpoints, so that correction counts exist.
