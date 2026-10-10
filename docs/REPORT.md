# Judging a research agent by its process: a study on real literature

**Scope.** Every number below comes from executed code and is copied from
[generated/results.md](generated/results.md) and
[generated/error_analysis.md](generated/error_analysis.md), which `discoverylab results` and
`scripts/error_analysis.py` regenerate from the versioned result directories in `results/`. The
literature is real: arXiv, Crossref, Semantic Scholar and OpenAlex, retrieved on 2026-10-08 and
2026-10-09.

**Which models.** The language-model agents, the model critic and the judges ran on **local
open-weights models**, not on Claude: Qwen3-4B-Instruct-2507 (4-bit, Apache-2.0) as the agents and
model critic, and Phi-4-mini-instruct (4-bit, MIT) as the judge (DECISIONS D17). No
`ANTHROPIC_API_KEY` was available, and the project owner asked for the study to be finished
without the API. The Claude arms stay in the configuration and are recorded as skipped; they are
*Status: pending* and appear nowhere below as results. A 4-billion-parameter quantised model is
far weaker than the model the study was designed for, so the model results answer "what does a
small open model do inside this process", not "what does a frontier model do". No person reviewed
any run.

## 1. Question

A Research Director agent takes a research question through literature search, evidence
extraction, gap identification, hypothesis generation, experiment design, analysis, critique and
conclusion. Reports written by such systems can read well whatever happened underneath, so we
judge the **process**: is every cited source real and correctly described, is every quote in it,
is every experiment valid and actually run, does every conclusion follow from the statistics, and
can the run be replayed?

| | Question | Experiment |
|---|---|---|
| RQ1 | Do runs on real questions cite real sources, quote them correctly and design valid experiments, and how do a rule baseline and a language-model agent compare? | E1, E1 compare |
| RQ2 | Does the independent verifier reject fabricated, altered and misattributed citations without rejecting real ones? | E2 |
| RQ3 | Which process faults does the rule critic catch, which can it not see, and does a model critic see them? | E3 |
| RQ4 | Does a fluent or convincing report predict a sound process? | E4 |
| RQ5 | Does replaying a recorded run reproduce every artefact? | E5 |

**Hypotheses stated before running.** H-RQ2: the verifier rejects every corruption type and no
benign variant. H-RQ3: the rule critic catches structural faults and misses semantic ones (it does
not read for meaning, D6); a model critic catches some semantic faults. H-RQ4: neither readability
nor judged convincingness is a proxy for process quality. H-RQ5: replay from the recorded cache
and transcripts reproduces every artefact.

## 2. Setup

**Questions.** Three machine-learning questions whose answers can be tested with the bundled
toolbox (configs/questions/): post-hoc calibration of random forests (q1), feature
standardisation for k-nearest neighbours (q2), and oversampling versus class weighting for
imbalanced logistic regression (q3).

**Arms (E1).**

| Arm | What it is | Runs |
|---|---|---|
| `rule` | Rule baseline (D4): BM25 evidence selection, gap rules, hypothesis and design from a person-written operationalisation. Deterministic. | 1 per question |
| `local` | Language-model agents on the local model; evidence read from all 25 abstracts in one prompt. | 3 per question |
| `local+mc` | As `local`, with the local model also acting as a second critic after every stage. | 3 per question |
| `local-b5` | As `local`, but evidence read 5 abstracts per call (ablation, D18). | 3 per question |
| `claude`, `claude+mc` | The arms the study was designed for. | skipped: no key |

Model arms sample at temperature 0.7 with a logged per-call seed; repeats differ by seed. The
judge samples at temperature 0 and is a different model family from the agents, so no model
grades its own output.

**Retrieval.** Up to 10 results per source per query, at most 25 papers per run. Every response
is cached by request hash with its status and retrieval time; each run records which version of
each response it consumed (`retrievals.json`, D13).

**Experiments run by the agents.** Stratified 5-fold cross-validation × 3 seeds on bundled
scikit-learn datasets, bootstrap 95% intervals, paired Wilcoxon tests, Holm correction across
comparisons. The verdict rule is fixed (critic.py): *supported* only if every dataset shows a
Holm-significant difference in the hypothesised direction; *not supported* if any shows one
against it; otherwise *inconclusive*. Model agents choose their own verdicts; the rule is the
yardstick they are scored against.

**Verification.** Every cited identifier is re-resolved through the indexes, independently of the
record the agent worked from; title (token similarity ≥ 0.85), first author (surname), year
(± 1) and the verbatim quote are checked against what the authoritative source returns.

## 3. Results

### RQ1 · E1 end-to-end runs and the arm comparison

29 runs completed: 3 rule, 9 `local`, 9 `local+mc` and 8 `local-b5`. One `local-b5` run on q2
failed when the toolbox could not build a design the validator had passed (`knn+class_weight`);
it stays in E1 as failed and the validator was fixed (D20). 18 Claude runs were skipped.

Mean ± standard deviation over runs, all three questions:

| Metric | Rule (n=3) | Local (n=9) | Local + model critic (n=9) | Local, 5 per read (n=8) |
|---|---:|---:|---:|---:|
| Evidence items | 24.3 ± 1.2 | 3.2 ± 4.4 | 2.1 ± 2.0 | 8.6 ± 2.8 |
| Citations verified | 97% ± 2% | 100% ± 0% | 100% ± 0% | 100% ± 0% |
| Evidence correct | 97% ± 2% | 100% ± 0% | 100% ± 0% | 100% ± 0% |
| Quotes judged to support claim | 100% ± 0% | 74% ± 43% | 74% ± 25% | 86% ± 11% |
| Hypothesis rubric (of 10, judged) | 8.3 ± 2.9 | 6.9 ± 0.7 | 7.1 ± 1.5 | 7.4 ± 1.6 |
| Design checks passed | 100% ± 0% | 86% ± 5% | 87% ± 3% | 83% ± 5% |
| Conclusions reached | 1.0 ± 0.0 | 0.9 ± 0.9 | 1.1 ± 0.9 | 1.0 ± 0.9 |
| Verdicts matching the statistics | 100% ± 0% | 70% ± 45% | 83% ± 26% | 60% ± 55% |
| Critiques raised | 0.0 ± 0.0 | 7.4 ± 4.3 | 10.0 ± 4.8 | 12.0 ± 5.8 |
| Run time (min) | 2.9 ± 0.9 | 13.8 ± 9.0 | 17.4 ± 7.3 | 14.8 ± 5.2 |

Per-question tables are in generated/results.md. The rule baseline's designs and verdicts come
from its operationalisation and the fixed rule, so its 100% scores are not its own reasoning;
they are the ceiling a reasoner reaches by following the protocol exactly.

**What the model agents got right.** Every citation the model agents emitted resolved to a real
paper with matching title, author and year, and every quote was found verbatim in the source:
100% over 26 completed model runs (the rule baseline: 97%, see error analysis). The grammar
constraint (D17) makes a malformed citation impossible, and the agents copy rather than
paraphrase quotes. Fabrication, the failure that motivated the verifier, did not occur.

**What they got wrong** (generated/error_analysis.md, 26 completed model runs, 40 designs):

1. *Reading many abstracts at once.* With all 25 abstracts in one prompt, 6 of 18 runs returned
   no evidence at all (4 `local`, 2 `local+mc`), so those runs had no gaps, hypotheses or
   conclusions. Reading 5 per call (`local-b5`) raised evidence from 3.2 to 8.6 items per run and
   no run came back empty (0 of 8). The ablation was added as a separate arm after the first
   result, not as a change to the existing arm (D18).
2. *Inventing catalogue names.* 14 of 40 designs could not be run: they named metrics (10),
   modifiers (10) or models (2) the toolbox does not have, e.g. `macro-F1`, `AURC` or a
   distance-contribution metric. The critic blocks them, and the agents did not repair them
   within the two critique rounds: 66 blocking `unrunnable_design` critiques are still open on
   the final states.
3. *Verdicts.* Of 26 conclusions with computed statistics, 7 differ from the fixed rule: 5 say
   *inconclusive* where a Holm-significant effect goes against the hypothesis (*not supported*),
   and 2 say *supported* where the rule says *not supported*. All 7 errors are on results the
   rule calls *not supported*: the model avoided reporting a negative result.
4. *Primary metric naming.* 39 of 40 designs name a primary metric that differs from the
   hypothesis's dependent variable by exact string. In 35 the design uses a catalogue metric that
   names the same quantity (`ece` for "Expected Calibration Error (ECE)", `minority_recall` for
   "minority-class recall"; the full list is in error_analysis.md, read by hand). The validity
   check compares names exactly, so it under-credits the model arms by about one check in nine;
   this is a limit of the metric, stated rather than changed after seeing the result.

**The model critic (ablation `local` vs `local+mc`).** Adding the model critic raised critiques
from 7.4 to 10.0 per run, and verdict validity from 70% to 83%, but with standard deviations of
45 and 26 points over 9 runs each that difference is not distinguishable from sampling noise.
No metric moved by more than its spread.

**The rule runs' own experiments** (unchanged from the first report; the toolbox is seeded):

- **q1.** Platt scaling and isotonic regression lowered the Brier score of a random forest on all
  three datasets, but only on digits was the difference Holm-significant
  (Platt −0.0166 [−0.0182, −0.0151], isotonic −0.0162 [−0.0179, −0.0148], Holm p = 0.00037). On
  breast cancer and wine the intervals were within ±0.005 and Holm p ≥ 0.19. Verdict:
  inconclusive.
- **q2.** Standardisation raised kNN accuracy on breast cancer (+0.032 [+0.023, +0.042]) and wine
  (+0.081 [+0.058, +0.105]) and lowered it on digits (−0.0039 [−0.0059, −0.0020]), all
  Holm-significant. Digits' pixel features already share one scale. Under the fixed rule one
  significant reversal makes the verdict *not supported*; a reader of the same numbers would say
  "helps when scales differ". The gap between the verdict and the more useful reading is a
  finding about verdict rules, not about kNN.
- **q3.** With a 10:1 imbalance, random oversampling did not recover more minority recall than
  class weighting (−0.0095 [−0.029, 0.000] and −0.0074 [−0.019, 0.000], Holm p = 0.36).
  Verdict: inconclusive.

**Rule-baseline citation failures.** Two citations failed, both real papers, so both are verifier
false rejections in the conservative direction, the same two in the 2026-10-08 and 2026-10-09
runs: an online-first versus issue year two years apart (DOI 10.2174/0126662558347788241127051934,
D14), and a record with no author at Crossref (DOI 10.52465/joscex.v1i1.7, D15). Neither was
tuned away. Verification establishes that a source exists, matches its metadata and contains the
quote; it does not establish relevance. The rule baseline's verified evidence is often only
lexically related to the question (e.g. neonatal mortality and credit-card fraud papers for q3);
the model agents' evidence is fewer items, and the judge rated 74–86% of their quotes as
supporting the stated claim.

**Live variability.** Retrieval from live indexes is not stationary: the same query returned
different papers on 2026-10-08 and 2026-10-09 as Semantic Scholar and OpenAlex rate limits
changed (OpenAlex 20 and Semantic Scholar 17 requests answered "try later" in this E1). Only
replay from the recorded responses is exactly reproducible (RQ5).

### RQ2 · E2 citation verifier

148 verified evidence items from E1 (rule and model runs) were varied 1,619 times. The verifier
rejected **1035/1035** corruptions (fabricated identifier, wrong title, wrong first author, wrong
year, swapped identifier, misattributed quote, altered quote: 147–148 trials each) and wrongly
rejected **0/584** benign variants (unchanged, reformatted title, preprint year, shorter quote).
H-RQ2 holds on this sample. E2 varies only items that verified, so it bounds false rejections of
*perturbations*, not of real metadata disagreements between indexes like the two above.

### RQ3 · E3 critic fault injection

779 faults were injected into the 29 completed runs. The rule critic caught **578/578**
structural faults (12 types) and **0/201** semantic faults (claim contradicting its quote, stance
flipped, hypothesis direction flipped). H-RQ3 holds for the rule critic. "Stance flipped" now has
70 trials because the model runs emit a stance.

The model critic reviewed a sample of the same faulted states (4 trials per fault type):

| | Rule critic | Model critic |
|---|---:|---:|
| Structural faults flagged | 40/40 | 8/40 |
| Semantic faults flagged | 0/12 | 2/12 (both on hypothesis direction, 2/4) |

The model critic caught 2 semantic faults the rule critic cannot see, and missed 32 of 40
structural faults the rule critic catches every time, including all altered quotes, missing
quotes and citations to unretrieved sources. On 6 unmodified final states it raised 12 critiques,
none blocking. H-RQ3's second half holds only weakly: a 4B model critic is a complement to the
rule critic on meaning, not a substitute for it on structure. The rule critic's 80 critiques on
unmodified final states are all on model runs and all correspond to the problems in the error
analysis above (39 metric-name mismatches, 22 unrunnable-design problems, 7 verdict errors, and 6
runs each with no evidence and no gaps); none is on a rule run.

Two harness defects were found and fixed before this result: the verdict fault could turn a wrong
model verdict into a right one (D21; first result kept), and the toolbox validator passed
unbuildable conditions (D20).

### RQ4 · E4 fluency versus process

For each completed run, the template report (D11) was corrupted three ways and all three combined;
readability is Flesch reading ease, convincingness is the judge's 1–10 rating of the report
without access to the sources.

| Variant | Reports | Readability | Convincingness | Process score |
|---|---:|---:|---:|---:|
| original | 29 | 1.97 | 5.18 | 0.900 |
| swapped citations | 21 | 0.66 | 6.05 | 0.321 |
| fabricated citations | 23 | 2.02 | 6.05 | 0.315 |
| inflated conclusions | 19 | −2.79 | 7.00 | 0.537 |
| all three | 17 | −2.86 | 7.13 | 0.044 |

Spearman correlation with process score over the rated reports: readability **0.08**, judged
convincingness **−0.37**. Corrupting a report made the judge find it *more* convincing, most of
all when conclusions were inflated; the report with every part of the process destroyed scored
highest. H-RQ4 holds, more strongly than stated: a small model judge rewards confident claims
regardless of whether they are supported. The judge gave no usable rating for 6 of 109 reports;
they are left out, not imputed. Variants with no change to make (a run with no citations to swap)
are not generated, which is why counts differ.

### RQ5 · E5 reproducibility

All **29/29** runs, rule and model, replayed offline from the recorded responses and model
transcripts with every artefact group (8/8) and the verification identical. Model runs replay
from their transcripts, so this checks the process code, not the model's determinism; how much a
live re-run differs is what E1's repeats measure.

### Reproduction from a fresh clone

On 2026-10-10 a fresh clone of commit 2c4f693 (the analysis code and every result above) ran
`make setup` and `make test` (52 passed) and `make setup-local`, with the model weights checked
against `configs/models.sha256` (the weights were linked from an earlier download rather than
fetched again). With the archived E1 runs and literature cache restored, it reran E1 compare, E2,
the error analysis and E5 offline:

| | Committed | Fresh clone |
|---|---|---|
| E1 compare: every finding and table | as above | identical |
| E2 corruptions rejected / benign rejected | 1035/1035, 0/584 | identical |
| Error analysis (generated/error_analysis.md) | as above | byte-identical |
| E5 replay identical | 29/29 | 29/29, identical tables |

The clone exposed one defect: the weights' hash check failed when `models/` is a symbolic link,
because it named the hash file by a relative path; fixed in the Makefile. E1 itself, the E3 model
critic and the E4 judge were not rerun in the clone (about nine hours together on four cores); E1's live
variability is measured by its repeats, and the first report's fresh clone (commit e3319b4,
2026-10-09) reproduced every qualitative rule-baseline finding from an empty cache against the
live indexes.

## 4. Threats to validity

- **A small model stands in for the intended one.** The model arms use a 4B quantised model on a
  CPU. Its failures (empty evidence on long prompts, invented catalogue names, timid verdicts) may
  not occur with a larger model; the study measures them, it does not generalise them.
- **A small judge.** Quote support, the hypothesis rubric and convincingness come from a 3.8B
  model. E4 shows it rewards confident text; judged metrics are reported beside, never instead
  of, the verified ones.
- **Few runs.** 3 repeats per question and arm; arm differences smaller than the reported
  spreads are not claimed.
- **Exact-name metric check.** The design-validity metric under-credits model designs that name
  the right quantity in other words (35 of 40).
- **Index availability shaped the evidence.** Rate limits changed what was retrieved from day to
  day.
- **Abstracts only.** Quotes are checked against titles and abstracts.
- **E2 and E3 test the faults we thought of.** Perfect detection of designed corruptions is an
  upper bound on detection of real errors; E1 found two real disagreements outside E2's design.
- **Small toolbox.** The q1–q3 effects are measurements on two or three bundled datasets.

## 5. Reproduce

```bash
make setup setup-local models   # Python and web dependencies, llama-cpp-python, model weights (4.9 GB, hash-checked)
cp .env.example .env
make e1 e1-compare e2 e3 e4 e5  # E1 needs network access to the indexes and about 7 hours on four CPU cores for the model arms
make results                     # regenerates docs/generated/results.md
uv run python scripts/error_analysis.py
make screenshots
```

E1-compare, E2 and E5 are offline once E1's runs and cache exist; E3's model critic and E4's
judge need the weights.

## 6. Pending

- E1 Claude and Claude-with-model-critic arms (configured; need `ANTHROPIC_API_KEY`).
- A run with a human reviewer at the four checkpoints, so correction counts exist.
- A larger model critic and judge, to separate "a model critic cannot do this" from "a 4B model
  cannot do this".
