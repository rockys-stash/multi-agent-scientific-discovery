# Internal review board

Reviewed 2026-10-10 against [REPORT.md](REPORT.md), at the commit that changed this file. The
first review (2026-10-09) failed the Completion Gate because no model arm had run. The model arms
have now run on local open-weights models (D17), at the owner's request to finish without the
Anthropic API. **Overall: pass**, with the Claude arms recorded as pending and the model
substitution stated in the first paragraph of the README and the report.

| Reviewer | Verdict | Required fixes |
|---|---|---|
| Research | Pass | None open; Claude arms and a larger judge listed as future work |
| Engineering | Pass | None open |
| QA | Pass | None open |
| Security | Pass | None open |
| UX | Pass | None open |
| Recruiter | Pass | None open |
| PhD supervisor | Pass | None open |

## Research reviewer

*Is this scientifically defensible?*

- The model/condition comparison now exists: rule baseline against three local-model arms on the
  same three questions, 3 repeats per model arm with logged seeds, mean and standard deviation
  per arm and per question (E1 compare). Arm differences smaller than their spreads (the model
  critic's effect) are reported as not distinguishable, not as findings.
- The ablation (`local-b5`) was added as a new arm after seeing the first result, and is labelled
  as such (D18); the original arm was not changed.
- Negative results are reported: 14/40 unrunnable model designs, 7/26 verdicts against the
  statistics, a model critic that misses 32/40 structural faults, a judge that prefers corrupted
  reports (ρ = −0.37), one failed run kept as failed (D20).
- Two harness defects found on the model runs were fixed and the experiment rerun, with the first
  result kept (D21 verdict fault, D20 validator).
- A metric limitation found in the error analysis (exact-name primary-metric check) is stated in
  the report and METRICS.md rather than changed after seeing results.
- Substituting a 4B local model for the designed model is the largest threat; it is stated in the
  scope paragraph, in threats to validity, and in every model-arm label.

## Engineering reviewer

*Is this technically sound?*

- The local client implements the same `messages.parse` call as the Anthropic client, so agents,
  critic and judges have one code path; output is grammar-constrained to each schema, an unusable
  answer is retried once and then fails the step, never repaired.
- Weights are hash-pinned (`configs/models.sha256`) and recorded in each experiment's provenance
  with package versions.
- Live runs exposed defects that fixtures had not: container restarts from double-resident
  weights (D19, fixed with no-mmap loading and smaller contexts), an E1 without checkpoints (D18,
  now checkpointed after every run, `reuse_from` takes a list), the validator gap (D20). The
  fresh clone exposed a hash check that broke when `models/` is a symlink (fixed in this review).
- mypy strict, ruff, ESLint and TypeScript strict are clean.

## QA reviewer

*Does it actually work, end to end?*

- 52 offline tests (including the local adapter against a fake llama.cpp model) and 27 browser
  tests pass; CI runs lint, the offline tests, the web build and the browser tests on every push.
- Fresh clone of the pushed commit: `make setup`, `make test` (52 passed), `make setup-local`,
  then E1 compare, E2, the error analysis and E5 from the archived runs and cache; results in
  REPORT §3, "Reproduction from a fresh clone".
- Screenshots regenerated from the console serving the real runs, including a local-model run's
  network and critiques.

## Security reviewer

*Any obvious security or privacy problems?*

- No credentials in code or history; `.env` is ignored. The local path needs no key and makes no
  network call after the weights are downloaded.
- Weights are downloaded over HTTPS from a fixed URL and checked against pinned hashes before
  use.
- API, CSP and token rules unchanged from the first review.

## UX reviewer

*Does it feel like a professionally designed product?*

- Model arms are labelled in plain words in the console ("Local model, 5 abstracts per read") and
  the Evaluation page shows the arm comparison. 0 axe violations in both themes and viewports.

## Recruiter reviewer

*Does this repository demonstrate meaningful skill?*

- A working multi-agent pipeline over four real scholarly APIs, run end to end with both a rule
  baseline and an open model on a CPU, with independent verification, fault injection, replay and
  a measured account of where the model failed. The decision log shows problems found by
  measurement and fixed.

## PhD supervisor reviewer

*Does it demonstrate research ability, not merely software development?*

- Yes: the instruments are validated before being trusted (E2, E3, E5), the agent comparison is
  run with repeats and reported with spreads, the ablation is labelled as post hoc, and the most
  interesting finding (a small judge rates corrupted reports as more convincing) is reported with
  its limits. What a frontier model would do remains open and is said to be open.
