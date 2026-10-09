# Internal review board

Reviewed 2026-10-09 at the commit that added this file, against the rule-baseline results in
[REPORT.md](REPORT.md). **Overall: not complete.** Every reviewer below passes what exists, but
the project's central comparison, a language-model research agent against the rule baseline,
has not run because no `ANTHROPIC_API_KEY` reached this build environment. Until it runs, the
Completion Gate's "model/condition comparisons executed" item fails.

| Reviewer | Verdict | Required fixes |
|---|---|---|
| Research | Pass for what ran; **fail for the gate** | Run E1's Claude arms (3 repeats each), the judges and the model critic |
| Engineering | Pass | None open |
| QA | Pass | None open |
| Security | Pass | None open |
| UX | Pass | None open |
| Recruiter | Pass | None open |
| PhD supervisor | Pass with reservation | Same as Research |

## Research reviewer

*Is this scientifically defensible?*

- Questions, hypotheses and a baseline were fixed before the runs (README, configs, D4). Metrics
  are defined in METRICS.md and computed only from artefacts and independent verification.
- Findings that held: verifier 482/482 corruptions rejected and 0/273 benign variants rejected;
  rule critic 72/72 structural and 0/18 semantic faults; readability uncorrelated with process
  (ρ = 0.04); replay 3/3 identical.
- Negative and awkward results are reported, not tuned: two real false rejections by the verifier
  (D14, D15), a verdict rule that calls q2 "not supported" although the effect appears where
  feature scales differ, and the first E5 failing 0/3 (D13).
- Variance: the rule baseline is deterministic, so run-to-run variance is undefined and stated as
  such; the agents' own experiments report bootstrap intervals over 15 paired folds.
- **Open:** the comparison that answers "does a multi-agent LLM system do research soundly" is
  pending. E3's semantic faults and E4's convincingness need a model. Verified evidence is often
  only lexically related to the question (REPORT §3), which the rule baseline cannot fix.

## Engineering reviewer

*Is this technically sound?*

- Live runs exposed three defects that offline fixtures had not: a 429 read as "no record"
  (D12), replay serving newer cache versions (D13), and the E2 same-author corruption (D15). Each
  was fixed with a regression test before results were regenerated.
- arXiv requests are now spaced 3 s apart as its API terms ask; DATASET.md said so but the code
  used 1 s (fixed in this review).
- Typed core (mypy strict on src, tests, scripts), ruff, ESLint, TypeScript strict; results are
  versioned with config, commit and a dirty flag; a failed experiment leaves `FAILED` and never
  moves `LATEST`.

## QA reviewer

*Does it actually work, end to end?*

- 43 unit, workflow, harness and API tests and 27 browser tests pass; CI runs lint and tests on
  every push.
- Fresh clone of the repository → `make setup` → `make test` passed, then E1–E5 were reproduced
  from scratch against the live indexes (results in that clone's `results/`, summarised in the
  registry). Replay from the recorded cache reproduced every artefact.
- The console was driven against the real runs to capture docs/screenshots (desktop light and
  dark, mobile); loading, empty, error and not-found states are covered by the browser tests.

## Security reviewer

*Any obvious security or privacy problems?*

- No credentials in code or history; configuration via `.env` (ignored) and `.env.example`.
- The API refuses a non-loopback host without `DISCOVERYLAB_API_TOKEN`, rate-limits clients and
  requires the token for writes; same-origin CSP and security headers on the console.
- The literature cache holds publishers' abstracts and is not committed (DATASET.md). No contact
  address is sent to the indexes unless `DISCOVERYLAB_MAILTO` is set.

## UX reviewer

*Does it feel like a professionally designed product?*

- The research network reads left to right like the workflow; the citation table shows what was
  cited next to what the source says, with the failing field named.
- Fixed in this review: experiments with a pending part (the Claude arms, judges) were labelled
  "Complete"; they now show "Partial: part pending".
- 0 axe violations in both themes and both viewports (browser tests).

## Recruiter reviewer

*Does this repository demonstrate meaningful skill?*

- A working multi-agent pipeline over four real scholarly APIs, an independent verifier, a fixed
  experiment toolbox with proper statistics, fault injection, replay, and a designed console. The
  decision log shows defects found by measurement and fixed, which is the strongest signal here.

## PhD supervisor reviewer

*Does it demonstrate research ability, not merely software development?*

- Yes in method: the project measures the instruments (verifier, critic, replay) before trusting
  them, reports their failures, and separates "verified" from "relevant". The substantive
  scientific question about LLM agents is still open; the report says so in its first paragraph.
