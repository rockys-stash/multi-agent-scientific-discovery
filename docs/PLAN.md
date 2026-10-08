# Plan: Multi-Agent Scientific Discovery Lab

## 1. Scope

A research environment in which a **Research Director** agent coordinates **Literature**, **Hypothesis** and **Experiment** agents. A **Critic** agent and a **human researcher** are in the loop. The workflow is fixed by the spec:

```
research question → literature search → evidence extraction → gap identification
                  → hypothesis generation → experiment design → analysis → critique
```

The evaluation principle is also fixed: **judge the process, not how fluent the report sounds.** Every citation an agent emits is checked against the real source it names.

Deliverables:

1. **Agent runtime** (`discoverylab.agents`). The Director and four specialist agents. Each one reads typed inputs and writes typed artefacts (`Paper`, `Evidence`, `Gap`, `Hypothesis`, `ExperimentDesign`, `AnalysisResult`, `Critique`). Model calls go through one `Reasoner` interface with two implementations:
   - `ClaudeReasoner`: a language model with structured outputs (Anthropic SDK);
   - `RuleReasoner`: a transparent, non-LLM baseline (retrieval ranking, verbatim sentence extraction, concept-coverage gaps, templated hypotheses, a rule-based critic).
   
   The baseline is not the system under study. It exists so that every stage, the evaluation and the console can be tested deterministically, and it gives the language-model agents a floor to beat.
2. **Literature layer** (`discoverylab.literature`). Adapters for OpenAlex, Crossref, arXiv and Semantic Scholar behind one interface. Every raw response is stored in a content-addressed cache with its URL and retrieval time, so a run can be replayed exactly from the cache.
3. **Citation verifier** (`discoverylab.verify`). Resolves each emitted citation (DOI, arXiv id or OpenAlex id) against the source registry, independently of the agent that cited it. It then checks the title, authors and year, and checks that every quoted evidence span appears in the source text the verifier retrieved.
4. **Experiment toolbox** (`discoverylab.toolbox`). The datasets, models, metrics and statistical tests the Experiment agent may compose into a design. Analyses run for real; agents never write numbers.
5. **Process log** (`discoverylab.log`). An append-only, hash-chained JSONL record of every agent message, tool call, artefact, critique round and human action.
6. **Human-in-the-loop checkpoints.** The researcher can approve, edit or reject the question framing, the evidence set, the hypotheses and the design. Every edit is logged as a correction, with its stage and reason.
7. **Console "Lattice"** (`web/`). The research network as a knowledge graph, an inspector for each node, the verification status of every citation, the critique cycle and the correction log.
8. **Process evaluation** (`discoverylab.evaluate`), experiments E1–E5, a research report and the documentation set.

## 2. Research questions and hypotheses

| RQ | Question | Hypothesis | Experiment |
|---|---|---|---|
| RQ1 | Can a multi-agent pipeline answer a real research question with **correct evidence** and **citations that resolve**? | With retrieval grounded in real sources and an independent verifier, ≥ 95% of citations resolve with matching metadata. Quoted evidence is found verbatim in the source for ≥ 90% of claims. | E1 |
| RQ2 | Is the verifier itself trustworthy? | Seeded with real citations corrupted in known ways (fabricated identifier, wrong year, wrong author, wrong title, misattributed or altered quote), the verifier flags ≥ 99% of corruptions with no false alarms on clean citations. | E2 |
| RQ3 | Does the **critique cycle** catch process errors? | When known faults are injected into stage outputs, the Critic catches most of them. The faults are an unsupported claim, a citation to an unretrieved paper, a hypothesis with no gap, a design without a control or with leakage, an arithmetic error and a conclusion that overreaches. Rule checks catch the structural faults; the language-model critic adds the semantic ones. | E3 |
| RQ4 | Does **fluency** predict process quality? | No. Report variants whose process has been corrupted (swapped citations, unsupported conclusions) keep their readability and "convincingness" scores while their process scores fall. | E4 |
| RQ5 | Is a run **reproducible**? | Replaying from the retrieval cache reproduces the process graph exactly. A fresh live run reproduces the evidence set only partly; we measure how much. | E5 |

The hypothesis-quality rubric, novelty and human-correction count are reported for every end-to-end run (§4).

## 3. Research questions for end-to-end runs

The acceptance criterion needs an end-to-end run on a real research question. Questions are chosen so that both halves of the workflow can be done for real here:
- the literature exists and is indexed by the open APIs;
- the experiment can run on data available without credentials.

Each question is a YAML file in `configs/questions/`. The primary question is:

> **Q1.** Does post-hoc probability calibration (Platt scaling or isotonic regression) improve the probability estimates of tree ensembles on tabular classification, and at what cost in discrimination?

Literature on this question exists (for example Niculescu-Mizil and Caruana, 2005), and the experiment can run on the datasets bundled with scikit-learn. Q2 (feature scaling and nearest-neighbour classifiers) and Q3 (class weighting against oversampling for minority recall) are secondary questions. They exist so that results are not tied to one question.

## 4. Metrics

Defined formally in `docs/METRICS.md`.

| Metric | How it is measured |
|---|---|
| Citation accuracy | Share of emitted citations that resolve and whose title, first author and year match the source. |
| Evidence correctness | Share of evidence items whose quote appears verbatim (after whitespace and Unicode normalisation) in the source. A second, judged layer asks whether the quote supports the claim it is attached to: language-model judge, plus a human spot check. |
| Hypothesis quality | A 5-criterion rubric scored 0–2 each: specificity, testability, grounding in an identified gap, falsifiability, and consistency with the evidence. Scored by a language-model judge and, for the pilot, by the researcher. |
| Experimental validity | A checklist applied to the design: control or baseline present, held-out evaluation, no train/test leakage, a pre-stated metric, a statistical test matching the design, seeds, enough repeats. |
| Novelty | Maximum similarity (TF-IDF and embedding) of each hypothesis to the retrieved abstracts. High similarity means the hypothesis restates existing work. Reported as an indicator, not as proof of novelty. |
| Reproducibility | Graph identity under cache replay; Jaccard overlap of papers and evidence under a live re-run; metric agreement of the analysis across seeds. |
| Human-correction count | Number of human edits and rejections per stage in the process log. Reported only from real sessions. |

## 5. Experiments

| | Experiment | Data |
|---|---|---|
| E1 | End-to-end runs on Q1–Q3 with each reasoner, with citation and evidence verification. | Live APIs, then cache |
| E2 | Verifier validity: 7 corruption types applied to every real citation from E1. | E1 cache |
| E3 | Critic recall and precision per fault type, for rule and language-model critics. 6 fault types are injected into real stage outputs from E1. | E1 artefacts |
| E4 | Fluency against process: corrupted report variants, scored for readability (a formula) and convincingness (language-model judge) against process scores. | E1 reports |
| E5 | Reproducibility: cache replay and live re-run. | E1 |

Seeds and configurations are in YAML. Results go to `results/<exp>/<UTC time>-<commit>/`.

## 6. Risks and dependencies

- **Network.** The scholarly APIs are blocked by the current environment's network policy, as of 2026-10-08. Until they are allowed, the literature layer, the verifier and the console are built and tested against small recorded fixtures that are labelled as fixtures. No end-to-end result is reported. *Status of E1–E5: pending.*
- **Language model.** No API key is available in the environment. The `ClaudeReasoner` is implemented and unit-tested with a stubbed client. Every language-model result is *Status: pending* until a key is configured. Rule-baseline results are always labelled as such.
- **Human researcher.** Correction counts come only from real use of the console. A pilot run by the project owner is the minimum. Simulated corrections are never reported as human ones.
- **API terms.** Requests are polite: an identifying `User-Agent`, rate limits per source, and caching.

## 7. Stack

- Python 3.13 with uv; httpx; pydantic; scikit-learn; scipy; FastAPI; SQLite for the cache index.
- The Anthropic SDK for the language-model reasoner.
- The `web/` console uses React, TypeScript, Vite and a hand-written force layout (no graph library).
- Playwright and axe-core for browser tests.
