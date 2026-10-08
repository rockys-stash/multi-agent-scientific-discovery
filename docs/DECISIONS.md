# Decisions

**D1. A fixed workflow under a Director, not free-form agent chat.** The spec judges the process. A fixed sequence of stages, each with one owner, a critique after every stage and logged hand-offs, makes the process inspectable and comparable across runs. Agents still decide content at each stage.

**D2. Citations are verified by an independent resolver, not by the agent's own records.** The verifier re-resolves each identifier through the indexes and compares title, first author, year and quote with what comes back. The agent cannot make its own citations look verified.

**D3. Layered layout instead of the planned force layout.** DESIGN first called for a seeded force layout. The graph is a stage pipeline, so a layered layout (one column per stage, barycentre ordering) reads left to right like the workflow, is deterministic without a seed, keeps keyboard movement predictable (up/down within a stage, left/right across stages) and needs no layout library. DESIGN was updated.

**D4. A rule-based reasoner as baseline.** Without a language model the system still runs end to end: query planning from the question's concepts, BM25 evidence selection, gap rules over concept co-occurrence, hypothesis and design from a person-written operationalisation per question. This gives a defensible baseline and lets everything run offline in CI. Its advantage is stated plainly: the operationalisation is written by a person, so its designs are not the baseline's own reasoning. It is labelled "Rule baseline (no language model)" wherever it appears.

**D5. Experiments limited to a fixed toolbox.** Agents design experiments by naming datasets, models, modifiers, metrics and a test from a catalogue; the toolbox runs them with stratified cross-validation, bootstrap intervals, paired tests and Holm correction. Agents cannot run arbitrary code, so every number is produced by audited code, and a hypothesis outside the toolbox is rejected at design time instead of being approximated. Leakage is prevented by construction: every modifier is fitted inside the training fold, so the planned "design with leakage" fault cannot be expressed and is not injected in E3.

**D6. Rule critic plus optional model critic.** Structural checks need no model and cannot be talked round by fluent text. Semantic checks (does the quote support the claim?) need judgement and are left to the model critic and the person. E3 includes semantic faults on purpose to measure what the rule critic misses.

**D7. Human corrections are counted only when a person reviewed.** A stage nobody reviewed reports "no review", not 0, so an unreviewed run cannot look like a flawless one.

**D8. Content-addressed retrieval cache and a transcript for model calls.** Runs record every index response and every model prompt and answer. Replay re-executes everything downstream, so reproducibility is measured (E5) rather than assumed.

**D9. Novelty by TF-IDF only.** PLAN listed TF-IDF and embedding similarity. Embedding models cannot be downloaded in this environment (model hubs are blocked), so only the TF-IDF indicator is computed. It is labelled an indicator, not proof.

**D10. Network and API key blockers (2026-10-08).** The environment's network policy blocks every scholarly index and no `ANTHROPIC_API_KEY` is configured. Everything that can be built and tested without them is built and tested on invented fixtures labelled as such. No end-to-end result, experiment result or screenshot from a run is reported until a real run exists.

**D11. Report fluency is scored on a template report.** The report renderer (`report.py`) writes the report from the artefacts only, so E4's corrupted variants differ in content but not in style. That isolates the question "does the process change how the report reads?" A model-written report would add stylistic variance on top.
