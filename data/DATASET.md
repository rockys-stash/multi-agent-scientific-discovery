# Data

## Literature (retrieved at run time)

| Source | Access | Terms |
|---|---|---|
| OpenAlex (`api.openalex.org`) | Open API, no key; polite pool with `DISCOVERYLAB_MAILTO` | Metadata CC0 |
| Crossref (`api.crossref.org`) | Open API, no key | Metadata is facts, no restriction; abstracts are the publishers' and are used only for quote checking |
| arXiv (`export.arxiv.org`) | Open API, no key, ≥ 3 s between requests | Metadata CC0; abstracts under each paper's licence |
| Semantic Scholar (`api.semanticscholar.org`) | Open API, no key (rate limited) | API licence: research use, attribution |

Every response is cached under `cache/` by a content hash of the request, with the URL, status
and retrieval time, so runs replay offline. The cache is not committed: abstracts belong to
their publishers. Literature was retrieved on 2026-10-08 and 2026-10-09 (E1, `results/e1_runs/`). That day OpenAlex's
anonymous daily credit was exhausted and Semantic Scholar rate-limited the shared address, so
those two indexes contributed fewer records; such answers are recorded as "could not check",
never as missing papers (docs/DECISIONS.md D12). Requests are spaced at least 1 s per host, 3 s
for arXiv.

Known biases: the indexes over-represent English-language, recent and open-access work;
abstracts are the only text, so evidence from a paper's body cannot be quoted or checked.

## Experiment data (bundled, no download)

The experiment toolbox uses datasets shipped with scikit-learn 1.9.1 (BSD-3-Clause package; the
underlying datasets are from the UCI repository).

| Name | Source | Rows × features | Classes (counts) |
|---|---|---|---|
| `breast_cancer` | Wisconsin Diagnostic Breast Cancer | 569 × 30 | 212 / 357 |
| `wine_binary` | UCI Wine, class 0 vs the rest | 178 × 13 | 119 / 59 |
| `digits_binary` | UCI optical digits, ≥ 5 vs < 5 | 1797 × 64 | 901 / 896 |
| `breast_cancer_imb10` | `breast_cancer`, minority subsampled to 10% of the majority (seed 12345) | 393 × 30 | 36 / 357 |
| `digits_binary_imb10` | `digits_binary`, same procedure | 991 × 64 | 901 / 90 |

Counts above were printed from the loaders. Splits are stratified k-fold inside each design
(folds and seeds chosen by the design); every metric is computed out of fold, and calibration
and oversampling are fitted inside the training folds only, so no test fold leaks into training.

## Model weights (downloaded by `make models`, not committed)

| Role | Model | File | Source | Licence |
|---|---|---|---|---|
| Agents and model critic | Qwen3-4B-Instruct-2507, Q4_0 | `qwen3-4b-instruct-2507-q4_0.gguf` (2.38 GB) | huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF | Apache-2.0 |
| Judges | Phi-4-mini-instruct, Q4_K_M | `phi-4-mini-instruct-q4_k_m.gguf` (2.49 GB) | huggingface.co/unsloth/Phi-4-mini-instruct-GGUF | MIT |

SHA-256 hashes are in `configs/models.sha256`; `make models` checks them and every experiment's
`provenance.json` records the file, size and hash of the weights it used. The models' training
data is the developers' and is not documented here; what they know about the literature is not
used as evidence, because every evidence item must quote a retrieved abstract (DECISIONS D17).

## Test fixtures

`tests/conftest.py` contains four invented papers with the reserved test DOI prefix 10.5555.
They exist only to test the machinery and are never reported as literature or as results.
