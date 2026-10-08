# Design: "Lattice", a research network console

## Visual concept

A field notebook crossed with a citation index. The main view is the run's **research network**: the question at the root, then papers, evidence, gaps, hypotheses, designs, results and critiques as nodes. Edges are typed relations: *retrieved for*, *quotes*, *supports*, *reveals gap*, *motivates*, *tests*, *produced*, *critiques*.

The graph is not decoration. Every edge that rests on a citation shows its verification status, so a reader can see at a glance which parts of a conclusion stand on verified sources.

- Graphite neutrals.
- One accent, **ochre**, used for selection, focus and the "verified" state.
- Status colours are reserved and always come with a glyph and a label.

## Information architecture

| Route | Page | Purpose |
|---|---|---|
| `/runs` | Runs | Every run: question, reasoner, stage reached, citation accuracy, corrections. |
| `/runs/:id` | Network | The knowledge graph for one run, with a node inspector. |
| `/runs/:id/process` | Process | The workflow stages in order, the Director's hand-offs, each critique round and its resolution. |
| `/runs/:id/citations` | Citations | Every emitted citation with its resolution, metadata match and quote check, linked to the source record. |
| `/runs/:id/corrections` | Corrections | The human-correction log: stage, action, before and after, reason. |
| `/evaluation` | Evaluation | E1–E5 tables and charts from committed results, or a pending state. |
| `/method` | Method | How verification and scoring work, and what is not measured. |

## Navigation model

- A top bar holds the run selector and the page tabs.
- The network view has a side inspector: selecting a node opens its artefact, sources and critiques.
- Keyboard:
  - `/` focuses search;
  - arrow keys move between connected nodes;
  - `Enter` opens the inspector;
  - `Esc` closes it;
  - `[` and `]` step through critique rounds.

## Typography

- **IBM Plex Sans** for the interface and **IBM Plex Mono** for identifiers (DOIs, hashes).
- Body 14/20; inspector headings 16/24; page titles 22/28.
- Tabular figures in every table.

## Color tokens

| Token | Light | Dark |
|---|---|---|
| `--bg` | `#f4f3f0` | `#16171a` |
| `--surface` | `#ffffff` | `#1f2024` |
| `--ink` | `#16171a` | `#ecebe7` |
| `--ink-2` | `#4a4b50` | `#b4b3ae` |
| `--rule` | `#dcdad4` | `#34353a` |
| `--accent` (ochre) | `#8a5410` | `#e2a24e` |
| `--ok` | `#2f6b3a` | `#7cc489` |
| `--warn` | `#8a5a12` | `#e0b45c` |
| `--bad` | `#a3302a` | `#f08a80` |

- Node kinds are distinguished by **shape and label**, never by colour alone:
  - circle: paper;
  - square: evidence;
  - diamond: gap;
  - hexagon: hypothesis;
  - rounded rectangle: design and result;
  - critiques: a small triangle with a count on the node they target.
- All nodes are graphite, apart from the selected node (ochre) and status rings.

## Spacing, shape and components

- A 4 px base grid; 1 px rules; 6 px radii.
- No gradients, glass, neon or emoji.
- Reuses the console components from the earlier projects: segmented controls, data tables with a tabbable scroll region, status marks and provenance lines.

## Data visualisation

- The **network** uses a deterministic layered layout: one column per workflow stage, left to right, with nodes ordered inside each column by the mean position of their neighbours (two barycentre sweeps). The same run always draws the same way. Edges leave a node after its label, so they do not cross text. (A force layout was planned first; see DECISIONS D3.)
- **Evaluation** shows each experiment's tables as they were written; numbers appear only for experiments that ran.

## Motion

Layout settles once. Selection uses 120 ms transitions. Everything is disabled under `prefers-reduced-motion`.

## States

Loading, empty (no runs yet, or a run stopped before a stage), error (API unreachable, run failed) and pending (an experiment not run) all have their own message and an action where one exists.

## Responsive behaviour

- Below 760 px, the network becomes a stage-ordered list with expandable nodes, because a dense graph is not usable on a phone. A Graph/List toggle offers the list on desktop too.
- The inspector becomes a full-width sheet.

## Accessibility

- The graph has a list alternative, announced in place of the SVG.
- Every node is focusable, with an accessible name built from its kind, label and status.
- WCAG AA contrast in both themes, checked with axe-core in the browser tests.
