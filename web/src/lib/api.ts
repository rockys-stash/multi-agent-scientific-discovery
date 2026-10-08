import { useCallback, useEffect, useRef, useState } from "react";

/* ---------- types mirrored from the API (src/discoverylab/api.py) ---------- */
export type Stage =
  | "question"
  | "literature"
  | "evidence"
  | "gaps"
  | "hypotheses"
  | "design"
  | "analysis"
  | "critique"
  | "conclusion";
export type NodeKind =
  | "question"
  | "paper"
  | "evidence"
  | "gap"
  | "hypothesis"
  | "design"
  | "result"
  | "conclusion"
  | "critique";

export interface Rate {
  k: number;
  n: number;
  rate: number | null;
}

export interface RunSummary {
  run_id: string;
  question: string;
  question_id: string;
  reasoner: string;
  status: "running" | "awaiting_review" | "complete" | "failed";
  stage: Stage;
  citation_accuracy: Rate;
  evidence_correctness: Rate;
  corrections: Record<string, number | null>;
  human_reviewed: boolean;
  verdicts: Record<string, string>;
}

export interface Citation {
  identifier: string;
  title: string;
  first_author: string;
  year: number | null;
}

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  venue: string;
  abstract: string;
  url: string;
  source: string;
  cache_key: string;
  query: string;
}

export interface Evidence {
  id: string;
  claim: string;
  quote: string;
  citation: Citation;
  concepts: string[];
  stance: "supports" | "contradicts" | "context";
}

export interface Gap {
  id: string;
  description: string;
  concepts: string[];
  evidence_ids: string[];
  kind: string;
}

export interface Hypothesis {
  id: string;
  statement: string;
  independent_variable: string;
  dependent_variable: string;
  expected_direction: string;
  gap_ids: string[];
  evidence_ids: string[];
  rationale: string;
}

export interface Design {
  id: string;
  hypothesis_id: string;
  datasets: string[];
  model: string;
  control: string;
  treatments: string[];
  metrics: string[];
  test: string;
  split: string;
  folds: number;
  seeds: number[];
  rationale: string;
}

export interface ConditionResult {
  condition: string;
  dataset: string;
  metric: string;
  values: number[];
  mean: number;
  lo: number;
  hi: number;
}

export interface Comparison {
  dataset: string;
  metric: string;
  treatment: string;
  control: string;
  mean_diff: number;
  lo: number;
  hi: number;
  p: number;
  p_holm: number | null;
  effect_size_dz: number | null;
  n: number;
  higher_is_better: boolean;
}

export interface AnalysisResult {
  id: string;
  design_id: string;
  conditions: ConditionResult[];
  comparisons: Comparison[];
  runtime_seconds: number;
}

export interface Critique {
  id: string;
  round: number;
  target_id: string;
  stage: Stage;
  issue: string;
  severity: "blocking" | "major" | "minor";
  message: string;
  reviewer: "rule" | "model" | "human";
  resolution: "open" | "fixed" | "accepted" | "dismissed";
  resolution_note: string;
}

export interface Conclusion {
  id: string;
  hypothesis_id: string;
  result_id: string;
  verdict: "supported" | "not_supported" | "inconclusive";
  statement: string;
  evidence_ids: string[];
  limitations: string[];
}

export interface Correction {
  stage: Stage;
  target_id: string;
  action: "approve" | "edit" | "reject" | "add";
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  reason: string;
  actor: string;
}

export interface RunState {
  run_id: string;
  question: {
    id: string;
    text: string;
    keywords: string[];
    concepts: string[];
  };
  reasoner: string;
  stage: Stage;
  status: RunSummary["status"];
  papers: Paper[];
  evidence: Evidence[];
  gaps: Gap[];
  hypotheses: Hypothesis[];
  designs: Design[];
  results: AnalysisResult[];
  critiques: Critique[];
  conclusions: Conclusion[];
  corrections: Correction[];
  completed: Stage[];
  reviewed: Record<string, string>;
  error: string | null;
}

export interface CitationCheck {
  identifier: string;
  normalised: string | null;
  status: "verified" | "metadata_mismatch" | "unresolvable" | "unverifiable" | "malformed";
  title_similarity: number | null;
  author_match: boolean | null;
  year_match: boolean | null;
  resolved_title: string;
  resolved_first_author: string;
  resolved_year: number | null;
  resolved_by: string;
  tried: string[];
  problems: string[];
}

export interface EvidenceCheck {
  evidence_id: string;
  citation: CitationCheck;
  quote: "found" | "not_found" | "no_source_text";
  found_in: string[];
  correct: boolean;
}

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  stage: string;
  meta?: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  rel: string;
  verification?: string | null;
}

export interface Metrics {
  papers: number;
  papers_with_abstract: number;
  evidence: number;
  citation_accuracy: Rate;
  citation_status: Record<string, number>;
  evidence_correctness: Rate;
  quote_status: Record<string, number>;
  gaps: number;
  hypotheses: number;
  design_validity: Record<
    string,
    { passed: number; of: number; checks: Record<string, boolean> }
  >;
  novelty_max_tfidf_similarity: Record<string, number | null>;
  verdicts: Record<string, string>;
  critiques: { stage: string; reviewer: string; severity: string; n: number }[];
  critique_resolution: Record<string, number>;
  human_corrections: Record<string, number | null>;
  human_reviewed: boolean;
}

export interface RunDetail {
  state: RunState;
  verification: { evidence: EvidenceCheck[]; citations_emitted: number } | null;
  metrics: Metrics;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  log_integrity: {
    valid: boolean;
    events: number;
    first_bad_seq: number | null;
    problem: string | null;
  };
  checkpoints: string[];
}

export interface LogEvent {
  seq: number;
  ts: string;
  actor: string;
  kind: string;
  stage: string;
  data: Record<string, unknown>;
  prev: string;
  hash: string;
}

export interface ExperimentInfo {
  experiment: string;
  status: "pending" | "complete";
  run_id?: string;
  commit?: string;
  created_at?: string;
  runtime_seconds?: number;
  description?: string;
}

/* ---------- fetching ---------- */
export class ApiError extends Error {
  constructor(
    public url: string,
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const cache = new Map<string, unknown>();
// Runs can change while one is paused or resumed, so they are always fetched fresh.
const UNCACHED = /^\/api\/runs/;

async function errorFrom(url: string, res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail))
      detail = body.detail
        .map((d: { msg?: string }) => d.msg ?? "invalid input")
        .join("; ");
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(url, res.status, detail);
}

export async function getJson<T>(url: string): Promise<T> {
  if (cache.has(url)) return cache.get(url) as T;
  let res: Response;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(url, 0, "The lab API could not be reached.");
  }
  if (!res.ok) throw await errorFrom(url, res);
  const data = (await res.json()) as T;
  if (!UNCACHED.test(url)) cache.set(url, data);
  return data;
}

export interface ApiState<T> {
  data: T | undefined;
  error: ApiError | undefined;
  loading: boolean;
  retry: () => void;
}

/** Fetches JSON for ``url``; keeps the previous data while a new URL loads. */
export function useApi<T>(url: string | null): ApiState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(url !== null);
  const [nonce, setNonce] = useState(0);
  const current = useRef(url);

  useEffect(() => {
    current.current = url;
    if (url === null) {
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    setError(undefined);
    getJson<T>(url)
      .then((d) => {
        if (alive && current.current === url) setData(d);
      })
      .catch((e: unknown) => {
        if (alive && current.current === url)
          setError(e instanceof ApiError ? e : new ApiError(url, 0, String(e)));
      })
      .finally(() => {
        if (alive && current.current === url) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [url, nonce]);

  const retry = useCallback(() => {
    if (url) cache.delete(url);
    setNonce((n) => n + 1);
  }, [url]);

  return { data, error, loading, retry };
}

export const api = {
  runs: "/api/runs",
  run: (id: string) => `/api/runs/${encodeURIComponent(id)}`,
  log: (id: string) => `/api/runs/${encodeURIComponent(id)}/log`,
  experiments: "/api/experiments",
  experiment: (exp: string) => `/api/experiments/${encodeURIComponent(exp)}`,
};

/** Every experiment writes the same summary shape, so the console renders any of them. */
export interface SummaryTable {
  id: string;
  caption: string;
  columns: { key: string; label: string; align?: "l" | "r" }[];
  rows: Record<string, string | number | null>[];
}

export interface ExperimentSummary {
  title: string;
  question: string;
  findings: string[];
  tables: SummaryTable[];
  notes: string[];
  provenance: ExperimentInfo;
}
