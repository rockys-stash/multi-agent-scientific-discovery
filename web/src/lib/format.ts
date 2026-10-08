import type { NodeKind, Rate, Stage } from "./api";

export const STAGES: { id: Stage; label: string }[] = [
  { id: "literature", label: "Literature search" },
  { id: "evidence", label: "Evidence extraction" },
  { id: "gaps", label: "Gap identification" },
  { id: "hypotheses", label: "Hypothesis generation" },
  { id: "design", label: "Experiment design" },
  { id: "analysis", label: "Analysis" },
  { id: "conclusion", label: "Conclusion" },
  { id: "critique", label: "Final critique" },
];

export const STAGE_LABEL: Record<string, string> = Object.fromEntries(
  STAGES.map((s) => [s.id, s.label]),
);

export const KIND_LABEL: Record<NodeKind, string> = {
  question: "Question",
  paper: "Paper",
  evidence: "Evidence",
  gap: "Gap",
  hypothesis: "Hypothesis",
  design: "Design",
  result: "Result",
  conclusion: "Conclusion",
  critique: "Critique",
};

export const OWNER: Record<string, string> = {
  literature: "Literature agent",
  evidence: "Literature agent",
  gaps: "Hypothesis agent",
  hypotheses: "Hypothesis agent",
  design: "Experiment agent",
  analysis: "Experiment agent",
  conclusion: "Experiment agent",
  critique: "Critic",
};

export const ACTOR_LABEL = (a: string) =>
  a.startsWith("human:")
    ? `Researcher (${a.slice(6)})`
    : ((
        {
          director: "Research Director",
          literature: "Literature agent",
          hypothesis: "Hypothesis agent",
          experiment: "Experiment agent",
          critic: "Critic",
          system: "System",
        } as Record<string, string>
      )[a] ?? a);

export type Tone = "ok" | "attest" | "bad" | "neutral";

/** Verification shown on evidence and citations: a glyph, a word and a tone, never colour alone. */
export function verificationMark(v: string | null | undefined): {
  tone: Tone;
  glyph: string;
  word: string;
  help: string;
} {
  switch (v) {
    case "verified":
      return {
        tone: "ok",
        glyph: "✓",
        word: "Verified",
        help: "Identifier resolved, metadata matches and the quote is in the source.",
      };
    case "quote_not_found":
      return {
        tone: "bad",
        glyph: "✕",
        word: "Quote not in source",
        help: "The source exists but the quoted text is not in it.",
      };
    case "quote_no_source_text":
      return {
        tone: "attest",
        glyph: "◐",
        word: "No source text",
        help: "The source resolved but has no abstract to check the quote against.",
      };
    case "metadata_mismatch":
      return {
        tone: "bad",
        glyph: "✕",
        word: "Metadata mismatch",
        help: "The identifier resolves to a different title, author or year.",
      };
    case "unresolvable":
      return {
        tone: "bad",
        glyph: "✕",
        word: "Does not resolve",
        help: "No index returned a record for this identifier.",
      };
    case "malformed":
      return {
        tone: "bad",
        glyph: "✕",
        word: "Not an identifier",
        help: "Not a DOI, arXiv id or OpenAlex id.",
      };
    default:
      return {
        tone: "neutral",
        glyph: "–",
        word: "Not verified",
        help: "The run has not been verified yet.",
      };
  }
}

export const VERDICT: Record<
  string,
  { tone: Tone; glyph: string; word: string }
> = {
  supported: { tone: "ok", glyph: "✓", word: "Supported" },
  not_supported: { tone: "bad", glyph: "✕", word: "Not supported" },
  inconclusive: { tone: "attest", glyph: "◐", word: "Inconclusive" },
};

export const RUN_STATUS: Record<
  string,
  { tone: Tone; glyph: string; word: string }
> = {
  complete: { tone: "ok", glyph: "✓", word: "Complete" },
  awaiting_review: { tone: "attest", glyph: "◐", word: "Waiting for review" },
  running: { tone: "neutral", glyph: "…", word: "Running" },
  failed: { tone: "bad", glyph: "✕", word: "Failed" },
};

export const pct = (r: Rate | null | undefined, digits = 0) =>
  r && r.rate !== null ? `${(100 * r.rate).toFixed(digits)}%` : "–";

export const frac = (r: Rate | null | undefined) =>
  r && r.n ? `${r.k}/${r.n}` : "none";

export const num = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined ? "–" : x.toFixed(d);

export const signed = (x: number, d = 4) =>
  `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;

export const pval = (p: number | null | undefined) =>
  p === null || p === undefined ? "–" : p < 0.001 ? "< 0.001" : p.toFixed(3);

export const REASONER: Record<string, string> = {
  rule: "Rule baseline (no language model)",
  claude: "Claude",
  "claude+mc": "Claude with model critic",
};

export const verdictOf = (v: string) =>
  VERDICT[v] ?? {
    tone: "neutral" as Tone,
    glyph: "–",
    word: v.replace(/_/g, " "),
  };
export const statusOf = (s: string) =>
  RUN_STATUS[s] ?? {
    tone: "neutral" as Tone,
    glyph: "–",
    word: s.replace(/_/g, " "),
  };
