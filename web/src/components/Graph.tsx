import { useMemo, useRef, type KeyboardEvent } from "react";
import type { GraphEdge, GraphNode, NodeKind } from "../lib/api";
import { KIND_LABEL, verificationMark } from "../lib/format";

/* A layered layout: one column per workflow stage, left to right, with nodes ordered inside each
   column by the mean position of their neighbours (two barycentre sweeps). Deterministic: the same
   run always draws the same way. Critiques are not columns; they show as a count on their target. */

const COLUMN: Partial<Record<NodeKind, number>> = {
  question: 0,
  paper: 1,
  evidence: 2,
  gap: 3,
  hypothesis: 4,
  design: 5,
  result: 6,
  conclusion: 7,
};
const COL_W = 196;
const ROW_H = 34;
const PAD_X = 24;
const PAD_Y = 40;
const LABEL_CHARS = 22;
const EDGE_OUT = COL_W - 26; // edges leave a node after its label, so they never cross text

export interface Placed extends GraphNode {
  col: number;
  row: number;
  x: number;
  y: number;
}

export function layout(nodes: GraphNode[], edges: GraphEdge[]) {
  const cols: GraphNode[][] = Array.from({ length: 8 }, () => []);
  for (const n of nodes) {
    const c = COLUMN[n.kind];
    if (c !== undefined) cols[c]?.push(n);
  }
  const nbrs = new Map<string, Set<string>>();
  for (const e of edges) {
    for (const [a, b] of [
      [e.source, e.target],
      [e.target, e.source],
    ] as const) {
      const set = nbrs.get(a) ?? new Set<string>();
      set.add(b);
      nbrs.set(a, set);
    }
  }
  const pos = new Map<string, number>();
  cols.forEach((col) => col.forEach((n, i) => pos.set(n.id, i)));
  const sweep = (order: number[], ref: (c: number) => number) => {
    for (const c of order) {
      const other = new Set(cols[ref(c)]?.map((n) => n.id) ?? []);
      const score = (n: GraphNode) => {
        const ps = [...(nbrs.get(n.id) ?? [])]
          .filter((m) => other.has(m))
          .map((m) => pos.get(m) ?? 0);
        return ps.length
          ? ps.reduce((a, b) => a + b, 0) / ps.length
          : (pos.get(n.id) ?? 0);
      };
      const col = cols[c] ?? [];
      col.sort(
        (a, b) =>
          score(a) - score(b) ||
          a.id.localeCompare(b.id, "en", { numeric: true }),
      );
      col.forEach((n, i) => pos.set(n.id, i));
    }
  };
  sweep([2, 3, 4, 5, 6, 7], (c) => c - 1); // evidence follows papers, gaps follow evidence...
  sweep([1], () => 2); // then papers follow the evidence that quotes them
  const tallest = Math.max(1, ...cols.map((c) => c.length));
  const placed = new Map<string, Placed>();
  cols.forEach((col, c) => {
    const offset = ((tallest - col.length) * ROW_H) / 2;
    col.forEach((n, r) =>
      placed.set(n.id, {
        ...n,
        col: c,
        row: r,
        x: PAD_X + c * COL_W,
        y: PAD_Y + offset + r * ROW_H,
      }),
    );
  });
  return {
    placed,
    cols,
    width: PAD_X * 2 + 7 * COL_W + 150,
    height: PAD_Y * 2 + tallest * ROW_H,
    nbrs,
  };
}

function Shape({ kind }: { kind: NodeKind }) {
  switch (kind) {
    case "question":
      return (
        <>
          <circle r={9} className="shape" />
          <circle r={5} className="shape inner" />
        </>
      );
    case "paper":
      return <circle r={6.5} className="shape" />;
    case "evidence":
      return <rect x={-6} y={-6} width={12} height={12} className="shape" />;
    case "gap":
      return <path d="M0 -8 8 0 0 8 -8 0z" className="shape" />;
    case "hypothesis":
      return <path d="M-7 -4 0 -8 7 -4 7 4 0 8 -7 4z" className="shape" />;
    case "conclusion":
      return (
        <rect
          x={-8}
          y={-6}
          width={16}
          height={12}
          rx={3}
          className="shape strong"
        />
      );
    default:
      return (
        <rect x={-8} y={-6} width={16} height={12} rx={3} className="shape" />
      );
  }
}

const short = (s: string, n = LABEL_CHARS) =>
  s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s;

export function nodeName(n: GraphNode): string {
  const status =
    n.kind === "evidence"
      ? `, ${verificationMark(n.meta?.verification as string | undefined).word}`
      : "";
  const shortId = n.kind === "paper" ? "" : ` ${n.id}`;
  return `${KIND_LABEL[n.kind]}${shortId}: ${n.label}${status}`;
}

export function Graph({
  nodes,
  edges,
  selected,
  onSelect,
  critiqueCount,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  critiqueCount: Map<string, number>;
}) {
  const { placed, cols, width, height, nbrs } = useMemo(
    () => layout(nodes, edges),
    [nodes, edges],
  );
  const refs = useRef(new Map<string, SVGGElement>());
  const near = selected ? (nbrs.get(selected) ?? new Set<string>()) : null;

  const focus = (id: string | undefined) => {
    if (!id) return;
    refs.current.get(id)?.focus();
  };

  const onKey = (e: KeyboardEvent<SVGGElement>, n: Placed) => {
    const col = cols[n.col] ?? [];
    const toCol = (c: number) => {
      const target = cols[c];
      if (!target?.length) return undefined;
      const linked = target.filter((m) => nbrs.get(n.id)?.has(m.id));
      const pool = linked.length ? linked : target;
      const dy = (m: GraphNode) => Math.abs((placed.get(m.id)?.y ?? 0) - n.y);
      return pool.reduce((best, m) => (dy(m) < dy(best) ? m : best)).id;
    };
    const keys: Record<string, () => string | undefined> = {
      ArrowDown: () => col[n.row + 1]?.id,
      ArrowUp: () => col[n.row - 1]?.id,
      ArrowRight: () => toCol(n.col + 1),
      ArrowLeft: () => toCol(n.col - 1),
    };
    const move = keys[e.key];
    if (move) {
      e.preventDefault();
      focus(move());
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(n.id);
    } else if (e.key === "Escape") {
      onSelect(null);
    }
  };

  const heads = [
    "Question",
    "Papers",
    "Evidence",
    "Gaps",
    "Hypotheses",
    "Designs",
    "Results",
    "Conclusions",
  ];

  return (
    <div
      className="graph-scroll"
      role="region"
      aria-label="Research network. Arrow keys move between nodes, Enter opens one."
      tabIndex={-1}
    >
      <svg
        className="graph"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label="Research network"
      >
        {heads.map((h, i) =>
          cols[i]?.length ? (
            <text key={h} x={PAD_X + i * COL_W - 8} y={18} className="col-head">
              {h} <tspan className="muted">{cols[i]?.length}</tspan>
            </text>
          ) : null,
        )}
        <g className="edges" aria-hidden="true">
          {edges.map((e, i) => {
            const a = placed.get(e.source);
            const b = placed.get(e.target);
            if (!a || !b) return null;
            const [l, r] = a.x <= b.x ? [a, b] : [b, a];
            const mx = (l.x + EDGE_OUT + r.x) / 2;
            const on =
              selected !== null &&
              (e.source === selected || e.target === selected);
            const fail = e.verification && e.verification !== "verified";
            return (
              <path
                key={i}
                d={`M${l.x + EDGE_OUT} ${l.y} C${mx} ${l.y} ${mx} ${r.y} ${r.x - 10} ${r.y}`}
                className={`edge${on ? " on" : ""}${selected && !on ? " dim" : ""}${fail ? " fail" : ""}${e.rel === "quotes" && e.verification === "verified" ? " verified" : ""}`}
              />
            );
          })}
        </g>
        {[...placed.values()].map((n) => {
          const isSel = n.id === selected;
          const dim = selected !== null && !isSel && !near?.has(n.id);
          const v =
            n.kind === "evidence"
              ? verificationMark(n.meta?.verification as string | undefined)
              : null;
          const cited = n.kind !== "paper" || n.meta?.cited !== false;
          const crit = critiqueCount.get(n.id) ?? 0;
          return (
            <g
              key={n.id}
              ref={(el) => {
                if (el) refs.current.set(n.id, el);
                else refs.current.delete(n.id);
              }}
              transform={`translate(${n.x} ${n.y})`}
              className={`node k-${n.kind}${isSel ? " sel" : ""}${dim ? " dim" : ""}${cited ? "" : " uncited"}${v ? ` v-${v.tone}` : ""}`}
              role="button"
              tabIndex={0}
              aria-pressed={isSel}
              aria-label={
                nodeName(n) +
                (crit ? `, ${crit} critique${crit > 1 ? "s" : ""}` : "")
              }
              onClick={() => onSelect(isSel ? null : n.id)}
              onKeyDown={(e) => onKey(e, n)}
            >
              <title>{nodeName(n)}</title>
              <rect
                x={-12}
                y={-14}
                width={COL_W - 14}
                height={28}
                className="hit"
              />
              <Shape kind={n.kind} />
              {v && v.tone !== "neutral" && (
                <circle r={10.5} className={`ring ${v.tone}`} />
              )}
              <text x={14} y={4} className="node-label">
                {n.kind === "paper"
                  ? short(n.label)
                  : `${n.id} · ${short(n.label, LABEL_CHARS - n.id.length - 3)}`}
              </text>
              {crit > 0 && (
                <g
                  transform={`translate(${-14} ${-12})`}
                  className="crit-badge"
                  aria-hidden="true"
                >
                  <path d="M0 -5 5 4 -5 4z" />
                  <text x={7} y={4}>
                    {crit}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
