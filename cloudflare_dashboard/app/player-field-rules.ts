/**
 * Pure row building for the whole-field board (/weekly-players): one comparison row per player for the chosen market.
 * Probabilities arrive as fractions (0.24 = 24%); the board shows percent. Gap = model minus book, in percentage points
 * (positive = our chance is above the book's). No runtime imports, so node's strip-types test loader can run it.
 */
import type { ExplainDoc, ExPlayer, Market } from "./explain-rules";
import { playerDisplayName } from "./player-name.ts";

export type BoardSort = "model" | "gap" | "book" | "skill" | "name";
export type BoardRow = {
  id: number; name: string; country: string | null; model: number | null; book: number | null; gap: number | null;
  skill: number | null; why: string; player: ExPlayer;
};

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
/** Fraction to percent, rounded so tiny float noise never decides a sort or a sign. */
export const toPts = (v: number | null | undefined): number | null => (fin(v) ? Math.round(v * 1e6) / 1e4 : null);

/** Skill against the field average, strokes per round (live runs use the live skill). Null when any player lacks a number. */
export function fieldSkills(doc: Pick<ExplainDoc, "kind" | "players">): Map<number, number> {
  const active = doc.players.filter((p) => !p.withdrawn);
  const metric = (p: ExPlayer) => (doc.kind === "live" ? p.live?.mu_live : p.mu);
  const vals = active.map(metric).filter(fin);
  const out = new Map<number, number>();
  if (!vals.length || vals.length !== active.length) return out;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  for (const p of active) out.set(p.id, (metric(p) as number) - mean);
  return out;
}

export function boardRows(doc: Pick<ExplainDoc, "kind" | "players">, market: Market): BoardRow[] {
  const skills = fieldSkills(doc);
  return doc.players.filter((p) => !p.withdrawn).map((p) => {
    const m = p.probs?.[market];
    const model = toPts(m?.model), book = toPts(m?.market);
    return { id: p.id, name: playerDisplayName(p.name), country: p.country, model, book, gap: model !== null && book !== null ? Math.round((model - book) * 100) / 100 : null, skill: skills.get(p.id) ?? null, why: p.why ?? "", player: p };
  });
}

export function sortBoard(rows: BoardRow[], sort: BoardSort): BoardRow[] {
  const desc = (f: (r: BoardRow) => number | null) => [...rows].sort((a, b) => (f(b) ?? -Infinity) - (f(a) ?? -Infinity) || a.name.localeCompare(b.name));
  if (sort === "name") return [...rows].sort((a, b) => a.name.localeCompare(b.name));
  if (sort === "gap") return desc((r) => (r.gap === null ? null : Math.abs(r.gap)));
  if (sort === "book") return desc((r) => r.book);
  if (sort === "skill") return desc((r) => r.skill);
  return desc((r) => r.model);
}

/** Half-width of the gap rail for a market: the 90th percentile of |gap|, so most rows use the bar's length and one outlier does not flatten the rest. */
export function railRange(rows: BoardRow[]): number {
  const g = rows.map((r) => (r.gap === null ? null : Math.abs(r.gap))).filter(fin).sort((a, b) => a - b);
  if (!g.length) return 1;
  const p90 = g[Math.min(g.length - 1, Math.floor(g.length * 0.9))];
  return Math.max(0.5, Math.ceil(p90 * 2) / 2);
}

/** The largest disagreements: `above` = we are higher than the book, `below` = lower. Books under `minBook` points are skipped (long-shot noise). */
export function biggestGaps(rows: BoardRow[], n = 3, minBook = 0.5): { above: BoardRow[]; below: BoardRow[] } {
  const ok = rows.filter((r) => r.gap !== null && r.book !== null && r.book >= minBook);
  return {
    above: ok.filter((r) => (r.gap as number) > 0).sort((a, b) => (b.gap as number) - (a.gap as number)).slice(0, n),
    below: ok.filter((r) => (r.gap as number) < 0).sort((a, b) => (a.gap as number) - (b.gap as number)).slice(0, n),
  };
}

export function searchBoard(rows: BoardRow[], q: string): BoardRow[] {
  const t = q.trim().toLowerCase();
  return t ? rows.filter((r) => r.name.toLowerCase().includes(t) || (r.country ?? "").toLowerCase().includes(t)) : rows;
}
