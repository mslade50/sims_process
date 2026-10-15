/**
 * Presentation logic for the "Why we differ" section of the player drawer (DivergeView.tsx). The numbers and sentences are built by golfprice/explain_export.py
 * (explain version 11); this file only picks and orders what to show. Pure functions, no runtime APIs: shared with tests/explain.test.mjs.
 */
import type { DivergeCompare, EdgeSplit, ExPlayer, ExplainDoc, FormWindow, Market, Outlier, PeerSet, SourceChances } from "./explain-rules";
import { FORM_WINDOWS, MARKETS, MARKET_LABEL } from "./explain-rules.ts";

export const SOURCE_LABEL: Record<"ours" | "dg" | "book", string> = { ours: "Ours", dg: "Data Golf", book: "Sportsbooks" };
export const SOURCE_KEYS = ["ours", "dg", "book"] as const;

export type ChanceRow = { market: Market; label: string; values: SourceChances };
/** The chance table: only bet types where at least two sources have a number (a single number has nothing to be compared with). */
export function chanceRows(c: DivergeCompare | null | undefined): ChanceRow[] {
  if (!c) return [];
  return MARKETS.flatMap((m) => {
    const v = c.markets[m];
    if (!v) return [];
    const have = SOURCE_KEYS.filter((k) => typeof v[k] === "number").length;
    return have >= 2 ? [{ market: m, label: MARKET_LABEL[m], values: v }] : [];
  });
}
/** The column to mark as the odd one out; only when the verdict names one of the three (all_agree / all_differ / missing sources name nobody). */
export function oddColumn(c: DivergeCompare | null | undefined): Outlier | null {
  if (!c || (c.verdict !== "ours_outlier" && c.verdict !== "dg_outlier" && c.verdict !== "book_outlier")) return null;
  return c.outlier;
}

export type SplitSegment = { key: "reference" | "explained" | "unexplained"; label: string; value: number };
export type SplitView = { edge: number; segments: SplitSegment[]; scale: number; unexplainedShare: number | null; referenceLabel: string };
/** Three numbers that add up to the gap: what players priced like this one get, our parts, and what no part of our model explains. */
export function splitView(es: EdgeSplit | null | undefined): SplitView | null {
  if (!es) return null;
  const referenceLabel = es.reference === "peers" ? "players priced like them" : "field average";
  const segments: SplitSegment[] = [];
  if (es.reference_edge !== null) segments.push({ key: "reference", label: referenceLabel, value: es.reference_edge });
  if (es.explained !== null) segments.push({ key: "explained", label: "our parts", value: es.explained });
  if (es.unexplained !== null) segments.push({ key: "unexplained", label: "not explained by our model", value: es.unexplained });
  if (!segments.length) return null;
  // Share of what is particular to this player (the gap beyond the reference) that no part of our model explains; defined when both pieces push the same way.
  const ex = es.explained ?? 0, un = es.unexplained ?? 0;
  const share = es.unexplained !== null && ex * un >= 0 && Math.abs(ex + un) > 1e-9 ? Math.abs(un) / Math.abs(ex + un) : null;
  return { edge: es.edge, segments, scale: Math.max(0.1, Math.abs(es.edge), ...segments.map((s) => Math.abs(s.value))), unexplainedShare: share, referenceLabel };
}

export type PeerPartRow = { key: string; label: string; value: number | null; peer: number | null; diff: number | null; standout: boolean };
/** Shared vs distinctive parts: rows keep the exported order; the ones that set the player apart (difference of at least 0.05 strokes, or the biggest) are flagged. */
export function peerPartRows(p: PeerSet | null | undefined): PeerPartRow[] {
  if (!p) return [];
  const big = Math.max(0, ...p.parts.map((x) => Math.abs(x.distinctive ?? 0)));
  return p.parts.map((x) => ({ key: x.key, label: x.label, value: x.value, peer: x.peer_median, diff: x.distinctive, standout: x.distinctive !== null && (Math.abs(x.distinctive) >= 0.1 || (Math.abs(x.distinctive) >= 0.05 && Math.abs(x.distinctive) >= big * 0.99)) }));
}

export type PeerLink = { id: number; name: string };
/** Peer buttons: names from the document's own player list when the id is there, the exported name otherwise. */
export function peerLinks(p: PeerSet | null | undefined, doc: Pick<ExplainDoc, "players">): PeerLink[] {
  if (!p) return [];
  const byId = new Map(doc.players.map((x) => [x.id, x] as const));
  return p.ids.flatMap((id, i) => { const name = byId.get(id)?.name ?? p.names[i]; return name ? [{ id, name }] : []; });
}

export type FormRow = { id: number; name: string; self: boolean; windows: Array<number | null>; seasons: Array<number | null>; ours: number | null; market: number | null };
export type FormTable = { windows: readonly FormWindow[]; years: number[]; rows: FormRow[] };
/** One table for this player and their price peers. Hidden (null) when nobody shown has form. Seasons: the last three years any shown player has. */
export function formTable(doc: Pick<ExplainDoc, "players">, player: ExPlayer, peerIds: number[], skillMarket: (p: ExPlayer) => number | null): FormTable | null {
  const byId = new Map(doc.players.map((x) => [x.id, x] as const));
  const shown = [player, ...peerIds.map((id) => byId.get(id)).filter((x): x is ExPlayer => !!x && x.id !== player.id)];
  if (!shown.some((p) => p.form)) return null;
  const years = [...new Set(shown.flatMap((p) => p.form?.seasons.map((s) => s.year) ?? []))].sort((a, b) => a - b).slice(-3);
  const rows = shown.map((p) => ({ id: p.id, name: p.name, self: p.id === player.id, windows: FORM_WINDOWS.map((w) => p.form?.windows[w]?.mean ?? null),
    seasons: years.map((y) => p.form?.seasons.find((s) => s.year === y)?.mean ?? null), ours: p.mu_rel, market: skillMarket(p) }));
  return { windows: FORM_WINDOWS, years, rows };
}

/** The sentence-level headline of the section: the first exported sentence (the chance comparison). */
export const leadSentence = (sentences: string[]): string | null => sentences[0] ?? null;
