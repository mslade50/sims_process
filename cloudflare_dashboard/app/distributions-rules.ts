import type { ExplainDoc, ExPlayer, Market } from "./explain-rules";
import { etTime } from "./lib.ts";

export type Checkpoint = { run: string; kind: "week" | "live"; as_of: string | null; after_round: number | null; key: string; validated: boolean };
export type DistributionEvent = { event_uid: string; name: string; tour: string; date_start: string; explain_key?: string; distribution_runs?: Checkpoint[] };
export type JointPrices = { method: string; tie_rule: string; rows: Array<[number, number, number, number | null]> };
export type DistributionDoc = ExplainDoc & { probability_basis?: string; head_to_heads?: JointPrices };

export function checkpoints(event: DistributionEvent): Checkpoint[] {
  return [...(event.distribution_runs ?? [])].filter((r) => r.validated && (r.kind === "week" || (r.after_round ?? 0) >= 1))
    .sort((a, b) => (a.as_of ?? "").localeCompare(b.as_of ?? "") || Number(a.kind === "live") - Number(b.kind === "live"));
}
export function runLabel(run: Checkpoint): string {
  const stamp = etTime(run.as_of, "time unknown");
  return `${run.kind === "week" ? "Before the event" : `After round ${run.after_round}`} · ${stamp}`;
}
export function matchup(doc: DistributionDoc | null, a: number, b: number): { p: number; tie: number | null } | null {
  if (!doc || a === b || doc.head_to_heads?.method !== "joint_simulation") return null;
  const row = doc.head_to_heads.rows.find((r) => (r[0] === a && r[1] === b) || (r[0] === b && r[1] === a));
  if (!row || !Number.isFinite(row[2]) || row[2] < 0 || row[2] > 1) return null;
  return { p: row[0] === a ? row[2] : 1 - row[2], tie: row[3] };
}
/** The finish arrays (finish.pos) are settlement rank: players who miss the cut are ranked below the field. */
export const SETTLEMENT_RANK_LABEL = "finish position, with ties shared and missed cuts ranked below the field";
/** One plain sentence under the finish chart. */
export const FINISH_CHART_CAPTION = "Each bar is the chance the player finishes in that range of finish positions (ties shared), out of 100%; hover a bar for the exact figure.";
/** Marker label: the cut line is "Make cut" (the priced make-cut probability), not an ordinary top-N. */
export function markerLabel(k: number, cutTopN: number | null | undefined): string {
  return cutTopN && k === cutTopN ? `Make cut (top ${k} and ties)` : `Top ${k}`;
}
/** Marker probability. The cut line uses the priced p_make_cut (ties at the cut make it, so top-N slicing would understate it); top 1/5/10/20 use their model prices. */
export function marker(player: ExPlayer, k: number, cutTopN?: number | null): number | null {
  if (cutTopN && k === cutTopN && !([1, 5, 10, 20].includes(k))) {
    const made = player.probs.make_cut?.model;
    if (typeof made === "number" && Number.isFinite(made)) return made;
  }
  const market = ({ 1: "win", 5: "top_5", 10: "top_10", 20: "top_20" } as Record<number, Market>)[k];
  if (market) return player.probs[market].model;
  return player.finish?.pos?.length ? player.finish.pos.slice(0, k).reduce((a, b) => a + b, 0) : null;
}
/** Fixed cut reference (independent of the user's finish marker): the cut position and each player's priced make-cut probability. Null when the event has no cut or no price. */
export function cutReference(player: ExPlayer, cutTopN: number | null | undefined): { x: number; label: string; prob: number } | null {
  if (!cutTopN || cutTopN < 1) return null;
  const made = player.probs.make_cut?.model;
  if (typeof made !== "number" || !Number.isFinite(made)) return null;
  return { x: cutTopN, label: markerLabel(cutTopN, cutTopN), prob: made };
}
export function curveRows(players: ExPlayer[], previous: ExPlayer[], cumulative: boolean): Record<string, number>[] {
  const size = Math.max(0, ...players.map((p) => p.finish?.pos.length ?? 0), ...previous.map((p) => p.finish?.pos.length ?? 0));
  const sums: Record<string, number> = {};
  return Array.from({ length: size }, (_, i) => {
    const row: Record<string, number> = { position: i + 1 };
    for (const [prefix, group] of [["now", players], ["then", previous]] as const) for (const p of group) {
      const key = `${prefix}_${p.id}`;
      if (p.finish && i < p.finish.pos.length) {
        sums[key] = (sums[key] ?? 0) + p.finish.pos[i];
        row[key] = 100 * (cumulative ? sums[key] : p.finish.pos[i]);
      }
    }
    return row;
  });
}

/* ------------------------------------------------------------------ finish buckets (histogram) */
export type FinishBucket = { key: string; label: string; lo: number; hi: number; missed?: boolean };
const BUCKET_EDGES: Array<[number, number, string]> = [[1, 1, "Win"], [2, 5, "2-5"], [6, 10, "6-10"], [11, 20, "11-20"], [21, 30, "21-30"], [31, 50, "31-50"]];
/** Win, 2-5, 6-10, 11-20, 21-30, 31-50, 51+ and (events with a cut) Missed cut. Buckets beyond the field size are dropped. */
export function finishBuckets(fieldSize: number, hasCut: boolean): FinishBucket[] {
  const out: FinishBucket[] = [];
  for (const [lo, hi, label] of BUCKET_EDGES) {
    if (lo > fieldSize) break;
    out.push({ key: label, label, lo, hi: Math.min(hi, fieldSize) });
  }
  if (fieldSize > 50) out.push({ key: "51+", label: "51+", lo: 51, hi: fieldSize });
  if (hasCut) out.push({ key: "missed", label: "Missed cut", lo: fieldSize + 1, hi: fieldSize + 1, missed: true });
  return out;
}
/** Probability (0-1) of each bucket. Missed cuts are ranked below the field in `pos`, so they are removed from the last finishing bucket and reported on their own. */
export function bucketProbs(pos: number[], pMiss: number | null | undefined, buckets: FinishBucket[]): number[] {
  const miss = Number.isFinite(pMiss ?? NaN) ? Math.max(0, pMiss as number) : 0;
  const hasMissBucket = buckets.some((b) => b.missed);
  const out = buckets.map((b) => {
    if (b.missed) return miss;
    let s = 0;
    for (let k = b.lo; k <= Math.min(b.hi, pos.length); k++) s += pos[k - 1] ?? 0;
    return s;
  });
  if (hasMissBucket) {
    const last = buckets.map((b) => !b.missed).lastIndexOf(true);
    if (last >= 0) out[last] = Math.max(0, out[last] - miss);
  }
  return out;
}
/** Chart rows for the finish histogram, in percent. Cumulative mode leaves out the missed-cut bar and counts "this range or better". */
export function bucketRows(players: ExPlayer[], previous: ExPlayer[], cumulative: boolean, fieldSize: number, hasCut: boolean): Record<string, number | string>[] {
  const buckets = finishBuckets(fieldSize, hasCut).filter((b) => !(cumulative && b.missed));
  const series: Array<[string, ExPlayer]> = [...players.map((p) => ["now", p] as [string, ExPlayer]), ...previous.map((p) => ["then", p] as [string, ExPlayer])];
  const probs = new Map<string, number[]>();
  for (const [prefix, p] of series) if (p.finish?.pos?.length) {
    const raw = bucketProbs(p.finish.pos, p.finish.p_miss_cut, finishBuckets(fieldSize, hasCut));
    const kept = raw.filter((_, i) => !(cumulative && finishBuckets(fieldSize, hasCut)[i].missed));
    let c = 0;
    probs.set(`${prefix}_${p.id}`, kept.map((v) => 100 * (cumulative ? (c += v) : v)));
  }
  return buckets.map((b, i) => {
    const row: Record<string, number | string> = { label: b.label };
    for (const [k, v] of probs) row[k] = +v[i].toFixed(3);
    return row;
  });
}
