import type { ExplainDoc, ExPlayer, Market } from "./explain-rules";

export type Checkpoint = { run: string; kind: "week" | "live"; as_of: string | null; after_round: number | null; key: string; validated: boolean };
export type DistributionEvent = { event_uid: string; name: string; tour: string; date_start: string; explain_key?: string; distribution_runs?: Checkpoint[] };
export type JointPrices = { method: string; tie_rule: string; rows: Array<[number, number, number, number | null]> };
export type DistributionDoc = ExplainDoc & { probability_basis?: string; head_to_heads?: JointPrices };

export function checkpoints(event: DistributionEvent): Checkpoint[] {
  return [...(event.distribution_runs ?? [])].filter((r) => r.validated && (r.kind === "week" || (r.after_round ?? 0) >= 1))
    .sort((a, b) => (a.as_of ?? "").localeCompare(b.as_of ?? "") || Number(a.kind === "live") - Number(b.kind === "live"));
}
export function runLabel(run: Checkpoint): string {
  const date = run.as_of ? new Date(run.as_of) : null;
  const stamp = date && Number.isFinite(date.getTime()) ? date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : run.run;
  return `${run.kind === "week" ? "Pre-event" : `After R${run.after_round}`} · ${stamp}`;
}
export function matchup(doc: DistributionDoc | null, a: number, b: number): { p: number; tie: number | null } | null {
  if (!doc || a === b || doc.head_to_heads?.method !== "joint_simulation") return null;
  const row = doc.head_to_heads.rows.find((r) => (r[0] === a && r[1] === b) || (r[0] === b && r[1] === a));
  if (!row || !Number.isFinite(row[2]) || row[2] < 0 || row[2] > 1) return null;
  return { p: row[0] === a ? row[2] : 1 - row[2], tie: row[3] };
}
export function marker(player: ExPlayer, k: number): number | null {
  const market = ({ 1: "win", 5: "top_5", 10: "top_10", 20: "top_20" } as Record<number, Market>)[k];
  if (market) return player.probs[market].model;
  return player.finish?.pos?.length ? player.finish.pos.slice(0, k).reduce((a, b) => a + b, 0) : null;
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
