// Pure rules for the /players?layout=next hero, SG bars, filter summary bar and method text (site audit B1/B2/A3-A7).
// Only type imports from other modules: this file is loaded directly by node tests (strip-types, no resolver).
import type {PgaBenchmark} from "./player-profile-rules";
import type {PerformancePoint} from "./player-performance-rules";

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export const signed = (v: unknown, digits = 2): string => finite(v) ? `${v > 0 ? "+" : ""}${v.toFixed(digits)}` : "n/a";

/** B2 caveat rows 1 and 2: one line each, directly under the headline. */
export const CAVEAT_REFERENCE_ONLY = "Reference only; prices use the event model";
export const CAVEAT_DESCRIPTIVE = "Descriptive, not a forecast";
/** B2 row 8: the label carries the zero point. */
export const ZERO_LABEL = "vs 2025 PGA avg round";
/** B2 row 9: one clause beside the interval. */
export const INTERVAL_CLAUSE = "excludes future-round variance and anchor uncertainty";
/** B2 row 10 / A6. */
export const CROSS_TOUR_BAND = "cross-tour band ±0.2";
export const NO_CATEGORY_DATA = "n/a (no category data on this tour)";
export const RADAR_LABEL = "within-field style, PGA+LIV data only";
export const WEEKEND_ROUNDS_NOTE = "includes weekend rounds (selected by the cut)";

/** A catalog may carry one stored "typical PGA regular" tick (A3). Never computed client side. */
export type RegularTick = {value?: number; unshrunk?: number; as_of?: string; label?: string};
export function regularTick(catalog: unknown): {value: number; asOf: string | null} | null {
  const t = (catalog as {regular_tick?: RegularTick | number} | null | undefined)?.regular_tick;
  if (finite(t)) return {value: t, asOf: null};
  if (!t || typeof t !== "object") return null;
  const value = finite(t.value) ? t.value : finite(t.unshrunk) ? t.unshrunk : null;
  return value === null ? null : {value, asOf: typeof t.as_of === "string" ? t.as_of : null};
}

export type SupportBadge = {tone: "neutral" | "warning" | "negative"; text: string};
/** A: n badge. Amber under 30 rounds; "n<12 rounds" when the benchmark is unsupported. */
export function supportBadge(rating: Pick<PgaBenchmark, "n_rounds" | "status"> | undefined, minRounds = 12): SupportBadge {
  const n = rating?.n_rounds ?? 0;
  if (!rating || rating.status !== "available" || n < minRounds) return {tone: "negative", text: `n<${minRounds} rounds`};
  if (n < 30) return {tone: "warning", text: `n=${n} rounds · thin sample`};
  return {tone: "neutral", text: `n=${n} rounds`};
}

/** A6: tour-mix badge only when the primary tour differs from the comparison event's tour or the PGA share is under half. */
export function tourMixBadge(weights: Array<{tour: string; weight_fraction: number}> | undefined, eventTour?: string | null): string | null {
  const w = (weights ?? []).filter(t => finite(t.weight_fraction));
  if (!w.length) return null;
  const primary = [...w].sort((a, b) => b.weight_fraction - a.weight_fraction)[0];
  const pgaShare = w.filter(t => t.tour.toLowerCase() === "pga").reduce((s, t) => s + t.weight_fraction, 0);
  const differs = !!eventTour && primary.tour.toLowerCase() !== eventTour.toLowerCase();
  if (!differs && pgaShare >= 0.5) return null;
  return `mostly ${primary.tour.toUpperCase()} rounds (${Math.round(primary.weight_fraction * 100)}%) · ${CROSS_TOUR_BAND}`;
}

/** Horizontal bar geometry on a symmetric domain; zero sits in the middle. Values clamp to the domain. */
export function barGeometry(value: number | null, tick: number | null, floor = 2): {lo: number; hi: number; zero: number; value: number | null; tick: number | null; left: number; width: number} {
  const reach = Math.max(floor, Math.ceil(Math.max(Math.abs(finite(value) ? value : 0), Math.abs(finite(tick) ? tick : 0))));
  const lo = -reach, hi = reach, pos = (v: number) => (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * 100;
  const v = finite(value) ? pos(value) : null;
  return {lo, hi, zero: pos(0), value: v, tick: finite(tick) ? pos(tick) : null, left: v === null ? 50 : Math.min(50, v), width: v === null ? 0 : Math.abs(v - 50)};
}

/** A4: the benchmark and the SG bars are R1-R2 by default; any other round set is called out. */
export function roundSetBadge(rounds: number[]): string | null {
  const set = [...new Set(rounds)].sort();
  if (set.length === 2 && set[0] === 1 && set[1] === 2) return null;
  return set.some(r => r >= 3) ? WEEKEND_ROUNDS_NOTE : null;
}
function roundsLabel(rounds: number[]): string {
  const r = [...new Set(rounds)].sort((a, b) => a - b);
  if (!r.length) return "no rounds";
  if (r.length === 1) return `R${r[0]}`;
  return r[r.length - 1] - r[0] === r.length - 1 ? `R${r[0]}-R${r[r.length - 1]}` : r.map(n => `R${n}`).join(", ");
}
type FilterLike = {window?: string; season: string; tour: string; major: boolean; difficulty: string; strength: string; search: string; rounds: number[]; contention: string; event: string; course: string};
/** Summary bar text: "R1-R2 · last 2 years · no situational filter". */
export function describeFilters(f: FilterLike): {text: string; situational: string[]; roundBadge: string | null} {
  const situational = [
    f.contention !== "all" ? (f.contention === "near" ? "in contention (reconstructed)" : "outside contention (reconstructed)") : "",
    f.major ? "majors only" : "", f.difficulty ? `${f.difficulty} scoring` : "", f.strength ? `${f.strength} field` : "",
    f.event ? "one event" : "", f.course ? "one course" : "", f.search ? `search "${f.search}"` : "",
  ].filter(Boolean);
  const when = f.season ? `${f.season} season` : f.window === "all" ? "all published history" : "last 2 years";
  const parts = [roundsLabel(f.rounds), when, ...(f.tour ? [`${f.tour.toUpperCase()} only`] : []), situational.length ? situational.join(", ") : "no situational filter"];
  return {text: parts.join(" · "), situational, roundBadge: roundSetBadge(f.rounds)};
}

/** A7: live-checkpoint chips from the stored per-player live block. Null unless the transient latent exists. */
export type LiveBlock = {tier?: number | null; mu_live?: number | null; week_latent_mean?: number | null; contention?: number | null} | null | undefined;
/** A7: the stored week latent is centred over every entrant who played the completed rounds, not over the players still active. After a cut the active-field mean is above zero (survivors), so the chips are not centred on the field still playing. */
export const LIVE_CENTRING_NOTE = "centred on all entrants, not the active field";
export function liveChips(live: LiveBlock): {transient: number; priced: number | null; contentionIncluded: boolean} | null {
  if (!live || !finite(live.week_latent_mean)) return null;
  // Tier 0 is active; 1 (MDF), 2 (missed cut) and 3/4 (WD) play no further round, so "next round" chips would mislead.
  if (finite(live.tier) && live.tier !== 0) return null;
  const hasMu = finite(live.mu_live), hasC = finite(live.contention);
  return {transient: live.week_latent_mean, priced: hasMu ? live.mu_live! + live.week_latent_mean + (hasC ? live.contention! : 0) : null, contentionIncluded: hasC};
}

/** Trend teaser: tiny SMA20/SMA50 paths over the visible R1-R2 series. */
export type TeaserPoint = Pick<PerformancePoint, "time" | "sma20" | "sma50">;
export function sparklinePaths(points: TeaserPoint[], width = 180, height = 44): {sma20: string; sma50: string; zero: number | null; n: number} | null {
  const vals = points.flatMap(p => [p.sma20, p.sma50]).filter(finite);
  if (points.length < 2 || !vals.length) return null;
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = Math.max(0.1, hi - lo);
  const t0 = points[0].time, t1 = points[points.length - 1].time;
  const x = (t: number) => (t - t0) / Math.max(1, t1 - t0) * width, y = (v: number) => height - (v - lo) / span * height;
  const path = (key: "sma20" | "sma50") => points.filter(p => finite(p[key])).map((p, i) => `${i ? "L" : "M"}${x(p.time).toFixed(1)},${y(p[key]!).toFixed(1)}`).join(" ");
  return {sma20: path("sma20"), sma50: path("sma50"), zero: lo < 0 && hi > 0 ? y(0) : null, n: points.length};
}

/** ROW 2: category bars. Rows come from the filteredSkillProfile radar metrics. */
export type BarMetric = {key: string; label: string; value: number | null; z: number | null; n: number | null};
export type BarRow = {key: string; label: string; value: number | null; z: number | null; n: number; status: "ok" | "thin" | "none"; left: number; width: number; zero: number};
export function sgBarRows(metrics: BarMetric[], minRounds = 3): {rows: BarRow[]; available: boolean} {
  const rows = metrics.map(m => {
    const n = finite(m.n) ? m.n : 0, ok = n >= minRounds && finite(m.z);
    const g = barGeometry(ok ? m.z : null, null, 3);
    return {key: m.key, label: m.label, value: finite(m.value) ? m.value : null, z: ok ? m.z : null, n, status: ok ? "ok" as const : n > 0 ? "thin" as const : "none" as const, left: g.left, width: g.width, zero: g.zero};
  });
  return {rows, available: rows.some(r => r.status === "ok")};
}
export const NO_CATEGORY_REFERENCE = "n/a (category reference unavailable)";
export const NO_CATEGORY_ROUNDS = (minRounds: number) => `n/a (fewer than ${minRounds} category rounds in this window)`;
/** Name the real cause of an empty bar block: tour coverage first, then a missing reference, then too few rounds. */
export function noCategoryReason(tours: string[] | undefined, ctx?: {hasReference?: boolean; deepLoaded?: boolean; minRounds?: number}): string {
  const list = (tours ?? []).map(t => t.toLowerCase());
  const covered = list.some(t => t === "pga" || t === "liv");
  if (list.length > 0 && !covered) return `${NO_CATEGORY_DATA}: category strokes gained are published for PGA and LIV rounds only; this player's rounds are ${list.map(t => t.toUpperCase()).join(", ")}`;
  if (!ctx) return list.length ? `${NO_CATEGORY_DATA}: category strokes gained are published for PGA and LIV rounds only; this player's rounds are ${list.map(t => t.toUpperCase()).join(", ")}` : NO_CATEGORY_DATA;
  if (ctx.deepLoaded === false) return "n/a (deep history has not loaded)";
  if (ctx.hasReference === false) return `${NO_CATEGORY_REFERENCE}: the fixed PGA category reference was not published, so category rounds cannot be scored`;
  return NO_CATEGORY_ROUNDS(ctx.minRounds ?? 3);
}

/** B5: next-action wording for the /players error and absent states. */
export const CATALOG_UNAVAILABLE_HINT = "The catalog request failed or returned nothing. Try again; if it persists, the profile publish may not have run (check the Run page).";
