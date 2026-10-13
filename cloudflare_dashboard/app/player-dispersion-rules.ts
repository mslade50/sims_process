/**
 * Shot dispersion ("driving range") rules: parsing of the published per-player payload, selection of rounds / yardage bands,
 * per-band width statistics and the PGA comparison. Pure functions only; the component draws what these return.
 *
 * Payload (golfprice/player_dispersion.py): integers are half-yards. Lateral is right-positive. Tee shots are measured against the
 * estimated fairway center line (tee to the median landing spot of the field's fairway-finding drives on that hole and day);
 * approach shots against the straight line from the ball to the pin. One-character lies: F fairway, R rough, S sand, G green,
 * T tee, X other, ? unknown (rounds before the 2024 lie convention).
 *
 * v2 ("situational"): tee lateral is measured from the field's dogleg-aware landing line, and each shot may carry one-character
 * situation codes (tee d dogleg L/S/R/x, w water/OB, b bunkers, t trees: 0 none, 1 left only, 2 right only, 3 both; approach p pin zone
 * 0..8 = left/center/right + 3 * front/middle/back, h hex bits of water/OB near the green: 1 left, 2 right, 4 short, 8 behind; ? unknown).
 * The reference names the buckets (dimension -> codes) and holds the PGA mean and spread for each, overall and per yardage band.
 * Everything situational is ESTIMATED from published shot coordinates; there is no official course geometry.
 */
export type Kind = "tee" | "app";
export type DispersionIndex = { schema_version: string; generation: string; prefix: string; reference_key: string; as_of: string; count: number; player_ids: number[] };
export type TeeBlock = { r: number[]; f: number[]; l: number[]; z: string; d?: string; w?: string; b?: string; t?: string };
export type AppBlock = { r: number[]; s: number[]; l: number[]; a: number[]; z: string; o: string; p?: string; h?: string };
export type DispersionPayload = { schema_version: string; dg_id: number; as_of: string; rounds: Array<[string, number, string, string?]>; /* [date, round, event name, deep-history event id] */ tee?: TeeBlock; app?: AppBlock };
export type Spread = { n: number; mean: number; sd: number | null; q10: number; q90: number; player_sd?: { n_players: number; p10: number; p50: number; p90: number } };
export type RefBand = { n_shots: number; n_players: number; lat: Spread; fwd?: Spread; along?: Spread; green_rate?: number; fairway_rate?: number; end_lie_known_n?: number;
  prox_med?: number; fairway_lat?: { n: number; q02: number; q98: number } };
export type SitSpread = { mean: number; sd: number | null; q10: number; q90: number; player_sd_p50?: number; player_sd_n?: number };
export type SitStat = { n: number; n_players: number; lat: SitSpread; fwd?: SitSpread; along?: SitSpread };
export type SituationBucket = { key: string; label: string; codes: string[] };
export type SituationDim = { key: string; label: string; field: string; buckets: SituationBucket[] };
export type SituationRef = { dims: SituationDim[]; stats: Record<string, Record<string, { all: SitStat; bands: Record<string, SitStat> }>> };
export type Situations = { tee: SituationRef; app: SituationRef; min_shots: { tee: number; app: number }; method?: string };
export type Reference = {
  schema_version: string; as_of: string; max_rounds: number; n_players: number; min_player_shots: number;
  tee_bands: number[][]; app_bands: number[][]; tee: Record<string, RefBand>; app: Record<string, Record<string, RefBand>>; situations?: Situations;
};
export type Point = { lat: number; y: number; dist: number; along: number; band: number; round: number; lie: string; start: string; c?: Record<string, string> };
export type BandStat = {
  band: number; n: number; meanLat: number; sdLat: number | null; q10: number; q90: number; width: number;
  meanY: number; meanAlong: number; rate: number | null; rateN: number;
  medLat: number; proxMed: number; alongQ10: number; alongQ90: number;
};
export type Verdict = { text: string; tone: "narrower" | "similar" | "wider" | "unknown"; ratio: number | null };

export const PAYLOAD_SCHEMAS = ["player_dispersion.v1", "player_dispersion.v2"];
export const ROUND_CHOICES = [20, 50, 100] as const;
export const MIN_BAND_SHOTS = 10;
export const MIN_VERDICT_SHOTS = 30;
export const LIE_OPTIONS: Array<{ key: string; label: string }> = [
  { key: "all", label: "Any lie" }, { key: "F", label: "Fairway" }, { key: "R", label: "Rough" }, { key: "T", label: "Tee (par 3s)" }, { key: "S", label: "Sand" },
];
export const DEFAULT_TEE_BANDS = [[0, 260], [260, 290], [290, 310], [310, 330], [330, 9999]];
export const DEFAULT_APP_BANDS = [[50, 100], [100, 150], [150, 200], [200, 301]];

const isInts = (v: unknown, n?: number): v is number[] => Array.isArray(v) && (n === undefined || v.length === n) && v.every((x) => Number.isInteger(x));
const sameLength = (...vals: Array<ArrayLike<unknown>>) => vals.every((v) => v.length === vals[0].length);

export function parseIndex(raw: unknown): DispersionIndex | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as DispersionIndex;
  if (d.schema_version !== "player_dispersion.index.v1" || typeof d.prefix !== "string" || typeof d.reference_key !== "string") return null;
  if (!d.prefix.startsWith("golfprice/player_profiles/dispersion-") || d.prefix.includes("..") || !d.reference_key.startsWith(d.prefix)) return null;
  return isInts(d.player_ids) ? d : null;
}

export function dispersionKey(index: DispersionIndex, dgId: number | null | undefined): string | null {
  if (!Number.isSafeInteger(dgId) || !index.player_ids.includes(dgId as number)) return null;
  return `${index.prefix}${dgId}.json`;
}

export function parsePayload(raw: unknown, dgId: number): DispersionPayload | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as DispersionPayload;
  if (!PAYLOAD_SCHEMAS.includes(d.schema_version) || d.dg_id !== dgId || !Array.isArray(d.rounds)) return null;
  const t = d.tee, a = d.app;
  if (t && !(isInts(t.r) && isInts(t.f, t.r.length) && isInts(t.l, t.r.length) && typeof t.z === "string" && t.z.length === t.r.length)) return null;
  if (a && !(isInts(a.r) && isInts(a.s, a.r.length) && isInts(a.l, a.r.length) && isInts(a.a, a.r.length) && typeof a.z === "string" && typeof a.o === "string" && sameLength(a.z, a.r) && sameLength(a.o, a.r))) return null;
  const codesOk = (blk: object | undefined, fields: string[], n: number) => !blk || fields.every((f) => { const v = (blk as Record<string, unknown>)[f]; return v === undefined || (typeof v === "string" && v.length === n); });
  if (!codesOk(t, ["d", "w", "b", "t"], t?.r.length ?? 0) || !codesOk(a, ["p", "h"], a?.r.length ?? 0)) return null;
  return t || a ? d : null;
}

export function parseReference(raw: unknown): Reference | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Reference;
  return PAYLOAD_SCHEMAS.includes(d.schema_version) && d.tee && d.app && Array.isArray(d.tee_bands) && Array.isArray(d.app_bands) ? d : null;
}

export function bandLabel(kind: Kind, bands: number[][], i: number): string {
  const b = bands[i];
  if (!b) return "";
  if (kind === "tee") return i === 0 ? `Under ${b[1]}` : b[1] >= 999 ? `${b[0]}+` : `${b[0]}–${b[1]}`;
  return b[1] > 300 ? `${b[0]}+` : `${b[0]}–${b[1]}`;
}

export function laterality(v: number, digits = 0): string {
  const a = Math.abs(v);
  if (a < 0.5 * 10 ** -digits) return "on the line";
  return `${a.toFixed(digits)} yd ${v > 0 ? "right" : "left"}`;
}

/** Rounds actually available for the type (ordered newest first), and the date span of the first `rounds` of them. */
export function roundCount(p: DispersionPayload, kind: Kind): number {
  const block = kind === "tee" ? p.tee : p.app;
  if (!block || !block.r.length) return 0;
  return new Set(block.r).size;
}

/** Spans over the payload's newest `rounds` rounds (payload round index 0 = newest). */
export function windowSpan(p: DispersionPayload, rounds: number): { first: string; last: string; n: number } | null {
  const n = Math.min(rounds, p.rounds.length);
  if (!n) return null;
  return { first: p.rounds[n - 1][0], last: p.rounds[0][0], n };
}

/** `event id|round` keys of a sliced history (same keys FilteredPlayerShapes uses); structural type so this file stays import-light. */
export function sliceKeys(events: Array<{ id: string; round_data: Array<{ round?: number; round_num?: number }> }>): Set<string> {
  return new Set(events.flatMap((e) => e.round_data.map((r) => `${e.id}|${r.round ?? r.round_num}`)));
}

/** Payload round indexes (0 = newest) whose history event id + round number are in the slice. */
export function sliceRounds(p: DispersionPayload, keys: Set<string>): Set<number> {
  const out = new Set<number>();
  p.rounds.forEach((r, i) => { if (r[3] && keys.has(`${r[3]}|${r[1]}`)) out.add(i); });
  return out;
}

/** True when the payload was published with history event ids (older publishes have only the name). */
export const hasEventIds = (p: DispersionPayload) => p.rounds.some((r) => !!r[3]);

/** Rounds of the slice that have shots of this type, and their date span. */
export function sliceSpan(p: DispersionPayload, kind: Kind, allowed: Set<number>): { n: number; first: string; last: string } | null {
  const block = kind === "tee" ? p.tee : p.app;
  if (!block) return null;
  const have = [...new Set(block.r)].filter((i) => allowed.has(i));
  if (!have.length) return null;
  const dates = have.map((i) => p.rounds[i][0]).sort();
  return { n: have.length, first: dates[0], last: dates[dates.length - 1] };
}

type SliceFilters = { window?: string; season?: string; tour?: string; major?: boolean; event?: string; course?: string; rounds?: number[]; contention?: string; difficulty?: string; strength?: string; search?: string };

/** The slice in plain words for the caption, e.g. "last 2 years · PGA · majors only · rounds 1, 2". */
export function sliceLabel(f: SliceFilters, eventName?: string): string {
  const out: string[] = [f.window === "recent2y" ? "last 2 years" : "all history"];
  if (f.season) out.push(`${f.season} season`);
  if (f.tour) out.push(f.tour.toUpperCase());
  if (f.major) out.push("majors only");
  if (f.event) out.push(eventName || "one event");
  if (f.course) out.push(f.course);
  if (f.rounds && f.rounds.length && f.rounds.length < 4) out.push(`round${f.rounds.length > 1 ? "s" : ""} ${[...f.rounds].sort().join(", ")}`);
  if (f.contention === "near") out.push("within 2 shots of the lead");
  if (f.contention === "outside") out.push("more than 2 shots back");
  if (f.difficulty) out.push(`${f.difficulty} scoring conditions`);
  if (f.strength) out.push(`${f.strength} fields`);
  if (f.search) out.push(`“${f.search}”`);
  return out.join(" · ");
}

/** `allowed` (payload round indexes) replaces the last-`rounds` cut when given: the history-filter slice. */
export function selectPoints(p: DispersionPayload, kind: Kind, rounds: number, bands: number[][], lie = "all", allowed?: Set<number>): Point[] {
  const out: Point[] = [];
  const skip = (r: number) => (allowed ? !allowed.has(r) : r >= rounds);
  if (kind === "tee" && p.tee) {
    const t = p.tee;
    for (let i = 0; i < t.r.length; i++) {
      if (skip(t.r[i])) continue;
      const dist = t.f[i] / 2, lat = t.l[i] / 2;
      const band = bandIndex(dist, bands);
      if (band < 0) continue;
      const pt: Point = { lat, y: dist, dist, along: 0, band, round: t.r[i], lie: t.z[i], start: "T" };
      if (t.d || t.w || t.b || t.t) pt.c = { d: t.d?.[i] ?? "?", w: t.w?.[i] ?? "?", b: t.b?.[i] ?? "?", t: t.t?.[i] ?? "?" };
      out.push(pt);
    }
  } else if (kind === "app" && p.app) {
    const a = p.app;
    for (let i = 0; i < a.r.length; i++) {
      if (skip(a.r[i])) continue;
      if (lie !== "all" && a.o[i] !== lie) continue;
      const dist = a.s[i] / 2, along = a.a[i] / 2;
      const band = bandIndex(dist, bands);
      if (band < 0) continue;
      const pt: Point = { lat: a.l[i] / 2, y: dist + along, dist, along, band, round: a.r[i], lie: a.z[i], start: a.o[i] };
      if (a.p || a.h) pt.c = { p: a.p?.[i] ?? "?", h: a.h?.[i] ?? "?" };
      out.push(pt);
    }
  }
  return out;
}

export function bandIndex(value: number, bands: number[][]): number {
  for (let i = 0; i < bands.length; i++) if (value >= bands[i][0] && value < bands[i][1]) return i;
  return -1;
}

export function quantile(sorted: number[], p: number): number {
  if (!sorted.length) return NaN;
  const x = (sorted.length - 1) * p, i = Math.floor(x), f = x - i;
  return sorted[i] + (sorted[Math.min(i + 1, sorted.length - 1)] - sorted[i]) * f;
}

export function bandStats(points: Point[], kind: Kind, bands: number[][]): BandStat[] {
  const out: BandStat[] = [];
  for (let b = 0; b < bands.length; b++) {
    const pts = points.filter((p) => p.band === b);
    if (!pts.length) continue;
    const lat = pts.map((p) => p.lat).sort((x, y) => x - y);
    const mean = lat.reduce((s, v) => s + v, 0) / lat.length;
    const sd = lat.length > 1 ? Math.sqrt(lat.reduce((s, v) => s + (v - mean) ** 2, 0) / (lat.length - 1)) : null;
    const known = pts.filter((p) => p.lie !== "?");
    const hit = known.filter((p) => p.lie === (kind === "tee" ? "F" : "G")).length;
    out.push({
      band: b, n: pts.length, meanLat: mean, sdLat: sd, q10: quantile(lat, 0.1), q90: quantile(lat, 0.9), width: quantile(lat, 0.9) - quantile(lat, 0.1),
      medLat: quantile(lat, 0.5), proxMed: quantile(pts.map((p) => Math.hypot(p.lat, p.along)).sort((x, y) => x - y), 0.5),
      alongQ10: quantile(pts.map((p) => p.along).sort((x, y) => x - y), 0.1), alongQ90: quantile(pts.map((p) => p.along).sort((x, y) => x - y), 0.9),
      meanY: pts.reduce((s, p) => s + p.y, 0) / pts.length, meanAlong: pts.reduce((s, p) => s + p.along, 0) / pts.length,
      rate: known.length >= MIN_BAND_SHOTS ? hit / known.length : null, rateN: known.length,
    });
  }
  return out;
}

export function referenceBand(ref: Reference | null, kind: Kind, band: number, lie = "all"): RefBand | null {
  if (!ref) return null;
  const r = kind === "tee" ? ref.tee[String(band)] : ref.app[lie]?.[String(band)];
  return r ?? null;
}

/** Typical PGA player's side-to-side spread (SD) for the band: the median player where published, else all shots pooled. */
export function tourSd(ref: RefBand | null): number | null {
  if (!ref) return null;
  return ref.lat.player_sd?.p50 ?? ref.lat.sd ?? null;
}

export function compare(stat: BandStat, ref: RefBand | null): Verdict {
  const t = tourSd(ref);
  if (stat.n < MIN_VERDICT_SHOTS || stat.sdLat === null || t === null || t <= 0) return { text: stat.n < MIN_VERDICT_SHOTS ? "Too few shots to compare" : "No PGA comparison", tone: "unknown", ratio: null };
  const ratio = stat.sdLat / t, pct = Math.round(Math.abs(ratio - 1) * 100);
  if (pct < 5) return { text: "About typical for the PGA Tour", tone: "similar", ratio };
  return ratio < 1 ? { text: `${pct}% tighter than the typical PGA player`, tone: "narrower", ratio } : { text: `${pct}% wider than the typical PGA player`, tone: "wider", ratio };
}

/** Plain-English headline for the selected view. */
export function headline(kind: Kind, stats: BandStat[], bands: number[][], ref: Reference | null, lie = "all"): string {
  const usable = stats.filter((s) => s.n >= MIN_VERDICT_SHOTS && s.sdLat !== null);
  if (!usable.length) return "Not enough shots in this view to draw a conclusion.";
  let w = 0, tw = 0, n = 0;
  for (const s of usable) {
    const t = tourSd(referenceBand(ref, kind, s.band, lie));
    if (t === null) continue;
    w += (s.sdLat as number) * s.n; tw += t * s.n; n += s.n;
  }
  const what = kind === "tee" ? "tee shots" : "approach shots";
  if (!n || tw <= 0) return `${n || usable.reduce((a, s) => a + s.n, 0)} ${what} shown.`;
  const ratio = w / tw, pct = Math.round(Math.abs(ratio - 1) * 100);
  const tail = kind === "tee" ? "the middle of the fairway" : "the line to the pin";
  if (pct < 5) return `Side to side, these ${what} spread about as widely around ${tail} as the typical PGA player's (${n.toLocaleString("en-US")} shots).`;
  return `Side to side, these ${what} spread ${pct}% ${ratio < 1 ? "tighter" : "wider"} around ${tail} than the typical PGA player's, at the same distances (${n.toLocaleString("en-US")} shots).`;
}

export function niceStep(span: number, target = 5): number {
  const raw = span / target, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
}

export function ticks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return out;
}

/** View window for the chart: forward range and symmetric lateral half-width, from the shots in view (and the PGA bars). */
export function viewWindow(points: Point[], kind: Kind): { yLo: number; yHi: number; xMax: number; outside: number } {
  const lats = points.map((p) => Math.abs(p.lat)).sort((a, b) => a - b);
  const ys = points.map((p) => p.y).sort((a, b) => a - b);
  const xRaw = lats.length ? quantile(lats, 0.985) : 30;
  const xMax = Math.min(kind === "tee" ? 70 : 60, Math.max(kind === "tee" ? 40 : 20, Math.ceil((xRaw + 2) / 10) * 10));
  const lo = ys.length ? quantile(ys, 0.01) : kind === "tee" ? 220 : 40, hi = ys.length ? quantile(ys, 0.995) : kind === "tee" ? 340 : 300;
  const step = kind === "tee" ? 10 : 25;
  const yLo = Math.max(0, Math.floor((lo - 5) / step) * step), yHi = Math.ceil((hi + 5) / step) * step;
  const outside = points.filter((p) => Math.abs(p.lat) > xMax || p.y < yLo || p.y > yHi).length;
  return { yLo, yHi: Math.max(yHi, yLo + step), xMax, outside };
}

export function fmt(v: number | null | undefined, digits = 1): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

/** Distance in plain units: feet when under 10 yards, else yards. */
export function fmtDist(yd: number | null | undefined): string {
  if (typeof yd !== "number" || !Number.isFinite(yd)) return "—";
  return Math.abs(yd) < 10 ? `${Math.round(Math.abs(yd) * 3)} ft` : `${Math.round(Math.abs(yd))} yd`;
}

/** Proximity is quoted in feet, as golf does. */
export function fmtFt(yd: number | null | undefined): string {
  return typeof yd === "number" && Number.isFinite(yd) ? `${Math.round(Math.abs(yd) * 3)} ft` : "—";
}

/** "9 yd left to 8 yd right" from a left(-)/right(+) pair. */
export function sideRange(lo: number, hi: number): string {
  const one = (v: number) => (Math.abs(v) < 0.5 ? "the line" : `${fmtDist(v)} ${v < 0 ? "left" : "right"}`);
  return `${one(lo)} to ${one(hi)}`;
}

/** "12 yd short to 6 yd long" from a short(-)/long(+) pair. */
export function depthRange(lo: number, hi: number): string {
  const one = (v: number) => (Math.abs(v) < 0.5 ? "the pin" : `${fmtDist(v)} ${v < 0 ? "short" : "long"}`);
  return `${one(lo)} to ${one(hi)}`;
}

export const GREEN_RINGS = [5, 10, 20, 30, 45, 60];
export const GREEN_HALF_WIDTH = 11, GREEN_HALF_DEPTH = 15; // stylised green, yards (about 22 x 30 yd)

/** Green-view scale: half-width (yards) of the square view, from the shots in view, and the rings that fit. */
export function greenWindow(points: Point[]): { radius: number; rings: number[]; outside: number } {
  const d = points.map((p) => Math.max(Math.abs(p.lat), Math.abs(p.along))).sort((a, b) => a - b);
  const raw = d.length ? quantile(d, 0.97) : 20;
  const radius = [15, 20, 30, 40, 60, 90].find((r) => raw <= r) ?? 90;
  return { radius, rings: GREEN_RINGS.filter((r) => r < radius), outside: points.filter((p) => Math.abs(p.lat) > radius || Math.abs(p.along) > radius).length };
}

/** Fairway-width guide for the tee view: one strip over the whole distance range, at the median of the field's per-band fairway extents. */
export function fairwayGuide(ref: Reference | null, bands: number[][], yLo: number, yHi: number): Array<{ y0: number; y1: number; left: number; right: number }> {
  if (!ref) return [];
  const lefts: number[] = [], rights: number[] = [];
  bands.forEach((_, i) => {
    const f = ref.tee[String(i)]?.fairway_lat;
    if (f) { lefts.push(f.q02); rights.push(f.q98); }
  });
  if (!lefts.length || yHi <= yLo) return [];
  return [{ y0: yLo, y1: yHi, left: quantile(lefts.sort((a, b) => a - b), 0.5), right: quantile(rights.sort((a, b) => a - b), 0.5) }];
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Situations (v2): one dimension at a time, never crossed with the yardage bands
// ---------------------------------------------------------------------------------------------------------------------------------
export const SIT_MIN_SHOTS = { tee: 60, app: 100 } as const; // width is shown only from this many shots
export const SIT_MIN_MEAN = 15; // below this the average itself is not reported

/** How each dimension is detected, in plain words. The data are estimates built from shot coordinates, not official course maps. */
export const SIT_HELP: Record<string, string> = {
  dog: "Dogleg: the angle between the line from the tee to where the field's drives land and the line from there to the pin. 20 degrees or more is a dogleg; under 10 is straight; 10 to 20 (slight bends) are left out. Estimated from shot coordinates, not an official course map.",
  wat: "Water or out of bounds on a side: at least 1% of the field's drives on that hole have finished in water or out of bounds on that side of the landing line. Estimated from where shots ended, not from a course map.",
  bun: "Bunkers on a side: at least 5% of the field's drives on that hole finish in a bunker on that side. Bunkers sit where the field misses, so this describes the hole, not a golfer steering away from them.",
  tre: "Trees on a side: at least 2% of drives finish in trees on that side. The data provider only labels trees from 2024, so holes without 2024 or later rounds are left out.",
  pin: "Pin side: the green's outline is estimated from where shots ended and putts began on that hole across tournaments, and each day's pin is placed in its left, center or right third as the golfer faces the green. The outline is an estimate, not a surveyed green.",
  dep: "Pin depth: the front, middle or back third of the same estimated green (front is the side the golfer approaches from).",
  wtr: "Water or out of bounds near the green: at least 0.7% of approaches on that hole finish in water or out of bounds within 30 yards of the estimated green, on that side (short, left, right or behind).",
};

const PHRASES: Record<string, Record<string, string>> = {
  dog: { L: "left doglegs", S: "straight holes", R: "right doglegs" },
  wat: { "1": "holes with water or out of bounds on the left", "2": "holes with water or out of bounds on the right", "3": "holes with water or out of bounds on both sides", "0": "holes without water or out of bounds" },
  bun: { "1": "holes with bunkers on the left", "2": "holes with bunkers on the right", "3": "holes with bunkers on both sides", "0": "holes without bunkers off the tee" },
  tre: { "1": "holes with trees on the left", "2": "holes with trees on the right", "3": "holes with trees on both sides", "0": "holes without trees" },
  pin: { L: "the pin in the left third of the green", C: "the pin in the middle third of the green", R: "the pin in the right third of the green" },
  dep: { F: "the pin in the front third of the green", M: "the pin in the middle third of the green", B: "the pin in the back third of the green" },
  wtr: { N: "no water or out of bounds near the green", A: "water or out of bounds near the green", L: "water or out of bounds left of the green", R: "water or out of bounds right of the green", S: "water or out of bounds short of the green", B: "water or out of bounds behind the green" },
};

export function situationRef(ref: Reference | null, kind: Kind): SituationRef | null {
  const s = ref?.situations?.[kind];
  return s && Array.isArray(s.dims) && s.stats ? s : null;
}

/** Dimensions worth offering: the payload carries the code and the reference knows at least one bucket for it. */
export function availableDims(ref: Reference | null, kind: Kind, payload: DispersionPayload | null): SituationDim[] {
  const sr = situationRef(ref, kind), block = payload ? (kind === "tee" ? payload.tee : payload.app) : undefined;
  if (!sr || !block) return [];
  return sr.dims.filter((d) => (block as unknown as Record<string, unknown>)[d.field] !== undefined && Object.keys(sr.stats[d.key] ?? {}).length > 0);
}

export function inBucket(p: Point, dim: SituationDim, bucket: SituationBucket): boolean {
  const c = p.c?.[dim.field];
  return c !== undefined && bucket.codes.includes(c);
}

export function bucketCounts(points: Point[], dim: SituationDim): Record<string, number> {
  const out: Record<string, number> = {};
  for (const b of dim.buckets) out[b.key] = 0;
  for (const p of points) for (const b of dim.buckets) if (inBucket(p, dim, b)) out[b.key]++;
  return out;
}

export type SitResult = {
  n: number; gate: number; enough: boolean; showMean: boolean; meanLat: number; meanAlong: number; meanDist: number;
  pgaLat: number | null; pgaAlong: number | null; width: number; pgaWidth: number | null; sd: number | null; pgaSd: number | null;
};

const wavg = (pairs: Array<[number, number | null | undefined]>): number | null => {
  let w = 0, s = 0;
  for (const [n, v] of pairs) if (typeof v === "number" && Number.isFinite(v)) { w += n; s += n * v; }
  return w > 0 ? s / w : null;
};

/** Averages and spread of the shots in one bucket, with the PGA figures matched to this golfer's own yardage mix (band by band). */
export function situationStat(points: Point[], kind: Kind, dim: SituationDim, bucket: SituationBucket, ref: Reference | null): SitResult | null {
  const sr = situationRef(ref, kind);
  const pts = points.filter((p) => inBucket(p, dim, bucket));
  if (!pts.length) return null;
  const lat = pts.map((p) => p.lat).sort((a, b) => a - b);
  const n = pts.length, mean = lat.reduce((s, v) => s + v, 0) / n;
  const sd = n > 1 ? Math.sqrt(lat.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : null;
  const cell = sr?.stats[dim.key]?.[bucket.key];
  const perBand = new Map<number, number>();
  for (const p of pts) perBand.set(p.band, (perBand.get(p.band) ?? 0) + 1);
  const bands = [...perBand.entries()];
  const val = (f: (s: SitStat) => number | null | undefined) => {
    if (!cell) return null;
    return wavg(bands.map(([b, c]): [number, number | null | undefined] => { const st = cell.bands[String(b)] ?? cell.all; return [c, st ? f(st) : null]; }));
  };
  const gate = SIT_MIN_SHOTS[kind];
  return {
    n, gate, enough: n >= gate, showMean: n >= SIT_MIN_MEAN, meanLat: mean, meanAlong: pts.reduce((s, p) => s + p.along, 0) / n, meanDist: pts.reduce((s, p) => s + p.y, 0) / n,
    pgaLat: val((s) => s.lat.mean), pgaAlong: kind === "app" ? val((s) => s.along?.mean) : null,
    width: quantile(lat, 0.9) - quantile(lat, 0.1), pgaWidth: val((s) => s.lat.q90 - s.lat.q10),
    sd, pgaSd: val((s) => s.lat.player_sd_p50 ?? s.lat.sd),
  };
}

const yd = (v: number) => `${Math.round(Math.abs(v))} yd`;

/** One plain-English line for the chosen situation, and a separate width clause (shown only once there are enough shots). */
export function situationReadout(kind: Kind, who: string, dim: SituationDim, bucket: SituationBucket, r: SitResult): { text: string; width: string } {
  const what = PHRASES[dim.key]?.[bucket.key] ?? bucket.label.toLowerCase();
  if (!r.showMean) return { text: `Only ${r.n} ${kind === "tee" ? "drives" : "approaches"} in this situation, too few to say anything.`, width: "" };
  const alongFocus = kind === "app" && dim.key === "dep";
  const val = alongFocus ? r.meanAlong : r.meanLat, pga = alongFocus ? r.pgaAlong : r.pgaLat;
  const neg = alongFocus ? "short" : "left", pos = alongFocus ? "long" : "right";
  const target = kind === "tee" ? "the field's landing line" : "the pin";
  const where = Math.round(Math.abs(val)) < 1 ? `right on ${target}` : `${yd(val)} ${val < 0 ? neg : pos} of ${target}`;
  let diff = "";
  if (pga !== null) {
    const shift = val - pga;
    diff = Math.round(Math.abs(shift)) < 1 ? ", about the same as a typical PGA player" : `, ${yd(shift)} further ${shift < 0 ? neg : pos} than a typical PGA player`;
  }
  const lead = kind === "tee" ? `On ${what}, ` : `With ${what}, `;
  const text = `${lead}${who} finishes ${where}${diff}.`;
  const width = r.enough
    ? `Middle 80% is ${Math.round(r.width)} yd wide${r.pgaWidth !== null ? ` (PGA ${Math.round(r.pgaWidth)} yd)` : ""}, from ${r.n} shots.`
    : `Width: not enough shots (${r.n}; needs ${r.gate}).`;
  return { text, width };
}

/** "McIlroy, Rory" -> "Rory McIlroy"; falls back to a neutral phrase. */
export function displayName(name: string | null | undefined): string {
  const n = (name ?? "").trim();
  if (!n) return "this golfer";
  const m = n.split(",");
  return m.length === 2 && m[0].trim() && m[1].trim() ? `${m[1].trim()} ${m[0].trim()}` : n;
}
