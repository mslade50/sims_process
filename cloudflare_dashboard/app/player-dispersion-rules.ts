/**
 * Shot dispersion ("driving range") rules: parsing of the published per-player payload, selection of rounds / yardage bands,
 * per-band width statistics and the PGA comparison. Pure functions only; the component draws what these return.
 *
 * Payload (golfprice/player_dispersion.py): integers are half-yards. Lateral is right-positive. Tee shots are measured against the
 * estimated fairway centre line (tee to the median landing spot of the field's fairway-finding drives on that hole and day);
 * approach shots against the straight line from the ball to the pin. One-character lies: F fairway, R rough, S sand, G green,
 * T tee, X other, ? unknown (rounds before the 2024 lie convention).
 */
export type Kind = "tee" | "app";
export type DispersionIndex = { schema_version: string; generation: string; prefix: string; reference_key: string; as_of: string; count: number; player_ids: number[] };
export type TeeBlock = { r: number[]; f: number[]; l: number[]; z: string };
export type AppBlock = { r: number[]; s: number[]; l: number[]; a: number[]; z: string; o: string };
export type DispersionPayload = { schema_version: string; dg_id: number; as_of: string; rounds: Array<[string, number, string]>; tee?: TeeBlock; app?: AppBlock };
export type Spread = { n: number; mean: number; sd: number | null; q10: number; q90: number; player_sd?: { n_players: number; p10: number; p50: number; p90: number } };
export type RefBand = { n_shots: number; n_players: number; lat: Spread; fwd?: Spread; along?: Spread; green_rate?: number; fairway_rate?: number; end_lie_known_n?: number;
  prox_med?: number; fairway_lat?: { n: number; q02: number; q98: number } };
export type Reference = {
  schema_version: string; as_of: string; max_rounds: number; n_players: number; min_player_shots: number;
  tee_bands: number[][]; app_bands: number[][]; tee: Record<string, RefBand>; app: Record<string, Record<string, RefBand>>;
};
export type Point = { lat: number; y: number; dist: number; along: number; band: number; round: number; lie: string; start: string };
export type BandStat = {
  band: number; n: number; meanLat: number; sdLat: number | null; q10: number; q90: number; width: number;
  meanY: number; meanAlong: number; rate: number | null; rateN: number;
  medLat: number; proxMed: number; alongQ10: number; alongQ90: number;
};
export type Verdict = { text: string; tone: "narrower" | "similar" | "wider" | "unknown"; ratio: number | null };

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
  if (d.schema_version !== "player_dispersion.v1" || d.dg_id !== dgId || !Array.isArray(d.rounds)) return null;
  const t = d.tee, a = d.app;
  if (t && !(isInts(t.r) && isInts(t.f, t.r.length) && isInts(t.l, t.r.length) && typeof t.z === "string" && t.z.length === t.r.length)) return null;
  if (a && !(isInts(a.r) && isInts(a.s, a.r.length) && isInts(a.l, a.r.length) && isInts(a.a, a.r.length) && typeof a.z === "string" && typeof a.o === "string" && sameLength(a.z, a.r) && sameLength(a.o, a.r))) return null;
  return t || a ? d : null;
}

export function parseReference(raw: unknown): Reference | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Reference;
  return d.schema_version === "player_dispersion.v1" && d.tee && d.app && Array.isArray(d.tee_bands) && Array.isArray(d.app_bands) ? d : null;
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

export function selectPoints(p: DispersionPayload, kind: Kind, rounds: number, bands: number[][], lie = "all"): Point[] {
  const out: Point[] = [];
  if (kind === "tee" && p.tee) {
    const t = p.tee;
    for (let i = 0; i < t.r.length; i++) {
      if (t.r[i] >= rounds) continue;
      const dist = t.f[i] / 2, lat = t.l[i] / 2;
      const band = bandIndex(dist, bands);
      if (band < 0) continue;
      out.push({ lat, y: dist, dist, along: 0, band, round: t.r[i], lie: t.z[i], start: "T" });
    }
  } else if (kind === "app" && p.app) {
    const a = p.app;
    for (let i = 0; i < a.r.length; i++) {
      if (a.r[i] >= rounds) continue;
      if (lie !== "all" && a.o[i] !== lie) continue;
      const dist = a.s[i] / 2, along = a.a[i] / 2;
      const band = bandIndex(dist, bands);
      if (band < 0) continue;
      out.push({ lat: a.l[i] / 2, y: dist + along, dist, along, band, round: a.r[i], lie: a.z[i], start: a.o[i] });
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
