/** Strict, presentation-ready helpers for golfprice's scoring expectation block. */
export const SCORING_SCHEMA = "golfprice.scoring.v1";
export type PmfRow = [number, number];
export type ScoreComponent = { key: string; label: string; strokes: number; note: string | null };
export type ScoringPlayer = {
  id: number; name: string; mean: number; sd: number; p10: number; p50: number; p90: number;
  pmf: PmfRow[]; n_draws: number; components: ScoreComponent[]; data_depth: number | null; tee_time: string | null;
};
export type ScoringDoc = {
  schema: string; status: "available" | "unavailable"; round: number | null; run_id: string; probability_basis: string;
  method: string | null; n_draws: number;
  field: { mean: number | null; sd: number | null; p10: number | null; p50: number | null; p90: number | null; players: number; active_players_min: number | null; active_players_max: number | null; outcome_label: string; quantile_curve: Array<[number, number]> | null };
  baseline: { course_score: number | null; common_weather: number | null; source: string | null; detail: string | null; median_hole_observations: number | null; round_par: number | null; arithmetic_residual: number | null };
  players: ScoringPlayer[];
  confidence: { mean_interval: unknown; status: string; reasons: string[]; monte_carlo_se: { field: number | null; basis: string } | null };
  drivers: Array<{ key: string; label: string; mean_score_shift: number }>;
  source_notes: string[]; weather: { source: string | null; freshness: string | null; status: string | null; applied_in_prices: boolean | null; label: string | null; relative_offsets_status: string | null; relative_offsets_reason: string | null };
};
const obj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown): v is string => typeof v === "string";
const nullableNumber = (v: unknown): v is number | null => v === null || finite(v);

/** Unavailable is a first-class exporter result; malformed available payloads fail closed. */
export function parseScoring(raw: unknown): ScoringDoc | null {
  if (!obj(raw) || raw.schema !== SCORING_SCHEMA || !["available", "unavailable"].includes(String(raw.status)) || !Array.isArray(raw.players) || !obj(raw.field) || !obj(raw.baseline) || !obj(raw.confidence)) return null;
  const field = raw.field as Record<string, unknown>, baseline = raw.baseline as Record<string, unknown>, confidence = raw.confidence as Record<string, unknown>;
  if (!text(raw.run_id) || !text(raw.probability_basis) || !finite(raw.n_draws) || !nullableNumber(raw.round) || !(raw.method === null || text(raw.method))) return null;
  const summaryKeys = ["mean", "sd", "p10", "p50", "p90"];
  if (!summaryKeys.every((k) => nullableNumber(field[k])) || !finite(field.players) || !text(field.outcome_label)) return null;
  if (["active_players_min", "active_players_max"].some((k) => Object.hasOwn(field, k) && !nullableNumber(field[k]))) return null;
  const baselineKeys = ["course_score", "common_weather", "median_hole_observations", "round_par", "arithmetic_residual"];
  if (!baselineKeys.every((k) => nullableNumber(baseline[k]))) return null;
  if (!text(confidence.status) || !Array.isArray(confidence.reasons)) return null;
  const rawMc = confidence.monte_carlo_se;
  let monteCarlo: ScoringDoc["confidence"]["monte_carlo_se"] = null;
  if (rawMc !== null && rawMc !== undefined) {
    if (!obj(rawMc) || !nullableNumber(rawMc.field) || !text(rawMc.basis)) return null;
    monteCarlo = { field: rawMc.field, basis: rawMc.basis };
  }
  if (raw.status === "unavailable") {
    if (raw.players.length !== 0 || raw.round !== null || raw.method !== null || !summaryKeys.every((k) => field[k] === null)) return null;
    return { schema: SCORING_SCHEMA, status: "unavailable", round: null, run_id: raw.run_id, probability_basis: raw.probability_basis, method: null, n_draws: raw.n_draws,
      field: { mean: null, sd: null, p10: null, p50: null, p90: null, players: field.players as number, active_players_min: null, active_players_max: null, outcome_label: field.outcome_label, quantile_curve: null },
      baseline: { course_score: baseline.course_score as number | null, common_weather: baseline.common_weather as number | null, source: text(baseline.source) ? baseline.source : null, detail: text(baseline.detail) ? baseline.detail : null, median_hole_observations: baseline.median_hole_observations as number | null, round_par: baseline.round_par as number | null, arithmetic_residual: baseline.arithmetic_residual as number | null },
      players: [], confidence: { mean_interval: confidence.mean_interval ?? null, status: confidence.status, reasons: confidence.reasons.filter(text), monte_carlo_se: monteCarlo }, drivers: [],
      source_notes: Array.isArray(raw.source_notes) ? raw.source_notes.filter(text) : [], weather: parseWeather(raw.weather) };
  }
  if (!finite(raw.round) || !Number.isInteger(raw.round) || !text(raw.method) || !Number.isInteger(raw.n_draws) || raw.n_draws <= 0 || !summaryKeys.every((k) => finite(field[k])) || !Number.isInteger(field.players) || field.players <= 0 || raw.players.length !== field.players || raw.players.length <= 0 || monteCarlo === null || !Array.isArray(raw.drivers) || !Array.isArray(raw.source_notes) || !obj(raw.weather)) return null;
  const quantileCurve: Array<[number, number]> | null = field.quantile_curve === undefined || field.quantile_curve === null ? null : Array.isArray(field.quantile_curve) ? field.quantile_curve.flatMap((row): Array<[number, number]> => Array.isArray(row) && row.length === 2 && finite(row[0]) && row[0] >= 0 && row[0] <= 1 && finite(row[1]) ? [[row[0], row[1]]] : []) : null;
  if (field.quantile_curve !== undefined && field.quantile_curve !== null && (!Array.isArray(field.quantile_curve) || quantileCurve?.length !== field.quantile_curve.length || !quantileCurve?.length || quantileCurve.some((p, i) => i > 0 && p[0] < quantileCurve[i - 1][0]))) return null;
  const players: ScoringPlayer[] = [];
  for (const candidate of raw.players) {
    if (!obj(candidate) || !finite(candidate.id) || !Number.isInteger(candidate.id) || !text(candidate.name) || ![candidate.mean, candidate.sd, candidate.p10, candidate.p50, candidate.p90, candidate.n_draws].every(finite) || !Number.isInteger(candidate.n_draws) || !nullableNumber(candidate.data_depth) || !(candidate.tee_time === null || text(candidate.tee_time)) || !Array.isArray(candidate.components) || !Array.isArray(candidate.pmf)) return null;
    const pmf: PmfRow[] = [];
    for (const row of candidate.pmf) {
      if (!Array.isArray(row) || row.length !== 2 || !finite(row[0]) || !Number.isInteger(row[0]) || !finite(row[1]) || row[1] < 0) return null;
      pmf.push([row[0], row[1]]);
    }
    if (!pmf.length || Math.abs(pmf.reduce((s, [, p]) => s + p, 0) - 1) > 1e-6) return null;
    const components = Array.isArray(candidate.components) ? candidate.components.filter((c): c is Record<string, unknown> => obj(c)).flatMap((c) => finite(c.strokes) && text(c.label) && text(c.key) ? [{ key: c.key, label: c.label, strokes: c.strokes, note: text(c.note) ? c.note : null }] : []) : [];
    if (Array.isArray(candidate.components) && components.length !== candidate.components.length) return null;
    players.push({ id: candidate.id, name: candidate.name, mean: candidate.mean as number, sd: candidate.sd as number, p10: candidate.p10 as number, p50: candidate.p50 as number, p90: candidate.p90 as number, pmf, n_draws: candidate.n_draws as number, components, data_depth: candidate.data_depth as number | null, tee_time: candidate.tee_time as string | null });
  }
  const drivers = Array.isArray(raw.drivers) ? raw.drivers.filter((d): d is Record<string, unknown> => obj(d)).map((d) => {
    if (!text(d.key) || !text(d.label) || !finite(d.mean_score_shift)) return null;
    return { key: d.key, label: d.label, mean_score_shift: d.mean_score_shift };
  }) : [];
  if (drivers.some((d) => d === null)) return null;
  return { schema: SCORING_SCHEMA, status: "available", round: raw.round, run_id: raw.run_id, probability_basis: raw.probability_basis, method: raw.method, n_draws: raw.n_draws,
    field: { mean: field.mean as number, sd: field.sd as number, p10: field.p10 as number, p50: field.p50 as number, p90: field.p90 as number, players: field.players as number, active_players_min: nullableNumber(field.active_players_min) ? field.active_players_min : null, active_players_max: nullableNumber(field.active_players_max) ? field.active_players_max : null, outcome_label: field.outcome_label, quantile_curve: quantileCurve },
    baseline: { course_score: baseline.course_score as number | null, common_weather: baseline.common_weather as number | null, source: text(baseline.source) ? baseline.source : null, detail: text(baseline.detail) ? baseline.detail : null, median_hole_observations: baseline.median_hole_observations as number | null, round_par: baseline.round_par as number | null, arithmetic_residual: baseline.arithmetic_residual as number | null },
    players, confidence: { mean_interval: confidence.mean_interval ?? null, status: confidence.status, reasons: confidence.reasons.filter(text), monte_carlo_se: monteCarlo }, drivers: drivers as ScoringDoc["drivers"],
    source_notes: Array.isArray(raw.source_notes) ? raw.source_notes.filter(text) : [], weather: parseWeather(raw.weather) };
}
function parseWeather(raw: unknown): ScoringDoc["weather"] {
  if (!obj(raw)) return { source: null, freshness: null, status: null, applied_in_prices: null, label: null, relative_offsets_status: null, relative_offsets_reason: null };
  return { source: text(raw.source) ? raw.source : null, freshness: text(raw.freshness) ? raw.freshness : null, status: text(raw.status) ? raw.status : null, applied_in_prices: typeof raw.applied_in_prices === "boolean" ? raw.applied_in_prices : null, label: text(raw.label) ? raw.label : null, relative_offsets_status: text(raw.relative_offsets_status) ? raw.relative_offsets_status : null, relative_offsets_reason: text(raw.relative_offsets_reason) ? raw.relative_offsets_reason : null };
}
export function prettyName(name: string): string { return name.includes(",") ? name.split(",").reverse().join(" ").trim() : name; }
export function normalizePmf(rows: PmfRow[]): PmfRow[] { const merged = new Map<number, number>(); for (const [score, p] of rows) if (Number.isInteger(score) && finite(p) && p >= 0) merged.set(score, (merged.get(score) ?? 0) + p); const total = [...merged.values()].reduce((a, b) => a + b, 0); return total > 0 ? [...merged].sort(([a], [b]) => a - b).map(([s, p]) => [s, p / total]) : []; }
export function overUnder(rows: PmfRow[], line: number): { over: number; under: number; push: number } { if (!finite(line)) return { over: 0, under: 0, push: 0 }; return normalizePmf(rows).reduce((a, [score, p]) => { if (score > line) a.over += p; else if (score < line) a.under += p; else a.push += p; return a; }, { over: 0, under: 0, push: 0 }); }
/** EV per unit stake. Exact integer lines retain push probability at zero profit/loss. */
export function betEv(p: { over: number; under: number; push: number }, side: "over" | "under", decimalOdds: number): number | null { if (!finite(decimalOdds) || decimalOdds <= 1) return null; const win = p[side], lose = p[side === "over" ? "under" : "over"]; return win * (decimalOdds - 1) - lose; }
export function breakEvenDecimal(p: { over: number; under: number; push: number }, side: "over" | "under"): number | null { const win = p[side], lose = p[side === "over" ? "under" : "over"]; return win > 0 ? 1 + lose / win : null; }
export function decimalToAmerican(decimal: number | null): string { if (decimal == null || !finite(decimal) || decimal <= 1) return "—"; return decimal >= 2 ? `+${Math.round((decimal - 1) * 100)}` : `−${Math.round(100 / (decimal - 1))}`; }
/** Shift translates PMF linearly between adjacent integer scores; it never reruns the model. */
export function shiftedOverUnder(rows: PmfRow[], line: number, shift: number): { over: number; under: number; push: number } { if (!finite(shift)) return { over: 0, under: 0, push: 0 }; const pmf = normalizePmf(rows), lo = Math.floor(shift), frac = shift - lo, moved: PmfRow[] = []; for (const [score, p] of pmf) { if (frac < 1e-9) moved.push([score + lo, p]); else moved.push([score + lo, p * (1 - frac)], [score + lo + 1, p * frac]); } return overUnder(moved, line); }
export function marketNoVig(overDecimal: number, underDecimal: number): { over: number; under: number } | null { if (![overDecimal, underDecimal].every(finite) || overDecimal <= 1 || underDecimal <= 1) return null; const a = 1 / overDecimal, b = 1 / underDecimal, total = a + b; return { over: a / total, under: b / total }; }
export function curveRows(players: Array<{ id: number; pmf: PmfRow[] }>, cumulative: boolean): Array<Record<string, number>> {
  const all = players.flatMap((p) => p.pmf.map(([s]) => s)); if (!all.length) return [];
  const lo = Math.min(...all), hi = Math.max(...all), cumulativeById = new Map<number, number>();
  return Array.from({ length: hi - lo + 1 }, (_, i) => { const score = lo + i, row: Record<string, number> = { score };
    for (const p of players) { const mass = normalizePmf(p.pmf).find(([s]) => s === score)?.[1] ?? 0; const next = (cumulativeById.get(p.id) ?? 0) + mass; cumulativeById.set(p.id, next); row[`p_${p.id}`] = 100 * (cumulative ? next : mass); }
    return row;
  });
}
/** Solve the scenario shift at which a side's unit EV crosses zero. */
export function evBreakEvenShift(rows: PmfRow[], line: number, side: "over" | "under", odds: number, extent = 5, step = 0.01): number | null {
  if (!rows.length || !finite(extent) || extent <= 0 || !finite(step) || step <= 0 || betEv(overUnder(rows, line), side, odds) === null) return null;
  const target = (shift: number) => betEv(shiftedOverUnder(rows, line, shift), side, odds) ?? Number.NaN;
  let prevShift = -extent, prev = target(prevShift); if (prev === 0) return prevShift;
  for (let s = -extent + step; s <= extent + 1e-9; s += step) { const value = target(s); if (Number.isFinite(value) && ((value <= 0 && prev >= 0) || (value >= 0 && prev <= 0))) { const fraction = prev === value ? 0 : prev / (prev - value); return prevShift + fraction * (s - prevShift); } prev = value; prevShift = s; }
  return null;
}
