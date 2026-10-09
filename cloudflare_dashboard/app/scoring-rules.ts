/** Strict, presentation-ready helpers for golfprice's scoring expectation block. */
export const SCORING_SCHEMA = "golfprice.scoring.v1";
export type PmfRow = [number, number];
export type ScoreComponent = { key: string; label: string; strokes: number; note: string | null };
export type ScoringPlayer = {
  id: number; name: string; mean: number; sd: number; p10: number; p50: number; p90: number;
  pmf: PmfRow[]; n_draws: number; components: ScoreComponent[]; data_depth: number | null; tee_time: string | null;
};
export type CourseProvenance = {
  model_version: string | null;
  layout: { status: string | null; source: string | null; year: number | null; current_event_confirmed: boolean | null };
  general_fit: { start: string | null; end: string | null; cutoff: string | null; observations: number | null; method: string | null; skill_response_cutoff: string | null };
  venue_history: { observations: number | null; editions: string[]; cutoff: string | null; year_min: number | null; year_max: number | null };
  historical_weather: { reference: string | null; field_adjustment: string | null; status: string | null; method: string | null; note: string | null };
  field_strength: { status: string | null; detail: string | null; method: string | null };
};
export type RoundUpdate = {
  status: string | null; applied: boolean | null; target_round: number | null; source_rounds: number[];
  adjustment_strokes: number | null; observations: number | null; reason: string | null; method: string | null;
  observed_rounds: Array<{ round: number; field_actual: number | null; field_expected: number | null; weather_strokes: number | null; residual: number | null; players: number | null }>;
};
export type ScoringDoc = {
  schema: string; status: "available" | "unavailable"; round: number | null; run_id: string; probability_basis: string;
  method: string | null; n_draws: number;
  field: { mean: number | null; sd: number | null; p10: number | null; p50: number | null; p90: number | null; players: number; active_players_min: number | null; active_players_max: number | null; outcome_label: string; quantile_curve: Array<[number, number]> | null };
  baseline: { course_score: number | null; common_weather: number | null; source: string | null; detail: string | null; median_hole_observations: number | null; round_par: number | null; arithmetic_residual: number | null; course_provenance: CourseProvenance | null };
  round_update: RoundUpdate | null;
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
      baseline: { course_score: baseline.course_score as number | null, common_weather: baseline.common_weather as number | null, source: text(baseline.source) ? baseline.source : null, detail: text(baseline.detail) ? baseline.detail : null, median_hole_observations: baseline.median_hole_observations as number | null, round_par: baseline.round_par as number | null, arithmetic_residual: baseline.arithmetic_residual as number | null, course_provenance: parseCourseProvenance(baseline.course_provenance) },
      round_update: parseRoundUpdate(raw.round_update),
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
    baseline: { course_score: baseline.course_score as number | null, common_weather: baseline.common_weather as number | null, source: text(baseline.source) ? baseline.source : null, detail: text(baseline.detail) ? baseline.detail : null, median_hole_observations: baseline.median_hole_observations as number | null, round_par: baseline.round_par as number | null, arithmetic_residual: baseline.arithmetic_residual as number | null, course_provenance: parseCourseProvenance(baseline.course_provenance) },
      round_update: parseRoundUpdate(raw.round_update),
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

/** Field residual is a separate reconciliation term; player residual is already in its component list. */
export function expectationBridge(doc: ScoringDoc, player: ScoringPlayer | null = null): { components: ScoreComponent[]; residual: number | null; total: number | null } {
  const components = player ? player.components : doc.drivers.filter((d) => !["unexplained_residual", "engine_posterior_residual"].includes(d.key)).map((d) => ({ key: d.key, label: d.label, strokes: d.mean_score_shift, note: null }));
  const residual = player ? null : doc.baseline.arithmetic_residual;
  const { course_score: course, common_weather: weather } = doc.baseline;
  const total = course == null || weather == null || (!player && residual == null) ? null : course + weather + components.reduce((sum, c) => sum + c.strokes, 0) + (residual ?? 0);
  return { components, residual, total };
}

/** Two-way no-vig market shares are conditional on resolution, so exclude pushes on both sides. */
export function resolvedOverShare(p: { over: number; under: number; push: number }): number | null {
  const resolved = p.over + p.under;
  return finite(resolved) && resolved > 0 ? p.over / resolved : null;
}

// Additive metadata never makes an otherwise valid legacy score disappear.
const stringValue = (v: unknown): string | null => text(v) ? v : finite(v) ? String(v) : null;
const numberValue = (v: unknown): number | null => finite(v) ? v : null;
const record = (v: unknown): Record<string, unknown> => obj(v) ? v : {};
function parseCourseProvenance(raw: unknown): CourseProvenance | null {
  if (!obj(raw)) return null;
  const layout = record(raw.layout), fit = record(raw.general_fit), history = record(raw.venue_history), weather = record(raw.historical_weather), strength = record(raw.field_strength);
  return {
    model_version: stringValue(raw.model_version),
    layout: { status: stringValue(layout.status), source: stringValue(layout.source), year: numberValue(layout.year), current_event_confirmed: typeof layout.current_event_confirmed === "boolean" ? layout.current_event_confirmed : null },
    general_fit: { start: stringValue(fit.start), end: stringValue(fit.end), cutoff: stringValue(fit.cutoff), observations: numberValue(fit.observations), method: stringValue(fit.method), skill_response_cutoff: stringValue(fit.skill_response_cutoff) },
    venue_history: { observations: numberValue(history.observations), editions: Array.isArray(history.editions) ? history.editions.flatMap((e) => stringValue(e) === null ? [] : [stringValue(e)!]) : [], cutoff: stringValue(history.cutoff), year_min: numberValue(history.year_min), year_max: numberValue(history.year_max) },
    historical_weather: { reference: stringValue(weather.reference), field_adjustment: stringValue(weather.field_adjustment), status: stringValue(weather.status), method: stringValue(weather.method), note: stringValue(weather.note) },
    field_strength: { status: stringValue(strength.status), detail: stringValue(strength.detail), method: stringValue(strength.method) },
  };
}
function parseRoundUpdate(raw: unknown): RoundUpdate | null {
  if (!obj(raw)) return null;
  return { status: stringValue(raw.status), applied: typeof raw.applied === "boolean" ? raw.applied : null, target_round: numberValue(raw.target_round), source_rounds: Array.isArray(raw.source_rounds) ? raw.source_rounds.filter((r): r is number => finite(r) && Number.isInteger(r) && r >= 1 && r <= 4) : [], adjustment_strokes: numberValue(raw.adjustment_strokes), observations: numberValue(raw.observations), reason: stringValue(raw.reason), method: stringValue(raw.method),
    observed_rounds: Array.isArray(raw.observed_rounds) ? raw.observed_rounds.flatMap((r) => obj(r) && finite(r.round) ? [{ round: r.round, field_actual: numberValue(r.field_actual), field_expected: numberValue(r.field_expected), weather_strokes: numberValue(r.weather_strokes), residual: numberValue(r.residual), players: numberValue(r.players) }] : []) : [] };
}
export type ScoringEvidenceRow = { label: string; value: string; detail: string };
export type EvidenceContext = { courseName?: string | null; eventYear?: number | null; forecastTime?: string | null };

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2026-10-07" (or "2026-10-08 (strictly earlier events)") to "Oct 7, 2026"; anything else is returned unchanged. A calendar date has no time zone, so it is never run through Date. */
export function plainDate(value: string | null | undefined): string | null {
  if (!value) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  return m ? `${MONTHS[Number(m[2]) - 1] ?? m[2]} ${Number(m[3])}, ${m[1]}` : value;
}
/** 1581678 -> "1.58M", 3204 -> "3,204". */
export function compactCount(n: number): string { return n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n.toLocaleString("en-US"); }
/** "V_EU_campo_madrid_villa" -> "Campo Madrid Villa". Only a fallback for when the event has no course name. */
export function venueNameFromSource(source: string | null | undefined): string | null {
  const m = source ? /\b(V_[A-Za-z]{2,3}_[A-Za-z0-9_]+)/.exec(source) : null;
  if (!m) return null;
  const words = m[1].replace(/^V_[A-Za-z]{2,3}_/, "").split("_").filter(Boolean);
  return words.length ? words.map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ") : null;
}
const yearSpan = (a: number | string | null | undefined, b: number | string | null | undefined): string | null => {
  const x = a == null || a === "" ? null : String(a), y = b == null || b === "" ? null : String(b);
  return x && y ? (x === y ? x : `${x}–${y}`) : x ?? y;
};

/** Short, plain-English bullets describing where the course number comes from. Each row's `detail` is a hover definition. */
export function scoringEvidence(doc: ScoringDoc, ctx: EvidenceContext = {}): { rows: ScoringEvidenceRow[]; warning: string | null; updateApplied: boolean } {
  const p = doc.baseline.course_provenance, u = doc.round_update;
  const applied = u?.status === "applied" && u.applied === true && u.target_round === doc.round && u.adjustment_strokes !== null;
  const rows: ScoringEvidenceRow[] = [];
  const venue = ctx.courseName ?? venueNameFromSource(doc.baseline.source) ?? "This course";

  // Course and which hole layout the model used.
  let layoutText: string;
  if (p?.layout.current_event_confirmed === true) layoutText = "using this year's confirmed hole layout";
  else if (p?.layout.year != null) layoutText = `using ${ctx.eventYear != null && p.layout.year === ctx.eventYear - 1 ? "last year's" : `the ${p.layout.year}`} hole layout (this year's setup not confirmed)`;
  else layoutText = "hole layout not recorded for this older checkpoint";
  rows.push({ label: "Course", value: `${venue}, ${layoutText}.`, detail: "Which course and which version of its 18 holes the model scored. Hole lengths and pars come from this layout." });

  // Hole difficulty: own-venue history vs tour-wide model.
  const fit = p?.general_fit, hist = p?.venue_history;
  const fitBits = fit ? [fit.observations != null ? `${compactCount(fit.observations)} holes` : null, yearSpan(fit.start, fit.end)].filter(Boolean).join(", ") : "";
  const tourModel = `a tour-wide ${fit?.method && /par\s*[×x]\s*yardage/i.test(fit.method) ? "par/yardage " : ""}model${fitBits ? ` (${fitBits})` : ""}`;
  const venueObs = hist?.observations ?? (p ? null : doc.baseline.median_hole_observations);
  if (venueObs === 0) rows.push({ label: "Hole difficulty", value: `No past hole data at this venue, so hole difficulty comes from ${tourModel}.`, detail: "With no earlier rounds recorded on this course, each hole's expected score is estimated from similar par and yardage holes across the tour." });
  else if (venueObs != null && venueObs > 0) {
    const span = hist && yearSpan(hist.year_min, hist.year_max);
    rows.push({ label: "Hole difficulty", value: hist?.observations != null ? `Uses ${venueObs.toLocaleString("en-US")} past holes played here${span ? ` (${span})` : ""}, together with ${tourModel}.` : `Uses about ${venueObs.toLocaleString("en-US")} past plays per hole here, together with ${tourModel}.`, detail: "How hard each hole plays comes from the scores players actually made on this course, anchored by a tour-wide model of par and yardage." });
  } else rows.push({ label: "Hole difficulty", value: p ? `Hole difficulty comes from ${tourModel}; how much past data exists at this venue was not reported.` : "This older checkpoint does not record how much past hole data supported the course number.", detail: "Past rounds at this venue make the course number more reliable." });

  // Weather.
  const w = doc.baseline.common_weather;
  const time = ctx.forecastTime ? ` (forecast as of ${ctx.forecastTime})` : "";
  if (w == null) rows.push({ label: "Weather", value: "No weather adjustment was recorded for this round.", detail: "The shift in scoring the model expects from the forecast conditions, for every player alike." });
  else rows.push({ label: "Weather", value: Math.abs(w) < 0.005 ? `Conditions are forecast to play about normal${time}.` : `Conditions are forecast to play ${Math.abs(w).toFixed(2)} strokes ${w < 0 ? "easier" : "harder"} than normal${time}.`, detail: "How much the forecast wind, rain and temperature move the whole field's average score compared with a typical round at this venue." });

  // Completed rounds and whether they changed the course number.
  const done = u?.observed_rounds.filter((r) => r.field_actual != null) ?? [];
  if (done.length) rows.push({ label: "Completed rounds", value: `${done.map((r) => `Round ${r.round} field average ${r.field_actual!.toFixed(2)}${r.players != null ? ` (${r.players} players)` : ""}`).join("; ")}.`, detail: "The actual average score of the players who have finished that round." });
  if (u) {
    const t = u.target_round;
    const text = applied ? `Completed rounds moved the round ${t} course number by ${u.adjustment_strokes! >= 0 ? "+" : ""}${u.adjustment_strokes!.toFixed(2)} strokes.`
      : u.status === "staged_not_applied" ? `These results have not changed the round ${t ?? "next"} course number yet. Scores alone cannot tell a hard course from bad weather, so the adjustment waits on a validated method.`
      : `These results were not used to adjust the round ${t ?? "next"} course number.`;
    rows.push({ label: "Effect on next round", value: text, detail: "Whether what happened in earlier rounds was fed back into the course number for the next round." });
  }

  const warning = p?.venue_history.observations === 0 || doc.baseline.median_hole_observations === 0 ? "There is no past hole data for this venue, so the course number leans on a tour-wide model and the earlier layout. Treat it as less certain than at a course with history."
    : p && p.layout.current_event_confirmed !== true ? "This year's hole layout is not confirmed, so the course number uses an earlier layout. Check it before relying on the number."
    : !p ? "This older checkpoint does not record where its course number came from. The stored expectation has not been recalculated with a later release."
    : doc.confidence.reasons[0] ?? null;
  return { rows, warning, updateApplied: applied };
}
