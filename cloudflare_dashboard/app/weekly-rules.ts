/**
 * Weekly field table (/weekly-players): column definitions, default column sets and row building.
 * Pure and dependency-free at runtime (only `import type`), so node's strip-types test loader can run it.
 * The view injects the shared vocabulary and the profile helpers through `WeeklyDeps`.
 */
import type { ExplainDoc, ExPlayer } from "./explain-rules";
import type { PgaBenchmark, PgaBenchmarkReference, ProfileCatalog } from "./player-profile-rules";
import type { LABELS as LabelsType } from "./labels";

type Obj = Record<string, unknown>;
export type WeeklyRow = Record<string, unknown>;

export type WeeklyDeps = {
  LABELS: typeof LabelsType;
  plainName: (key: string, legend?: Record<string, { label?: string } | undefined>) => string;
  signedZ: (v: unknown, digits?: number) => string;
  pct: (p: number | null | undefined, digits?: number) => string;
  savedFieldSkill: (doc: ExplainDoc | null, id: number | null) => { value: number | null };
  checkpointBenchmarkValue: (rating: PgaBenchmark | undefined, reference: PgaBenchmarkReference | undefined, checkpoint?: string | null) => number | null;
  savedSkill: (row: unknown) => { base: number | null; final: number | null; sum: number | null; residual: number | null; families: Record<string, unknown>; weather: Record<string, unknown> };
};

const obj = (v: unknown): Obj => (v && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : {});
const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** The one missing-value token; benchmark cells use the badge instead. */
export const NA = "n/a";
export const THIN_BADGE = "n<12 rounds";
/** Per-event accuracy of the field offset (tour_scale.md: 308-event re-run against provider field strength). */
export const ACCURACY_SENTENCE = "The field offset is accurate to about SD 0.12 per event; single events can be off by up to 0.4.";
export const ESTIMATOR_NAME = "F02 theta (ridge-shrunk, 365-day half-life), mean over the field";

/** Fixed column names (these are the table headers; DataTable title-cases keys). */
export const COL = {
  player: "Player",
  nTour: "n (tour mix)",
  win: "Win",
  makeCut: "Make cut",
  top10: "Top 10",
  liveSkill: "Live skill",
  b8: "B8 live shift",
  contention: "Contention shift",
  preEventSkill: "Pre-event skill (field-centred)",
  gap: "Model minus form gap",
  base: "CHL base",
  location: "Location (saved)",
  override: "Override (saved)",
  sum: "Component sum",
  baseline: "Baseline skill",
  residual: "Residual",
} as const;

export function nameOf(p: { name: string }): string {
  return p.name.includes(",") ? p.name.split(",").reverse().join(" ").trim() : p.name;
}

export type WeeklyInputs = { players: Obj[]; fieldStrength: Obj; kind: string } | null;

/** Parse the fields we use from a validated model_inputs document (identity checking stays in the caller). */
export function parseInputs(doc: unknown): WeeklyInputs {
  const d = obj(doc);
  if (!Array.isArray(d.players)) return null;
  return { players: d.players.map(obj), fieldStrength: obj(obj(d.event).field_strength), kind: String(d.kind ?? "") };
}

export type TourWeight = { tour: string; weight_fraction: number };
export function tourWeights(rating: PgaBenchmark | undefined): TourWeight[] {
  const w = obj(obj(rating).decomposition).tour_weights;
  return Array.isArray(w) ? w.map(obj).filter((x) => typeof x.tour === "string" && fin(x.weight_fraction)).map((x) => ({ tour: x.tour as string, weight_fraction: x.weight_fraction as number })) : [];
}

/** Badge only when the player's primary tour differs from the event tour or the PGA share is under 50%. */
export function tourMixBadge(rating: PgaBenchmark | undefined, eventTour: string | null): string | null {
  const weights = tourWeights(rating);
  if (!weights.length) return null;
  const primary = [...weights].sort((a, b) => b.weight_fraction - a.weight_fraction)[0];
  const pgaShare = weights.filter((w) => w.tour === "pga").reduce((a, w) => a + w.weight_fraction, 0);
  const differs = !!eventTour && primary.tour !== eventTour.toLowerCase();
  if (!differs && pgaShare >= 0.5) return null;
  return `${primary.tour.toUpperCase()}-led, PGA ${Math.round(pgaShare * 100)}%`;
}

export function nCell(rating: PgaBenchmark | undefined, fallbackN: number | null, eventTour: string | null): string {
  const n = fin(rating?.n_rounds) ? rating!.n_rounds : fallbackN;
  const badge = tourMixBadge(rating, eventTour);
  const base = fin(n) ? String(n) : NA;
  return badge ? `${base} · ${badge}` : base;
}

/** "vs PGA avg" cell: as-was gated; a missing value says why. */
export function benchmarkCell(rating: PgaBenchmark | undefined, reference: PgaBenchmarkReference | undefined, checkpoint: string | null, deps: Pick<WeeklyDeps, "checkpointBenchmarkValue" | "signedZ">): string {
  const value = deps.checkpointBenchmarkValue(rating, reference, checkpoint);
  if (value !== null) return deps.signedZ(value, 3);
  if (!rating) return NA;
  if (rating.status !== "available" || (fin(rating.n_rounds) && rating.n_rounds < 12)) return THIN_BADGE;
  return "after this checkpoint";
}

export const RESIDUAL_EPS = 1e-5;
export const residualFlag = (res: number | null): boolean => fin(res) && Math.abs(res) > RESIDUAL_EPS;
/** Four decimals, or a rounding note; flagged only when the residual is real. */
export function residualText(res: number | null, signedZ: (v: unknown, digits?: number) => string): string {
  if (!fin(res)) return "—";
  if (!residualFlag(res)) return "<1e-5 (rounding)";
  return `FLAG ${signedZ(res, 4)}`;
}

export type FieldMethod = { offsetText: string; offsetLabel: string; estimator: string; vintage: string; imputed: string; nField: string; fieldAverage: string };
export function fieldMethod(inputs: WeeklyInputs, kind: string): FieldMethod {
  const s = obj(inputs?.fieldStrength);
  const offset = fin(s.field_offset) ? s.field_offset : null;
  const mus = (inputs?.players ?? []).map((p) => obj(obj(p).challenger).mu_tour).filter(fin);
  const avg = mus.length ? mus.reduce((a, b) => a + b, 0) / mus.length : null;
  const sign = (v: number | null) => (v === null ? NA : `${v > 0 ? "+" : ""}${v.toFixed(3)}`);
  return {
    offsetText: sign(offset),
    offsetLabel: kind === "live" ? "pre-event field offset" : "field offset",
    estimator: ESTIMATOR_NAME,
    vintage: typeof s.vintage === "string" && s.vintage ? s.vintage.slice(0, 10) : NA,
    imputed: fin(s.n_imputed) ? String(s.n_imputed) : NA,
    nField: fin(s.n_field) ? String(s.n_field) : NA,
    fieldAverage: sign(avg),
  };
}

/** The This-week column guide (a header tooltip is not available in DataTable, so the same text sits in the column guide). */
export function thisWeekGuide(m: FieldMethod, deps: Pick<WeeklyDeps, "LABELS">): string {
  return `${deps.LABELS.thisWeekPga.long}. Field average ${m.fieldAverage}; ${m.offsetLabel} ${m.offsetText}; estimator ${m.estimator}; skills vintage ${m.vintage}.`;
}

/** Default and picker-only columns for a checkpoint kind. Defaults come first so DataTable's own ordering keeps them in front. */
export function columnPlan(kind: string, deps: Pick<WeeklyDeps, "LABELS">): { defaults: string[]; picker: string[] } {
  const L = deps.LABELS;
  const defaults: string[] = [COL.player, L.thisWeekPga.short, L.vsField.short, COL.nTour, COL.win, COL.makeCut];
  if (kind === "live") defaults.push(COL.liveSkill, COL.b8, COL.contention);
  const picker = [L.vsPgaAvg.short, COL.gap, ...(kind === "live" ? [COL.preEventSkill] : []), COL.top10];
  return { defaults, picker };
}

function uniqueLabel(label: string, taken: Set<string>): string {
  let out = label;
  if (taken.has(out)) out = `${label} (component)`;
  while (taken.has(out)) out += "*";
  taken.add(out);
  return out;
}

export type BuildArgs = {
  doc: ExplainDoc;
  inputs: WeeklyInputs;
  catalog: ProfileCatalog | null;
  players: ExPlayer[];
  deps: WeeklyDeps;
};
export type BuiltTable = { rows: WeeklyRow[]; defaults: string[]; ordered: string[] };

export function buildTable({ doc, inputs, catalog, players, deps }: BuildArgs): BuiltTable {
  const { LABELS: L, plainName, signedZ, savedFieldSkill, savedSkill } = deps;
  const plan = columnPlan(doc.kind, deps);
  const live = doc.kind === "live";
  const byInput = new Map((inputs?.players ?? []).map((p) => [Number(p.dg_id), p]));
  const byCatalog = new Map((catalog?.players ?? []).map((p) => [p.dg_id, p]));
  const reference = catalog?.pga_benchmark_reference;
  const keys = [...new Set(doc.players.flatMap((p) => Object.keys(p.components)))];
  const familyKeys = [...new Set((inputs?.players ?? []).flatMap((p) => Object.keys(savedSkill(p).families)))];
  const weatherRounds = [...new Set((inputs?.players ?? []).flatMap((p) => Object.keys(savedSkill(p).weather)))];
  const noCut = doc.event.cut_round === 0 || doc.event.cut_rule === "no cut";
  const taken = new Set<string>([...plan.defaults, ...plan.picker, COL.base, COL.location, COL.override, COL.sum, COL.baseline, COL.residual]);
  const componentLabel = new Map(keys.map((k) => [k, uniqueLabel(plainName(k, doc.components_legend), taken)]));
  const familyLabel = new Map(familyKeys.map((k) => [k, uniqueLabel(`CHL ${plainName(k)}`, taken)]));
  // Until DataTable accepts a default-column prop it shows the first 9 columns: the three that follow the defaults are deliberately harmless (never "vs PGA avg").
  const ordered: string[] = [...plan.defaults, COL.top10, COL.base, COL.location, COL.override, L.vsPgaAvg.short, COL.gap, ...(live ? [COL.preEventSkill] : []), ...familyLabel.values(), COL.sum, COL.baseline, COL.residual, ...weatherRounds.map((r) => `Weather ${r}`), ...componentLabel.values()];
  const rows = players.map((p): WeeklyRow => {
    const input = byInput.get(p.id);
    const ch = obj(input?.challenger);
    const rating = byCatalog.get(p.id)?.pga_benchmark;
    const tw = fin(ch.mu_tour) ? ch.mu_tour : null;
    const vsPga = deps.checkpointBenchmarkValue(rating, reference, doc.as_of);
    const s = input ? savedSkill(input) : null;
    const bd = obj(ch.breakdown);
    const row: WeeklyRow = {
      [COL.player]: nameOf(p),
      [L.thisWeekPga.short]: tw === null ? NA : signedZ(tw, 3),
      [L.vsField.short]: signedZ(savedFieldSkill(doc, p.id).value, 3),
      [COL.nTour]: nCell(rating, p.n_prior_rounds, doc.event.tour),
      [COL.win]: deps.pct(p.probs.win.model),
      [COL.makeCut]: noCut ? "No cut" : deps.pct(p.probs.make_cut.model),
    };
    if (live) {
      row[COL.liveSkill] = signedZ(p.live?.mu_live, 3);
      row[COL.b8] = signedZ(p.live?.b8_delta, 3);
      row[COL.contention] = signedZ(p.live?.contention, 3);
    }
    row[L.vsPgaAvg.short] = benchmarkCell(rating, reference, doc.as_of, deps);
    row[COL.gap] = tw !== null && vsPga !== null ? signedZ(tw - vsPga, 3) : NA;
    if (live) row[COL.preEventSkill] = signedZ(p.mu, 3);
    row[COL.top10] = deps.pct(p.probs.top_10.model);
    row[COL.base] = signedZ(s?.base, 3);
    for (const k of familyKeys) row[familyLabel.get(k)!] = signedZ(s?.families[k], 3);
    row[COL.location] = signedZ(obj(bd.location).total, 3);
    row[COL.override] = signedZ(obj(bd.override).total, 3);
    row[COL.sum] = signedZ(s?.sum, 3);
    row[COL.baseline] = signedZ(s?.final, 3);
    row[COL.residual] = residualText(s?.residual ?? null, signedZ);
    for (const r of weatherRounds) row[`Weather ${r}`] = s && r in s.weather ? signedZ(s.weather[r], 3) : NA;
    for (const k of keys) row[componentLabel.get(k)!] = signedZ(p.components[k], 3);
    // Object-valued columns never reach the picker or the CSV: this carries the row identity for tap-to-expand.
    row.__row = { id: p.id };
    return row;
  });
  return { rows, defaults: plan.defaults, ordered };
}

/** Mobile priority (Player, This week, Win) as a stable list for the expand card and CSS hooks. */
export const MOBILE_COLUMNS = (deps: Pick<WeeklyDeps, "LABELS">): string[] => [COL.player, deps.LABELS.thisWeekPga.short, COL.win];

/** The expand card for a tapped row: every column except the three already visible on a phone. */
export function expandEntries(row: WeeklyRow, deps: Pick<WeeklyDeps, "LABELS">): Array<[string, string]> {
  const skip = new Set(MOBILE_COLUMNS(deps));
  return Object.entries(row).filter(([k, v]) => k !== "__row" && !skip.has(k) && typeof v !== "object").map(([k, v]) => [k, String(v)]);
}

/** Plain-name headers for the closed reconciliation table's family columns (never the raw keys). */
export function reconciliationHeaders(keys: string[], legend: Record<string, { label?: string } | undefined> | undefined, plain: WeeklyDeps["plainName"]): Array<{ key: string; label: string }> {
  return keys.map((key) => ({ key, label: plain(key, legend) }));
}
