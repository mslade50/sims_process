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
  /** The standing "form versus model" sentence (labels.ts GAP_METHOD_SENTENCE), used in column hover text. */
  gapSentence?: string;
  plainName: (key: string, legend?: Record<string, { label?: string } | undefined>) => string;
  /** Display-side translation of publisher phrases (labels.ts plainPhrase). */
  plainPhrase?: (text: string) => string;
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
export const THIN_BADGE = "under 12 rounds";
/** Per-event accuracy of the field offset (tour_scale.md: 308-event re-run against provider field strength). */
export const ACCURACY_SENTENCE = "The field offset is typically within about 0.12 strokes of the true figure for an event; a single event can be off by up to 0.4.";
export const ESTIMATOR_NAME = "Smoothed player strength ratings, averaged over the field";

/** Fixed column names (these are the table headers; DataTable title-cases keys). */
export const COL = {
  player: "Player",
  nTour: "Rounds (tour mix)",
  win: "Win",
  makeCut: "Make cut",
  top10: "Top 10",
  liveSkill: "Live skill",
  b8: "Live shift (strokes/rd)",
  contention: "Leaderboard shift (strokes/rd)",
  preEventSkill: "Pre-event skill (vs field)",
  gap: "This week minus form",
  base: "Base skill",
  location: "Course location",
  override: "Owner override",
  sum: "Sum of parts",
  baseline: "Saved skill (all parts)",
  residual: "Unexplained",
} as const;

/** One plain sentence per fixed column, shown on hover over the header and in the column picker. */
export function columnTitles(deps: Pick<WeeklyDeps, "LABELS">, gapSentence: string): Record<string, string> {
  const L = deps.LABELS;
  return {
    [COL.player]: "The golfer. Click a row for details and a link to the full profile.",
    [L.thisWeekPga.short]: L.thisWeekPga.long,
    [L.vsField.short]: L.vsField.long,
    [L.vsPgaAvg.short]: `${L.vsPgaAvg.long}. ${gapSentence}`,
    [COL.nTour]: "Rounds behind the player's recent-form rating. A note appears when the player mostly plays another tour, or has a thin record.",
    [COL.win]: "The model's chance of winning the event.",
    [COL.makeCut]: "The model's chance of making the cut.",
    [COL.top10]: "The model's chance of a top-10 finish.",
    [COL.liveSkill]: "The model's skill estimate for the rest of the event, updated for scoring so far, in strokes per round against the field.",
    [COL.b8]: "How much the player's scoring so far this week has moved the skill estimate up (+) or down (-), in strokes per round.",
    [COL.contention]: "Adjustment for where the player sits on the leaderboard (+ helps, - hurts), in strokes per round.",
    [COL.preEventSkill]: "The model's skill estimate before the event began, minus the field's average.",
    [COL.gap]: `${L.thisWeekPga.short} minus ${L.vsPgaAvg.short}. ${gapSentence}`,
    [COL.base]: "The model's skill estimate before course-location and owner adjustments.",
    [COL.location]: "Adjustment for travel, home region and nationality at this event's location.",
    [COL.override]: "A one-off change to the player's skill set by the owner on the private site.",
    [COL.sum]: "Base skill plus every adjustment added up.",
    [COL.baseline]: "The model's final saved skill for the event, before any live updates. Weather is shown separately and is not included.",
    [COL.residual]: "How far the saved skill differs from the sum of its parts. It should be zero; anything shown is a flagged mismatch.",
  };
}

/** One precision for chances: "<0.1%" for tiny values, otherwise one decimal. Display only. */
export function chance(p: number | null | undefined): string {
  if (!fin(p)) return NA;
  if (p <= 0) return "0%";
  if (p < 0.001) return "<0.1%";
  return `${(p * 100).toFixed(1)}%`;
}

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
  const parts = [base];
  if (fin(n) && n < 12) parts.push("thin record");
  if (badge) parts.push(badge);
  return parts.join(" · ");
}

/** "vs PGA avg" cell: as-was gated; a missing value says why. */
export function benchmarkCell(rating: PgaBenchmark | undefined, reference: PgaBenchmarkReference | undefined, checkpoint: string | null, deps: Pick<WeeklyDeps, "checkpointBenchmarkValue" | "signedZ">): string {
  const value = deps.checkpointBenchmarkValue(rating, reference, checkpoint);
  if (value !== null) return deps.signedZ(value, 3);
  if (!rating) return NA;
  if (rating.status !== "available" || (fin(rating.n_rounds) && rating.n_rounds < 12)) return THIN_BADGE;
  return "after this run";
}

export const RESIDUAL_EPS = 1e-5;
export const residualFlag = (res: number | null): boolean => fin(res) && Math.abs(res) > RESIDUAL_EPS;
/** Four decimals, or zero; flagged only when the residual is real. */
export function residualText(res: number | null, signedZ: (v: unknown, digits?: number) => string): string {
  if (!fin(res)) return "—";
  if (!residualFlag(res)) return ROUNDING;
  return `Mismatch ${signedZ(res, 4)}`;
}
const ROUNDING = "0 (rounding)";

/** Calendar date ("2026-09-28") as "Sep 28, 2026"; no clock time, so no timezone. */
function plainDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? NA : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
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
    vintage: typeof s.vintage === "string" && s.vintage ? plainDate(s.vintage.slice(0, 10)) : NA,
    imputed: fin(s.n_imputed) ? String(s.n_imputed) : NA,
    nField: fin(s.n_field) ? String(s.n_field) : NA,
    fieldAverage: sign(avg),
  };
}

/** The This-week column guide, also used as that column's hover text. */
export function thisWeekGuide(m: FieldMethod, deps: Pick<WeeklyDeps, "LABELS">): string {
  return `${deps.LABELS.thisWeekPga.long}. This field averages ${m.fieldAverage}; its ${m.offsetLabel} is ${m.offsetText}.`;
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
export type BuiltTable = { rows: WeeklyRow[]; defaults: string[]; ordered: string[]; titles: Record<string, string>; notes: string[] };

const isEmptyCell = (v: unknown): boolean => v == null || v === "" || v === "—" || v === NA || v === ROUNDING;

export function buildTable({ doc, inputs, catalog, players, deps }: BuildArgs): BuiltTable {
  const { LABELS: L, plainName, signedZ, savedFieldSkill, savedSkill } = deps;
  const plan = columnPlan(doc.kind, deps);
  const live = doc.kind === "live";
  const byInput = new Map((inputs?.players ?? []).map((p) => [Number(p.dg_id), p]));
  const byCatalog = new Map((catalog?.players ?? []).map((p) => [p.dg_id, p]));
  const reference = catalog?.pga_benchmark_reference;
  const keys = [...new Set(doc.players.flatMap((p) => Object.keys(p.components)))];
  const familyKeys = [...new Set((inputs?.players ?? []).flatMap((p) => Object.keys(savedSkill(p).families)))];
  const weatherRounds = activeWeatherRounds((inputs?.players ?? []).map((p) => ({ weather: savedSkill(p).weather })));
  const noCut = doc.event.cut_round === 0 || doc.event.cut_rule === "no cut";
  const taken = new Set<string>([...plan.defaults, ...plan.picker, COL.base, COL.location, COL.override, COL.sum, COL.baseline, COL.residual]);
  const componentLabel = new Map(keys.map((k) => [k, uniqueLabel(plainName(k, doc.components_legend), taken)]));
  const familyLabel = new Map(familyKeys.map((k) => [k, uniqueLabel(`Base skill: ${plainName(k, doc.components_legend)}`, taken)]));
  // Until DataTable accepts a default-column prop it shows the first 9 columns: the three that follow the defaults are deliberately harmless (never "vs PGA avg").
  const ordered: string[] = [...plan.defaults, COL.top10, COL.base, COL.location, COL.override, L.vsPgaAvg.short, COL.gap, ...(live ? [COL.preEventSkill] : []), ...familyLabel.values(), COL.sum, COL.baseline, COL.residual, ...weatherRounds.map((r) => `Weather ${r}`), ...componentLabel.values()];
  const titles = columnTitles(deps, deps.gapSentence ?? "");
  const say = deps.plainPhrase ?? ((t: string) => t);
  for (const [k, label] of componentLabel) titles[label] = say(doc.components_legend?.[k]?.meaning ?? "A saved part of the model's skill estimate, in strokes per round against the field.");
  for (const [k, label] of familyLabel) titles[label] = say(doc.components_legend?.[k]?.meaning ?? plainName(k, doc.components_legend));
  for (const r of weatherRounds) titles[`Weather ${r}`] = `Effect of the forecast weather on the model's skill estimate for round ${r}.`;
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
      [COL.win]: chance(p.probs.win.model),
      [COL.makeCut]: noCut ? "No cut" : chance(p.probs.make_cut.model),
    };
    if (live) {
      row[COL.liveSkill] = signedZ(p.live?.mu_live, 3);
      row[COL.b8] = signedZ(p.live?.b8_delta, 3);
      row[COL.contention] = signedZ(p.live?.contention, 3);
    }
    row[L.vsPgaAvg.short] = benchmarkCell(rating, reference, doc.as_of, deps);
    row[COL.gap] = tw !== null && vsPga !== null ? signedZ(tw - vsPga, 3) : NA;
    if (live) row[COL.preEventSkill] = signedZ(p.mu, 3);
    row[COL.top10] = chance(p.probs.top_10.model);
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
  // No empty columns: a column that is blank, n/a or pure rounding for every player is dropped, and the reason is said once.
  const notes: string[] = [];
  const keep = (col: string) => rows.some((r) => !isEmptyCell(r[col]));
  if (noCut) notes.push("This event has no cut, so make-cut odds are not shown.");
  const gone = new Set<string>();
  for (const col of ordered) {
    if (col === COL.player) continue;
    if (!keep(col) || (col === COL.makeCut && noCut)) gone.add(col);
  }
  // Two names for one number: when Live skill equals vs field on every row, show it once.
  if (live && !gone.has(COL.liveSkill) && !gone.has(L.vsField.short) && rows.every((r) => r[COL.liveSkill] === r[L.vsField.short])) {
    gone.add(COL.liveSkill);
    notes.push("Live skill matches the vs field column for every player, so it is shown once.");
  }
  // The residual and sum columns repeat the saved skill when nothing is unexplained.
  if (!gone.has(COL.sum) && !gone.has(COL.baseline) && rows.every((r) => r[COL.sum] === r[COL.baseline])) gone.add(COL.sum);
  const cleanRows = rows.map((r) => { const o: WeeklyRow = {}; for (const [k, v] of Object.entries(r)) if (!gone.has(k)) o[k] = v; return o; });
  return { rows: cleanRows, defaults: plan.defaults.filter((c) => !gone.has(c)), ordered: ordered.filter((c) => !gone.has(c)), titles, notes };
}

/** Mobile priority (Player, This week, Win) as a stable list for the expand card and CSS hooks. */
export const MOBILE_COLUMNS = (deps: Pick<WeeklyDeps, "LABELS">): string[] => [COL.player, deps.LABELS.thisWeekPga.short, COL.win];

/** The expand card for a tapped row: every column except the three already visible on a phone. */
export function expandEntries(row: WeeklyRow, deps: Pick<WeeklyDeps, "LABELS">): Array<[string, string]> {
  const skip = new Set(MOBILE_COLUMNS(deps));
  return Object.entries(row).filter(([k, v]) => k !== "__row" && !skip.has(k) && typeof v !== "object" && !isEmptyCell(v)).map(([k, v]) => [k, String(v)]);
}

/** Plain-name headers for the closed reconciliation table's family columns (never the raw keys). */
/** Weather rounds that carry a real effect; rounds that are zero for everyone (no forecast yet) are left out. */
export function activeWeatherRounds(players: Array<{ weather: Record<string, unknown> }>): string[] {
  const rounds = [...new Set(players.flatMap((p) => Object.keys(p.weather)))];
  return rounds.filter((r) => players.some((p) => fin(p.weather[r]) && Math.abs(p.weather[r] as number) > 1e-9));
}

export function reconciliationHeaders(keys: string[], legend: Record<string, { label?: string } | undefined> | undefined, plain: WeeklyDeps["plainName"]): Array<{ key: string; label: string }> {
  return keys.map((key) => ({ key, label: plain(key, legend) }));
}
