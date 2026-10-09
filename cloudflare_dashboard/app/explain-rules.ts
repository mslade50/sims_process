/**
 * Pure parsing and table/chart logic for the golfprice explain document (golfprice/explain/<event>/latest.json, schema golfprice.explain.v1, built by
 * golfprice/explain_export.py). No runtime APIs: shared by ExplainViews.tsx / ExplainCharts.tsx and tests/explain.test.mjs.
 * Everything optional is tolerated missing; anything that is not the document gives null (the views then show their empty state).
 */

export const EXPLAIN_SCHEMA = "golfprice.explain.v1";
export const MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"] as const;
export type Market = (typeof MARKETS)[number];
export const MARKET_LABEL: Record<Market, string> = { win: "Win", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", make_cut: "Make cut" };

export type MarketProb = { model: number | null; market: number | null; fair: number | null; n_books: number | null; edge: number | null; rel: number | null; logit: number | null };
export type Pctl = { mean: number; sd: number; p10: number; p50: number; p90: number };
export type TotalScore = { mean: number; sd: number; p5: number; p25: number; p50: number; p75: number; p95: number; n_made: number };
export type LiveRound = { round: number; score: number | null; sg_total: number | null; sg_ott: number | null; sg_app: number | null; sg_arg: number | null; sg_putt: number | null };
export type LiveState = {
  tier: number; cum_strokes: number | null; to_par: number | null; pos: string | null; pos_n: number | null; mu_pre: number | null; mu_live: number | null;
  b8_delta: number | null; contention: number | null; strokes_behind_leader: number | null; d_mu_since_last: number | null; d_b8_since_last: number | null;
  d_contention_since_last: number | null; pre_event: Partial<Record<Market, number | null>>; since_last: Partial<Record<Market, number | null>>; rounds: LiveRound[];
};
export type ExPlayer = {
  id: number; name: string; country: string | null; amateur: boolean; withdrawn: boolean; n_prior_rounds: number | null; wave: string | null; tee_r1: string | null; tee_r2: string | null;
  mu: number | null; mu_rel: number | null; sd: number | null; components: Record<string, number>; probs: Record<Market, MarketProb>;
  edge_sg: number | null; attribution: Record<string, number>; why: string; why_edge: string | null; tags: string[]; neighbors: number[];
  finish: { pos: number[]; p_miss_cut: number; mean_pos: number; p10: number; median: number; p90: number } | null;
  scores: Record<string, Pctl | TotalScore>; live: LiveState | null;
};
export type LegendEntry = { label: string; group: string; meaning: string };
export type BiasRow = {
  dimension: string; dimension_label: string; bucket: string; label: string; n: number; edge_sg: number | null; edge_sg_se: number | null;
  p_win_model: number | null; p_win_market: number | null; drivers: Array<{ component: string; label: string; attr_sg: number | null; avg_rel: number | null }>;
};
export type HoleRow = { hole: number; par: number; yd: number | null; exp: number | null; bird: number | null; bog: number | null; sens: number | null };
export type CourseCard = {
  course: string; par_per_round: number[]; yardage: number | null; scoring_avg_vs_par: number | null; hole_source: string | null; holes: HoleRow[];
  by_par: Record<string, { holes: number; avg_exp_vs_par: number | null; avg_birdie: number | null; avg_bogey: number | null; avg_sensitivity: number | null }>;
  cut_rule: string | null; what_drives_scoring: string[];
  weather: { status: string | null; applied_in_prices: boolean | null; label: string | null; venue_class?: string | null; wind_slope?: number | null; wind_slope_source?: string | null; issued_at?: string | null;
    waves?: Array<{ round: number; wave: string; rel: number | null; mean_wind: number | null }>; field?: Array<{ round: number; common: number | null; cut_line_shift: number | null }> };
};
export type ExplainDoc = {
  schema: string; kind: "week" | "live"; event_uid: string; run: string; as_of: string | null; after_round: number | null;
  event: { name: string; tour: string | null; course: string | null; date_start: string | null; rounds: number | null; field_size: number | null; par_per_round: number[]; cut_round: number | null; cut_top_n: number | null; cut_rule: string | null };
  model: { weather_layer: string | null; overrides_applied: string[]; n_sims: number | null };
  components_legend: Record<string, LegendEntry>; component_field_mean: Record<string, number | null>;
  course_card: CourseCard;
  finish: { method: string; n_draws?: number; reason: string | null; calibration?: Record<string, { mean_abs: number; max_abs: number }> };
  edge_model: { ridge_r2: number | null; strokes_per_logit: Record<string, number>; note: string };
  bias: BiasRow[]; top_edges: number[]; players: ExPlayer[]; since_last: { prev_run: string; prev_as_of: string | null; prev_after_round: number | null } | null;
};

/** Accent/case-insensitive type-ahead: every typed word must start a word of "Last, First" or "First Last". */
const foldText = (t: string): string => t.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/['’`]/g, "").replace(/[^a-z0-9]+/g, " ").trim();
export function nameMatches(name: string, query: string): boolean {
  const q = foldText(query).split(" ").filter(Boolean);
  if (!q.length) return true;
  const parts = name.split(",");
  const words = [...new Set([...foldText(name).split(" "), ...foldText(parts.length === 2 ? `${parts[1]} ${parts[0]}` : name).split(" ")])];
  return q.every((t) => words.some((w) => w.startsWith(t)));
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function neutralLegend(raw: unknown): Record<string, LegendEntry> {
  if (!isObject(raw)) return {};
  const out: Record<string, LegendEntry> = {};
  for (const [k, v] of Object.entries(raw)) {
    const e = v as LegendEntry;
    const plain = COMPONENT_PLAIN[k];
    out[k] = isObject(v) ? { ...e, label: plain?.label ?? plainText(String(e.label ?? k)), meaning: plain?.meaning ?? (typeof e.meaning === "string" ? plainText(neutralPronouns(e.meaning)) : "") } : e;
  }
  return out;
}

export function parseExplain(raw: unknown): ExplainDoc | null {
  if (!isObject(raw) || raw.schema !== EXPLAIN_SCHEMA || !Array.isArray(raw.players) || !isObject(raw.event)) return null;
  const d = raw as unknown as ExplainDoc;
  const players = (raw.players as unknown[]).filter((p): p is Record<string, unknown> => isObject(p) && typeof p.id === "number").map((p) => {
    const probs = (isObject(p.probs) ? p.probs : {}) as Record<string, Partial<MarketProb>>;
    const full = {} as Record<Market, MarketProb>;
    for (const m of MARKETS) {
      const e = probs[m] ?? {};
      full[m] = { model: num(e.model), market: num(e.market), fair: num(e.fair), n_books: num(e.n_books), edge: num(e.edge), rel: num(e.rel), logit: num(e.logit) };
    }
    return { ...(p as object), probs: full, components: isObject(p.components) ? p.components : {}, attribution: isObject(p.attribution) ? p.attribution : {},
      tags: Array.isArray(p.tags) ? p.tags : [], neighbors: Array.isArray(p.neighbors) ? p.neighbors : [], scores: isObject(p.scores) ? p.scores : {},
      finish: isObject(p.finish) && Array.isArray((p.finish as { pos?: unknown }).pos) ? p.finish : null, live: isObject(p.live) ? p.live : null, why: typeof p.why === "string" ? plainWhy(neutralPronouns(p.why)) : "",
      why_edge: typeof p.why_edge === "string" ? plainWhyEdge(neutralPronouns(p.why_edge)) : null } as unknown as ExPlayer;
  });
  return { ...d, players, bias: Array.isArray(d.bias) ? plainBias(d.bias) : [], top_edges: Array.isArray(d.top_edges) ? d.top_edges : [], components_legend: neutralLegend(d.components_legend),
    finish: isObject(d.finish) ? d.finish : { method: "unavailable", reason: null }, course_card: (isObject(d.course_card) ? d.course_card : { holes: [], what_drives_scoring: [], by_par: {}, weather: {} }) as CourseCard,
    edge_model: isObject(d.edge_model) ? d.edge_model : { ridge_r2: null, strokes_per_logit: {}, note: "" }, since_last: isObject(d.since_last) ? d.since_last : null };
}

/* ------------------------------------------------------------------ index / event selection */
export type DistributionRun = { key?: string | null; kind?: string | null; run?: string | null; as_of?: string | null; after_round?: number | null; validated?: boolean | null };
export type IndexEvent = {
  event_uid: string; name?: string; tour?: string; date_start?: string; course?: string; explain_key?: string | null; explain_run?: string | null;
  distribution_runs?: DistributionRun[] | null;
};
export const explainKeyFor = (e: IndexEvent): string => e.explain_key || `golfprice/explain/${e.event_uid.replace(/:/g, "_")}/latest.json`;

/** Which explanation documents exist for an event. Every run keeps its own document (distribution_runs), so the last pre-tournament run survives live updates. */
export type ExplainChoices = { preKey: string | null; liveKey: string | null; liveRound: number | null; hasLive: boolean };
export function explainChoices(e: IndexEvent): ExplainChoices {
  const runs = (e.distribution_runs ?? []).filter((r): r is DistributionRun & { key: string } => !!r && typeof r.key === "string" && r.key.length > 0 && r.validated !== false);
  const newest = (kind: string) => runs.filter((r) => r.kind === kind).sort((a, b) => String(a.as_of ?? "").localeCompare(String(b.as_of ?? ""))).at(-1) ?? null;
  const pre = newest("week");
  const live = newest("live");
  const currentIsLive = (e.explain_run ?? "").startsWith("live_") || (!!e.explain_key && /\/live_[^/]*$/.test(e.explain_key));
  const liveKey = currentIsLive ? explainKeyFor(e) : live?.key ?? null;
  const fromRun = /live_R(\d+)_/.exec(e.explain_run ?? "")?.[1];
  const liveRound = live?.after_round ?? (fromRun ? Number(fromRun) : null);
  return { preKey: pre?.key ?? null, liveKey, liveRound, hasLive: currentIsLive || live !== null };
}

/** The published text was written in the masculine ("his mean skill"); players can be anyone, so read it as "their". */
export function neutralPronouns(text: string): string {
  return text
    .replace(/\b(He|he) (has|is|was|plays|hasn't|isn't)\b/g, (_m, h: string, v: string) => `${h === "He" ? "They" : "they"} ${{ has: "have", is: "are", was: "were", plays: "play", "hasn't": "haven't", "isn't": "aren't" }[v]}`)
    .replace(/\bHe\b/g, "They").replace(/\bhe\b/g, "they").replace(/\bHis\b/g, "Their").replace(/\bhis\b/g, "their").replace(/\bhim\b/g, "them");
}

/** Events of the newest week first (all sharing the newest start date), then older ones; events without an explain document are kept (the page says so). */
export function orderEvents(events: IndexEvent[]): IndexEvent[] {
  const withDate = events.filter((e) => e.date_start);
  const newest = withDate.map((e) => e.date_start as string).sort().at(-1);
  return [...events].sort((a, b) => {
    const an = a.date_start === newest ? 0 : 1;
    const bn = b.date_start === newest ? 0 : 1;
    return an - bn || (b.date_start ?? "").localeCompare(a.date_start ?? "") || (a.tour ?? "").localeCompare(b.tour ?? "");
  });
}

/* ------------------------------------------------------------------ numbers */
export const pct = (p: number | null | undefined, digits = 1): string => (p === null || p === undefined ? "-" : p === 0 ? "0%" : p < 0.0001 ? "<0.01%" : p < 0.001 ? `${(p * 100).toFixed(2)}%` : `${(p * 100).toFixed(digits)}%`);
/** Two chances shown side by side with the same number of decimals, so "0.2% vs 0.07%" reads "0.20% vs 0.07%". */
export function pctPair(a: number | null | undefined, b: number | null | undefined): [string, string] {
  const small = [a, b].some((x) => x !== null && x !== undefined && x > 0 && x < 0.001);
  return small ? [pct(a, 2), pct(b, 2)] : [pct(a), pct(b)];
}
/** Signed number; a value that rounds to zero is shown as plain "0.00" (never "-0.00" or "+0.00"). */
export const signed = (v: number | null | undefined, digits = 2): string => {
  if (v === null || v === undefined) return "-";
  const r = Number(Math.abs(v).toFixed(digits));
  return `${r === 0 ? "" : v > 0 ? "+" : "−"}${r.toFixed(digits)}`;
};
/** The publisher clips the strokes gap at +/-2.00; those are floors, not measurements. */
export const GAP_CAP = 2;
export const isCapped = (v: number | null | undefined): boolean => v !== null && v !== undefined && Math.abs(v) >= GAP_CAP - 0.005;
export const gapText = (v: number | null | undefined, digits = 2): string => (isCapped(v) ? `${v! > 0 ? "+" : "−"}${GAP_CAP.toFixed(digits)} or more` : signed(v, digits));
export const toParText = (v: number | null | undefined): string => (v === null || v === undefined ? "-" : v === 0 ? "E" : v > 0 ? `+${v}` : `−${Math.abs(v)}`);

/** Decimal-odds style label for a probability ("+1900" American). */
export function american(p: number | null | undefined): string {
  if (p === null || p === undefined || p <= 0 || p >= 1) return "-";
  return p < 0.5 ? `+${Math.round((1 / p - 1) * 100)}` : `−${Math.round((p / (1 - p)) * 100)}`;
}

/* ------------------------------------------------------------------ finish shape */
export type Bin = { key: string; label: string; lo: number; hi: number };

/** Finish-position bins for the chart: 1..5 singly, then 6-10, 11-15, 16-20, 21-30 ... up to the cut line (or the field), ending in one "rest" bin. */
export function finishBins(fieldSize: number, cutTopN: number | null): Bin[] {
  const top = Math.max(5, Math.min(fieldSize, cutTopN && cutTopN > 0 ? cutTopN : fieldSize));
  const edges = [5, 10, 15, 20, 30, 40, 50, 65, 80, 100, 125, 160];
  const bins: Bin[] = [];
  for (let k = 1; k <= Math.min(5, top); k++) bins.push({ key: String(k), label: k === 1 ? "1st" : String(k), lo: k, hi: k });
  let lo = 6;
  for (const e of edges) {
    if (lo > top) break;
    const hi = Math.min(e, top);
    if (hi >= lo) bins.push({ key: `${lo}-${hi}`, label: `${lo}-${hi}`, lo, hi });
    lo = hi + 1;
    if (hi >= top) break;
  }
  if (lo <= fieldSize) bins.push({ key: `${lo}+`, label: `${lo}+`, lo, hi: fieldSize });
  return bins;
}

export type FinishBars = { bins: Bin[]; p: number[]; cum: number[]; missCut: number | null };
/** Probability mass per bin (the tail beyond the cut line shares the last bin; missed-cut mass is reported separately) and the cumulative P(finish <= bin end). */
export function binFinish(pos: number[], bins: Bin[], missCut: number | null): FinishBars {
  const p = bins.map((b) => {
    let s = 0;
    for (let k = b.lo; k <= Math.min(b.hi, pos.length); k++) s += pos[k - 1] ?? 0;
    return s;
  });
  let c = 0;
  const cum = p.map((v) => (c += v));
  return { bins, p, cum, missCut };
}

/** P(finish in the top k) for k in `at` (from the full position array). */
export function topK(pos: number[], at: number[]): number[] {
  return at.map((k) => {
    let s = 0;
    for (let i = 0; i < Math.min(k, pos.length); i++) s += pos[i] ?? 0;
    return s;
  });
}

/* ------------------------------------------------------------------ waterfall */
export type WaterfallRow = { key: string; label: string; group: string; value: number; meaning: string };
const GROUP_ORDER = ["skill", "course", "location", "override", "live", "weather"];
/** Components of one player, field-relative strokes per round, grouped and then by size; near-zero rows are folded into one "smaller items" row. */
export type WaterfallResult = { rows: WaterfallRow[]; total: number; small: number; smallCount: number; skillTotal: number; weatherTotal: number; smallSkill: number; smallSkillCount: number; unexplained: number };
export function waterfall(player: ExPlayer, legend: Record<string, LegendEntry>, minAbs = 0.015): WaterfallResult {
  const rows: WaterfallRow[] = [];
  let small = 0;
  let smallCount = 0;
  let smallSkill = 0;
  let smallSkillCount = 0;
  let weatherSmall = 0;
  for (const [key, value] of Object.entries(player.components)) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const lg = legend[key] ?? { label: key, group: "skill", meaning: "" };
    if (Math.abs(value) < minAbs) {
      small += value; smallCount += 1;
      if (lg.group === "weather") weatherSmall += value; else { smallSkill += value; smallSkillCount += 1; }
      continue;
    }
    rows.push({ key, label: lg.label, group: lg.group, value, meaning: lg.meaning });
  }
  rows.sort((a, b) => GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group) || Math.abs(b.value) - Math.abs(a.value));
  const total = rows.reduce((s, r) => s + r.value, 0) + small;
  // Weather rows are applied on top of the skill rating, so the skill rows alone are what add up to the displayed expected skill.
  const weatherTotal = rows.filter((r) => r.group === "weather").reduce((s, r) => s + r.value, 0) + weatherSmall;
  const skillTotal = total - weatherTotal;
  const unexplained = player.mu_rel === null || player.mu_rel === undefined ? 0 : player.mu_rel - skillTotal;
  return { rows, total, small, smallCount, skillTotal, weatherTotal, smallSkill, smallSkillCount, unexplained };
}

/* ------------------------------------------------------------------ why-priced table */
export type SortKey = "name" | "model" | "market" | "edge" | "rel" | "edge_sg" | "mu" | "sd" | "pos";
export type WhyFilter = { query: string; tags: string[]; market: Market; onlyEdge: "all" | "above" | "below"; top?: number | null };
export const FILTER_TAGS = ["thin data", "light data", "deep data", "x-tour dependent", "USA", "international", "amateur", "early", "late", "OTT strong", "OTT weak", "approach strong", "approach weak",
  "putting hot (regression risk)", "putting cold", "hot form", "cold form", "long layoff", "course fit +", "course fit -", "continent change", "long jet lag"];

export function playerHasTag(p: ExPlayer, tag: string): boolean {
  return p.tags.includes(tag) || (tag === "early" || tag === "late" ? p.wave === tag : false);
}
export function availableTags(players: ExPlayer[]): string[] {
  return FILTER_TAGS.filter((t) => players.some((p) => playerHasTag(p, t)));
}
/** Players ranked by market win probability (model where there is no market); `top` keeps the first N, so long-shot noise can be hidden. */
export function priceRanks(players: ExPlayer[]): Map<number, number> {
  const order = [...players].filter((p) => !p.withdrawn).sort((a, b) => (b.probs.win.market ?? b.probs.win.model ?? 0) - (a.probs.win.market ?? a.probs.win.model ?? 0));
  return new Map(order.map((p, i) => [p.id, i + 1] as const));
}
export function filterWhy(players: ExPlayer[], f: WhyFilter): ExPlayer[] {
  const ranks = f.top ? priceRanks(players) : null;
  return players.filter((p) => {
    if (p.withdrawn) return false;
    if (ranks && (ranks.get(p.id) ?? 9999) > (f.top as number)) return false;
    if (f.query.trim() && !nameMatches(p.name, f.query)) return false;
    if (f.tags.length && !f.tags.every((t) => playerHasTag(p, t))) return false;
    const e = p.probs[f.market];
    if (f.onlyEdge === "above" && !((e.edge ?? 0) > 0)) return false;
    if (f.onlyEdge === "below" && !((e.edge ?? 0) < 0)) return false;
    return true;
  });
}
export function sortValue(p: ExPlayer, key: SortKey, market: Market): number | string | null {
  const e = p.probs[market];
  switch (key) {
    case "name": return p.name;
    case "model": return e.model;
    case "market": return e.market;
    case "edge": return e.edge;
    case "rel": return e.rel;
    case "edge_sg": return p.edge_sg;
    case "mu": return p.mu_rel;
    case "sd": return p.sd;
    case "pos": return p.live?.pos_n ?? null;
  }
}
/** Sorts (nulls always last); stable on ties by model win probability then name. */
export function sortWhy(players: ExPlayer[], key: SortKey, dir: "asc" | "desc", market: Market): ExPlayer[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...players].sort((a, b) => {
    const x = sortValue(a, key, market);
    const y = sortValue(b, key, market);
    if (x === null && y === null) return a.name.localeCompare(b.name);
    if (x === null) return 1;
    if (y === null) return -1;
    const c = typeof x === "string" || typeof y === "string" ? String(x).localeCompare(String(y)) : x - y;
    return c * sign || (b.probs.win.model ?? 0) - (a.probs.win.model ?? 0) || a.name.localeCompare(b.name);
  });
}

/* ------------------------------------------------------------------ headline lists */
export function topEdges(doc: ExplainDoc, n = 6): { above: ExPlayer[]; below: ExPlayer[] } {
  const byId = new Map(doc.players.map((p) => [p.id, p] as const));
  const ranked = doc.players.filter((p) => !p.withdrawn && p.edge_sg !== null && (p.probs.win.market ?? 0) >= 0.005 && !isCapped(p.edge_sg));   // priced contenders only: a long shot's clipped 2-stroke gap is noise
  const above = [...ranked].filter((p) => (p.edge_sg ?? 0) > 0).sort((a, b) => (b.edge_sg ?? 0) - (a.edge_sg ?? 0)).slice(0, n);
  const below = [...ranked].filter((p) => (p.edge_sg ?? 0) < 0).sort((a, b) => (a.edge_sg ?? 0) - (b.edge_sg ?? 0)).slice(0, n);
  void byId;
  return { above, below };
}
export function favourites(doc: ExplainDoc, n = 12): ExPlayer[] {
  // In play the favourite is whoever has the best current chance; before the event it is the shortest-priced player in the market.
  const live = doc.kind === "live";
  const score = (p: ExPlayer) => (live ? p.probs.win.model ?? p.probs.win.market : p.probs.win.market ?? p.probs.win.model) ?? 0;
  return [...doc.players].filter((p) => !p.withdrawn).sort((a, b) => score(b) - score(a)).slice(0, n);
}
export function biasForDimension(doc: ExplainDoc, dimension: string | "all", minN = 3): BiasRow[] {
  return doc.bias.filter((r) => r.n >= minN && (dimension === "all" || r.dimension === dimension)).sort((a, b) => Math.abs((b.edge_sg ?? 0) * Math.sqrt(b.n)) - Math.abs((a.edge_sg ?? 0) * Math.sqrt(a.n)));
}
/** A dimension where one group holds nearly everyone ("No travel 127 vs Changed continent 5") says nothing, so it is not offered. */
function trivialDimensions(rows: BiasRow[]): Set<string> {
  const byDim = new Map<string, number[]>();
  for (const r of rows) byDim.set(r.dimension, [...(byDim.get(r.dimension) ?? []), r.n]);
  const out = new Set<string>();
  for (const [dim, ns] of byDim) { const tot = ns.reduce((a, b) => a + b, 0); if (tot > 0 && Math.max(...ns) / tot >= 0.9) out.add(dim); }
  return out;
}
export function biasDimensions(doc: ExplainDoc): Array<{ value: string; label: string }> {
  const skip = trivialDimensions(doc.bias);
  const seen = new Map<string, string>();
  for (const r of doc.bias) if (!skip.has(r.dimension) && !seen.has(r.dimension)) seen.set(r.dimension, r.dimension_label);
  return [...seen].map(([value, label]) => ({ value, label }));
}
/** Our average gap to the market across the whole field (the price tiers partition it). Group bars are shown against this, so a field-wide offset does not make every group look "below the market". */
export function biasBaseline(doc: ExplainDoc): number {
  for (const dim of ["price", "depth", "xtour"]) {
    const rows = doc.bias.filter((r) => r.dimension === dim && r.edge_sg !== null && r.n > 0);
    const n = rows.reduce((a, r) => a + r.n, 0);
    if (rows.length >= 2 && n > 0) return rows.reduce((a, r) => a + (r.edge_sg as number) * r.n, 0) / n;
  }
  return 0;
}
/** Player-type rows for the chart: each group's gap relative to the field baseline, duplicates (the same players under two names) and negligible differences removed. */
export function biasView(doc: ExplainDoc, dimension: string | "all", limit: number, baseline = biasBaseline(doc), minDiff = 0.03): BiasRow[] {
  const skip = trivialDimensions(doc.bias);
  const seen = new Set<string>();
  const rel = biasForDimension(doc, dimension).filter((r) => !skip.has(r.dimension)).filter((r) => {
    const sig = `${r.n}|${r.edge_sg}|${r.p_win_model}`;
    if (seen.has(sig)) return false;
    seen.add(sig);
    return true;
  }).map((r) => ({ ...r, edge_sg: r.edge_sg === null ? null : r.edge_sg - baseline }));
  return rel.filter((r) => Math.abs(r.edge_sg ?? 0) >= minDiff).sort((a, b) => Math.abs((b.edge_sg ?? 0) * Math.sqrt(b.n)) - Math.abs((a.edge_sg ?? 0) * Math.sqrt(a.n))).slice(0, limit);
}
/** One-sentence reading of a bias row, with the gap already relative to the field ("Thin data: compared with the rest of the field, we rate them lower than the market does, mainly on tour strength adjustment"). */
export function biasSentence(r: BiasRow): string {
  const e = r.edge_sg ?? 0;
  const d = r.drivers.filter((x) => Math.abs(x.attr_sg ?? 0) >= 0.01)[0];
  if (Math.abs(e) < 0.03) return `${r.label}: in line with the rest of the field (${r.n} players).`;
  return `${r.label} (${r.n} players): compared with the rest of the field, we rate them ${e > 0 ? "higher" : "lower"} than the market does${d ? `, mainly on ${d.label.toLowerCase()}` : ""}.`;
}
/** Plain-English one-liner for each player-type tab, shown under the tabs. */
export const DIMENSION_HELP: Record<string, string> = {
  all: "The groups where our view differs most from the market.",
  depth: "How many rounds of results we have on a player. Fewer rounds means a less certain rating.",
  xtour: "Whether a player's rating leans on results from other tours (feeder, Asian and so on) rather than PGA or European Tour rounds.",
  price: "Players grouped by how short they are in the betting market.",
  wave: "Players grouped by whether they tee off early or late in round 1.",
  travel: "Whether a player travelled from another continent to play this week.",
  region: "Where the player is from.",
};

/** Leaderboard (live runs): active players by position then win probability; missed-cut / withdrawn players last. */
export function leaderboard(doc: ExplainDoc, n = 15): ExPlayer[] {
  return [...doc.players].filter((p) => p.live && !p.withdrawn).sort((a, b) => {
    const at = a.live?.tier ?? 0;
    const bt = b.live?.tier ?? 0;
    if ((at === 0) !== (bt === 0)) return at === 0 ? -1 : 1;
    return (a.live?.pos_n ?? 999) - (b.live?.pos_n ?? 999) || (b.probs.win.model ?? 0) - (a.probs.win.model ?? 0);
  }).slice(0, n);
}
/** Biggest movers since the last run (by absolute win-probability change). */
export function movers(doc: ExplainDoc, n = 8): ExPlayer[] {
  return doc.players.filter((p) => p.live && Math.abs(p.live.since_last.win ?? 0) > 0).sort((a, b) => Math.abs(b.live?.since_last.win ?? 0) - Math.abs(a.live?.since_last.win ?? 0)).slice(0, n);
}

/* ------------------------------------------------------------------ overlay */
export const MAX_OVERLAY = 5;
/** Preselected comparison: the player's same-odds neighbours (as exported), restricted to players that have a finish shape. */
export function defaultCompetitors(doc: ExplainDoc, id: number, n = 3): number[] {
  const p = doc.players.find((x) => x.id === id);
  if (!p) return [];
  const ok = new Set(doc.players.filter((x) => x.finish && !x.withdrawn).map((x) => x.id));
  return p.neighbors.filter((x) => ok.has(x) && x !== id).slice(0, n);
}
export function toggleCompetitor(list: number[], id: number, max = MAX_OVERLAY - 1): number[] {
  return list.includes(id) ? list.filter((x) => x !== id) : list.length >= max ? list : [...list, id];
}

/** Rounded summary numbers for a player: how far from his market the model is, in the market the user picked. */
export function edgeBadge(p: ExPlayer, market: Market): { tone: "positive" | "negative" | "neutral"; text: string } {
  const e = p.probs[market];
  if (e.rel === null) return { tone: "neutral", text: "no market" };
  const tone = e.rel > 0.02 ? "positive" : e.rel < -0.02 ? "negative" : "neutral";
  const r = Math.round(Math.abs(e.rel * 100));
  return { tone, text: `${r === 0 ? "" : e.rel > 0 ? "+" : "−"}${r}%` };
}

/* ------------------------------------------------------------------ plain-English display layer
 * The publisher writes its own labels ("kalman trend", "cross-tour", "b8"). The views show these plain versions instead; the true lineage stays in the Technical details sections. */
export const COMPONENT_PLAIN: Record<string, { label: string; meaning: string }> = {
  level_form: { label: "Recent scoring level", meaning: "How well they have scored in recent rounds, and whether they are running hot or cold." },
  kalman: { label: "Recent form trend", meaning: "A skill rating that moves up or down after every round, so a hot or cold streak counts." },
  xtour: { label: "Tour strength adjustment", meaning: "Results on other tours (feeder, Asian and others), adjusted for how strong those tours are. It matters most for players with few PGA or European results." },
  category: { label: "Skill by shot type", meaning: "Driving, approach, short game and putting skill, plus driving distance." },
  sit: { label: "Layoff and age", meaning: "How long since they last played, and how age changes what we expect." },
  act: { label: "Recent playing time", meaning: "How much they have played lately." },
  sklv: { label: "Long-run skill estimate", meaning: "A steadier skill estimate that blends their long record with their recent hot or cold run." },
  course: { label: "Course fit and history", meaning: "How well their game suits this course, and how they have done here before." },
  thin: { label: "Small-sample pullback", meaning: "For players with few rounds on record, the rating is pulled toward what is typical for similar players." },
  disp: { label: "Blow-up risk", meaning: "How often they have big bad rounds." },
  other: { label: "Other", meaning: "Small leftovers, such as a flag for players with no history." },
  loc_travel: { label: "Travel and home base", meaning: "The effect of playing near or far from where they usually play, including time zones." },
  loc_nat: { label: "Nationality", meaning: "The effect of playing in or near their home country." },
  loc_refit: { label: "Location correction", meaning: "A small correction so the location effects fit with the skill ratings." },
  override: { label: "Manual adjustment (owner)", meaning: "A one-off change to this player's rating made by the owner on the private site." },
  b8: { label: "Form this week", meaning: "How well they have played in this event so far, which changes what we expect from the remaining rounds." },
  live_other: { label: "Other in-play changes", meaning: "Small leftover live changes." },
  wx_wind: { label: "Weather: wind", meaning: "How the wind affects their tee times." },
  wx_gust: { label: "Weather: gusts", meaning: "How wind gusts affect their tee times." },
  wx_temp: { label: "Weather: temperature", meaning: "How the temperature affects their tee times." },
  wx_rain: { label: "Weather: rain", meaning: "How rain affects their tee times." },
  wx_tod: { label: "Weather: time of day", meaning: "How the time of day (firm greens, light) affects their tee times." },
  wx_traj: { label: "Weather: ball flight", meaning: "Whether a high or low ball flight is helped or hurt by the wind." },
};
const PHRASES: Array<[RegExp, string]> = [
  [/\bin-event form \(b8\)/gi, "form this week"], [/\bb8\b/gi, "form this week"],
  [/\bkalman trend\b/gi, "recent form trend"], [/\bcross-tour dependent\b/gi, "relies on other-tour results"], [/\bx-tour dependent\b/gi, "relies on other-tour results"],
  [/\bcross-tour\b/gi, "tour strength adjustment"], [/\bx-tour\b/gi, "other-tour"], [/\bthin-data shrink\b/gi, "small-sample pullback"],
  [/\blevel and form\b/gi, "recent scoring level"], [/\bcategory skills\b/gi, "skill by shot type"], [/\bskill level\b/gi, "long-run skill estimate"],
  [/\bsituation\b/gi, "layoff and age"], [/\bactivity\b/gi, "recent playing time"], [/\blocation re-fit\b/gi, "location correction"], [/\bowner override\b/gi, "manual adjustment (owner)"],
  [/\bstrokes\/round\b/gi, "strokes a round"], [/\bempirical-bayes\b/gi, "shrunk"], [/\bstate-space\b/gi, "running"], [/\bposterior\b/gi, "estimate"],
];
/** Rewrites publisher jargon into plain words (display only) and uses a real minus sign. */
export function plainText(text: string): string {
  let t = text;
  for (const [re, to] of PHRASES) t = t.replace(re, to);
  return t.replace(/(^|[^\w.])-(?=\d)/g, "$1−");
}
const TAG_LABEL: Record<string, string> = {
  "thin data": "few rounds (under 15)", "light data": "some rounds (15-49)", "deep data": "many rounds (50+)", "x-tour dependent": "relies on other-tour results",
  USA: "American", international: "non-American", amateur: "amateur", early: "early tee time", late: "late tee time",
  "OTT strong": "strong driving", "OTT weak": "weak driving", "approach strong": "strong approach play", "approach weak": "weak approach play",
  "putting hot (regression risk)": "putting running hot", "putting cold": "putting running cold", "hot form": "hot form", "cold form": "cold form", "long layoff": "long layoff",
  "course fit +": "good course fit", "course fit -": "poor course fit", "continent change": "flew in from another continent", "long jet lag": "long jet lag",
};
export const tagLabel = (t: string): string => TAG_LABEL[t] ?? plainText(t);
/** The filter chips, grouped so each kind of filter has its own label. */
export const TAG_GROUPS: Array<{ label: string; help: string; tags: string[] }> = [
  { label: "Rounds on record", help: "How much recent history we have on the player.", tags: ["thin data", "light data", "deep data", "x-tour dependent"] },
  { label: "Nationality", help: "Whether the player is from the USA, and whether they are an amateur.", tags: ["USA", "international", "amateur"] },
  { label: "Tee time", help: "Early or late wave in round 1.", tags: ["early", "late"] },
  { label: "Strengths and weaknesses", help: "Where the player gains or loses strokes.", tags: ["OTT strong", "OTT weak", "approach strong", "approach weak", "putting hot (regression risk)", "putting cold"] },
  { label: "Form", help: "Recent form and time away.", tags: ["hot form", "cold form", "long layoff"] },
  { label: "Course and travel", help: "Fit with this course and effects of travel.", tags: ["course fit +", "course fit -", "continent change", "long jet lag"] },
];
const DIM_LABEL: Record<string, string> = { depth: "Rounds on record", xtour: "Other-tour reliance", price: "Betting price", wave: "Tee-time wave", travel: "Travel", region: "Region" };
const BUCKET_LABEL: Array<[RegExp, string]> = [
  [/^thin data \(<15 rounds\)$/i, "Thin data (under 15 rounds)"], [/^light data \(15-49\)$/i, "Light data (15-49 rounds)"], [/^deep data \(50\+\)$/i, "Deep data (50+ rounds)"],
  [/^cross-tour dependent$/i, "Rely on other-tour results"], [/^core-tour records$/i, "Mostly PGA or European records"],
  [/^early wave r(\d)$/i, "Early tee times, round $1"], [/^late wave r(\d)$/i, "Late tee times, round $1"],
  [/^favourites \(top 10 by market\)$/i, "Favourites (ten shortest prices)"], [/^no travel$/i, "No travel"], [/^changed continent$/i, "Flew in from another continent"],
  [/^usa$/i, "American players"], [/^international$/i, "Non-American players"],
];
function plainBias(rows: BiasRow[]): BiasRow[] {
  return rows.map((r) => {
    let label = r.label;
    for (const [re, to] of BUCKET_LABEL) if (re.test(label)) { label = label.replace(re, to); break; }
    return { ...r, label: plainText(label), dimension_label: DIM_LABEL[r.dimension] ?? plainText(r.dimension_label),
      drivers: (r.drivers ?? []).map((d) => ({ ...d, label: COMPONENT_PLAIN[d.component]?.label ?? plainText(d.label) })) };
  });
}
export const tourName = (t: string | null | undefined): string => (t === "euro" ? "European Tour" : t === "pga" ? "PGA Tour" : t ? t.toUpperCase() : "");

const itemRe = /^\s*(.*?)\s*([+\-−]\d+(?:\.\d+)?)\s*$/;
function parseReasons(list: string): Array<{ label: string; value: number }> {
  return list.split(/[;,]/).map((x) => x.trim().replace(/\.$/, "")).filter(Boolean).flatMap((x) => {
    const m = itemRe.exec(x);
    return m ? [{ label: plainText(m[1]).replace(/^mostly\s+/i, ""), value: Number(m[2].replace("−", "-")) }] : [];
  });
}
const joinList = (xs: string[]): string => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);
/** "kalman trend +0.93; cross-tour +0.62" -> "Rated up by recent form trend and tour strength adjustment." (the numbers have no unit, so they are left out). */
export function plainWhy(text: string): string {
  const items = parseReasons(text);
  if (!items.length) return plainText(text);
  const up = items.filter((i) => i.value > 0).slice(0, 3).map((i) => i.label);
  const down = items.filter((i) => i.value < 0).slice(0, 2).map((i) => i.label);
  return [up.length ? `Rated up by ${joinList(up)}` : "", down.length ? `${up.length ? "held back" : "Held back"} by ${joinList(down)}` : ""].filter(Boolean).join("; ") + ".";
}
/** "We are above the market by 0.26 strokes/round: mostly cross-tour +0.36, kalman trend -0.19." -> one sentence pattern for every player (the size of the gap is in its own column). */
export function plainWhyEdge(text: string | null): string | null {
  if (!text) return null;
  const m = /we are (above|below) the market by ([\d.]+)[^:]*(?::\s*(.*))?$/i.exec(text.trim());
  if (!m) return plainText(text);
  const higher = m[1].toLowerCase() === "above";
  const gap = Number(m[2]);
  const items = m[3] ? parseReasons(m[3]) : [];
  const withGap = items.filter((i) => (i.value > 0) === higher && i.value !== 0);
  const against = items.filter((i) => (i.value > 0) !== higher && i.value !== 0);
  const explained = withGap.reduce((a, i) => a + Math.abs(i.value), 0);
  const lead = `We rate them ${higher ? "higher" : "lower"} than the market`;
  const other = against.length ? `; ${against[0].label} points the other way` : "";
  if (!withGap.length || explained < 0.4 * gap) return `${lead}, with no single clear reason${other}.`;
  return `${lead}, mainly on ${joinList(withGap.slice(0, 2).map((i) => i.label))}${other}.`;
}

/* ---- course card wording */
export function plainCourseBullets(bullets: string[]): string[] {
  const out: string[] = [];
  for (const b of bullets) {
    if (/^venue class\b/i.test(b)) continue;
    let m = /^skill separates players most on par (\d)s/i.exec(b);
    if (m) { out.push(`Skill matters most on par ${m[1]}s.`); continue; }
    m = /^course variance multiplier \(median ([\d.]+)\)/i.exec(b);
    if (m) { const v = Number(m[1]); out.push(v < 0.97 ? "Results should be a bit more bunched than at a typical event." : v > 1.03 ? "Results should be more spread out than at a typical event." : "Results should be about as spread out as at a typical event."); continue; }
    m = /^field-average scoring [^;]*;\s*([\d.]+) birdies\+ and ([\d.]+) bogeys\+ per round/i.exec(b);
    if (m) { out.push(`The average player makes ${m[1]} birdies or better and ${m[2]} bogeys or worse a round.`); continue; }
    out.push(plainText(b));
  }
  return out;
}
export const plainHoleSource = (s: string | null | undefined): string => (!s ? "" : /prior/i.test(s) ? "Hole data comes from the previous edition of this event." : `Hole data source: ${plainText(s)}.`);

/* ---- weather wording */
const WX_NAME: Record<string, string> = { wx_wind: "wind", wx_gust: "gusts", wx_temp: "temperature", wx_rain: "rain", wx_tod: "time of day", wx_traj: "ball flight" };
/** Which weather effects actually move prices in this run (some player has a non-zero component), and which are listed but zero for everyone. */
export function weatherCoverage(doc: ExplainDoc): { included: string[]; excluded: string[] } {
  const keys = Object.keys(doc.components_legend).filter((k) => k.startsWith("wx_"));
  const used = (k: string) => doc.players.some((p) => Math.abs(p.components[k] ?? 0) >= 0.005);
  return { included: keys.filter(used).map((k) => WX_NAME[k] ?? k), excluded: keys.filter((k) => !used(k)).map((k) => WX_NAME[k] ?? k) };
}
export function weatherStatus(doc: ExplainDoc): { short: string; long: string } {
  const w = doc.course_card.weather;
  const { included, excluded } = weatherCoverage(doc);
  if (!w || (!w.label && !w.status && !included.length)) return { short: "", long: "" };
  if (!included.length) return { short: "Weather not in prices", long: "Weather is not included in these prices." };
  if (!excluded.length) return { short: "Weather included", long: `Weather is included in the prices: ${joinList(included)}.` };
  return { short: "Weather partly included", long: `In the prices: ${joinList(included)}. Not in the prices: ${joinList(excluded)}.` };
}
/** Cut-line shift rows worth showing: only events with a cut, only up to the cut round, and only rounds still to come when the event is in play. */
export function cutLineRows(doc: ExplainDoc): Array<{ round: number; shift: number | null }> {
  const e = doc.event;
  if (!e.cut_round || e.cut_round < 1) return [];
  const after = doc.kind === "live" ? doc.after_round ?? 0 : 0;
  return (doc.course_card.weather?.field ?? []).filter((f) => f.round <= (e.cut_round as number) && f.round > after).map((f) => ({ round: f.round, shift: f.cut_line_shift }));
}
export const availableMarkets = (doc: ExplainDoc): Market[] => MARKETS.filter((m) => m !== "make_cut" || !!doc.event.cut_round);
