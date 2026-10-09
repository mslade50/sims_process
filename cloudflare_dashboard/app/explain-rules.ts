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

/* ------------------------------------------------------------------ where we differ from the market: contenders and the shape of their chances
 * Mirrors golfprice/explain_export.py (_shape_map / _invert_shape / _player_shape / _differ_block). The exported `shape` and `differ` fields are used when
 * present; otherwise the same numbers are computed here from the published chances, skill and spread (documents published before October 9, 2026).
 *
 * Method: across the whole field our chance in each bet type follows our skill and spread closely: logit(chance) = s * (skill - q) / spread, fitted per bet
 * type on our own chances (typically 96-99% of the variation). Running a player's sportsbook chances back through the same curves gives the skill and spread
 * the market's prices imply; doing the same with our own chances and taking the difference cancels the approximation, so the gaps are like for like. */
export type ShapeMap = Partial<Record<Market, { s: number; q: number; n: number }>>;
export type MarketEdge = { model: number; market: number; rel: number; side: "yes" | "miss_cut" | null; ev: number | null; tone: "pos" | "neg" | "flat" };
export type ShapePart = { key: string; label: string; value: number };
export type Shape = {
  skill_ours: number; skill_market: number; skill_gap: number; spread_ours: number; spread_market: number; spread_gap: number;
  markets: Partial<Record<Market, MarketEdge>>; higher: Market[]; lower: Market[]; pattern: string; reason: string; sentence: string;
  parts: ShapePart[]; base_gap: number; parts_sentence: string;
};
export type DifferGroup = { key: string; label: string; n: number; win_ours: number; win_market: number; win_rel: number; top20_ours: number | null; top20_market: number | null; top20_rel: number | null; skill_gap: number; spread_gap: number };
export type Differ = { rule: string; ids: number[]; groups: DifferGroup[]; spread_gap_median: number | null; skill_gap_median: number | null; source: "export" | "page" };
type Row = { player: ExPlayer; shape: Shape };

const SHAPE_P_LO = 0.002;
const SHAPE_P_HI = 0.995;
export const EDGE_MIN = 0.05;       // a gap under 5% of the sportsbook's chance counts as "in line"
export const SKILL_MIN = 0.05;      // strokes a round
export const SPREAD_MIN = 0.08;     // strokes a round of round-to-round spread
export const SHAPE_METHOD = "Where we differ: for each bet type, our chances across the whole field follow a smooth curve in each player's skill and spread (round-to-round swing). Running a player's sportsbook chances back through the same curves gives the skill and spread its prices imply; doing the same with our own chances and taking the difference keeps the comparison like for like. The skill gap is split into our week-specific adjustments (course, travel and home country, weather and tee times, owner adjustment) and the rest, which is a difference in base rating. Groups add up chances among the contenders. This is an explanation only; prices are unchanged.";
export const CONTENDER_N = 25;
export const CONTENDER_MIN_ROUNDS = 50;
const MARKET_WORD: Record<Market, string> = { win: "win", top_5: "top 5", top_10: "top 10", top_20: "top 20", make_cut: "make cut" };
const lgt = (p: number): number => Math.log(p / (1 - p));
const okP = (p: number | null | undefined): p is number => typeof p === "number" && Number.isFinite(p) && p > SHAPE_P_LO && p < SHAPE_P_HI;
const weatherSum = (p: ExPlayer): number => Object.entries(p.components).reduce((a, [k, v]) => a + (k.startsWith("wx_") && typeof v === "number" ? v : 0), 0);
/** The skill our chances are built on: expected skill against the field plus the weather and tee-time effect (which is in our chances too). */
export const effectiveSkill = (p: ExPlayer): number | null => (p.mu_rel === null || p.mu_rel === undefined ? null : p.mu_rel + weatherSum(p));

/** Per bet type: least squares of logit(our chance) on skill/spread and 1/spread across the field (no intercept), giving the curve's steepness s and midpoint q. */
export function shapeMap(doc: ExplainDoc): ShapeMap {
  const out: ShapeMap = {};
  const act = doc.players.filter((p) => !p.withdrawn && typeof p.sd === "number" && p.sd > 0 && effectiveSkill(p) !== null);
  for (const m of MARKETS) {
    let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0, n = 0;
    for (const p of act) {
      const pm = p.probs[m].model;
      if (!okP(pm)) continue;
      const sd = p.sd as number;
      const x1 = (effectiveSkill(p) as number) / sd, x2 = 1 / sd, y = lgt(pm);
      a11 += x1 * x1; a12 += x1 * x2; a22 += x2 * x2; b1 += x1 * y; b2 += x2 * y; n += 1;
    }
    const det = a11 * a22 - a12 * a12;
    if (n < 10 || Math.abs(det) < 1e-12) continue;
    const s = (b1 * a22 - b2 * a12) / det;
    const c = (a11 * b2 - a12 * b1) / det;
    if (!(s > 0.05)) continue;
    out[m] = { s, q: -c / s, n };
  }
  return out;
}

/** Skill and spread that reproduce a set of chances through the field curves: each bet type gives skill - spread * logit(p) / s = q; least squares over the bet types. */
export function invertShape(probs: Partial<Record<Market, number>>, map: ShapeMap): { mu: number; sd: number } | null {
  const A: number[] = [], Q: number[] = [];
  for (const m of MARKETS) {
    const c = map[m];
    const p = probs[m];
    if (!c || !okP(p)) continue;
    A.push(lgt(p) / c.s); Q.push(c.q);
  }
  if (A.length < 3) return null;
  const ma = A.reduce((x, y) => x + y, 0) / A.length, mq = Q.reduce((x, y) => x + y, 0) / Q.length;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < A.length; i++) { sxx += (A[i] - ma) ** 2; sxy += (A[i] - ma) * (Q[i] - mq); }
  if (sxx < 1e-9) return null;
  const sd = -sxy / sxx;
  const mu = mq + sd * ma;
  return Number.isFinite(mu) && Number.isFinite(sd) && sd > 0.5 && sd < 8 ? { mu, sd } : null;
}

const joinWords = (xs: string[]): string => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);
const WEEK_PARTS: Array<{ key: string; label: string; match: (k: string) => boolean }> = [
  { key: "course", label: "course fit and history", match: (k) => k === "course" },
  { key: "location", label: "travel and home country", match: (k) => k.startsWith("loc_") },
  { key: "weather", label: "weather and tee times", match: (k) => k.startsWith("wx_") },
  { key: "override", label: "owner adjustment", match: (k) => k === "override" },
];

/** Bet-by-bet comparison, the pattern sentence and the additive split of the skill gap for one player (week documents with sportsbook chances only). */
export function playerShape(p: ExPlayer, map: ShapeMap, markets: readonly Market[] = MARKETS): Shape | null {
  const skill = effectiveSkill(p);
  if (skill === null || typeof p.sd !== "number") return null;
  const pm: Partial<Record<Market, number>> = {}, pk: Partial<Record<Market, number>> = {};
  const edges: Partial<Record<Market, MarketEdge>> = {};
  for (const m of markets) {
    const e = p.probs[m];
    if (e.model === null || e.market === null || !(e.market > 0) || !(e.market < 1)) continue;
    const rel = e.model / e.market - 1;
    const miss = m === "make_cut" && e.model < e.market ? (1 - e.model) / (1 - e.market) - 1 : null;
    const ev = rel > 0 ? rel : miss;
    const sig = m === "make_cut" ? (ev ?? 0) >= EDGE_MIN : Math.abs(rel) >= EDGE_MIN;
    edges[m] = { model: e.model, market: e.market, rel, side: rel > 0 ? "yes" : miss !== null ? "miss_cut" : null, ev, tone: !sig ? "flat" : e.model > e.market ? "pos" : "neg" };
    if (okP(e.model) && okP(e.market) && map[m]) { pm[m] = e.model; pk[m] = e.market; }
  }
  const fm = invertShape(pm, map), fk = invertShape(pk, map);
  if (!fm || !fk) return null;
  const skillGap = fm.mu - fk.mu, spreadGap = fm.sd - fk.sd;
  const avail = markets.filter((m) => edges[m]);
  const higher = avail.filter((m) => edges[m]?.tone === "pos"), lower = avail.filter((m) => edges[m]?.tone === "neg");
  const rest = avail.filter((m) => edges[m]?.tone === "flat");
  // consecutive runs of three or more bet types read as a range ("win to top 20")
  const words = (ms: Market[]) => {
    const idx = ms.map((m) => avail.indexOf(m));
    return ms.length >= 3 && idx.every((v, i) => i === 0 || v === idx[i - 1] + 1) ? `${MARKET_WORD[ms[0]]} to ${MARKET_WORD[ms[ms.length - 1]]}` : joinWords(ms.map((m) => MARKET_WORD[m]));
  };
  let pattern: string;
  if (!higher.length && !lower.length) pattern = "In line with the market on every bet";
  else if (!lower.length) pattern = higher.length === avail.length ? "Higher across the board" : `Higher on ${words(higher)}, in line on ${words(rest)}`;
  else if (!higher.length) pattern = lower.length === avail.length ? "Lower across the board" : `Lower on ${words(lower)}, in line on ${words(rest)}`;
  else pattern = `Higher on ${words(higher)}, lower on ${words(lower)}`;
  const sk = Math.abs(skillGap) >= SKILL_MIN, sp = Math.abs(spreadGap) >= SPREAD_MIN;
  const spreadMarket = p.sd - spreadGap;
  const spreadTxt = `spread ${p.sd.toFixed(2)} vs the market's ${spreadMarket.toFixed(2)}`;
  const steady = spreadGap < 0;
  // the "so it shows more in ..." clause only when the bet-by-bet pattern itself shows it (win side weaker or stronger than the top 20 / make cut side)
  const rank = (m: Market) => (edges[m] ? { neg: -1, flat: 0, pos: 1 }[edges[m]!.tone] : null);
  const winSide = rank("win");
  const placeSide = Math.max(...(["top_20", "make_cut"] as Market[]).map(rank).filter((v): v is -1 | 0 | 1 => v !== null), -2);
  const places = avail.includes("make_cut") ? "top 20 and make cut" : "top 20";
  const consequence = !sp || winSide === null || placeSide === -2 ? "" : steady && winSide < placeSide ? `, so it shows more in ${places} than in wins` : !steady && winSide > placeSide ? `, so it shows more in wins than in ${places}` : "";
  let reason: string;
  if (sk && !sp) reason = `we rate their skill ${skillGap > 0 ? "higher" : "lower"} (${signed(skillGap)} strokes a round)`;
  else if (!sk && sp) reason = `same skill as the market, but we see a ${steady ? "steadier, lower-ceiling" : "more boom-or-bust"} player (${spreadTxt})${consequence}`;
  else if (sk && sp) reason = `we rate them ${skillGap > 0 ? "better" : "worse"} (${signed(skillGap)} strokes a round) and ${steady ? "steadier" : "more boom-or-bust"} (${spreadTxt})${consequence}`;
  else reason = "no clear difference in skill or spread, so the gap is specific to that bet";
  if (!higher.length && !lower.length) reason = "";
  const sentence = reason ? `${pattern}: ${reason}.` : `${pattern}.`;
  // additive split of the skill gap: our week-specific adjustments, then whatever is left over (a different base rating)
  const parts: ShapePart[] = WEEK_PARTS.map((w) => ({ key: w.key, label: w.label, value: Object.entries(p.components).reduce((a, [k, v]) => a + (w.match(k) && typeof v === "number" ? v : 0), 0) }));
  return { skill_ours: skill, skill_market: skill - skillGap, skill_gap: skillGap, spread_ours: p.sd, spread_market: spreadMarket, spread_gap: spreadGap,
    markets: edges, higher, lower, pattern, reason, sentence, parts, ...partsSplit(skillGap, parts) };
}

/** "Skill gap +0.10 a round: course fit and history +0.04 and weather +0.02 come from our week-specific adjustments; the other +0.04 is ..." (the pieces add up to the gap). */
export function partsSplit(skillGap: number, parts: ShapePart[]): { base_gap: number; parts_sentence: string } {
  const partsTotal = parts.reduce((a, x) => a + x.value, 0);
  const base = skillGap - partsTotal;
  const shown = parts.filter((x) => Math.abs(x.value) >= 0.02).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const small = partsTotal - shown.reduce((a, x) => a + x.value, 0);
  const items = shown.map((x) => `${x.label} ${signed(x.value)}`);
  if (items.length && Math.abs(small) >= 0.005) items.push(`smaller week items ${signed(small)}`);
  const lead = `Skill gap ${signed(skillGap)} a round`;
  const rest = Math.abs(base) < 0.02 ? "nothing is left over" : base > 0 ? `the other ${signed(base)} is a higher base rating than the market's (recent form and long-run skill)` : `the market rates their base form ${Math.abs(base).toFixed(2)} higher than we do`;
  const s = items.length
    ? `${lead}: ${joinWords(items)} come from our week-specific adjustments; ${rest}.`
    : `${lead}: our course, travel, weather and owner adjustments add ${signed(partsTotal)}; ${rest}.`;
  return { base_gap: base, parts_sentence: s };
}

/** The exported shape when the document carries one, otherwise computed here. */
export function shapeOf(p: ExPlayer, map: ShapeMap, markets: readonly Market[] = MARKETS): Shape | null {
  const ex = (p as ExPlayer & { shape?: Shape | null }).shape;
  return ex && typeof ex === "object" && typeof ex.skill_gap === "number" && typeof ex.sentence === "string" && isObject(ex.markets) ? ex : playerShape(p, map, markets);
}

export const CONTENDER_RULE = `The ${CONTENDER_N} players with the best average win chance (ours and the sportsbook's), leaving out anyone with fewer than ${CONTENDER_MIN_ROUNDS} rounds on record unless the sportsbook has them in its top 10.`;
/** "Players who matter": ranked by the average of our and the sportsbook's win chance; thin records out unless the sportsbook has them in its top 10. */
export function contenders(doc: ExplainDoc, map: ShapeMap, n = CONTENDER_N, minRounds = CONTENDER_MIN_ROUNDS): Row[] {
  const act = doc.players.filter((p) => !p.withdrawn && p.probs.win.market !== null);
  const mrank = new Map([...act].sort((a, b) => (b.probs.win.market ?? 0) - (a.probs.win.market ?? 0)).map((p, i) => [p.id, i + 1] as const));
  const score = (p: ExPlayer) => ((p.probs.win.market ?? 0) + (p.probs.win.model ?? p.probs.win.market ?? 0)) / 2;
  const markets = availableMarkets(doc);
  const out: Row[] = [];
  for (const p of [...act].sort((a, b) => score(b) - score(a) || a.id - b.id)) {
    if (out.length >= n) break;
    const thin = typeof p.n_prior_rounds === "number" && p.n_prior_rounds < minRounds;
    if (thin && (mrank.get(p.id) ?? 999) > 10) continue;
    const shape = shapeOf(p, map, markets);
    if (shape) out.push({ player: p, shape });
  }
  return out;
}

const median = (xs: number[]): number | null => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
/** Player-type groups among the contenders only, compared by summed chances (sums of chances net out honestly), with the average skill and spread gaps; small differences hidden. */
export function differGroups(rows: Row[], minN = 3, minWin = 0.1, minTop20 = 0.05, limit = 6): DifferGroup[] {
  const keys = new Set<string>();
  rows.forEach((r) => r.player.tags.forEach((t) => keys.add(t)));
  rows.forEach((r) => { if (r.player.wave === "early" || r.player.wave === "late") keys.add(r.player.wave); });
  const out: DifferGroup[] = [];
  for (const key of [...keys].sort()) {
    const g = rows.filter((r) => playerHasTag(r.player, key));
    if (g.length < minN || g.length >= rows.length) continue;
    const total = (f: (r: Row) => number | null): number | null => { let s = 0; for (const r of g) { const v = f(r); if (v === null) return null; s += v; } return s; };
    const wo = total((r) => r.player.probs.win.model), wm = total((r) => r.player.probs.win.market);
    if (wo === null || wm === null || !(wm > 0)) continue;
    const to = total((r) => r.player.probs.top_20.model), tm = total((r) => r.player.probs.top_20.market);
    const winRel = wo / wm - 1;
    const t20Rel = to !== null && tm !== null && tm > 0 ? to / tm - 1 : null;
    if (Math.abs(winRel) < minWin && Math.abs(t20Rel ?? 0) < minTop20) continue;
    out.push({ key, label: tagLabel(key), n: g.length, win_ours: wo, win_market: wm, win_rel: winRel, top20_ours: to, top20_market: tm, top20_rel: t20Rel,
      skill_gap: g.reduce((a, r) => a + r.shape.skill_gap, 0) / g.length, spread_gap: g.reduce((a, r) => a + r.shape.spread_gap, 0) / g.length });
  }
  return out.sort((a, b) => Math.abs(b.win_rel) * Math.sqrt(b.n) - Math.abs(a.win_rel) * Math.sqrt(a.n) || a.key.localeCompare(b.key)).slice(0, limit);
}

/** Everything the "Where we differ" panel needs: the exported block when present, the page-side computation otherwise. */
export function differView(doc: ExplainDoc): { rows: Row[]; differ: Differ } {
  if (doc.kind === "live") return { rows: [], differ: { rule: CONTENDER_RULE, ids: [], groups: [], spread_gap_median: null, skill_gap_median: null, source: "page" } };
  const map = shapeMap(doc);
  const ex = (doc as ExplainDoc & { differ?: Partial<Differ> | null }).differ;
  if (ex && isObject(ex) && Array.isArray(ex.ids) && ex.ids.length) {
    const byId = new Map(doc.players.map((p) => [p.id, p] as const));
    const markets = availableMarkets(doc);
    const rows = ex.ids.map((id) => byId.get(id)).filter((p): p is ExPlayer => !!p).flatMap((p) => { const s = shapeOf(p, map, markets); return s ? [{ player: p, shape: s }] : []; });
    const groups = Array.isArray(ex.groups) ? ex.groups.map((g) => ({ ...g, label: tagLabel(g.key) })) : differGroups(rows);
    return { rows, differ: { rule: typeof ex.rule === "string" ? ex.rule : CONTENDER_RULE, ids: ex.ids, groups, spread_gap_median: ex.spread_gap_median ?? median(rows.map((r) => r.shape.spread_gap)),
      skill_gap_median: ex.skill_gap_median ?? median(rows.map((r) => r.shape.skill_gap)), source: "export" } };
  }
  const rows = contenders(doc, map);
  return { rows, differ: { rule: CONTENDER_RULE, ids: rows.map((r) => r.player.id), groups: differGroups(rows), spread_gap_median: median(rows.map((r) => r.shape.spread_gap)),
    skill_gap_median: median(rows.map((r) => r.shape.skill_gap)), source: "page" } };
}

/** Text for one bet-type cell: how far our chance is from the no-margin sportsbook chance, and on make cut the miss-cut side when that is where the value is. */
export function edgeCellText(e: MarketEdge | undefined): string {
  if (!e) return "-";
  if (e.side === "miss_cut" && e.tone === "neg") return `miss cut +${Math.round((e.ev ?? 0) * 100)}%`;
  const r = Math.round(Math.abs(e.rel) * 100);
  return r === 0 ? "0%" : `${e.rel > 0 ? "+" : "−"}${r}%`;
}
/** Bets where our chance beats the sportsbook's by at least EDGE_MIN (the side to back). */
export function valueBets(s: Shape): string[] {
  return MARKETS.flatMap((m) => {
    const e = s.markets[m];
    if (!e || e.ev === null || e.ev < EDGE_MIN) return [];
    return [e.side === "miss_cut" ? "miss cut" : MARKET_WORD[m]];
  });
}
/** One-line group reading. */
export function groupSentence(g: DifferGroup): string {
  const r = (v: number) => `${v > 0 ? "+" : "−"}${Math.round(Math.abs(v) * 100)}%`;
  return `${g.label} (${g.n}): together ${pct(g.win_ours)} to win with us vs ${pct(g.win_market)} in the sportsbook (${r(g.win_rel)})${g.top20_rel !== null ? `, top 20 ${r(g.top20_rel)}` : ""}.`;
}
