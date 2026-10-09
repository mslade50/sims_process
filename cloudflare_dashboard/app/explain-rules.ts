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
    out[k] = isObject(v) ? { ...e, meaning: typeof e.meaning === "string" ? neutralPronouns(e.meaning) : "" } : e;
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
      finish: isObject(p.finish) && Array.isArray((p.finish as { pos?: unknown }).pos) ? p.finish : null, live: isObject(p.live) ? p.live : null, why: typeof p.why === "string" ? neutralPronouns(p.why) : "",
      why_edge: typeof p.why_edge === "string" ? neutralPronouns(p.why_edge) : null } as unknown as ExPlayer;
  });
  return { ...d, players, bias: Array.isArray(d.bias) ? d.bias : [], top_edges: Array.isArray(d.top_edges) ? d.top_edges : [], components_legend: neutralLegend(d.components_legend),
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
export const pct = (p: number | null | undefined, digits = 1): string => (p === null || p === undefined ? "-" : p < 0.001 ? `${(p * 100).toFixed(2)}%` : `${(p * 100).toFixed(digits)}%`);
export const signed = (v: number | null | undefined, digits = 2): string => (v === null || v === undefined ? "-" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`);
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
export function waterfall(player: ExPlayer, legend: Record<string, LegendEntry>, minAbs = 0.015): { rows: WaterfallRow[]; total: number; small: number; smallCount: number } {
  const rows: WaterfallRow[] = [];
  let small = 0;
  let smallCount = 0;
  for (const [key, value] of Object.entries(player.components)) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const lg = legend[key] ?? { label: key, group: "skill", meaning: "" };
    if (Math.abs(value) < minAbs) { small += value; smallCount += 1; continue; }
    rows.push({ key, label: lg.label, group: lg.group, value, meaning: lg.meaning });
  }
  rows.sort((a, b) => GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group) || Math.abs(b.value) - Math.abs(a.value));
  const total = rows.reduce((s, r) => s + r.value, 0) + small;
  return { rows, total, small, smallCount };
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
  const ranked = doc.players.filter((p) => !p.withdrawn && p.edge_sg !== null && (p.probs.win.market ?? 0) >= 0.0005);   // priced contenders: a 0.01% long shot's 2-stroke gap is noise
  const above = [...ranked].filter((p) => (p.edge_sg ?? 0) > 0).sort((a, b) => (b.edge_sg ?? 0) - (a.edge_sg ?? 0)).slice(0, n);
  const below = [...ranked].filter((p) => (p.edge_sg ?? 0) < 0).sort((a, b) => (a.edge_sg ?? 0) - (b.edge_sg ?? 0)).slice(0, n);
  void byId;
  return { above, below };
}
export function favourites(doc: ExplainDoc, n = 12): ExPlayer[] {
  return [...doc.players].filter((p) => !p.withdrawn).sort((a, b) => (b.probs.win.market ?? b.probs.win.model ?? 0) - (a.probs.win.market ?? a.probs.win.model ?? 0)).slice(0, n);
}
export function biasForDimension(doc: ExplainDoc, dimension: string | "all", minN = 3): BiasRow[] {
  return doc.bias.filter((r) => r.n >= minN && (dimension === "all" || r.dimension === dimension)).sort((a, b) => Math.abs((b.edge_sg ?? 0) * Math.sqrt(b.n)) - Math.abs((a.edge_sg ?? 0) * Math.sqrt(a.n)));
}
export function biasDimensions(doc: ExplainDoc): Array<{ value: string; label: string }> {
  const seen = new Map<string, string>();
  for (const r of doc.bias) if (!seen.has(r.dimension)) seen.set(r.dimension, r.dimension_label);
  return [...seen].map(([value, label]) => ({ value, label }));
}
/** One-sentence reading of a bias row ("We are 0.4 strokes/round below the market on thin data, mostly cross-tour"). */
export function biasSentence(r: BiasRow): string {
  const e = r.edge_sg ?? 0;
  if (Math.abs(e) < 0.03) return `${r.label}: in line with the market (${r.n} players).`;
  const d = r.drivers.filter((x) => Math.abs(x.attr_sg ?? 0) >= 0.01)[0];
  return `${r.label}: we are ${Math.abs(e).toFixed(2)} strokes/round ${e > 0 ? "above" : "below"} the market (${r.n} players)${d ? `, mostly ${d.label.toLowerCase()} (${signed(d.attr_sg)})` : ""}.`;
}

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
  return { tone, text: `${e.rel > 0 ? "+" : e.rel < 0 ? "−" : ""}${Math.abs(e.rel * 100).toFixed(0)}%` };
}
