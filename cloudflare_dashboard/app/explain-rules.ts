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
  /** DataGolf reference (week documents from October 15, 2026): their pre-tournament chances and skill estimate. Reference only, never in our price. */
  dg?: DgRef | null;
  /** "Why we differ" block (week documents from explain version 11): ours vs Data Golf vs the sportsbooks, the Data Golf odds check, the edge split and the price peers. Absent on older documents, live runs and withdrawn players. */
  diverge?: Diverge | null;
  /** Adjusted strokes gained a round against an average PGA Tour field, from rounds before the event (version 11 week documents). */
  form?: Form | null;
};
/** DataGolf's chances and skill estimate for one player. Skills are strokes a round against the average of the players both models rate (before weather); skill_gap = ours - DataGolf's. */
export type DgRef = Partial<Record<Market, number | null>> & { skill: number | null; skill_ours: number | null; skill_gap: number | null };
export type LegendEntry = { label: string; group: string; meaning: string;
  /** Set when this week has no data for the component ("fit and history", "fit" or "history"); the label and meaning are then already relabelled. */
  unavailable?: string };

/* ---- "Why we differ" (explain version 11). Parsed defensively: a malformed block is null and the view skips it. */
export type DivergeVerdict = "all_agree" | "ours_outlier" | "dg_outlier" | "book_outlier" | "all_differ" | "no_dg" | "agree_book_no_dg" | "no_book" | "insufficient";
export type Outlier = "ours" | "dg" | "book";
export type SourceChances = { ours: number | null; dg: number | null; book: number | null };
export type DivergeCompare = { markets: Partial<Record<Market, SourceChances>>; mean_logit_gap: { ours_dg: number | null; ours_book: number | null; dg_book: number | null }; verdict: DivergeVerdict; outlier: Outlier | null };
export type DgCheck = { skill_stated: number | null; skill_from_odds: number | null; stated_minus_odds: number | null; market: string | null; flag: boolean; skill_ours: number | null; gap_vs_odds: number | null;
  /** Ours minus Data Golf's stated skill number; gap_vs_odds is the same gap measured on chances, in Data Golf strokes. */
  skill_number_gap: number | null; sentence: string | null };
/** Our gap to the market (strokes a round, + = we are above). kind "in_line" and "long_shot" carry only the sentence; "normal" also splits the gap against the closest-priced players:
 *  particular = edge - reference_edge; ours_vs_peers / book_vs_peers = our / the sportsbooks' implied skill minus the peers' median. */
export type EdgeSplit = { kind: "in_line" | "long_shot" | "normal"; edge: number | null; reference: "peers" | "none"; reference_edge: number | null; particular: number | null; ours_vs_peers: number | null; book_vs_peers: number | null; sentence: string };
export type PeerPart = { key: string; label: string; value: number | null; peer_median: number | null; distinctive: number | null };
export type PeerSet = { ids: number[]; names: string[]; peer_edge_median: number | null; parts: PeerPart[]; sentence: string };
export type Diverge = { compare: DivergeCompare | null; dg_check: DgCheck | null; edge_split: EdgeSplit | null; peers: PeerSet | null; sentences: string[] };
export const FORM_WINDOWS = ["L8", "L24", "L50", "L100"] as const;
export type FormWindow = (typeof FORM_WINDOWS)[number];
export type Form = { n: number | null; last_round: string | null; windows: Partial<Record<FormWindow, { mean: number | null; n: number | null }>>; seasons: Array<{ year: number; mean: number | null; n: number | null }> };
export type Divergence = {
  version: number | null;
  course_data: { n_players: number | null; fit_available: number | null; history_available: number | null; venue_editions: number | null; blank: boolean; label: string | null; note: string | null;
    /** What is missing this week: "fit and history", "fit", "history" or "some players"; null when nothing is. */
    unavailable: string | null } | null;
  dg_odds_fit: { market: string | null; slope: number | null; intercept: number | null; n: number | null; resid_sd_skill: number | null; flag_threshold: number | null; n_flagged: number | null; flagged: number[] } | null;
  field_edge_mean: number | null; form_source: { table: string | null; before: string | null; available: boolean; scale: string | null } | null; method: string | null;
};
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
    waves?: Array<{ round: number; wave: string; rel: number | null; mean_wind: number | null }>; field?: Array<{ round: number; common: number | null; cut_line_shift: number | null }>;
    /** Per round: whether tee times were available ("ok") or only the wave-free field-wide shift applies ("common_only"). Published from explain v8 on. */
    rounds?: Array<{ round: number; status: string | null; tee_coverage: number | null; reason: string | null; venue_seen: boolean | null }>; m_v?: number | null; wind_prior_source?: string | null; layer?: string | null };
};
export type FieldScores = {
  rounds: Array<{ round: number; mean_strokes: number; mean_vs_par: number; players: number; basis: "simulated" | "actual" }>;
  winning: { mode: number; mode_share: number; median: number; mean: number; p10: number; p90: number; n_draws: number } | null;
};
export type ExplainDoc = {
  schema: string; kind: "week" | "live"; event_uid: string; run: string; as_of: string | null; after_round: number | null;
  event: { name: string; tour: string | null; course: string | null; date_start: string | null; rounds: number | null; field_size: number | null; par_per_round: number[]; cut_round: number | null; cut_top_n: number | null; cut_rule: string | null;
    /** Field mean skill, strokes a round vs an average PGA Tour player (+ = stronger). */
    field_strength?: number | null; field_strength_detail?: { field_offset?: number | null; n_field?: number | null; n_imputed?: number | null; typical_ours_minus_dg?: { mean?: number | null } | null } | null };
  model: { weather_layer: string | null; overrides_applied: string[]; n_sims: number | null };
  components_legend: Record<string, LegendEntry>; component_field_mean: Record<string, number | null>;
  course_card: CourseCard;
  finish: { method: string; n_draws?: number; reason: string | null; calibration?: Record<string, { mean_abs: number; max_abs: number }> };
  /** Field-level view of the same simulation draws (explain export from October 13, 2026): scoring average per round and the winning score. */
  field_scores?: FieldScores | null;
  edge_model: { ridge_r2: number | null; strokes_per_logit: Record<string, number>; note: string };
  /** Version 11 week documents: how course data, Data Golf's odds fit and form were sourced. */
  divergence?: Divergence | null;
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
    // A relabelled entry (nothing to go on this week) keeps the label and meaning the publisher wrote for it.
    const own = isObject(v) && typeof e.unavailable === "string" && e.unavailable !== "";
    out[k] = isObject(v) ? { ...e, label: (own ? undefined : plain?.label) ?? plainText(String(e.label ?? k)), meaning: (own ? undefined : plain?.meaning) ?? (typeof e.meaning === "string" ? plainText(neutralPronouns(e.meaning)) : "") } : e;
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
      why_edge: typeof p.why_edge === "string" ? (isObject(p.diverge) && isObject(p.diverge.edge_split) ? neutralPronouns(p.why_edge) : plainWhyEdge(neutralPronouns(p.why_edge))) : null, dg: parseDg(p.dg),
      diverge: parseDiverge(p.diverge), form: parseForm(p.form) } as unknown as ExPlayer;
  });
  return { ...d, players, bias: Array.isArray(d.bias) ? plainBias(d.bias) : [], top_edges: Array.isArray(d.top_edges) ? d.top_edges : [], components_legend: neutralLegend(d.components_legend),
    finish: isObject(d.finish) ? d.finish : { method: "unavailable", reason: null }, course_card: (isObject(d.course_card) ? d.course_card : { holes: [], what_drives_scoring: [], by_par: {}, weather: {} }) as CourseCard,
    edge_model: isObject(d.edge_model) ? d.edge_model : { ridge_r2: null, strokes_per_logit: {}, note: "" }, since_last: isObject(d.since_last) ? d.since_last : null,
    field_scores: parseFieldScores(d.field_scores), divergence: parseDivergence(d.divergence) };
}

const str = (v: unknown): string | null => (typeof v === "string" && v !== "" ? v : null);
const OUTLIERS: readonly string[] = ["ours", "dg", "book"];
const VERDICTS: readonly string[] = ["all_agree", "ours_outlier", "dg_outlier", "book_outlier", "all_differ", "no_dg", "agree_book_no_dg", "no_book", "insufficient"];
function parseCompare(raw: unknown): DivergeCompare | null {
  if (!isObject(raw)) return null;
  const mk = isObject(raw.markets) ? raw.markets : {};
  const markets: DivergeCompare["markets"] = {};
  for (const m of MARKETS) {
    const e = mk[m];
    if (isObject(e)) markets[m] = { ours: num(e.ours), dg: num(e.dg), book: num(e.book) };
  }
  const g = isObject(raw.mean_logit_gap) ? raw.mean_logit_gap : {};
  const verdict = (typeof raw.verdict === "string" && VERDICTS.includes(raw.verdict) ? raw.verdict : "insufficient") as DivergeVerdict;
  const outlier = typeof raw.outlier === "string" && OUTLIERS.includes(raw.outlier) ? (raw.outlier as Outlier) : null;
  return Object.keys(markets).length || verdict !== "insufficient" ? { markets, mean_logit_gap: { ours_dg: num(g.ours_dg), ours_book: num(g.ours_book), dg_book: num(g.dg_book) }, verdict, outlier } : null;
}
function parseDgCheck(raw: unknown): DgCheck | null {
  if (!isObject(raw)) return null;
  const flag = raw.flag === true;
  return { skill_stated: num(raw.skill_stated), skill_from_odds: num(raw.skill_from_odds), stated_minus_odds: num(raw.stated_minus_odds), market: str(raw.market), flag, skill_ours: num(raw.skill_ours),
    gap_vs_odds: num(raw.gap_vs_odds), skill_number_gap: num(raw.skill_number_gap), sentence: flag ? str(raw.sentence) : null };
}
function parseEdgeSplit(raw: unknown): EdgeSplit | null {
  if (!isObject(raw)) return null;
  const kind = raw.kind === "in_line" || raw.kind === "long_shot" ? raw.kind : "normal";
  const out: EdgeSplit = { kind, edge: num(raw.edge), reference: raw.reference === "peers" ? "peers" : "none", reference_edge: num(raw.reference_edge), particular: num(raw.particular),
    ours_vs_peers: num(raw.ours_vs_peers), book_vs_peers: num(raw.book_vs_peers), sentence: str(raw.sentence) ?? "" };
  return out.edge !== null || out.sentence ? out : null;
}
function parsePeers(raw: unknown): PeerSet | null {
  if (!isObject(raw)) return null;
  const ids = (Array.isArray(raw.ids) ? raw.ids : []).map(num).filter((n): n is number => n !== null);
  const names = (Array.isArray(raw.names) ? raw.names : []).filter((n): n is string => typeof n === "string");
  const parts = (Array.isArray(raw.parts) ? raw.parts : []).filter((x): x is Record<string, unknown> => isObject(x) && typeof x.label === "string" && typeof x.key === "string")
    .map((x) => ({ key: x.key as string, label: x.label as string, value: num(x.value), peer_median: num(x.peer_median), distinctive: num(x.distinctive) }));
  return ids.length || parts.length ? { ids, names, peer_edge_median: num(raw.peer_edge_median), parts, sentence: str(raw.sentence) ?? "" } : null;
}
function parseDiverge(raw: unknown): Diverge | null {
  if (!isObject(raw)) return null;
  const d: Diverge = { compare: parseCompare(raw.compare), dg_check: parseDgCheck(raw.dg_check), edge_split: parseEdgeSplit(raw.edge_split), peers: parsePeers(raw.peers),
    sentences: (Array.isArray(raw.sentences) ? raw.sentences : []).filter((x): x is string => typeof x === "string" && x !== "") };
  return d.compare || d.dg_check || d.edge_split || d.peers || d.sentences.length ? d : null;
}
function parseForm(raw: unknown): Form | null {
  if (!isObject(raw)) return null;
  const w = isObject(raw.windows) ? raw.windows : {};
  const windows: Form["windows"] = {};
  for (const k of FORM_WINDOWS) { const e = w[k]; if (isObject(e) && num(e.mean) !== null) windows[k] = { mean: num(e.mean), n: num(e.n) }; }
  const seasons = (Array.isArray(raw.seasons) ? raw.seasons : []).filter((x): x is Record<string, unknown> => isObject(x) && num(x.year) !== null && num(x.mean) !== null)
    .map((x) => ({ year: x.year as number, mean: num(x.mean), n: num(x.n) })).sort((a, b) => a.year - b.year);
  return Object.keys(windows).length || seasons.length ? { n: num(raw.n), last_round: str(raw.last_round), windows, seasons } : null;
}
function parseDivergence(raw: unknown): Divergence | null {
  if (!isObject(raw)) return null;
  const c = isObject(raw.course_data) ? raw.course_data : null;
  const f = isObject(raw.dg_odds_fit) ? raw.dg_odds_fit : null;
  const fs = isObject(raw.form_source) ? raw.form_source : null;
  const out: Divergence = {
    version: num(raw.version),
    course_data: c ? { n_players: num(c.n_players), fit_available: num(c.fit_available), history_available: num(c.history_available), venue_editions: num(c.venue_editions), blank: c.blank === true, label: str(c.label), note: str(c.note), unavailable: str(c.unavailable) } : null,
    dg_odds_fit: f ? { market: str(f.market), slope: num(f.slope), intercept: num(f.intercept), n: num(f.n), resid_sd_skill: num(f.resid_sd_skill), flag_threshold: num(f.flag_threshold), n_flagged: num(f.n_flagged),
      flagged: (Array.isArray(f.flagged) ? f.flagged : []).map(num).filter((n): n is number => n !== null) } : null,
    field_edge_mean: num(raw.field_edge_mean), form_source: fs ? { table: str(fs.table), before: str(fs.before), available: fs.available === true, scale: str(fs.scale) } : null, method: str(raw.method),
  };
  return out.course_data || out.dg_odds_fit || out.form_source || out.field_edge_mean !== null ? out : null;
}

function parseFieldScores(raw: unknown): FieldScores | null {
  if (!isObject(raw)) return null;
  const rounds = (Array.isArray(raw.rounds) ? raw.rounds : []).filter((r): r is Record<string, unknown> => isObject(r) && num(r.round) !== null && num(r.mean_strokes) !== null && num(r.mean_vs_par) !== null)
    .map((r) => ({ round: r.round as number, mean_strokes: r.mean_strokes as number, mean_vs_par: r.mean_vs_par as number, players: num(r.players) ?? 0, basis: r.basis === "actual" ? "actual" as const : "simulated" as const }));
  const w = isObject(raw.winning) && num(raw.winning.mode) !== null && num(raw.winning.median) !== null ? raw.winning as FieldScores["winning"] : null;
  return rounds.length || w ? { rounds, winning: w } : null;
}

/** Score vs par as golfers write it: -25, E, +3. */
export const toPar = (v: number, d = 0): string => {
  const r = Number(v.toFixed(d));
  return r === 0 ? "E" : `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(d)}`;
};
/** The field's skill against an average PGA Tour player, in words. */
export function fieldStrengthText(v: number): string {
  const a = Math.abs(v);
  if (a < 0.1) return "about the same as an average PGA Tour field";
  return `${a.toFixed(2)} strokes a round ${v < 0 ? "weaker" : "stronger"} than an average PGA Tour player`;
}

/* ------------------------------------------------------------------ index / event selection */
export type DistributionRun = { key?: string | null; kind?: string | null; run?: string | null; as_of?: string | null; after_round?: number | null; validated?: boolean | null };
export type IndexEvent = {
  event_uid: string; name?: string; tour?: string; date_start?: string; course?: string; explain_key?: string | null; explain_run?: string | null; weather_key?: string | null; weather_brief_key?: string | null;
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
export type WaterfallRow = { key: string; label: string; group: string; value: number; meaning: string; unavailable?: string };
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
    rows.push({ key, label: lg.label, group: lg.group, value, meaning: lg.meaning, ...(lg.unavailable ? { unavailable: lg.unavailable } : {}) });
  }
  rows.sort((a, b) => GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group) || Math.abs(b.value) - Math.abs(a.value));
  const total = rows.reduce((s, r) => s + r.value, 0) + small;
  // Weather rows are applied on top of the skill rating, so the skill rows alone are what add up to the displayed expected skill.
  const weatherTotal = rows.filter((r) => r.group === "weather").reduce((s, r) => s + r.value, 0) + weatherSmall;
  const skillTotal = total - weatherTotal;
  const unexplained = player.mu_rel === null || player.mu_rel === undefined ? 0 : player.mu_rel - skillTotal;
  return { rows, total, small, smallCount, skillTotal, weatherTotal, smallSkill, smallSkillCount, unexplained };
}

/** One bar scale shared by every player of an event, in strokes per round: the (near-)largest expected skill or component in the field, rounded up to a half stroke (at least 1). Bars beyond it are clipped and marked. */
export function waterfallScale(players: ExPlayer[]): number {
  const q = (xs: number[], f: number) => { const s = xs.filter(Number.isFinite).sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * f))] : 0; };
  const mu = players.filter((p) => !p.withdrawn).map((p) => Math.abs(p.mu_rel ?? 0));
  const comps = players.filter((p) => !p.withdrawn).flatMap((p) => Object.values(p.components ?? {}).filter((v): v is number => typeof v === "number").map(Math.abs));
  return Math.max(1, Math.ceil(Math.max(q(mu, 0.97), q(comps, 0.995)) * 2) / 2);
}

const NICE_STEPS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3];
/** The smallest round bar-end value (0.01, 0.02, 0.05, 0.1, 0.2 ...) that is at least x. */
export function niceScale(x: number): number {
  if (!Number.isFinite(x) || x <= 0) return NICE_STEPS[0];
  return NICE_STEPS.find((s) => s >= x - 1e-9) ?? Math.ceil(x * 2) / 2;
}
/** Each "Why this price" row has its own bar scale: how far that item reaches across the players of the event (95th percentile of the absolute values, rounded up to a round number). Expected skill keeps the event-wide waterfallScale. */
export type WaterfallScales = { rows: Record<string, number>; smallSkill: number; weatherSmall: number; unexplained: number; weatherTotal: number };
export function waterfallRowScales(players: ExPlayer[], legend: Record<string, LegendEntry>, minAbs = 0.015): WaterfallScales {
  const field = players.filter((p) => !p.withdrawn);
  const p95 = (xs: number[]): number => { const s = xs.filter(Number.isFinite).map(Math.abs).sort((a, b) => a - b); return niceScale(s.length ? s[Math.min(s.length - 1, Math.floor(s.length * 0.95))] : 0); };
  const keys = [...new Set(field.flatMap((p) => Object.keys(p.components ?? {})))];
  const rows: Record<string, number> = {};
  for (const k of keys) rows[k] = p95(field.map((p) => p.components[k]).filter((v): v is number => typeof v === "number"));
  const wfs = field.map((p) => waterfall(p, legend, minAbs));
  return { rows, smallSkill: p95(wfs.map((w) => w.smallSkill)), weatherSmall: p95(wfs.map((w) => w.small - w.smallSkill)), unexplained: p95(wfs.map((w) => w.unexplained)), weatherTotal: p95(wfs.map((w) => w.weatherTotal)) };
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
  // In play the favorite is whoever has the best current chance; before the event it is the shortest-priced player in the market.
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
  travel: "Whether a player traveled from another continent to play this week.",
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
/** finish.pos ranks missed cuts below the field, so their mass sits in the last positions: remove it from the tail (backwards) so finishing bars exclude it. */
export function finishingProbs(pos: number[], missCut: number | null | undefined, hasCut: boolean): number[] {
  const out = [...pos];
  let left = hasCut && Number.isFinite(missCut ?? NaN) ? Math.max(0, missCut as number) : 0;
  for (let i = out.length - 1; i >= 0 && left > 1e-12; i--) { const take = Math.min(out[i], left); out[i] -= take; left -= take; }
  return out;
}
/** Most likely (modal) finish position. Missing the cut is one outcome spread over many players, so it is not compared with single positions; "miss" only when no position has any chance. */
export type ModalFinish = { kind: "pos"; pos: number } | { kind: "miss" } | null;
export function modalFinish(finish: { pos: number[]; p_miss_cut?: number } | null | undefined, hasCut: boolean): ModalFinish {
  if (!finish?.pos?.length) return null;
  const pos = finishingProbs(finish.pos, finish.p_miss_cut, hasCut);
  let best = 0;
  pos.forEach((v, i) => { if (v > pos[best]) best = i; });
  const miss = hasCut && Number.isFinite(finish.p_miss_cut ?? NaN) ? (finish.p_miss_cut as number) : 0;
  return pos[best] <= 0 && miss > 0 ? { kind: "miss" } : { kind: "pos", pos: best + 1 };
}
export function modalFinishText(m: ModalFinish): string {
  return !m ? "" : m.kind === "miss" ? "Most likely: misses cut" : `Most likely finish: ${m.pos}`;
}
/** Preselected comparison: the player's same-odds neighbors (as exported), restricted to players that have a finish shape. */
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
  [/^favorites \(top 10 by market\)$/i, "Favorites (ten shortest prices)"], [/^no travel$/i, "No travel"], [/^changed continent$/i, "Flew in from another continent"],
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
const roundsText = (rs: number[]): string => (rs.length === 1 ? `round ${rs[0]}` : `rounds ${joinList(rs.map(String))}`);
const cap = (t: string): string => t.charAt(0).toUpperCase() + t.slice(1);
/** Rounds still to be played in this run, with the whole-field weather shift the model applies to everyone's scores (versus typical conditions). */
export function fieldWeatherRows(doc: ExplainDoc): Array<{ round: number; common: number }> {
  const after = doc.kind === "live" ? doc.after_round ?? 0 : 0;
  return (doc.course_card.weather?.field ?? []).filter((f) => f.round > after && typeof f.common === "number" && Number.isFinite(f.common)).map((f) => ({ round: f.round, common: f.common as number }));
}
/** Rounds still to come that have no tee times yet, so only the whole-field shift (no early/late wave effect) applies to them. */
export function roundsWithoutTeeTimes(doc: ExplainDoc): number[] {
  const after = doc.kind === "live" ? doc.after_round ?? 0 : 0;
  return (doc.course_card.weather?.rounds ?? []).filter((r) => r.round > after && r.status === "common_only").map((r) => r.round);
}
/**
 * Whether weather is in the prices. This follows the published `applied_in_prices` flag, not whether a player-level effect is non-zero:
 * the per-player weather rows only exist for rounds with published tee times (the early/late wave and tee-group wind parts), whereas the
 * whole-field shift (how hard the course plays for everyone, which also moves the cut line) comes from the forecast alone. Judging "in the prices" from the
 * player rows alone wrongly reported "Weather not in prices" before tee times came out and in later rounds of live runs.
 */
export function weatherStatus(doc: ExplainDoc): { short: string; long: string } {
  const w = doc.course_card.weather;
  const { included, excluded } = weatherCoverage(doc);
  const shifts = fieldWeatherRows(doc);
  const noTee = roundsWithoutTeeTimes(doc);
  if (!w || (!w.label && !w.status && !w.venue_class && !shifts.length && !included.length)) return { short: "", long: "" };
  if (w.applied_in_prices === false && !included.length) return { short: "Weather not in prices", long: "Weather is shown for review only; these prices do not use the forecast." };
  const shiftText = shifts.length ? ` The forecast changes how hard the course plays for everyone (${shifts.map((f) => `round ${f.round} ${signed(f.common)}`).join(", ")} strokes against typical conditions), which also moves the cut line.` : "";
  if (!included.length) {
    if (!shifts.length) return { short: "Weather has no effect yet", long: "The weather model is on, but it adds nothing to any price in this run: there is no forecast or tee-time information for the rounds still to play." };
    const why = noTee.length ? ` Early and late tee times are not separated for ${roundsText(noTee)} because tee times are not out yet.` : " No player is helped or hurt by their tee time or wind exposure.";
    return { short: "Weather in prices: whole field", long: `Weather is in the prices as a field-wide shift.${shiftText}${why}` };
  }
  const idle = excluded.length ? ` ${cap(joinList(excluded))} ${excluded.length > 1 ? "have" : "has"} no effect for anyone in this forecast.` : "";
  const gap = noTee.length ? ` ${cap(roundsText(noTee))} ${noTee.length > 1 ? "have" : "has"} only the field-wide shift because tee times are not out yet.` : "";
  return { short: "Weather included", long: `Weather is in the prices. It moves players' prices through ${joinList(included)}.${idle}${shiftText}${gap}` };
}

const WX_WHAT: Record<string, { label: string; what: string }> = {
  wx_wind: { label: "Wind", what: "The wind each player's tee group faces, scaled by how much wind matters at this kind of course." },
  wx_gust: { label: "Gusts", what: "Gusts during the round, which make club choice harder." },
  wx_temp: { label: "Temperature", what: "Cooler or warmer than the field's average conditions, which changes how far the ball flies." },
  wx_rain: { label: "Rain", what: "Rain during the round, which softens the course and changes scoring." },
  wx_tod: { label: "Time of day", what: "Early versus late: the course plays differently in the morning (firmness and light) than later in the day." },
  wx_traj: { label: "Ball flight in wind", what: "How the wind suits each player's usual ball flight, for players with launch-monitor data." },
};
export type WeatherEffectLine = { key: string; label: string; what: string; spread: number; helped: { name: string; value: number } | null; hurt: { name: string; value: number } | null };
/** The weather effects behind the prices, in strokes a round: how far apart the field is on each and who is helped or hurt most. */
export function weatherEffectLines(doc: ExplainDoc): WeatherEffectLine[] {
  const live = doc.players.filter((p) => !p.withdrawn);
  return Object.keys(doc.components_legend).filter((k) => k.startsWith("wx_")).map((key) => {
    const vals = live.map((p) => ({ name: p.name, value: p.components[key] })).filter((x): x is { name: string; value: number } => typeof x.value === "number" && Number.isFinite(x.value));
    const best = vals.reduce<{ name: string; value: number } | null>((a, b) => (!a || b.value > a.value ? b : a), null);
    const worst = vals.reduce<{ name: string; value: number } | null>((a, b) => (!a || b.value < a.value ? b : a), null);
    const meta = WX_WHAT[key] ?? { label: doc.components_legend[key].label.replace(/^Weather:\s*/i, ""), what: plainText(doc.components_legend[key].meaning) };
    return { key, label: meta.label, what: meta.what, spread: Math.max(Math.abs(best?.value ?? 0), Math.abs(worst?.value ?? 0)), helped: best && best.value >= 0.005 ? best : null, hurt: worst && worst.value <= -0.005 ? worst : null };
  });
}
const CLASS_WORD: Record<string, string> = { parkland: "Parkland", links: "Links", desert: "Desert", coastal: "Coastal", mountain: "Mountain", heathland: "Heathland", resort: "Resort" };
/** Plain-English course type and wind sensitivity for the This week page. */
export function venueClassLines(w: CourseCard["weather"] | undefined): string[] {
  if (!w) return [];
  const out: string[] = [];
  const cls = w.venue_class ? (CLASS_WORD[w.venue_class.toLowerCase()] ?? cap(w.venue_class)) : null;
  if (cls) out.push(`${cls} course.${w.wind_slope != null ? ` Each extra mph of wind costs the average round about ${w.wind_slope.toFixed(2)} strokes here.` : ""}`);
  if (w.m_v != null) {
    const src = w.wind_prior_source === "own history" ? "this course's own past events" : "courses of the same type";
    out.push(`The model scales every wind effect by ×${w.m_v.toFixed(2)} for this course, learned from ${src}${w.m_v > 1.05 ? ": wind matters more here than at the average course" : w.m_v < 0.95 ? ": wind matters less here than at the average course" : ": wind matters about as much as at the average course"}.`);
  }
  const seen = (w.rounds ?? []).find((r) => r.venue_seen != null)?.venue_seen;
  if (w.field?.length) out.push(`The whole-field number combines wind, gusts, temperature, rain and time of day. ${seen === true ? "It is measured against this course's usual weather in the model's history." : seen === false ? "This course is new to the weather model, so it is measured against the average weather across all events in the model's history." : "It is measured against typical conditions."}`);
  return out;
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
  /** spread_gap minus the contenders' median spread gap (documents from October 12, 2026); the steady / boom-or-bust wording reads this. Absent on older documents. */
  spread_gap_rel?: number;
  markets: Partial<Record<Market, MarketEdge>>; higher: Market[]; lower: Market[]; pattern: string; reason: string; sentence: string;
  parts: ShapePart[]; base_gap: number; parts_sentence: string;
};
export type DifferGroup = { key: string; label: string; n: number; win_ours: number; win_market: number; win_rel: number; top20_ours: number | null; top20_market: number | null; top20_rel: number | null; skill_gap: number; spread_gap: number };
export type Differ = { rule: string; ids: number[]; groups: DifferGroup[]; spread_gap_median: number | null; skill_gap_median: number | null; spread_gap_centered?: boolean; source: "export" | "page" };

/** The spread gap the steady / boom-or-bust wording reads: relative to this week's contenders when the document carries it, the raw gap on older documents. */
export const spreadGapOf = (s: Pick<Shape, "spread_gap" | "spread_gap_rel">): number => (typeof s.spread_gap_rel === "number" && Number.isFinite(s.spread_gap_rel) ? s.spread_gap_rel : s.spread_gap);

/** The line above the contenders table: the typical spread gap is small and moves from event to event, so it is not read as a lean; labels are relative to it. */
export function spreadLeanNote(median: number | null, centered: boolean): string | null {
  if (median === null || !Number.isFinite(median)) return null;
  const lead = `The typical spread gap across these players is small and moves from week to week (${signed(median)} strokes a round this week), so it is not read as a lean toward wins or toward top 20 and make cut.`;
  return centered ? `${lead} Steadier and more boom-or-bust labels are relative to this week's contenders.` : lead;
}
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
export function playerShape(p: ExPlayer, map: ShapeMap, markets: readonly Market[] = MARKETS, center: number | null = null): Shape | null {
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
  const relGap = center === null ? spreadGap : spreadGap - center; // the gap the wording reads
  const sk = Math.abs(skillGap) >= SKILL_MIN, sp = Math.abs(relGap) >= SPREAD_MIN;
  const spreadMarket = p.sd - spreadGap;
  const spreadTxt = `spread ${p.sd.toFixed(2)} vs the market's ${spreadMarket.toFixed(2)}${center === null ? "" : `, ${signed(relGap)} against this week's contenders`}`;
  const steady = relGap < 0;
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
    ...(center === null ? {} : { spread_gap_rel: relGap }), markets: edges, higher, lower, pattern, reason, sentence, parts, ...partsSplit(skillGap, parts) };
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
      skill_gap_median: ex.skill_gap_median ?? median(rows.map((r) => r.shape.skill_gap)), spread_gap_centered: ex.spread_gap_centered === true && rows.every((r) => typeof r.shape.spread_gap_rel === "number"), source: "export" } };
  }
  const rows = contenders(doc, map);
  return { rows, differ: { rule: CONTENDER_RULE, ids: rows.map((r) => r.player.id), groups: differGroups(rows), spread_gap_median: median(rows.map((r) => r.shape.spread_gap)),
    skill_gap_median: median(rows.map((r) => r.shape.skill_gap)), spread_gap_centered: false, source: "page" } };
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

/* ------------------------------------------------------------------ This week: what changed, key facts, plain-language helpers (October 14) */
/** "Last, First" as people say it: "First Last". Names without a comma are kept. */
export function plainName(name: string | null | undefined): string {
  if (!name) return "Unnamed player";
  const parts = name.split(",").map((s) => s.trim()).filter(Boolean);
  return parts.length === 2 ? `${parts[1]} ${parts[0]}` : name.trim();
}
export const stamp = (iso: string | null | undefined): number | null => {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
};
/** "just now", "12 min ago", "3 h ago", "2 days ago"; null when the age is unknown. A time in the future reads as "just now". */
export function ageText(ms: number | null | undefined): string | null {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return null;
  const m = Math.round(Math.max(0, ms) / 60000);
  if (m < 2) return "just now";
  if (m < 90) return `${m} min ago`;
  const h = m / 60;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} days ago`;
}
/** Length of a time span for "5 h earlier": same units as ageText, without the "ago". */
export function spanText(ms: number | null | undefined): string | null {
  const t = ageText(ms);
  return t === null ? null : t === "just now" ? "moments" : t.replace(/ ago$/, "");
}

export type RunRef = { key: string; kind: string; run: string; as_of: string | null; after_round: number | null; id: string };
const runRefs = (e: Pick<IndexEvent, "distribution_runs">): RunRef[] =>
  (e.distribution_runs ?? []).filter((r): r is DistributionRun & { key: string; kind: string; run: string } => !!r && typeof r.key === "string" && r.key.length > 0 && typeof r.kind === "string" && typeof r.run === "string" && r.validated !== false)
    .map((r) => ({ key: r.key, kind: r.kind, run: r.run, as_of: r.as_of ?? null, after_round: r.after_round ?? null, id: `${r.kind}_${r.run}` }));
/** The validated runs of an event published strictly before `doc`, newest first. */
export function runsBefore(event: Pick<IndexEvent, "distribution_runs">, doc: Pick<ExplainDoc, "kind" | "run" | "as_of">): RunRef[] {
  const at = stamp(doc.as_of);
  const self = `${doc.kind}_${doc.run}`;
  return runRefs(event).filter((r) => r.id !== self && at !== null && (stamp(r.as_of) ?? Infinity) < at).sort((a, b) => (stamp(b.as_of) ?? 0) - (stamp(a.as_of) ?? 0));
}
/** The run to compare with by default: the one published just before this document. */
export const previousRunRef = (event: Pick<IndexEvent, "distribution_runs">, doc: Pick<ExplainDoc, "kind" | "run" | "as_of">): RunRef | null => runsBefore(event, doc)[0] ?? null;
/** The run the owner last looked at (stored id "<kind>_<run>"), when it is older than this document and not simply the previous run. */
export function lastSeenRunRef(event: Pick<IndexEvent, "distribution_runs">, doc: Pick<ExplainDoc, "kind" | "run" | "as_of">, seenId: string | null | undefined): RunRef | null {
  if (!seenId) return null;
  const before = runsBefore(event, doc);
  const hit = before.find((r) => r.id === seenId);
  return hit && before[0] && hit.id !== before[0].id ? hit : null;
}

/** Smallest probability move, in percentage points, that counts as a change. */
export const MOVE_MIN_PTS: Record<Market, number> = { win: 0.1, top_5: 0.3, top_10: 0.4, top_20: 0.5, make_cut: 0.5 };
export const GAP_MIN_MARKET = 0.01; // only players the sportsbook gives at least 1% to win

export type HeroFacts = {
  favourite: { id: number; name: string; model: number | null; market: number | null; live: boolean } | null;
  /** Before the event: the contender whose win chance differs most from the sportsbook's. In play: the biggest mover since the last run. */
  second: { kind: "gap"; id: number; name: string; model: number; market: number; rel: number } | { kind: "move"; id: number; name: string; before: number; after: number; delta: number } | null;
  /** The winning score from the simulations, else the scoring average of the first simulated round, else nothing. */
  scoring: { kind: "winning"; mode: number; modeShare: number; median: number; p10: number; p90: number } | { kind: "round"; round: number; strokes: number; vsPar: number; basis: "simulated" | "actual" } | null;
};
/** The three facts to read first. `ids` limits the disagreement to the contenders (differView ids); without it, players the sportsbook gives at least 2% to win. */
export function heroFacts(doc: ExplainDoc, ids?: number[]): HeroFacts {
  const live = doc.kind === "live";
  const fav = favourites(doc, 1)[0];
  const favourite = fav ? { id: fav.id, name: plainName(fav.name), model: fav.probs.win.model, market: live ? null : fav.probs.win.market, live } : null;
  let second: HeroFacts["second"] = null;
  if (!live) {
    const pool = doc.players.filter((p) => !p.withdrawn && p.probs.win.model !== null && p.probs.win.market !== null && p.probs.win.rel !== null && (ids && ids.length ? ids.includes(p.id) : (p.probs.win.market ?? 0) >= 0.02));
    const top = [...pool].sort((a, b) => Math.abs(b.probs.win.rel ?? 0) - Math.abs(a.probs.win.rel ?? 0) || (a.name ?? "").localeCompare(b.name ?? ""))[0];
    if (top && Math.abs(top.probs.win.rel ?? 0) >= EDGE_MIN) second = { kind: "gap", id: top.id, name: plainName(top.name), model: top.probs.win.model!, market: top.probs.win.market!, rel: top.probs.win.rel! };
  } else {
    const mv = [...doc.players].filter((p) => !p.withdrawn && typeof p.live?.since_last?.win === "number" && p.probs.win.model !== null)
      .sort((x, y) => Math.abs(y.live!.since_last.win!) - Math.abs(x.live!.since_last.win!))[0];
    if (mv && Math.abs(mv.live!.since_last.win! * 100) >= MOVE_MIN_PTS.win) second = { kind: "move", id: mv.id, name: plainName(mv.name), before: mv.probs.win.model! - mv.live!.since_last.win!, after: mv.probs.win.model!, delta: mv.live!.since_last.win! * 100 };
  }
  const fsc = doc.field_scores;
  let scoring: HeroFacts["scoring"] = null;
  if (fsc?.winning) scoring = { kind: "winning", mode: fsc.winning.mode, modeShare: fsc.winning.mode_share, median: fsc.winning.median, p10: fsc.winning.p10, p90: fsc.winning.p90 };
  else if (fsc?.rounds.length) {
    const next = fsc.rounds.find((r) => r.basis === "simulated") ?? fsc.rounds[0];
    scoring = { kind: "round", round: next.round, strokes: next.mean_strokes, vsPar: next.mean_vs_par, basis: next.basis };
  }
  return { favourite, second, scoring };
}

export type GapRow = { id: number; name: string; model: number; market: number; rel: number };
/** The players we are furthest above and below the sportsbook on win chance (contenders if `ids` is given), for the glance list. `scale` is the largest gap shown, at least 25%, so bars share one scale. */
export function biggestGaps(doc: ExplainDoc, n = 3, ids?: number[]): { higher: GapRow[]; lower: GapRow[]; scale: number } {
  const rows: GapRow[] = doc.players.filter((p) => !p.withdrawn && p.probs.win.model !== null && p.probs.win.market !== null && p.probs.win.rel !== null && (ids && ids.length ? ids.includes(p.id) : (p.probs.win.market ?? 0) >= GAP_MIN_MARKET))
    .map((p) => ({ id: p.id, name: plainName(p.name), model: p.probs.win.model!, market: p.probs.win.market!, rel: p.probs.win.rel! }));
  const byGap = (a: GapRow, b: GapRow) => Math.abs(b.rel) - Math.abs(a.rel) || (a.name ?? "").localeCompare(b.name ?? "");
  const higher = rows.filter((r) => r.rel >= EDGE_MIN).sort(byGap).slice(0, n);
  const lower = rows.filter((r) => r.rel <= -EDGE_MIN).sort(byGap).slice(0, n);
  return { higher, lower, scale: Math.max(0.25, ...[...higher, ...lower].map((r) => Math.abs(r.rel))) };
}

/** Signed percentage points, "+0.8 pts"; a move that rounds to zero is "no change". */
export function ptsText(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  const r = Number(Math.abs(v).toFixed(digits));
  return r === 0 ? "no change" : `${v > 0 ? "+" : "−"}${r.toFixed(digits)} pts`;
}
/** Signed whole percent of the sportsbook's chance, "+39%". */
export function relText(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  const r = Math.round(Math.abs(v) * 100);
  return r === 0 ? "0%" : `${v > 0 ? "+" : "−"}${r}%`;
}

/** Why a missing winning score is missing, in one short phrase. */
export const WINNING_SCORE_MISSING = "Not in this run: it needs the simulation draws, which this run did not publish.";

/** One readable sentence about the course and field, for the lede. */
export function eventLede(doc: ExplainDoc): string {
  const cc = doc.course_card;
  const par = cc.par_per_round?.[0];
  const course = cc.course || doc.event.course || doc.event.name;
  const spec = [par ? `par ${par}` : "", cc.yardage ? `${Math.round(cc.yardage).toLocaleString("en-US")} yd` : ""].filter(Boolean).join(", ");
  const field = doc.players.filter((p) => !p.withdrawn).length;
  const fs = doc.event.field_strength ?? doc.event.field_strength_detail?.field_offset ?? null;
  const strength = fs === null || fs === undefined ? "" : Math.abs(fs) < 0.1 ? ", an average-strength field" : fs > 0 ? ", a stronger-than-average field" : ", a weaker-than-average field";
  const cut = doc.event.cut_rule ? doc.event.cut_rule.replace(/^./, (c) => c.toLowerCase()) : "no cut";
  return `${[course, spec].filter(Boolean).join(" · ")}. ${field} players, ${cut}${strength}.`;
}

/** A player's DataGolf block, numbers only; null when absent or empty. */
function parseDg(raw: unknown): DgRef | null {
  if (!isObject(raw)) return null;
  const r = raw as Record<string, unknown>;
  const out = { skill: num(r.skill), skill_ours: num(r.skill_ours), skill_gap: num(r.skill_gap) } as DgRef;
  for (const m of MARKETS) out[m] = num(r[m]);
  return Object.values(out).some((v) => v !== null) ? out : null;
}

export type DgGapKind = "win" | "top_20" | "skill";
export const DG_GAP_LABEL: Record<DgGapKind, string> = { win: "Win", top_20: "Top 20", skill: "Skill" };
/** ours / dg: chances (win, top 20) or skills in strokes a round; gap: relative for chances (ours / DataGolf's - 1), strokes for skill (ours - DataGolf's). */
export type DgGapRow = { id: number; name: string; ours: number; dg: number; gap: number;
  /** Skill rows only: Data Golf's stated skill disagrees with its own odds (diverge.dg_check.flag). `dg` and `gap` are then measured against the skill its odds imply, `stated` is the number it published. */
  flag?: boolean; stated?: number | null; numberGap?: number | null };
export const DG_SKILL_MIN = 0.1;      // a skill gap under 0.1 strokes a round counts as in line
export const DG_TOP20_MIN = 0.05;     // players DataGolf gives at least 5% to finish top 20, plus this week's contenders
export const DG_WIN_FLOOR = 0.01;     // win view: at least 1% to win with us or Data Golf
export const DG_TOP20_FLOOR = 0.1;    // top-20 view: at least 10% to finish top 20 with us or Data Golf
/** Whether the document carries any DataGolf reference (older documents and live runs do not). */
export function hasDg(doc: ExplainDoc): boolean { return doc.players.some((p) => p.dg != null); }
/** Biggest gaps against DataGolf, ours above and ours below, among this week's contenders and the players DataGolf gives a real top-20 chance. */
export function dgGaps(doc: ExplainDoc, kind: DgGapKind, n = 4, ids?: number[]): { higher: DgGapRow[]; lower: DgGapRow[]; scale: number } {
  // Chance gaps are relative, so long shots (0.06% vs 0.2%) would fill the list with noise: the chance views need a real chance on either side.
  const floor = kind === "win" ? DG_WIN_FLOOR : kind === "top_20" ? DG_TOP20_FLOOR : 0;
  const real = (p: ExPlayer) => kind === "skill" || Math.max(p.probs[kind].model ?? 0, p.dg?.[kind] ?? 0) >= floor;
  const pool = doc.players.filter((p) => !p.withdrawn && p.dg && ((ids ?? []).includes(p.id) || (p.dg.top_20 ?? 0) >= DG_TOP20_MIN) && real(p));
  const rows: DgGapRow[] = [];
  for (const p of pool) {
    const d = p.dg!;
    if (kind === "skill") {
      const ck = p.diverge?.dg_check;
      // With the odds check, compare against the skill Data Golf's own odds imply (its stated number can be off for some players); flagged players carry a warning.
      if (ck && ck.skill_from_odds !== null && ck.gap_vs_odds !== null && (ck.skill_ours ?? d.skill_ours) !== null) rows.push({ id: p.id, name: plainName(p.name), ours: (ck.skill_ours ?? d.skill_ours) as number, dg: ck.skill_from_odds, gap: ck.gap_vs_odds, flag: ck.flag, stated: ck.skill_stated ?? d.skill, numberGap: ck.skill_number_gap ?? d.skill_gap });
      else if (d.skill !== null && d.skill_ours !== null && d.skill_gap !== null) rows.push({ id: p.id, name: plainName(p.name), ours: d.skill_ours, dg: d.skill, gap: d.skill_gap, flag: ck?.flag === true, stated: d.skill });
    } else {
      const ours = p.probs[kind].model, dg = d[kind] ?? null;
      if (ours !== null && dg !== null && dg > 0 && dg < 1) rows.push({ id: p.id, name: plainName(p.name), ours, dg, gap: ours / dg - 1 });
    }
  }
  const min = kind === "skill" ? DG_SKILL_MIN : EDGE_MIN;
  const byGap = (a: DgGapRow, b: DgGapRow) => Math.abs(b.gap) - Math.abs(a.gap) || a.name.localeCompare(b.name);
  const higher = rows.filter((r) => r.gap >= min).sort(byGap).slice(0, n);
  const lower = rows.filter((r) => r.gap <= -min).sort(byGap).slice(0, n);
  return { higher, lower, scale: Math.max(kind === "skill" ? 0.3 : 0.25, ...[...higher, ...lower].map((r) => Math.abs(r.gap))) };
}
/** The single biggest skill gap against DataGolf among the same players, or null. Players whose stated skill disagrees with Data Golf's own odds are skipped (their gap is mostly Data Golf's inconsistency, not a disagreement); the rest are ranked on the odds-implied gap when the document has it. */
export function biggestDgSkillGap(doc: ExplainDoc, ids?: number[]): DgGapRow | null {
  const g = dgGaps(doc, "skill", 1000, ids);
  return [...g.higher, ...g.lower].filter((r) => !r.flag).sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))[0] ?? null;
}
