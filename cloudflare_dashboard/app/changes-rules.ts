/**
 * What changed between two runs of one event: price moves, the causes behind each move, the sportsbook's moves.
 * Pure TypeScript with no DOM, React or runtime APIs: the What changed strip (ExplainViews.tsx) and the Worker endpoint
 * /api/golfprice/changes.json (worker/changes-api.ts) import the same functions, so the stored diff and the client fallback agree.
 * Tests: tests/what-changed.test.mjs, tests/changes-api.test.mjs.
 */
import { EDGE_MIN, GAP_MIN_MARKET, MARKET_LABEL, MARKETS, MOVE_MIN_PTS, availableMarkets, plainName, relText, stamp, type ExPlayer, type ExplainDoc, type FieldScores, type Market } from "./explain-rules.ts";

export { MOVE_MIN_PTS };

/* ------------------------------------------------------------------ types */
export type CauseKey = "skill" | "course" | "travel" | "weather" | "override" | "in_play" | "field_level" | "field" | "noise" | "unseparated";
export type Cause = {
  key: CauseKey;
  label: string;
  /** Strokes a round this cause moved the player's expected scoring (+ = better for the player); null for causes that are not a strokes change (field and other players, noise). */
  strokes: number | null;
  /** Signed share of the probability move, in percentage points; null when strokes cannot be converted to probability. */
  pts: number | null;
  /** Extra words for the hover/aria text ("wave early → late", "scored 68 in round 2"). */
  detail: string | null;
  /** The short chip text: "weather +0.12", "Hatton WD", "override +0.20". */
  chip: string;
};
export type Mover = {
  id: number; name: string; market: Market; before: number; after: number; delta: number; marketDelta: number | null;
  /** Up to CAUSE_CHIPS causes, largest first. One "not separable" cause when nothing can be named honestly. */
  causes: Cause[];
  separable: boolean;
};
export type GapMove = { id: number; name: string; before: number; after: number; delta: number };
export type RoundShift = { round: number; before: number; after: number; delta: number };
export type BookKind = "book_toward" | "book_away" | "model_toward" | "model_away" | "both";
export type BookMove = {
  id: number; name: string; market: Market; kind: BookKind;
  modelBefore: number; modelAfter: number; marketBefore: number; marketAfter: number;
  /** Gap to the sportsbook as a fraction of its chance, (ours - theirs) / theirs. */
  gapBefore: number; gapAfter: number;
  /** "growing": the edge got larger and is worth a look now; "closing": the sportsbook came to us, act before it is gone; null otherwise (also always null when only our own price moved). */
  actionable: "growing" | "closing" | null;
  text: string;
};
export type WhatChanged = {
  /** True when only the in-play "since last run" numbers could be used (the earlier document was not available). */
  partial: boolean;
  baseline: { run: string | null; as_of: string | null; after_round: number | null; kind: string | null };
  /** Milliseconds between the baseline and this document, when both times are known. */
  gapMs: number | null;
  /** Probability moves in percentage points (after minus before), largest first, only those above MOVE_MIN_PTS. */
  movers: Partial<Record<Market, Mover[]>>;
  field: { added: string[]; removed: string[]; withdrawn: string[] };
  weather: { rounds: RoundShift[]; layer: { from: string; to: string } | null; reissued: boolean };
  scoring: { rounds: RoundShift[]; winning: { before: number; after: number; delta: number } | null };
  /** Change of the win gap to the sportsbook (relative, fraction of the sportsbook chance), largest first. */
  gaps: GapMove[];
  /** Who moved the sportsbook gap, before the event only: actionable cases first. */
  bookMoves: BookMove[];
  overrides: { added: string[]; removed: string[] };
  /** How the causes were worked out (null when the earlier document was not available). */
  attribution: { fieldLevelStrokes: number | null; logitPerStroke: Partial<Record<Market, number>> } | null;
  /** Number of material changes found. */
  count: number;
  quiet: boolean;
  /** Short phrases for the summary chips. */
  headline: string[];
};

/* ------------------------------------------------------------------ named thresholds */
export const WEATHER_MIN = 0.05;   // strokes a round
export const SCORING_MIN = 0.1;    // strokes a round
export const GAP_MOVE_MIN = 0.05;  // 5% of the sportsbook's chance
/** A cause is named only when it carries at least this share of the probability move. */
export const CAUSE_SHARE_MIN = 0.15;
/** ... and, when it is a strokes change, at least this many strokes a round. */
export const CAUSE_STROKES_MIN = 0.01;
/** Most causes shown as chips on a mover row. */
export const CAUSE_CHIPS = 2;
/** A move smaller than this many standard errors of the simulation (n_sims draws on each side) may be noise. */
export const NOISE_Z = 2;
/** The whole-field level shift is reported only when at least this big (strokes a round). */
export const FIELD_LEVEL_MIN = 0.01;
/** Fewest players needed to estimate a field-wide level shift or a probability slope. */
export const MIN_FIELD_FOR_FIT = 15;
/** The sportsbook counts as having moved when its chance changed by at least this fraction of itself ... */
export const BOOK_MOVE_REL_MIN = 0.03;
/** ... and by at least MOVE_MIN_PTS[market] percentage points. The same bar decides whether our own price moved. */
export const BOOK_GAP_CHANGE_MIN = GAP_MOVE_MIN;
/** Sportsbook prices below this chance are skipped (a 1-in-300 shot's gap is mostly rounding). */
export const BOOK_MIN_PROB: Record<Market, number> = { win: GAP_MIN_MARKET, top_5: 0.03, top_10: 0.05, top_20: 0.08, make_cut: 0.1 };
export const MAX_BOOK_MOVES = 40;

/* ------------------------------------------------------------------ small helpers */
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const logit = (p: number): number => Math.log(p / (1 - p));
const expit = (x: number): number => 1 / (1 + Math.exp(-x));
const median = (xs: number[]): number => { const s = [...xs].sort((a, b) => a - b); const n = s.length; return n === 0 ? 0 : n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const lastName = (raw: string): string => { const c = raw.split(",")[0]?.trim(); return c && raw.includes(",") ? c : plainName(raw).split(" ").slice(-1)[0] ?? raw; };

/* ------------------------------------------------------------------ causes */
const CAUSE_LABEL: Record<CauseKey, string> = {
  skill: "skill update", course: "course", travel: "travel", weather: "weather", override: "override", in_play: "in-play",
  field_level: "field level", field: "field and other players", noise: "simulation noise", unseparated: "not separable",
};
const SKILL_KEYS = new Set(["level_form", "kalman", "xtour", "category", "sit", "act", "sklv", "thin", "disp", "other"]);
const TRAVEL_KEYS = new Set(["loc_travel", "loc_nat", "loc_refit"]);
const IN_PLAY_KEYS = new Set(["b8", "live_other"]);
const isWx = (k: string): boolean => k.startsWith("wx_");
/** Which cause a published component belongs to; null for a component this code does not know (it then stays in the unexplained remainder). */
function causeOf(component: string): Exclude<CauseKey, "field_level" | "field" | "noise" | "unseparated"> | null {
  if (SKILL_KEYS.has(component)) return "skill";
  if (component === "course") return "course";
  if (TRAVEL_KEYS.has(component)) return "travel";
  if (component === "override") return "override";
  if (IN_PLAY_KEYS.has(component)) return "in_play";
  if (isWx(component)) return "weather";
  return null;
}

/**
 * Logit change per stroke a round for one market. The explain export publishes it as edge_model.strokes_per_logit (the name is the wrong way round:
 * the value is logit per stroke, checked against a fit across the field in tests). Where the export has no value (top 5), it is fitted across this document's players.
 */
export function logitPerStroke(doc: ExplainDoc, m: Market): number | null {
  const pub = doc.edge_model?.strokes_per_logit?.[m];
  if (finite(pub) && pub > 0) return pub;
  const rows = doc.players.filter((p) => !p.withdrawn && finite(p.mu) && finite(p.probs[m].model) && p.probs[m].model! > 0.0005 && p.probs[m].model! < 0.9995);
  if (rows.length < MIN_FIELD_FOR_FIT) return null;
  const xs = rows.map((p) => p.mu!);
  const ys = rows.map((p) => logit(p.probs[m].model!));
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i += 1) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return slope > 0 ? slope : null;
}

const skillMu = (p: ExPlayer): number | null => (finite(p.mu) ? p.mu : finite(p.mu_rel) ? p.mu_rel : null);
const compSum = (c: Record<string, number>, pick: (k: string) => boolean): number => Object.entries(c).reduce((s, [k, v]) => (finite(v) && pick(k) ? s + v : s), 0);

export type Leaver = { id: number; name: string; win: number };
export type AttributionContext = {
  /** Strokes a round the whole field's expected skill moved together (median of what each player's own components do not explain). */
  fieldLevel: number;
  leavers: Leaver[];
  joiners: string[];
  slope: Partial<Record<Market, number | null>>;
  nSims: { cur: number | null; prev: number | null };
  live: boolean;
  prevAfterRound: number;
};

/** Everything about the pair of documents that the per-player attribution needs, computed once. */
export function attributionContext(cur: ExplainDoc, prev: ExplainDoc | null, markets: readonly Market[] = MARKETS): AttributionContext {
  const slope: Partial<Record<Market, number | null>> = {};
  for (const m of markets) slope[m] = logitPerStroke(cur, m);
  const nSims = { cur: finite(cur.model?.n_sims) ? cur.model.n_sims : null, prev: finite(prev?.model?.n_sims) ? prev!.model.n_sims : null };
  if (!prev) return { fieldLevel: 0, leavers: [], joiners: [], slope, nSims, live: cur.kind === "live", prevAfterRound: 0 };
  const prevById = new Map(prev.players.map((p) => [p.id, p] as const));
  const curById = new Map(cur.players.map((p) => [p.id, p] as const));
  const resid: number[] = [];
  for (const p of cur.players) {
    const q = prevById.get(p.id);
    if (!q || p.withdrawn || q.withdrawn) continue;
    const a = skillMu(p), b = skillMu(q);
    if (a === null || b === null) continue;
    resid.push(a - b - (compSum(p.components, (k) => !isWx(k)) - compSum(q.components, (k) => !isWx(k))));
  }
  const fl = resid.length >= MIN_FIELD_FOR_FIT ? median(resid) : 0;
  const leavers: Leaver[] = [];
  for (const q of prev.players) {
    if (q.withdrawn) continue;
    const now = curById.get(q.id);
    if (!now || now.withdrawn) leavers.push({ id: q.id, name: q.name, win: q.probs.win.model ?? 0 });
  }
  leavers.sort((a, b) => b.win - a.win || a.name.localeCompare(b.name));
  const joiners = cur.players.filter((p) => !p.withdrawn && !prevById.has(p.id)).map((p) => p.name);
  return { fieldLevel: Math.abs(fl) >= FIELD_LEVEL_MIN ? fl : 0, leavers, joiners, slope, nSims, live: cur.kind === "live", prevAfterRound: prev.after_round ?? 0 };
}

/** "weather helps 0.12" / "travel hurts 0.05": components are strokes a round where positive helps the player, and "+0.12 strokes" reads as worse to a golfer. */
const strokeChip = (c: { key: CauseKey; label: string; strokes: number | null }): string => `${c.label} ${(c.strokes ?? 0) >= 0 ? "helps" : "hurts"} ${Math.abs(c.strokes ?? 0).toFixed(2)}`;

function fieldCause(ctx: AttributionContext, pts: number | null): Cause {
  let label = CAUSE_LABEL.field;
  let chip = label;
  let detail: string | null = "The part of the move that is not this player's own rating change: other players withdrawing, joining or changing, and simulation variation.";
  if (ctx.leavers.length) {
    const first = lastName(ctx.leavers[0].name);
    chip = ctx.leavers.length === 1 ? `${first} WD` : `${first} WD +${ctx.leavers.length - 1}`;
    label = `${plainName(ctx.leavers[0].name)} out of the field`;
    detail = `${ctx.leavers.map((l) => plainName(l.name)).join(", ")} left the field; ${detail}`;
  } else if (ctx.joiners.length) {
    chip = `${ctx.joiners.length} added to field`;
    detail = `${ctx.joiners.map(plainName).join(", ")} joined the field; ${detail}`;
  }
  return { key: "field", label, strokes: null, pts, detail, chip };
}

/** In-play detail: the rounds played since the earlier run, "scored 68 in round 2". */
function playedText(p: ExPlayer, afterRound: number): string | null {
  const rounds = (p.live?.rounds ?? []).filter((r) => r.round > afterRound && finite(r.score));
  return rounds.length ? `scored ${rounds.map((r) => `${r.score} in round ${r.round}`).join(", ")}` : null;
}

/** Binomial standard error of the difference between two simulated chances, in percentage points; null when the draw counts are unknown. */
function noiseSe(a: number, b: number, ctx: AttributionContext): number | null {
  if (!finite(ctx.nSims.cur)) return null;
  const nb = finite(ctx.nSims.prev) ? ctx.nSims.prev : ctx.nSims.cur;
  return 100 * Math.sqrt((a * (1 - a)) / ctx.nSims.cur + (b * (1 - b)) / nb);
}

/**
 * Split one player's move in one market into labelled causes.
 *
 * Strokes: the change of each published component, grouped (skill update, course, travel, weather, owner override, in-play form). Expected skill (mu) is the sum of the
 * non-weather components plus a field-wide constant; weather is applied on top, so the player's own strokes change is d(mu) + d(weather). Whatever d(mu) the components
 * do not explain is a shift common to the whole field ("field level", the median over players) plus a per-player remainder ("not separable").
 * Probability: the own strokes change is turned into the probability it alone would give, holding everyone else fixed, through the market's logit-per-stroke slope; each
 * cause gets its share of that move in proportion to its strokes. What is left of the probability move (field changes, other players' ratings, simulation variation)
 * is "field and other players", never relabelled as the player's own doing. When the slope or either probability is unusable, strokes are shown but no share is invented.
 */
export function attribute(p: ExPlayer, q: ExPlayer | null, m: Market, ctx: AttributionContext, cur: ExplainDoc): { causes: Cause[]; separable: boolean } {
  const after = p.probs[m].model;
  const total = q ? (after !== null && q.probs[m].model !== null ? (after - q.probs[m].model!) * 100 : null) : null;
  const notSeparable = (why: string): { causes: Cause[]; separable: boolean } => ({ separable: false, causes: [{ key: "unseparated", label: CAUSE_LABEL.unseparated, strokes: null, pts: null, detail: why, chip: CAUSE_LABEL.unseparated }] });
  const slope = ctx.slope[m] ?? null;
  let strokes: Partial<Record<CauseKey, number>> = {};
  let dOwn: number;           // d(mu) + d(weather), strokes a round
  let before: number;
  const detail: Partial<Record<CauseKey, string>> = {};

  if (q) {
    if (after === null || q.probs[m].model === null) return notSeparable("a probability is missing in one of the runs");
    before = q.probs[m].model!;
    const a = skillMu(p), b = skillMu(q);
    const hasC = Object.keys(p.components).length > 0 && Object.keys(q.components).length > 0;
    if (a === null || b === null || !hasC) return notSeparable("expected skill or its components are missing in one of the runs");
    const dMu = a - b;
    const groups: Partial<Record<CauseKey, number>> = {};
    for (const k of new Set([...Object.keys(p.components), ...Object.keys(q.components)])) {
      const g = causeOf(k);
      if (!g) continue;
      groups[g] = (groups[g] ?? 0) + ((finite(p.components[k]) ? p.components[k] : 0) - (finite(q.components[k]) ? q.components[k] : 0));
    }
    const nonWx = Object.entries(groups).filter(([g]) => g !== "weather").reduce((s, [, v]) => s + (v ?? 0), 0);
    const remainder = dMu - nonWx;
    strokes = { ...groups };
    strokes.field_level = ctx.fieldLevel;
    strokes.unseparated = remainder - ctx.fieldLevel;
    dOwn = dMu + (groups.weather ?? 0);
    if (p.wave && q.wave && p.wave !== q.wave) detail.weather = `wave ${q.wave} → ${p.wave}`;
    else if ((p.tee_r1 && q.tee_r1 && p.tee_r1 !== q.tee_r1) || (p.tee_r2 && q.tee_r2 && p.tee_r2 !== q.tee_r2)) detail.weather = "tee time changed";
    if (cur.kind === "live") {
      const t = playedText(p, ctx.prevAfterRound);
      if (t) detail.in_play = t;
    }
  } else {
    // Partial: only the in-play "since last run" numbers exist.
    const d = p.live?.since_last?.[m];
    const dm = p.live?.d_mu_since_last;
    if (after === null || !finite(d) || !finite(dm)) return notSeparable("only the price change since the last run is published");
    before = after - d;
    strokes = { in_play: dm };
    dOwn = dm;
    const t = playedText(p, ctx.prevAfterRound) ?? playedText(p, p.live?.rounds.length ? Math.max(0, Math.max(...p.live.rounds.map((r) => r.round)) - 1) : 0);
    if (t) detail.in_play = t;
  }

  const tot = total ?? (after - before) * 100;
  // Probability share per stroke.
  let ownPts: number | null = null;
  let perStroke: number | null = null;
  if (slope !== null && before > 0.0005 && before < 0.9995 && Number.isFinite(dOwn)) {
    ownPts = (expit(logit(before) + slope * dOwn) - before) * 100;
    perStroke = Math.abs(dOwn) > 1e-6 ? ownPts / dOwn : before * (1 - before) * slope * 100;
  }
  const causes: Cause[] = [];
  for (const [key, s] of Object.entries(strokes) as Array<[CauseKey, number]>) {
    if (!finite(s) || Math.abs(s) < 1e-9) continue;
    const pts = perStroke === null ? null : perStroke * s;
    const label = CAUSE_LABEL[key];
    causes.push({ key, label, strokes: s, pts, detail: detail[key] ?? null, chip: key === "unseparated" ? CAUSE_LABEL.unseparated : strokeChip({ key, label, strokes: s }) });
  }
  const fieldPts = ownPts === null ? null : tot - ownPts;
  if (fieldPts !== null) causes.push(fieldCause(ctx, fieldPts));
  // Name only what carries a real share. Without a probability share, fall back to size in strokes.
  const sizeOf = (c: Cause): number => (c.pts !== null ? Math.abs(c.pts) : 0);
  const enough = (c: Cause): boolean => {
    if (c.key === "unseparated") return c.pts !== null && Math.abs(c.pts) >= CAUSE_SHARE_MIN * Math.abs(tot);
    if (c.pts !== null) return Math.abs(c.pts) >= CAUSE_SHARE_MIN * Math.abs(tot) && (c.strokes === null || Math.abs(c.strokes) >= CAUSE_STROKES_MIN);
    return c.strokes !== null && Math.abs(c.strokes) >= CAUSE_STROKES_MIN;
  };
  const ranked = causes.filter(enough).sort((x, y) => sizeOf(y) - sizeOf(x) || Math.abs(y.strokes ?? 0) - Math.abs(x.strokes ?? 0));
  const onlyUnexplainedField = ranked.length > 0 && ranked.every((c) => c.key === "field") && ctx.leavers.length === 0 && ctx.joiners.length === 0;
  if (ranked.length === 0 || onlyUnexplainedField) {
    const se = noiseSe(after, before, ctx);
    if (se !== null && Math.abs(tot) < NOISE_Z * se) return { separable: true, causes: [{ key: "noise", label: CAUSE_LABEL.noise, strokes: null, pts: tot, detail: `Smaller than ${NOISE_Z} standard errors of the simulation (${se.toFixed(2)} pts).`, chip: "within simulation noise" }] };
    if (ranked.length === 0) return notSeparable("no single cause carries a clear share of the move");
  }
  const top = ranked.slice(0, CAUSE_CHIPS);
  return { separable: !top.every((c) => c.key === "unseparated"), causes: top };
}

/* ------------------------------------------------------------------ price movers */
function moversFor(cur: ExplainDoc, prevById: Map<number, ExPlayer> | null, markets: readonly Market[], ctx: AttributionContext): Partial<Record<Market, Mover[]>> {
  const out: Partial<Record<Market, Mover[]>> = {};
  for (const m of markets) {
    const rows: Mover[] = [];
    for (const p of cur.players) {
      if (p.withdrawn) continue;
      const after = p.probs[m].model;
      let before: number | null = null;
      let marketDelta: number | null = null;
      const q = prevById?.get(p.id) ?? null;
      if (q) {
        before = q.probs[m].model;
        if (p.probs[m].market !== null && q.probs[m].market !== null) marketDelta = (p.probs[m].market! - q.probs[m].market!) * 100;
      } else if (!prevById) {
        const d = p.live?.since_last?.[m];
        if (after !== null && typeof d === "number" && Number.isFinite(d)) before = after - d;
      }
      if (before === null || after === null) continue;
      const delta = (after - before) * 100;
      if (Math.abs(delta) >= MOVE_MIN_PTS[m]) {
        const a = prevById && !q ? { causes: [{ key: "field" as const, label: "new in the field", strokes: null, pts: null, detail: null, chip: "new in the field" }], separable: false } : attribute(p, q, m, ctx, cur);
        rows.push({ id: p.id, name: plainName(p.name), market: m, before, after, delta, marketDelta, causes: a.causes, separable: a.separable });
      }
    }
    rows.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || (a.name ?? "").localeCompare(b.name ?? ""));
    if (rows.length) out[m] = rows;
  }
  return out;
}

/* ------------------------------------------------------------------ sportsbook moves */
/** Recomputed from model and sportsbook when both exist (equals the published `rel`, and cannot go stale); `rel` is the fallback. */
const gapOf = (e: { model: number | null; market: number | null; rel: number | null }): number | null => (finite(e.model) && finite(e.market) && e.market > 0 ? (e.model - e.market) / e.market : finite(e.rel) ? e.rel : null);
const movedBy = (a: number, b: number, m: Market): boolean => Math.abs(a - b) * 100 >= MOVE_MIN_PTS[m] && b > 0 && Math.abs(a - b) / b >= BOOK_MOVE_REL_MIN;

function bookText(kind: BookKind, name: string, m: Market, bookUp: boolean, modelUp: boolean, gb: number, ga: number): string {
  const who = m === "win" ? name : `${name} (${MARKET_LABEL[m]})`;
  const gap = `gap ${relText(gb)} → ${relText(ga)}`;
  const verb = bookUp ? "shortened" : "drifted";
  switch (kind) {
    case "book_toward": return `Book ${verb} ${who} toward our price: ${gap}`;
    case "book_away": return `Book ${verb} ${who} away from our price: ${gap}`;
    case "model_toward": return `We moved ${who} toward the book: ${gap}`;
    case "model_away": return `We moved ${who} away from the book: ${gap}`;
    default: return `Both moved on ${who} (book ${verb}, our price ${modelUp ? "up" : "down"}): ${gap}`;
  }
}

/** Classify each change of the gap to the sportsbook: who moved, and whether the edge is growing or closing. Before the event only (in play the sportsbook's prices are not live). */
export function bookMoves(cur: ExplainDoc, prev: ExplainDoc, markets: readonly Market[] = availableMarkets(cur)): BookMove[] {
  if (cur.kind !== "week" || prev.kind !== "week") return [];
  const prevById = new Map(prev.players.map((p) => [p.id, p] as const));
  const out: BookMove[] = [];
  for (const p of cur.players) {
    const q = prevById.get(p.id);
    if (!q || p.withdrawn || q.withdrawn) continue;
    for (const m of markets) {
      const a = p.probs[m], b = q.probs[m];
      if (!finite(a.model) || !finite(b.model) || !finite(a.market) || !finite(b.market) || b.market < BOOK_MIN_PROB[m]) continue;
      const bookMoved = movedBy(a.market, b.market, m);
      const modelMoved = b.model > 0 && movedBy(a.model, b.model, m);
      if (!bookMoved && !modelMoved) continue;
      const gb = gapOf(b), ga = gapOf(a);
      if (gb === null || ga === null || Math.abs(ga - gb) < BOOK_GAP_CHANGE_MIN) continue;
      const crossed = Math.sign(ga) !== Math.sign(gb) && Math.min(Math.abs(ga), Math.abs(gb)) > 0;
      const closing = crossed || Math.abs(ga) < Math.abs(gb);
      const kind: BookKind = bookMoved && modelMoved ? "both" : bookMoved ? (closing ? "book_toward" : "book_away") : closing ? "model_toward" : "model_away";
      const actionable: BookMove["actionable"] = !bookMoved ? null : closing ? (Math.abs(gb) >= EDGE_MIN ? "closing" : null) : Math.abs(ga) >= EDGE_MIN ? "growing" : null;
      const name = plainName(p.name);
      out.push({ id: p.id, name, market: m, kind, modelBefore: b.model, modelAfter: a.model, marketBefore: b.market, marketAfter: a.market, gapBefore: gb, gapAfter: ga, actionable,
        text: bookText(kind, name, m, a.market > b.market, a.model > b.model, gb, ga) });
    }
  }
  out.sort((x, y) => Number(!!y.actionable) - Number(!!x.actionable) || Math.abs(y.gapAfter - y.gapBefore) - Math.abs(x.gapAfter - x.gapBefore) || x.name.localeCompare(y.name) || x.market.localeCompare(y.market));
  return out.slice(0, MAX_BOOK_MOVES);
}

/**
 * Text lines a Telegram alert could carry: only the actionable sportsbook moves, "Edge growing: ..." / "Edge closing: ...", at most `max`.
 * Nothing here sends anything; worker/alerts.ts would call this with the WhatChanged it builds (see the report for the wiring).
 */
export function sportsbookAlertLines(moves: readonly BookMove[], max = 5): string[] {
  return moves.filter((x) => x.actionable).slice(0, max).map((x) => `${x.actionable === "growing" ? "Edge growing" : "Edge closing"}: ${x.text}`);
}

/* ------------------------------------------------------------------ the diff */
const emptyChange = (partial: boolean, baseline: WhatChanged["baseline"], gapMs: number | null): WhatChanged => ({
  partial, baseline, gapMs, movers: {}, field: { added: [], removed: [], withdrawn: [] }, weather: { rounds: [], layer: null, reissued: false }, scoring: { rounds: [], winning: null }, gaps: [], bookMoves: [], overrides: { added: [], removed: [] }, attribution: null, count: 0, quiet: true, headline: [],
});

function roundShifts(a: Array<{ round: number; v: number | null }>, b: Array<{ round: number; v: number | null }>, min: number, from = 0): RoundShift[] {
  const out: RoundShift[] = [];
  for (const x of b) {
    if (x.round <= from || x.v === null) continue;
    const y = a.find((r) => r.round === x.round);
    if (!y || y.v === null) continue;
    const delta = x.v - y.v;
    if (Math.abs(delta) >= min) out.push({ round: x.round, before: y.v, after: x.v, delta });
  }
  return out.sort((p, q) => p.round - q.round);
}

function finishChange(c: WhatChanged): WhatChanged {
  const moved = Object.values(c.movers).reduce((s, v) => s + (v?.length ?? 0), 0);
  c.count = moved + c.field.added.length + c.field.removed.length + c.field.withdrawn.length + c.weather.rounds.length + (c.weather.layer ? 1 : 0) + c.scoring.rounds.length + (c.scoring.winning ? 1 : 0) + c.gaps.length + c.bookMoves.length + c.overrides.added.length + c.overrides.removed.length;
  c.quiet = c.count === 0;
  const h: string[] = [];
  const win = c.movers.win ?? [];
  if (win.length) h.push(`${win.length} win ${win.length === 1 ? "price" : "prices"} moved`);
  else if (moved) h.push(`${moved} ${moved === 1 ? "price" : "prices"} moved`);
  const gone = c.field.removed.length + c.field.withdrawn.length;
  if (gone) h.push(`${gone} ${gone === 1 ? "player" : "players"} out`);
  if (c.field.added.length) h.push(`${c.field.added.length} added`);
  if (c.weather.rounds.length || c.weather.layer) h.push("forecast shifted");
  if (c.scoring.winning) h.push(`winning score ${c.scoring.winning.delta < 0 ? "lower" : "higher"}`);
  if (c.gaps.length) h.push(`${c.gaps.length} market ${c.gaps.length === 1 ? "gap" : "gaps"} changed`);
  const act = c.bookMoves.filter((x) => x.actionable).length;
  if (act) h.push(`${act} sportsbook ${act === 1 ? "move" : "moves"} to act on`);
  if (c.overrides.added.length || c.overrides.removed.length) h.push("manual adjustments changed");
  c.headline = h;
  return c;
}

/**
 * What changed between the document shown and an earlier run of the same event.
 * `prev` null means the earlier document is not available: an in-play document still carries each player's change since the last run, which gives the price moves alone (partial);
 * anything else gives null so the strip stays hidden. A different event, or the same run, gives null.
 */
export function whatChanged(cur: ExplainDoc, prev: ExplainDoc | null): WhatChanged | null {
  if (prev && (prev.event_uid !== cur.event_uid || (prev.kind === cur.kind && prev.run === cur.run))) return null;
  const markets = availableMarkets(cur);
  if (!prev) {
    const sl = cur.since_last;
    if (cur.kind !== "live" || !sl) return null;
    if (!cur.players.some((p) => p.live?.since_last && Object.keys(p.live.since_last).length)) return null;
    const a = stamp(cur.as_of);
    const b = stamp(sl.prev_as_of);
    const c = emptyChange(true, { run: sl.prev_run ?? null, as_of: sl.prev_as_of ?? null, after_round: sl.prev_after_round ?? null, kind: null }, a !== null && b !== null ? a - b : null);
    const ctx = attributionContext(cur, null, markets);
    ctx.prevAfterRound = sl.prev_after_round ?? 0;
    c.movers = moversFor(cur, null, markets, ctx);
    return finishChange(c);
  }
  const a = stamp(cur.as_of);
  const b = stamp(prev.as_of);
  const c = emptyChange(false, { run: prev.run, as_of: prev.as_of, after_round: prev.after_round ?? null, kind: prev.kind }, a !== null && b !== null ? a - b : null);
  const prevById = new Map(prev.players.map((p) => [p.id, p] as const));
  const curById = new Map(cur.players.map((p) => [p.id, p] as const));
  const ctx = attributionContext(cur, prev, markets);
  c.movers = moversFor(cur, prevById, markets, ctx);
  c.attribution = { fieldLevelStrokes: ctx.fieldLevel || null, logitPerStroke: Object.fromEntries(Object.entries(ctx.slope).filter(([, v]) => finite(v))) as Partial<Record<Market, number>> };
  // field
  c.field.added = cur.players.filter((p) => !p.withdrawn && !prevById.has(p.id)).map((p) => plainName(p.name));
  c.field.removed = prev.players.filter((p) => !p.withdrawn && !curById.has(p.id)).map((p) => plainName(p.name));
  c.field.withdrawn = cur.players.filter((p) => p.withdrawn && prevById.has(p.id) && !prevById.get(p.id)!.withdrawn).map((p) => plainName(p.name));
  // weather: only rounds still to play
  const from = cur.kind === "live" ? cur.after_round ?? 0 : 0;
  const cw = cur.course_card.weather;
  const pw = prev.course_card.weather;
  c.weather.rounds = roundShifts((pw?.field ?? []).map((f) => ({ round: f.round, v: f.common })), (cw?.field ?? []).map((f) => ({ round: f.round, v: f.common })), WEATHER_MIN, from);
  if (cw?.layer && pw?.layer && cw.layer !== pw.layer) c.weather.layer = { from: pw.layer, to: cw.layer };
  c.weather.reissued = !!cw?.issued_at && !!pw?.issued_at && cw.issued_at !== pw.issued_at;
  // scoring (simulation draws)
  const cs = cur.field_scores;
  const ps = prev.field_scores;
  if (cs && ps) {
    const sim = (f: FieldScores) => f.rounds.filter((r) => r.basis === "simulated").map((r) => ({ round: r.round, v: r.mean_vs_par }));
    c.scoring.rounds = roundShifts(sim(ps), sim(cs), SCORING_MIN, from);
    if (cs.winning && ps.winning) {
      const d = cs.winning.median - ps.winning.median;
      if (Math.abs(d) >= 0.5) c.scoring.winning = { before: ps.winning.median, after: cs.winning.median, delta: d };
    }
  }
  // gap to the sportsbook (before the event only: in play the sportsbook's prices are not live)
  if (cur.kind === "week" && prev.kind === "week") {
    for (const p of cur.players) {
      const q = prevById.get(p.id);
      const x = p.probs.win.rel;
      const y = q?.probs.win.rel;
      if (p.withdrawn || x === null || y === null || y === undefined || (p.probs.win.market ?? 0) < GAP_MIN_MARKET) continue;
      if (Math.abs(x - y) >= GAP_MOVE_MIN) c.gaps.push({ id: p.id, name: plainName(p.name), before: y, after: x, delta: x - y });
    }
    c.gaps.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta) || (x.name ?? "").localeCompare(y.name ?? ""));
    c.bookMoves = bookMoves(cur, prev, markets);
  }
  // manual adjustments
  const co = cur.model.overrides_applied ?? [];
  const po = prev.model.overrides_applied ?? [];
  c.overrides.added = co.filter((x) => !po.includes(x));
  c.overrides.removed = po.filter((x) => !co.includes(x));
  return finishChange(c);
}

/* ------------------------------------------------------------------ wire format of the Worker endpoint */
export const CHANGES_SCHEMA = "golfprice.changes.v1";
/** Part of the Worker's cache key for stored diffs: bump whenever the diff, attribution or wording changes, or diffs cached by older code are served forever. */
export const CHANGES_LOGIC_VERSION = 2;
/** Most price movers per bet type kept in the stored/served diff; `moverTotals` keeps the true counts. */
export const MAX_MOVERS_SERVED = 12;
export type ChangesPayload = { schema: typeof CHANGES_SCHEMA; event: string; from: string; to: string; change: WhatChanged; moverTotals: Partial<Record<Market, number>> };

/** The diff as the endpoint stores it: the full WhatChanged with each bet type's movers cut to MAX_MOVERS_SERVED. */
export function changesPayload(event: string, from: string, to: string, change: WhatChanged): ChangesPayload {
  const moverTotals: Partial<Record<Market, number>> = {};
  const movers: WhatChanged["movers"] = {};
  for (const m of MARKETS) {
    const rows = change.movers[m];
    if (!rows?.length) continue;
    moverTotals[m] = rows.length;
    movers[m] = rows.slice(0, MAX_MOVERS_SERVED);
  }
  return { schema: CHANGES_SCHEMA, event, from, to, change: { ...change, movers }, moverTotals };
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
/** The WhatChanged inside an endpoint answer, or null when the answer is not the expected shape (the client then falls back to the document diff). */
export function parseChangesPayload(raw: unknown): WhatChanged | null {
  if (!isObj(raw) || raw.schema !== CHANGES_SCHEMA || !isObj(raw.change)) return null;
  const c = raw.change;
  if (!isObj(c.movers) || !isObj(c.baseline) || !isObj(c.field) || !isObj(c.weather) || !isObj(c.scoring) || !isObj(c.overrides)) return null;
  if (!Array.isArray(c.gaps) || !Array.isArray(c.headline) || typeof c.quiet !== "boolean" || typeof c.count !== "number") return null;
  const rowsOk = Object.values(c.movers).every((v) => Array.isArray(v) && v.every((r) => isObj(r) && typeof r.id === "number" && typeof r.delta === "number" && Array.isArray(r.causes)));
  if (!rowsOk) return null;
  return { ...(c as unknown as WhatChanged), bookMoves: Array.isArray(c.bookMoves) ? (c.bookMoves as BookMove[]) : [] };
}
