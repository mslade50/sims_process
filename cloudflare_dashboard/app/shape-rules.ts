/**
 * Distribution shapes: which top-30 players have an unusual finish profile for their win chance, and why.
 * Pure logic (no React). The "typical shape" is a robust curve logit(P(market)) = a + b * ln(P(win)) fitted across the
 * top 30 by model win chance; a player's residual in percentage points against that curve is his shape signal:
 * above it = "Steady" (more top-N finishes than the win chance implies, a high floor), below = "Boom-or-bust".
 * In the simulation the main mechanical driver is `sd` (round-to-round spread). Explanations use only real fields
 * (sd, mu, cut risk, wave, live position) and say plainly when nothing explains a gap.
 */
import type { ExplainDoc, ExPlayer } from "./explain-rules";
import { finishingProbs } from "./explain-rules.ts";
import { playerDisplayName } from "./player-name.ts";

export const SHAPE_MARKETS = ["top_5", "top_10", "top_20", "make_cut"] as const;
export type ShapeMarket = (typeof SHAPE_MARKETS)[number];
export const SHAPE_MARKET_LABEL: Record<ShapeMarket, string> = { top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", make_cut: "Make cut" };
export const SHAPE_TOP_N = 30;
/** Two players count as "the same chance to win" when the larger win chance is within this ratio of the smaller: 15% first, 20% only when nothing matches at 15%. */
export const MATCH_TIGHT = 1.15;
export const MATCH_WIDE = 1.2;
/** A residual smaller than this share of the typical-curve value (and under 0.3 pp) is called typical, not steady or boom-or-bust. */
export const TYPICAL_REL = 0.03;
export const TYPICAL_PP = 0.3;
const MIN_FIT_N = 5;
const MIN_DRIVER_N = 8;

export type ShapeKind = "steady" | "bust" | "typical";
export type FitMethod = "robust" | "mean" | "none";
export type ShapeFit = { a: number; b: number; n: number; method: FitMethod };

const clampP = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p));
export const logit = (p: number): number => Math.log(clampP(p) / (1 - clampP(p)));
export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
function median(xs: number[]): number | null {
  const s = xs.filter(isNum).sort((a, b) => a - b);
  if (!s.length) return null;
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function wls(xs: number[], ys: number[], ws: number[]): { a: number; b: number } | null {
  const sw = ws.reduce((s, w) => s + w, 0);
  if (sw <= 0) return null;
  const mx = ws.reduce((s, w, i) => s + w * xs[i], 0) / sw;
  const my = ws.reduce((s, w, i) => s + w * ys[i], 0) / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) { sxx += ws[i] * (xs[i] - mx) ** 2; sxy += ws[i] * (xs[i] - mx) * (ys[i] - my); }
  if (sxx < 1e-9) return null;
  const b = sxy / sxx;
  return Number.isFinite(b) ? { a: my - b * mx, b } : null;
}
/** Huber-weighted straight line through (x, y); falls back to the mean when x has no spread, and to "none" under 5 points. */
export function fitCurve(xs: number[], ys: number[]): ShapeFit {
  const pairs = xs.map((x, i) => [x, ys[i]] as const).filter(([x, y]) => isNum(x) && isNum(y));
  const x = pairs.map((p) => p[0]), y = pairs.map((p) => p[1]);
  const n = pairs.length;
  if (n < MIN_FIT_N) return { a: n ? mean(y) : 0, b: 0, n, method: "none" };
  let w = x.map(() => 1);
  let line = wls(x, y, w);
  if (!line) return { a: mean(y), b: 0, n, method: "mean" };
  for (let it = 0; it < 8; it++) {
    const r = y.map((v, i) => v - (line!.a + line!.b * x[i]));
    const mad = median(r.map((v) => Math.abs(v - (median(r) ?? 0)))) ?? 0;
    const k = 1.345 * Math.max(mad * 1.4826, 0.02);
    w = r.map((v) => (Math.abs(v) <= k ? 1 : k / Math.abs(v)));
    const next = wls(x, y, w);
    if (!next) break;
    line = next;
  }
  return { a: line.a, b: line.b, n, method: "robust" };
}
export const curveAt = (fit: ShapeFit, win: number): number => sigmoid(fit.a + fit.b * Math.log(clampP(win)));

/* ------------------------------------------------------------------ players */
export type ShapePlayer = {
  id: number; name: string; rank: number; win: number; p: number; expected: number; residualPp: number; rel: number; kind: ShapeKind;
  mu: number | null; sd: number | null; sdGap: number | null; sdPct: number | null; pMiss: number | null; wave: string | null;
  pos: string | null; posN: number | null; toPar: number | null; hasFinish: boolean;
};
export const shortName = (raw: string): string => playerDisplayName(raw);
export const hasCut = (doc: ExplainDoc): boolean => !!doc.event?.cut_round || !!doc.event?.cut_top_n;
const winOf = (p: ExPlayer): number | null => { const v = p.probs?.win?.model; return isNum(v) && v > 0 ? v : null; };
const probOf = (p: ExPlayer, m: ShapeMarket): number | null => { const v = p.probs?.[m]?.model; return isNum(v) && v > 0 && v < 1 ? v : null; };

/** The top 30 (withdrawn players and players without a win price excluded), best win chance first. */
export function topPlayers(doc: ExplainDoc, n = SHAPE_TOP_N): ExPlayer[] {
  return (doc.players ?? []).filter((p) => !p.withdrawn && winOf(p) !== null).sort((a, b) => (winOf(b) ?? 0) - (winOf(a) ?? 0)).slice(0, n);
}
/** Field median of round spread (all non-withdrawn players with an sd). */
export function fieldMedianSd(doc: ExplainDoc): number | null {
  return median((doc.players ?? []).filter((p) => !p.withdrawn && isNum(p.sd) && (p.sd as number) > 0).map((p) => p.sd as number));
}
/** Markets that have enough spread of real prices among the top 30 to define a shape (make cut needs an event with a cut). */
export function availableMarkets(doc: ExplainDoc): ShapeMarket[] {
  const top = topPlayers(doc);
  return SHAPE_MARKETS.filter((m) => {
    if (m === "make_cut" && !hasCut(doc)) return false;
    const vals = top.map((p) => probOf(p, m)).filter(isNum);
    return vals.length >= MIN_FIT_N && Math.max(...vals) - Math.min(...vals) > 0.002;
  });
}

/* ------------------------------------------------------------------ drivers */
export type DriverKey = "spread" | "cut" | "wave" | "position";
export type DriverFit = { key: DriverKey; label: string; slope: number; corr: number; r2: number; n: number; strength: "strong" | "some" | "weak" };
const strengthOf = (r: number): DriverFit["strength"] => (Math.abs(r) >= 0.5 ? "strong" : Math.abs(r) >= 0.3 ? "some" : "weak");
const DRIVER_LABEL: Record<DriverKey, string> = { spread: "Round-to-round spread", cut: "Cut risk", wave: "Tee-time wave", position: "Current leaderboard position" };

function driverX(p: ShapePlayer, key: DriverKey, live: boolean): number | null {
  if (key === "spread") return p.sd;
  if (key === "cut") return p.pMiss;
  if (key === "wave") return p.wave === "late" ? 1 : p.wave === "early" ? 0 : null;
  return live && isNum(p.posN) ? Math.log(Math.max(1, p.posN)) : null;
}
function simpleFit(xs: number[], ys: number[]): { slope: number; corr: number; n: number } | null {
  const n = xs.length;
  if (n < MIN_DRIVER_N) return null;
  const mx = mean(xs), my = mean(ys);
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  if (sxx < 1e-12 || syy < 1e-12) return null;
  return { slope: sxy / sxx, corr: sxy / Math.sqrt(sxx * syy), n };
}
export function fitDrivers(players: ShapePlayer[], live: boolean, cut: boolean): DriverFit[] {
  const out: DriverFit[] = [];
  for (const key of ["spread", "cut", "wave", "position"] as DriverKey[]) {
    if ((key === "cut" && !cut) || (key === "position" && !live)) continue;
    const rows = players.map((p) => [driverX(p, key, live), p.residualPp] as const).filter((r): r is readonly [number, number] => isNum(r[0]));
    if (key === "wave") { const late = rows.filter((r) => r[0] === 1).length; if (late < 3 || rows.length - late < 3) continue; }
    const f = simpleFit(rows.map((r) => r[0]), rows.map((r) => r[1]));
    if (f) out.push({ key, label: DRIVER_LABEL[key], slope: f.slope, corr: f.corr, r2: f.corr ** 2, n: f.n, strength: strengthOf(f.corr) });
  }
  return out.sort((a, b) => b.r2 - a.r2);
}

/* ------------------------------------------------------------------ analysis */
export type ShapeDriverEffect = { key: DriverKey; label: string; effectPp: number; slope: number; strength: DriverFit["strength"] };
export type ShapePair = {
  steady: ShapePlayer; bust: ShapePlayer; gapPp: number; winRatio: number; winPartPp: number; shapePartPp: number;
  effects: ShapeDriverEffect[]; primary: ShapeDriverEffect | null; why: string;
};
export type Variety = {
  tier: "narrow" | "moderate" | "wide" | "unknown"; sdMin: number | null; sdMax: number | null; sdMedian: number | null; maxGapPp: number | null; maxGapRel: number | null; residualSd: number | null; bandLoPp: number | null; bandHiPp: number | null;
  spreadR2: number | null; spreadCorr: number | null; text: string;
};
export type ShapeAnalysis = {
  market: ShapeMarket; available: ShapeMarket[]; players: ShapePlayer[]; fit: ShapeFit; cut: boolean; live: boolean; afterRound: number | null;
  unusual: ShapePlayer[]; pairs: ShapePair[]; drivers: DriverFit[]; variety: Variety; ok: boolean; reason: string | null;
};

export const kindLabel = (k: ShapeKind): string => (k === "steady" ? "Steady" : k === "bust" ? "Boom-or-bust" : "Typical");
export const signedPp = (v: number, digits = 1): string => { const r = Number(Math.abs(v).toFixed(digits)); return `${r === 0 ? "" : v > 0 ? "+" : "−"}${r.toFixed(digits)} pp`; };
const pc = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const strokes = (v: number) => v.toFixed(2);

function kindOf(p: number, expected: number): ShapeKind {
  const diff = p - expected;
  if (Math.abs(diff) * 100 < TYPICAL_PP || Math.abs(diff) / expected < TYPICAL_REL) return "typical";
  return diff > 0 ? "steady" : "bust";
}
const EMPTY_VARIETY: Variety = { tier: "unknown", sdMin: null, sdMax: null, sdMedian: null, maxGapPp: null, maxGapRel: null, residualSd: null, bandLoPp: null, bandHiPp: null, spreadR2: null, spreadCorr: null, text: "" };
function emptyAnalysis(doc: ExplainDoc, market: ShapeMarket, reason: string, available: ShapeMarket[] = []): ShapeAnalysis {
  return { market, available, players: [], fit: { a: 0, b: 0, n: 0, method: "none" }, cut: hasCut(doc), live: doc.kind === "live", afterRound: doc.after_round ?? null, unusual: [], pairs: [], drivers: [], variety: EMPTY_VARIETY, ok: false, reason };
}

/** Everything the shape views need for one market. Degenerate inputs return `ok: false` with a plain reason, never throw. */
export function analyzeShapes(doc: ExplainDoc | null | undefined, market: ShapeMarket = "top_20"): ShapeAnalysis {
  if (!doc || !Array.isArray(doc.players)) return emptyAnalysis({ players: [] } as unknown as ExplainDoc, market, "No simulation loaded.");
  const available = availableMarkets(doc);
  const top = topPlayers(doc);
  if (top.length < MIN_FIT_N) return emptyAnalysis(doc, market, "Fewer than five priced players, so there is no typical shape to compare against.", available);
  const useMarket: ShapeMarket = available.includes(market) ? market : (available.includes("top_20") ? "top_20" : available[0]);
  if (!useMarket) return emptyAnalysis(doc, market, "This simulation has no usable finish prices for the top players.", available);
  const live = doc.kind === "live";
  const cut = hasCut(doc);
  const rows = top.map((p, i) => ({ p, i, y: probOf(p, useMarket) })).filter((r): r is { p: ExPlayer; i: number; y: number } => r.y !== null);
  const fit = fitCurve(rows.map((r) => Math.log(winOf(r.p) as number)), rows.map((r) => logit(r.y)));
  if (fit.method === "none") return emptyAnalysis(doc, useMarket, "Fewer than five players have a price in this market.", available);
  const medSd = fieldMedianSd(doc);
  const players: ShapePlayer[] = rows.map(({ p, i, y }) => {
    const win = winOf(p) as number;
    const expected = curveAt(fit, win);
    const sd = isNum(p.sd) && p.sd > 0 ? p.sd : null;
    return {
      id: p.id, name: p.name, rank: i + 1, win, p: y, expected, residualPp: (y - expected) * 100, rel: (y - expected) / expected, kind: kindOf(y, expected),
      mu: isNum(p.mu) ? p.mu : null, sd, sdGap: sd !== null && medSd !== null ? sd - medSd : null, sdPct: sd !== null && medSd ? sd / medSd - 1 : null,
      pMiss: cut && isNum(p.finish?.p_miss_cut) ? p.finish!.p_miss_cut : null, wave: p.wave ?? null,
      pos: p.live?.pos ?? null, posN: isNum(p.live?.pos_n) ? p.live!.pos_n : null, toPar: isNum(p.live?.to_par) ? p.live!.to_par : null, hasFinish: !!p.finish?.pos?.length,
    };
  });
  const drivers = fitDrivers(players, live, cut);
  const unusual = [...players].sort((a, b) => Math.abs(b.residualPp) - Math.abs(a.residualPp));
  const pairs = pickPairs(players, drivers, live, cut, useMarket, doc.after_round ?? null);
  const variety = summarizeVariety(players, drivers, pairs, medSd, useMarket);
  return { market: useMarket, available, players, fit, cut, live, afterRound: doc.after_round ?? null, unusual, pairs, drivers, variety, ok: true, reason: null };
}

/* ------------------------------------------------------------------ pairs */
function effectsFor(a: ShapePlayer, b: ShapePlayer, drivers: DriverFit[], live: boolean): ShapeDriverEffect[] {
  const out: ShapeDriverEffect[] = [];
  for (const d of drivers) {
    const xa = driverX(a, d.key, live), xb = driverX(b, d.key, live);
    if (!isNum(xa) || !isNum(xb)) continue;
    out.push({ key: d.key, label: d.label, effectPp: d.slope * (xa - xb), slope: d.slope, strength: d.strength }); // effect on (steady residual - bust residual)
  }
  return out.sort((x, y) => y.effectPp - x.effectPp);
}
/** Best pair(s) of players whose win chances are within ~15% (then 20%) of each other and whose chosen-market chances differ most. Up to three, no player repeated. */
export function pickPairs(players: ShapePlayer[], drivers: DriverFit[], live: boolean, cut: boolean, market: ShapeMarket, afterRound: number | null, limit = 3): ShapePair[] {
  const out: ShapePair[] = [];
  for (const tol of [MATCH_TIGHT, MATCH_WIDE]) {
    const cands: Array<{ hi: ShapePlayer; lo: ShapePlayer; gap: number }> = [];
    for (let i = 0; i < players.length; i++) for (let j = i + 1; j < players.length; j++) {
      const a = players[i], b = players[j];
      if (Math.max(a.win, b.win) / Math.min(a.win, b.win) > tol) continue;
      const [hi, lo] = a.p >= b.p ? [a, b] : [b, a];
      if (hi.residualPp - lo.residualPp > 0) cands.push({ hi, lo, gap: hi.p - lo.p });
    }
    // Rank by the shape part of the gap (residual difference), so a pair is not chosen just because one win chance is a little higher.
    cands.sort((x, y) => (y.hi.residualPp - y.lo.residualPp) - (x.hi.residualPp - x.lo.residualPp));
    const used = new Set(out.flatMap((pr) => [pr.steady.id, pr.bust.id]));
    for (const c of cands) {
      if (out.length >= limit) break;
      if (used.has(c.hi.id) || used.has(c.lo.id)) continue;
      used.add(c.hi.id); used.add(c.lo.id);
      const effects = effectsFor(c.hi, c.lo, drivers, live);
      const shapePartPp = c.hi.residualPp - c.lo.residualPp;
      const pair: ShapePair = { steady: c.hi, bust: c.lo, gapPp: c.gap * 100, winRatio: Math.max(c.hi.win, c.lo.win) / Math.min(c.hi.win, c.lo.win), winPartPp: (c.hi.expected - c.lo.expected) * 100, shapePartPp, effects, primary: null, why: "" };
      pair.primary = primaryDriver(pair);
      pair.why = pairWhy(pair, market, cut, afterRound);
      out.push(pair);
    }
    if (out.length) break; // widen the win-match tolerance only when 15% found nothing
  }
  return out;
}
/** The driver that explains the most of the shape part of the gap, if it is positive, material and not weak. */
function primaryDriver(pair: ShapePair): ShapeDriverEffect | null {
  const { steady: a, bust: b } = pair;
  const consistent = (e: ShapeDriverEffect) => (e.key === "spread" ? a.sd !== null && b.sd !== null && b.sd > a.sd : e.key === "cut" ? a.pMiss !== null && b.pMiss !== null && b.pMiss > a.pMiss : e.key === "position" ? !!a.pos && !!b.pos : e.key === "wave" ? !!a.wave && !!b.wave : false);
  const best = pair.effects.find((e) => e.effectPp > 0 && consistent(e));
  if (!best) return null;
  if (best.effectPp < 0.25 || pair.shapePartPp <= 0 || best.effectPp < 0.2 * pair.shapePartPp || best.strength === "weak") return null;
  return best;
}
function cutLabel(cut: boolean, m: ShapeMarket) { return cut && m === "make_cut" ? "make-cut" : SHAPE_MARKET_LABEL[m].toLowerCase(); }

export function pairWhy(pair: ShapePair, market: ShapeMarket, cut: boolean, afterRound: number | null): string {
  const { steady: a, bust: b } = pair;
  const an = shortName(a.name), bn = shortName(b.name);
  const lead = `${bn} and ${an} have about the same win chance (${pc(b.win)} and ${pc(a.win)}), but ${an}'s ${cutLabel(cut, market)} chance is ${signedPp(pair.gapPp).replace("+", "")} higher.`;
  const eff = pair.primary;
  const parts: string[] = [lead];
  if (pair.winPartPp <= -0.5) parts.push(`That is despite ${an}'s slightly lower win chance, so the shape alone is worth ${pair.shapePartPp.toFixed(1)} pp.`);
  else if (pair.winPartPp >= 0.5) parts.push(`About ${pair.winPartPp.toFixed(1)} pp of that is just ${an}'s slightly higher win chance; ${pair.shapePartPp.toFixed(1)} pp is shape.`);
  if (eff?.key === "spread" && a.sd !== null && b.sd !== null) {
    const d = b.sd - a.sd;
    const mu = !(afterRound && afterRound >= 1) && a.mu !== null && b.mu !== null && a.mu - b.mu >= 0.05 ? ` ${bn} is rated ${strokes(a.mu - b.mu)} strokes a round weaker, which his wider swings offset in the win price.` : "";
    parts.push(`${bn}'s rounds swing ${strokes(d)} strokes more (${strokes(b.sd)} vs ${strokes(a.sd)}), about ${Math.round((b.sd / a.sd - 1) * 100)}% more, so more of his chance comes from rare big weeks and fewer from steady top finishes.${mu}`);
  } else if (eff?.key === "cut" && a.pMiss !== null && b.pMiss !== null) {
    parts.push(`${bn} is likelier to miss the cut (${pc(b.pMiss, 0)} vs ${pc(a.pMiss, 0)}), which costs him top-finish chances but not much win chance.`);
  } else if (eff?.key === "wave") {
    parts.push(`${bn} plays the ${b.wave} wave and ${an} the ${a.wave} wave; this week that wave split goes with about ${Math.abs(eff.slope).toFixed(1)} pp of difference in shape across the top 30, so the weather each meets matters.`);
  } else if (eff?.key === "position" && a.pos && b.pos) {
    parts.push(`After round ${afterRound ?? "?"} ${an} is ${a.pos} and ${bn} is ${b.pos}; where each stands on the leaderboard now shapes how many finishes are still in reach.`);
  } else if (pair.shapePartPp < 0.5) {
    parts.push(`Most of that gap is just the small difference in win chance, not a different shape.`);
  } else {
    parts.push(`No single input here explains it; it comes from the mix of skill, spread and tee times in the simulation.`);
  }
  const second = pair.effects.find((e) => e !== eff && e.effectPp >= 0.5 && e.strength !== "weak" && (e.key !== "spread" || (a.sd !== null && b.sd !== null && b.sd > a.sd)) && (e.key !== "cut" || (a.pMiss !== null && b.pMiss !== null && b.pMiss > a.pMiss)));
  if (eff && second) parts.push(`${second.label} also points the same way (about ${second.effectPp.toFixed(1)} pp).`);
  return parts.join(" ");
}

/* ------------------------------------------------------------------ variety */
function summarizeVariety(players: ShapePlayer[], drivers: DriverFit[], pairs: ShapePair[], medSd: number | null, market: ShapeMarket): Variety {
  const sds = players.map((p) => p.sd).filter(isNum);
  const sdMin = sds.length ? Math.min(...sds) : null, sdMax = sds.length ? Math.max(...sds) : null;
  const best = pairs[0];
  const maxGapPp = best ? best.gapPp : null;
  const maxGapRel = best ? best.gapPp / 100 / ((best.steady.p + best.bust.p) / 2) : null;
  const sd = drivers.find((d) => d.key === "spread");
  const resid = players.map((p) => logit(p.p) - logit(p.expected));
  const residualSd = resid.length >= MIN_FIT_N ? Math.sqrt(resid.reduce((s, v) => s + v * v, 0) / resid.length) : null;
  const pps = players.map((p) => p.residualPp).sort((x, y) => x - y);
  const q = (f: number) => pps[Math.min(pps.length - 1, Math.max(0, Math.round(f * (pps.length - 1))))];
  // Tier from how far players sit from the typical curve in logit units (market-neutral): under 0.05 narrow, under 0.09 moderate, otherwise wide.
  const tier: Variety["tier"] = residualSd === null || maxGapRel === null ? "unknown" : residualSd < 0.05 ? "narrow" : residualSd < 0.09 ? "moderate" : "wide";
  const band = sdMin !== null && sdMax !== null ? `Round spread among the top 30 runs from ${strokes(sdMin)} to ${strokes(sdMax)} strokes` : "Round spread is not available";
  const label = SHAPE_MARKET_LABEL[market].toLowerCase();
  let text: string;
  if (tier === "unknown") text = `${band}; there is no pair of players with matching win chances to compare.`;
  else if (tier === "narrow") text = `Shapes are alike this week. ${band}, and among players with matching win chances the biggest ${label} gap is only ${maxGapPp!.toFixed(1)} pp. Treat small differences as noise.`;
  else if (tier === "moderate") text = `Some real variety. ${band}; among players with matching win chances the biggest ${label} gap is ${maxGapPp!.toFixed(1)} pp.`;
  else text = `Clear variety this week. ${band}; among players with matching win chances the biggest ${label} gap is ${maxGapPp!.toFixed(1)} pp.`;
  if (sd && sd.corr < -0.25) text += ` Spread explains about ${Math.round(sd.r2 * 100)}% of who sits above or below the typical shape.`;
  else if (sd && tier !== "unknown") text += ` Spread does not line up cleanly with shape this week.`;
  return { tier, sdMin, sdMax, sdMedian: medSd, maxGapPp, maxGapRel, residualSd, bandLoPp: pps.length ? q(0.1) : null, bandHiPp: pps.length ? q(0.9) : null, spreadR2: sd ? sd.r2 : null, spreadCorr: sd ? sd.corr : null, text };
}

/** One or two plain sentences about a single player's shape and the real fields behind it. */
export function playerWhy(a: ShapeAnalysis, p: ShapePlayer): string {
  const label = SHAPE_MARKET_LABEL[a.market];
  const name = shortName(p.name);
  const head = p.kind === "typical"
    ? `${name}'s ${label} chance (${pc(p.p)}) is about what his ${pc(p.win)} win chance implies.`
    : `${name}'s ${label} chance (${pc(p.p)}) is ${signedPp(Math.abs(p.residualPp)).replace("+", "")} ${p.kind === "steady" ? "above" : "below"} the typical ${pc(p.expected)} for a ${pc(p.win)} win chance: ${p.kind === "steady" ? "a steadier, higher-floor week" : "more of his chance rides on a big week"}.`;
  const bits: string[] = [];
  if (p.sd !== null && p.sdGap !== null && p.sdPct !== null) {
    if (Math.abs(p.sdPct) < 0.02) bits.push(`His rounds swing about as much as the typical player's (${strokes(p.sd)} strokes).`);
    else bits.push(`His rounds swing ${strokes(Math.abs(p.sdGap))} strokes ${p.sdGap > 0 ? "more" : "less"} than the field median (${strokes(p.sd)} vs ${strokes(p.sd - p.sdGap)}), ${Math.round(Math.abs(p.sdPct) * 100)}% ${p.sdGap > 0 ? "more" : "less"}.`);
  }
  if (a.cut && p.pMiss !== null && p.pMiss >= 0.25) bits.push(`He has a ${pc(p.pMiss, 0)} chance of missing the cut.`);
  if (a.live && p.pos) bits.push(`He is ${p.pos} after round ${a.afterRound ?? "?"}.`);
  return [head, ...bits].join(" ");
}

/* ------------------------------------------------------------------ finish curves */
export type FinishCurve = { x: number[]; y: number[]; pMiss: number; n: number };
/**
 * Smoothed chance of finishing at each position (percent per position). Missed-cut mass sits at the end of `pos` and is removed
 * first (reported as `pMiss`); a Gaussian kernel (width scaled to the field) turns tie-lumpy bars into a readable shape.
 */
export function finishCurve(pos: number[] | null | undefined, pMiss: number | null | undefined, cut: boolean, xMax?: number): FinishCurve | null {
  if (!pos || !pos.length || pos.some((v) => !isNum(v))) return null;
  const raw = finishingProbs(pos, pMiss, cut);
  const n = xMax ? Math.min(Math.max(1, Math.floor(xMax)), raw.length) : raw.length;
  const sigma = Math.max(0.8, Math.min(2.2, raw.length / 60));
  const radius = Math.ceil(sigma * 3);
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    let s = 0, w = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(raw.length - 1, i + radius); j++) {
      const k = Math.exp(-0.5 * ((j - i) / sigma) ** 2);
      s += raw[j] * k; w += k;
    }
    y.push(w ? (100 * s) / w : 0);
  }
  return { x: y.map((_, i) => i + 1), y, pMiss: cut && isNum(pMiss) ? Math.max(0, pMiss) : 0, n: raw.length };
}
/** The last position worth drawing: where the pooled finishing mass of these players reaches 99.5%, but at least 25 (or the field size when smaller). */
export function curveXMax(players: Array<ExPlayer | undefined | null>, cut: boolean): number {
  let size = 0;
  const pooled: number[] = [];
  for (const p of players) {
    if (!p?.finish?.pos?.length) continue;
    const f = finishingProbs(p.finish.pos, p.finish.p_miss_cut, cut);
    f.forEach((v, i) => { pooled[i] = (pooled[i] ?? 0) + (isNum(v) ? v : 0); });
    size = Math.max(size, f.length);
  }
  if (!size) return 0;
  const total = pooled.reduce((s, v) => s + v, 0) || 1;
  let c = 0, k = pooled.length;
  for (let i = 0; i < pooled.length; i++) { c += pooled[i]; if (c / total >= 0.995) { k = i + 1; break; } }
  return Math.min(size, Math.max(k, Math.min(25, size)));
}
/** x position of the cut line to mark, or null when the event has no cut or the line is off the drawn range. */
export function cutLineAt(doc: ExplainDoc, xMax: number): number | null {
  const n = doc.event?.cut_top_n;
  return hasCut(doc) && isNum(n) && n > 0 && n <= xMax ? n : null;
}
