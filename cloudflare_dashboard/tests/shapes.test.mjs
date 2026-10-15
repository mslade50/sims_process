// Distribution shapes: typical-shape curve, residual signs, pair selection, why-text, degenerate inputs, render smoke.
// Fixtures (tests/fixtures/shape-*.json) are trimmed copies of the real Open de Espana pre-event run and the Baycurrent after-round-1 run.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { parseExplain } from "../app/explain-rules.ts";
import { analyzeShapes, availableMarkets, curveAt, curveXMax, cutLineAt, finishBuckets, finishCurve, fitCurve, kindLabel, MATCH_WIDE, pairWhy, playerWhy, signedPp, topPlayers } from "../app/shape-rules.ts";
import { renderView } from "./helpers/render-harness.mjs";

const load = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));
const weekRaw = load("shape-week.json");
const liveRaw = load("shape-live.json");
const week = parseExplain(weekRaw);
const live = parseExplain(liveRaw);
const clone = (o) => JSON.parse(JSON.stringify(o));
const doc = (raw) => parseExplain(clone(raw));

test("fixtures parse as explain documents", () => {
  assert.ok(week && live);
  assert.equal(week.kind, "week");
  assert.equal(live.kind, "live");
  assert.ok(topPlayers(week).length === 30 && topPlayers(live).length === 30);
});

/* ------------------------------------------------------------------ curve fit */
test("fitCurve recovers an exact line and ignores one wild outlier", () => {
  const xs = [-5, -4.5, -4, -3.5, -3, -2.5, -2];
  const ys = xs.map((x) => 0.7 * x + 2);
  const exact = fitCurve(xs, ys);
  assert.equal(exact.method, "robust");
  assert.ok(Math.abs(exact.b - 0.7) < 1e-6 && Math.abs(exact.a - 2) < 1e-6);
  const dirty = fitCurve(xs, ys.map((y, i) => (i === 3 ? y + 3 : y)));
  assert.ok(Math.abs(dirty.b - 0.7) < 0.12, `robust slope ${dirty.b}`);
});
test("fitCurve falls back gracefully: under five points, or no spread in x", () => {
  assert.equal(fitCurve([1, 2, 3], [1, 2, 3]).method, "none");
  assert.equal(fitCurve([], []).method, "none");
  const flat = fitCurve([-3, -3, -3, -3, -3, -3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  assert.equal(flat.method, "mean");
  assert.equal(flat.b, 0);
  assert.equal(fitCurve([1, 2, NaN, 4, 5, 6], [1, 2, 3, NaN, 5, 6]).n, 4);
});
test("the fitted shape curve rises with win chance and stays a probability", () => {
  const a = analyzeShapes(week, "top_20");
  assert.equal(a.fit.method, "robust");
  assert.ok(a.fit.b > 0.4 && a.fit.b < 1.1, `slope ${a.fit.b}`);
  assert.ok(curveAt(a.fit, 0.2) > curveAt(a.fit, 0.01));
  for (const w of [0.0001, 0.5, 0.999]) assert.ok(curveAt(a.fit, w) > 0 && curveAt(a.fit, w) < 1);
});

/* ------------------------------------------------------------------ residual sign and kinds */
test("residual sign matches the kind and the arithmetic", () => {
  for (const d of [week, live]) {
    const a = analyzeShapes(d, "top_20");
    assert.ok(a.ok);
    for (const p of a.players) {
      assert.ok(Math.abs(p.residualPp - (p.p - curveAt(a.fit, p.win)) * 100) < 1e-9);
      if (p.kind === "steady") assert.ok(p.residualPp > 0);
      if (p.kind === "bust") assert.ok(p.residualPp < 0);
      if (p.kind === "typical") assert.ok(Math.abs(p.residualPp) < 3 && (Math.abs(p.residualPp) < 0.3 || Math.abs(p.rel) < 0.03));
    }
    const kinds = new Set(a.players.map((p) => p.kind));
    assert.ok(kinds.has("steady") && kinds.has("bust"));
    assert.deepEqual(a.unusual.map((p) => Math.abs(p.residualPp)), [...a.unusual.map((p) => Math.abs(p.residualPp))].sort((x, y) => y - x));
  }
});
test("a player with more top-20 chance than his win chance implies is Steady, fewer is Boom-or-bust (synthetic)", () => {
  const d = doc(weekRaw);
  const top = topPlayers(d);
  const a0 = analyzeShapes(d, "top_20");
  const hi = top[10], lo = top[11];
  hi.probs.top_20.model = Math.min(0.95, curveAt(a0.fit, hi.probs.win.model) + 0.08);
  lo.probs.top_20.model = Math.max(0.01, curveAt(a0.fit, lo.probs.win.model) - 0.08);
  const a = analyzeShapes(d, "top_20");
  assert.equal(a.players.find((p) => p.id === hi.id).kind, "steady");
  assert.equal(a.players.find((p) => p.id === lo.id).kind, "bust");
  assert.equal(kindLabel("steady"), "Steady");
  assert.equal(kindLabel("bust"), "Boom-or-bust");
  assert.equal(signedPp(2.04), "+2.0 pp");
  assert.equal(signedPp(-0.01), "0.0 pp");
});

/* ------------------------------------------------------------------ pair selection */
test("pairs have matching win chances, the steadier one first, and no player twice", () => {
  for (const d of [week, live]) {
    const a = analyzeShapes(d, "top_20");
    assert.ok(a.pairs.length >= 1 && a.pairs.length <= 3);
    const ids = new Set();
    for (const pr of a.pairs) {
      assert.ok(pr.winRatio <= MATCH_WIDE + 1e-9);
      assert.ok(pr.steady.p > pr.bust.p);
      assert.ok(Math.abs(pr.gapPp - (pr.steady.p - pr.bust.p) * 100) < 1e-9);
      assert.ok(Math.abs(pr.gapPp - (pr.winPartPp + pr.shapePartPp)) < 1e-6, "gap = win part + shape part");
      assert.ok(pr.shapePartPp > 0);
      for (const id of [pr.steady.id, pr.bust.id]) { assert.ok(!ids.has(id)); ids.add(id); }
    }
    // the first pair has the largest shape part of any win-matched pair
    const best = a.pairs[0].shapePartPp;
    for (const x of a.players) for (const y of a.players) {
      if (x.id >= y.id || Math.max(x.win, y.win) / Math.min(x.win, y.win) > 1.15) continue;
      assert.ok(Math.abs(x.residualPp - y.residualPp) <= best + 1e-9);
    }
  }
});
test("market choice changes the fit and the labels", () => {
  const t5 = analyzeShapes(week, "top_5");
  const t20 = analyzeShapes(week, "top_20");
  assert.equal(t5.market, "top_5");
  assert.notDeepEqual(t5.fit, t20.fit);
  assert.ok(t5.pairs[0].gapPp < t20.pairs[0].gapPp + 5);
  assert.deepEqual(availableMarkets(week).sort(), ["make_cut", "top_10", "top_20", "top_5"]);
});

/* ------------------------------------------------------------------ field variety and the honest message */
test("pre-event shapes are alike: narrow spread band, narrow tier, plain wording", () => {
  const a = analyzeShapes(week, "top_20");
  assert.equal(a.variety.sdMin.toFixed(2), "2.39");
  assert.equal(a.variety.sdMax.toFixed(2), "2.74");
  assert.equal(a.variety.tier, "narrow");
  assert.match(a.variety.text, /Shapes are alike this week/);
  assert.match(a.variety.text, /2\.39 to 2\.74/);
  assert.ok(a.variety.maxGapPp < 5);
});
test("after round 1 the variety is wider and spread explains most of it", () => {
  const pre = analyzeShapes(week, "top_20");
  const a = analyzeShapes(live, "top_20");
  assert.notEqual(a.variety.tier, "narrow");
  assert.ok(a.variety.residualSd > 2 * pre.variety.residualSd);
  assert.ok(a.variety.spreadCorr < -0.8 && a.variety.spreadR2 > 0.6, `corr ${a.variety.spreadCorr}`);
  assert.match(a.variety.text, /Spread explains about \d+%/);
  assert.equal(a.drivers[0].key, "spread");
  assert.ok(a.live && a.afterRound === 1);
  assert.ok(a.drivers.some((d) => d.key === "position"));
  assert.ok(!a.cut && !a.drivers.some((d) => d.key === "cut"));
});

/* ------------------------------------------------------------------ why text */
test("a spread-driven pair's why-text names the stroke difference and uses only real numbers", () => {
  const a = analyzeShapes(live, "top_20");
  const pr = a.pairs[0];
  assert.equal(pr.primary?.key, "spread");
  const sdGap = pr.bust.sd - pr.steady.sd;
  assert.ok(sdGap > 0);
  assert.ok(pr.why.includes(`${sdGap.toFixed(2)} strokes more`), pr.why);
  assert.ok(pr.why.includes(pr.bust.sd.toFixed(2)) && pr.why.includes(pr.steady.sd.toFixed(2)));
  assert.match(pr.why, /rare big weeks/);
  assert.ok(!/NaN|undefined/.test(pr.why));
});
test("when no input explains the gap the why-text says so rather than inventing a driver", () => {
  const d = doc(weekRaw);
  for (const p of d.players) { p.sd = 2.6; p.wave = null; if (p.finish) p.finish.p_miss_cut = 0.3; }
  const a = analyzeShapes(d, "top_20");
  assert.ok(a.ok && a.pairs.length);
  for (const pr of a.pairs) {
    assert.equal(pr.primary, null);
    assert.match(pr.why, /No single input|Most of that gap is just/);
  }
  assert.equal(a.drivers.find((x) => x.key === "spread"), undefined, "no spread variation, so no spread driver");
  assert.match(a.variety.text, /alike|variety/);
});
test("a driver that points the wrong way is never used as the explanation", () => {
  const a = analyzeShapes(live, "top_20");
  const pr = { ...a.pairs[0], steady: { ...a.pairs[0].steady, sd: 3.0 }, bust: { ...a.pairs[0].bust, sd: 2.5 } };
  const text = pairWhy({ ...pr, primary: null }, "top_20", false, 1);
  assert.doesNotMatch(text, /strokes more/);
});
test("player why-text compares spread with the field median and flags cut risk and live position", () => {
  const a = analyzeShapes(week, "top_20");
  const p = a.players.find((x) => x.sdPct > 0.05) ?? a.players[0];
  const t = playerWhy(a, p);
  assert.match(t, /strokes (more|less) than the field median|about as much as the typical/);
  assert.ok(!/NaN|undefined/.test(t));
  const la = analyzeShapes(live, "top_20");
  assert.match(playerWhy(la, la.players.find((x) => x.pos)), /after round 1/);
  const risky = { ...a.players[0], pMiss: 0.4 };
  assert.match(playerWhy(a, risky), /40% chance of missing the cut/);
});

/* ------------------------------------------------------------------ degenerate inputs */
test("null, empty and tiny fields return a plain reason instead of throwing", () => {
  for (const bad of [null, undefined, { players: [] }, { kind: "week", players: undefined }]) {
    const a = analyzeShapes(bad, "top_20");
    assert.equal(a.ok, false);
    assert.ok(a.reason);
    assert.deepEqual(a.pairs, []);
  }
  const small = doc(weekRaw);
  small.players = topPlayers(small).slice(0, 4);
  const a = analyzeShapes(small, "top_20");
  assert.equal(a.ok, false);
  assert.match(a.reason, /five/);
});
test("a market with no prices falls back to another market; no usable market is a reason", () => {
  const d = doc(weekRaw);
  for (const p of d.players) p.probs.top_20.model = null;
  const a = analyzeShapes(d, "top_20");
  assert.ok(a.ok);
  assert.notEqual(a.market, "top_20");
  const none = doc(weekRaw);
  for (const p of none.players) for (const m of ["top_5", "top_10", "top_20", "make_cut"]) p.probs[m].model = null;
  const n = analyzeShapes(none, "top_20");
  assert.equal(n.ok, false);
  assert.match(n.reason, /no usable finish prices/);
});
test("an event without a cut has no make-cut market and no cut risk", () => {
  assert.ok(!availableMarkets(live).includes("make_cut"));
  const a = analyzeShapes(live, "make_cut");
  assert.equal(a.market, "top_20");
  assert.ok(a.players.every((p) => p.pMiss === null));
});
test("withdrawn players and players without a win price are not in the top 30", () => {
  const d = doc(weekRaw);
  const first = topPlayers(d)[0];
  d.players.find((p) => p.id === first.id).withdrawn = true;
  d.players[5].probs.win.model = null;
  const ids = topPlayers(d).map((p) => p.id);
  assert.ok(!ids.includes(first.id));
  assert.ok(!ids.includes(d.players[5].id));
  assert.equal(ids.length, 30);
  assert.ok(analyzeShapes(d, "top_20").players.every((p) => p.id !== first.id));
});
test("missing sd or finish arrays degrade to prices-only without NaN", () => {
  const d = doc(weekRaw);
  for (const p of topPlayers(d).slice(0, 12)) { p.sd = null; p.finish = null; }
  const a = analyzeShapes(d, "top_20");
  assert.ok(a.ok);
  const bare = a.players.filter((p) => p.sd === null);
  assert.ok(bare.length >= 12 && bare.every((p) => p.sdGap === null && p.sdPct === null && !p.hasFinish && p.pMiss === null));
  for (const pr of a.pairs) assert.ok(!/NaN|undefined/.test(pr.why));
  assert.ok(a.players.every((p) => Number.isFinite(p.residualPp)));
  const all = doc(weekRaw);
  for (const p of all.players) p.sd = null;
  const b = analyzeShapes(all, "top_20");
  assert.ok(b.ok);
  assert.equal(b.variety.sdMin, null);
  assert.match(b.variety.text, /not available/);
  assert.equal(b.drivers.find((x) => x.key === "spread"), undefined);
});
test("identical win chances do not break the fit", () => {
  const d = doc(weekRaw);
  for (const p of topPlayers(d)) p.probs.win.model = 0.02;
  const a = analyzeShapes(d, "top_20");
  assert.ok(a.ok);
  assert.equal(a.fit.method, "mean");
  assert.ok(a.players.every((p) => Number.isFinite(p.residualPp)));
});

/* ------------------------------------------------------------------ finish curves */
test("finish curves keep the finishing mass, drop missed-cut mass, and bound the axis", () => {
  const p = week.players.find((x) => x.id === analyzeShapes(week, "top_20").pairs[0].bust.id);
  const c = finishCurve(p.finish.pos, p.finish.p_miss_cut, true);
  const mass = c.y.reduce((s, v) => s + v, 0) / 100;
  assert.ok(Math.abs(mass - (1 - p.finish.p_miss_cut)) < 0.02, `mass ${mass} vs ${1 - p.finish.p_miss_cut}`);
  assert.equal(c.pMiss, p.finish.p_miss_cut);
  const short = finishCurve(p.finish.pos, p.finish.p_miss_cut, true, 40);
  assert.equal(short.y.length, 40);
  assert.equal(finishCurve(null, 0, true), null);
  assert.equal(finishCurve([], 0, true), null);
  const xmax = curveXMax([p], true);
  assert.ok(xmax >= 25 && xmax <= p.finish.pos.length);
  assert.equal(curveXMax([null, undefined], true), 0);
  assert.equal(cutLineAt(week, 100), 65);
  assert.equal(cutLineAt(week, 40), null);
  assert.equal(cutLineAt(live, 72), null);
});
test("the steadier player's curve is taller near the top 20 than the boom-or-bust player's", () => {
  const pr = analyzeShapes(live, "top_20").pairs[0];
  const get = (id) => live.players.find((x) => x.id === id).finish;
  const top = (id) => get(id).pos.slice(0, 20).reduce((s, v) => s + v, 0);
  assert.ok(top(pr.steady.id) > top(pr.bust.id));
});

/* ------------------------------------------------------------------ render smoke */
test("ShapeSpotlightCard renders the pair, the honest summary and no Recharts", () => {
  const html = renderView("ShapeViews.tsx", "ShapeSpotlightCard", { props: { doc: week, openPlayer: () => {}, onSeeMore: () => {} } });
  assert.match(html, /Same chance to win, different weeks/);
  assert.match(html, /Shapes are alike this week/);
  assert.match(html, /<svg/);
  assert.match(html, /See all 30 shapes/);
  assert.doesNotMatch(html, /\[object Object\]|undefined|NaN/);
  const src = readFileSync(new URL("../app/ShapeViews.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(src, /recharts|ExplainCharts/);
});
test("ShapeSpotlightCard renders nothing without a usable doc", () => {
  for (const bad of [null, { players: [] }]) assert.equal(renderView("ShapeViews.tsx", "ShapeSpotlightCard", { props: { doc: bad } }), "");
});
test("ShapeExplorer renders scatter, ranked list and table for both runs, and an explanation when empty", () => {
  for (const d of [week, live]) {
    const html = renderView("ShapeViews.tsx", "ShapeExplorer", { props: { doc: d, onPick: () => {} } });
    assert.match(html, /Most unusual shapes/);
    assert.match(html, /Win chance \(log scale\)/);
    assert.match(html, /All 30 players/);
    assert.match(html, /Compare these two in the finish chart below/);
    assert.doesNotMatch(html, /\[object Object\]|undefined|NaN/);
  }
  const empty = renderView("ShapeViews.tsx", "ShapeExplorer", { props: { doc: { ...week, players: [] } } });
  assert.match(empty, /Fewer than five|No simulation/);
});
test("the Distributions page shows the shapes panel above the player comparison", () => {
  const event = { event_uid: "e1", name: "Fixture Open", tour: "euro", date_start: "2026-10-08", explain_key: "golfprice/explain/e1/week.json", distribution_runs: [{ run: "w", kind: "week", as_of: "2026-10-08T02:31:00Z", after_round: null, key: "golfprice/explain/e1/week.json", validated: true }] };
  const html = renderView("DistributionView.tsx", "DistributionView", { data: { "golfprice/index.json": { events: [event] }, "golfprice/explain/e1/week.json": weekRaw }, pathname: "/distributions" });
  const shapes = html.indexOf("Same chance to win, different weeks");
  assert.ok(shapes > 0 && shapes < html.indexOf("Up to four on the same axes"));
  assert.doesNotMatch(html, /\[object Object\]|undefined|NaN/);
});

test("finish buckets for the paired histogram sum to one and split out the missed cut", () => {
  const p = week.players.find((x) => x.id === analyzeShapes(week, "top_20").pairs[0].bust.id);
  const b = finishBuckets(p.finish.pos, p.finish.p_miss_cut, true);
  assert.deepEqual(b.slice(0, 4).map((x) => x.label), ["Win", "2–5", "6–10", "11–20"]);
  assert.equal(b.at(-1).label, "Missed cut");
  assert.equal(b.at(-1).p, p.finish.p_miss_cut);
  assert.ok(Math.abs(b.reduce((s, x) => s + x.p, 0) - 1) < 0.01);
  assert.ok(Math.abs(b[0].p - p.finish.pos[0]) < 1e-9);
  const nc = finishBuckets([0.5, 0.3, 0.2], null, false);
  assert.deepEqual(nc.map((x) => [x.label, x.p]), [["Win", 0.5], ["2+", 0.5]]);
  assert.equal(finishBuckets(null, 0, true), null);
});
