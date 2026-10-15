import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  american, availableTags, biasDimensions, biasForDimension, biasSentence, binFinish, defaultCompetitors, explainKeyFor, filterWhy, finishBins, leaderboard, movers, nameMatches, orderEvents,
  biggestDgSkillGap, dgGaps, hasDg, parseExplain, pct, signed, sortWhy, toParText, toggleCompetitor, topEdges, topK, waterfall,
} from "../app/explain-rules.ts";

const read = async (n) => JSON.parse(await readFile(new URL(`./fixtures/${n}`, import.meta.url), "utf8"));
const week = parseExplain(await read("explain_week.json"));
const live = parseExplain(await read("explain_live.json"));
const await_week = await read("explain_week.json");

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

test("documents parse; anything else is rejected", () => {
  assert.ok(week && live);
  assert.equal(week.kind, "week");
  assert.equal(live.kind, "live");
  assert.equal(week.players.length, 40);
  assert.equal(parseExplain(null), null);
  assert.equal(parseExplain({ schema: "other", players: [] }), null);
  assert.equal(parseExplain({ schema: "golfprice.explain.v1", players: "x", event: {} }), null);
  const bare = parseExplain({ schema: "golfprice.explain.v1", event: { name: "E" }, players: [{ id: 1, name: "A" }] });
  assert.equal(bare.players[0].probs.win.model, null);
  assert.deepEqual(bare.players[0].tags, []);
  assert.equal(bare.players[0].finish, null);
});

test("number formatting never shows NaN and keeps signs", () => {
  assert.equal(pct(null), "-");
  assert.equal(pct(0.1234), "12.3%");
  assert.equal(pct(0.0004), "0.04%");
  assert.equal(signed(0.5), "+0.50");
  assert.equal(signed(-0.5), "−0.50");
  assert.equal(signed(undefined), "-");
  assert.equal(american(0.1), "+900");
  assert.equal(american(0.75), "−300");
  assert.equal(american(null), "-");
  assert.equal(toParText(0), "E");
  assert.equal(toParText(-3), "−3");
});

test("finish bins cover the field once and aggregate mass", () => {
  const bins = finishBins(120, 65);
  assert.equal(bins[0].lo, 1);
  for (let i = 1; i < bins.length; i++) assert.equal(bins[i].lo, bins[i - 1].hi + 1);
  assert.equal(bins.at(-1).hi, 120);
  const p = week.players.find((x) => x.finish);
  const bars = binFinish(p.finish.pos, finishBins(p.finish.pos.length, 25), p.finish.p_miss_cut);
  assert.ok(Math.abs(bars.cum.at(-1) - 1) < 0.01);
  assert.ok(bars.cum.every((v, i, a) => i === 0 || v >= a[i - 1] - 1e-12));
  assert.deepEqual(topK([0.5, 0.25, 0.25], [1, 2, 3]), [0.5, 0.75, 1]);
  assert.ok(finishBins(40, null).length >= 6);
});

test("waterfall folds small items and closes on the field-relative mean", () => {
  const p = week.players[0];
  const w = waterfall(p, week.components_legend);
  assert.ok(w.rows.length > 3);
  const skill = Object.entries(p.components).filter(([k]) => ["skill", "course", "location", "override"].includes(week.components_legend[k].group)).reduce((s, [, v]) => s + v, 0);
  assert.ok(Math.abs(skill - p.mu_rel) < 0.02);
  assert.ok(Math.abs(w.total - Object.values(p.components).reduce((s, v) => s + v, 0)) < 1e-9);
  const order = ["skill", "course", "location", "override", "live", "weather"];
  const idx = w.rows.map((r) => order.indexOf(r.group));
  assert.deepEqual(idx, [...idx].sort((a, b) => a - b));
});

test("why-priced filter, search and sort", () => {
  const all = week.players;
  assert.equal(filterWhy(all, { query: "", tags: [], market: "win", onlyEdge: "all" }).length, 40);
  const up = filterWhy(all, { query: "", tags: [], market: "top_10", onlyEdge: "above" });
  assert.ok(up.length > 0 && up.every((p) => p.probs.top_10.edge > 0));
  const first = all[0].name.split(",")[0];
  assert.ok(filterWhy(all, { query: first.slice(0, 4).toLowerCase(), tags: [], market: "win", onlyEdge: "all" }).some((p) => p.id === all[0].id));
  const top5 = filterWhy(all, { query: "", tags: [], market: "win", onlyEdge: "all", top: 5 });
  assert.equal(top5.length, 5);
  assert.ok(top5.every((p) => (p.probs.win.market ?? 0) >= (all.slice().sort((a, b) => (b.probs.win.market ?? 0) - (a.probs.win.market ?? 0))[5].probs.win.market ?? 0)));
  assert.equal(filterWhy(all, { query: "zzzz", tags: [], market: "win", onlyEdge: "all" }).length, 0);
  const tags = availableTags(all);
  assert.ok(tags.length > 3);
  const t = tags[0];
  assert.ok(filterWhy(all, { query: "", tags: [t], market: "win", onlyEdge: "all" }).every((p) => p.tags.includes(t) || p.wave === t));
  const s = sortWhy(all, "edge_sg", "desc", "win");
  for (let i = 1; i < s.length; i++) if (s[i].edge_sg !== null && s[i - 1].edge_sg !== null) assert.ok(s[i - 1].edge_sg >= s[i].edge_sg);
  const n = sortWhy(all, "name", "asc", "win");
  assert.ok(n[0].name.localeCompare(n[1].name) <= 0);
  const nul = sortWhy([{ ...all[0], edge_sg: null }, ...all.slice(1, 4)], "edge_sg", "asc", "win");
  assert.equal(nul.at(-1).edge_sg, null);
  assert.ok(nameMatches("Åberg, Ludvig", "lud ab"));
  assert.ok(nameMatches("O'Neill, Pat", "pat oneill"));
});

test("headline lists: top edges, bias rows and sentences", () => {
  const e = topEdges(week, 5);
  assert.ok(e.above.every((p) => p.edge_sg > 0) && e.below.every((p) => p.edge_sg < 0));
  assert.ok(e.above.length <= 5);
  const rows = biasForDimension(week, "all");
  assert.ok(rows.length > 3);
  assert.ok(biasDimensions(week).some((d) => d.value === "depth"));
  assert.ok(biasForDimension(week, "depth").every((r) => r.dimension === "depth"));
  assert.match(biasSentence(rows[0]), /(higher|lower) than the market does|in line with the rest of the field/);
});

test("competitor overlay: same-odds neighbors preselected, capped, toggled", () => {
  const p = week.players.find((x) => x.finish);
  const c = defaultCompetitors(week, p.id);
  assert.ok(c.length > 0 && c.length <= 3 && !c.includes(p.id));
  assert.deepEqual(toggleCompetitor([1, 2], 2), [1]);
  assert.deepEqual(toggleCompetitor([1, 2, 3, 4], 9), [1, 2, 3, 4]);
  assert.deepEqual(toggleCompetitor([], 9), [9]);
  assert.deepEqual(defaultCompetitors(week, -1), []);
});

test("in-play: leaderboard, movers and live fields", () => {
  const lb = leaderboard(live, 10);
  assert.ok(lb.length > 0);
  for (let i = 1; i < lb.length; i++) assert.ok((lb[i - 1].live.pos_n ?? 999) <= (lb[i].live.pos_n ?? 999));
  const p = live.players[0];
  assert.ok(p.live && "b8_delta" in p.live && Array.isArray(p.live.rounds));
  assert.ok(movers(live, 5).length <= 5);
  assert.equal(live.players[0].probs.win.market, null); // a live run without odds: edges are absent, not zero
  assert.equal(topEdges(live, 3).above.length, 0);
});

test("events are ordered newest week first and keyed to the explain object", () => {
  const ev = orderEvents([
    { event_uid: "pga:1:2026-09-24", date_start: "2026-09-24", tour: "pga" },
    { event_uid: "euro:2:2026-10-01", date_start: "2026-10-01", tour: "euro" },
    { event_uid: "pga:3:2026-10-01", date_start: "2026-10-01", tour: "pga" },
  ]);
  assert.equal(ev.at(-1).event_uid, "pga:1:2026-09-24");
  assert.deepEqual(ev.slice(0, 2).map((e) => e.event_uid).sort(), ["euro:2:2026-10-01", "pga:3:2026-10-01"]);
  assert.equal(explainKeyFor({ event_uid: "pga:554:2026-10-01" }), "golfprice/explain/pga_554_2026-10-01/latest.json");
  assert.equal(explainKeyFor({ event_uid: "x", explain_key: "golfprice/explain/y/latest.json" }), "golfprice/explain/y/latest.json");
});

test("the routes server-render and the nav lists the new views", async () => {
  const worker = await loadWorker();
  for (const path of ["/this-week", "/why-priced"]) {
    const response = await worker.fetch(new Request(`https://golf.example${path}`, { headers: { accept: "text/html" } }), { ASSETS: { fetch: async () => new Response("nf", { status: 404 }) } }, { waitUntil() {}, passThroughOnException() {} });
    assert.equal(response.status, 200, path);
    const html = await response.text();
    assert.match(html, /This week/);
    assert.match(html, /Why priced/);
  }
});

test("real v6 live explain document (D7b fixture): parsers return non-null with scoring, head_to_heads, probability_basis and week_latent_mean", async () => {
  const { parseScoring } = await import("../app/scoring-rules.ts");
  const raw = await read("explain_live_v6.json");
  const doc = parseExplain(raw);
  assert.ok(doc, "parseExplain accepts the real v6 document");
  assert.equal(doc.kind, "live");
  assert.equal(String(doc.version), "6");
  assert.equal(doc.probability_basis, "model");
  assert.equal(doc.head_to_heads.method, "joint_simulation");
  assert.deepEqual(doc.head_to_heads.columns, ["a", "b", "p_a", "p_tie"]);
  assert.ok(doc.head_to_heads.rows.length > 0 && doc.head_to_heads.rows.every((r) => r.length === 4));
  assert.ok(doc.players.length > 50);
  const latent = doc.players.map((p) => p.live?.week_latent_mean);
  assert.ok(latent.every((v) => typeof v === "number" && Number.isFinite(v)), "every player carries live.week_latent_mean");
  const scoring = parseScoring(raw.scoring);
  assert.ok(scoring, "parseScoring accepts the embedded scoring block");
  assert.equal(scoring.status, "available");
  assert.equal(scoring.probability_basis, "model");
  assert.ok(scoring.players.length > 0);
});

test("where we differ: contenders only, parts add up to the skill gap, groups net out by summed chances, live compares nothing", async () => {
  const R = await import("../app/explain-rules.ts");
  const { rows, differ } = R.differView(week);
  assert.ok(rows.length > 0 && rows.length <= R.CONTENDER_N);
  assert.equal(differ.source, "page");
  const map = R.shapeMap(week);
  assert.ok(map.win && map.top_20 && map.win.s > 0);
  for (const { player: p, shape: s } of rows) {
    assert.ok(Math.abs(s.parts.reduce((a, x) => a + x.value, 0) + s.base_gap - s.skill_gap) < 1e-9, "parts + remainder = skill gap");
    assert.ok(Math.abs(s.skill_ours - s.skill_gap - s.skill_market) < 1e-9 && Math.abs(s.spread_ours - s.spread_gap - s.spread_market) < 1e-9);
    for (const m of s.higher) assert.ok(s.markets[m].model > s.markets[m].market);
    for (const m of s.lower) assert.ok(s.markets[m].model < s.markets[m].market);
    assert.ok(s.sentence.startsWith(s.pattern) && !/NaN|undefined/.test(s.sentence + s.parts_sentence));
    assert.ok(p.n_prior_rounds === null || p.n_prior_rounds === undefined || p.n_prior_rounds >= R.CONTENDER_MIN_ROUNDS || (p.probs.win.market ?? 0) > 0);
  }
  for (const g of differ.groups) {
    const members = rows.filter((r) => R.playerHasTag(r.player, g.key));
    const wo = members.reduce((a, r) => a + r.player.probs.win.model, 0), wm = members.reduce((a, r) => a + r.player.probs.win.market, 0);
    assert.ok(Math.abs(g.win_ours - wo) < 1e-9 && Math.abs(g.win_market - wm) < 1e-9);
    assert.equal(Math.sign(g.win_rel), Math.sign(wo - wm));
  }
  // chances generated from the curves invert exactly
  const probs = Object.fromEntries(Object.entries(map).map(([m, c]) => [m, 1 / (1 + Math.exp(-c.s * (1.2 - c.q) / 2.6))]));
  const back = R.invertShape(probs, map);
  assert.ok(Math.abs(back.mu - 1.2) < 1e-9 && Math.abs(back.sd - 2.6) < 1e-9);
  // the exported block wins when present
  const exported = { ...week, differ: { rule: "r", ids: [rows[0].player.id], groups: [] } };
  assert.equal(R.differView(exported).differ.source, "export");
  assert.equal(R.differView(exported).rows.length, 1);
  assert.equal(R.differView(live).rows.length, 0);
  assert.equal(R.edgeCellText({ model: 0.8, market: 0.85, rel: -0.0588, side: "miss_cut", ev: 1 / 3, tone: "neg" }), "miss cut +33%");
});

test("modalFinish picks the most likely single finishing position; missing the cut is never compared with one position", async () => {
  const { modalFinish, modalFinishText, finishingProbs, waterfallScale } = await import("../app/explain-rules.ts");
  assert.deepEqual(modalFinish({ pos: [0.1, 0.3, 0.2, 0.4] }, false), { kind: "pos", pos: 4 });
  assert.deepEqual(modalFinish({ pos: [0.1, 0.3, 0.2, 0.4], p_miss_cut: 0 }, true), { kind: "pos", pos: 4 });
  // 45% miss the cut: that mass is ranked last and removed from the tail; the likeliest made-cut position still wins.
  assert.deepEqual(modalFinish({ pos: [0.1, 0.2, 0.2, 0.5], p_miss_cut: 0.45 }, true), { kind: "pos", pos: 2 });
  assert.deepEqual(modalFinish({ pos: [0, 0, 0, 1], p_miss_cut: 1 }, true), { kind: "miss" });
  assert.deepEqual(modalFinish({ pos: [0.2, 0.3, 0.2, 0.3], p_miss_cut: 0.25 }, true), { kind: "pos", pos: 2 });
  assert.equal(modalFinish(null, true), null);
  assert.equal(modalFinishText({ kind: "pos", pos: 12 }), "Most likely finish: 12");
  assert.equal(modalFinishText({ kind: "miss" }), "Most likely: misses cut");
  const f = finishingProbs([0.1, 0.2, 0.2, 0.5], 0.6, true);
  assert.ok(Math.abs(f.reduce((a, b) => a + b, 0) - 0.4) < 1e-9 && f[3] === 0);
  const players = [{ withdrawn: false, mu_rel: 0.3, components: { a: 0.1 } }, { withdrawn: false, mu_rel: 2.2, components: { a: 0.4, b: -0.7 } }];
  assert.equal(waterfallScale(players), 2.5);
  assert.equal(waterfallScale([]), 1);
});

test("each waterfall row gets its own small scale; expected skill keeps the event-wide one", async () => {
  const { niceScale, waterfallRowScales, waterfallScale } = await import("../app/explain-rules.ts");
  assert.equal(niceScale(0), 0.01);
  assert.equal(niceScale(0.04), 0.05);
  assert.equal(niceScale(0.11), 0.15);
  assert.equal(niceScale(0.9), 1);
  assert.equal(niceScale(3.2), 3.5);
  const legend = { a: { label: "A", group: "skill", meaning: "" }, b: { label: "B", group: "skill", meaning: "" }, wx_wind: { label: "Weather: wind", group: "weather", meaning: "" } };
  const players = Array.from({ length: 20 }, (_, i) => ({ withdrawn: false, mu_rel: i / 10, components: { a: 0.9 * (i % 2 ? 1 : -1), b: 0.01 * (i % 5), wx_wind: 0.02 * (i % 3) } }));
  const sc = waterfallRowScales(players, legend);
  assert.equal(sc.rows.a, 1);
  assert.equal(sc.rows.b, 0.05);
  assert.equal(sc.rows.wx_wind, 0.05);
  assert.ok(sc.rows.b < waterfallScale(players) / 10);
  assert.ok(sc.weatherTotal >= 0.01 && sc.unexplained >= 0.01);
  assert.equal(waterfallRowScales([], legend).rows.a, undefined);
});

test("weather status follows the published flag and the field-wide shift, not only the per-player rows", async () => {
  const { weatherStatus, weatherEffectLines, venueClassLines, fieldWeatherRows, roundsWithoutTeeTimes } = await import("../app/explain-rules.ts");
  const mk = (wx, comps, kind = "week", after = null) => {
    const base = structuredClone(week);
    base.kind = kind; base.after_round = after;
    base.course_card.weather = wx;
    base.components_legend = { ...base.components_legend, wx_wind: { label: "Weather: wind", group: "weather", meaning: "m" }, wx_rain: { label: "Weather: rain", group: "weather", meaning: "m" } };
    base.players = base.players.map((p, i) => ({ ...p, components: { ...p.components, wx_wind: comps ? 0.05 * (i % 2 ? 1 : -1) : 0, wx_rain: 0 } }));
    return base;
  };
  const field = [{ round: 1, common: -0.4, cut_line_shift: -0.4 }, { round: 2, common: -0.3, cut_line_shift: -0.7 }, { round: 3, common: -0.2, cut_line_shift: -0.9 }];
  const rounds = [{ round: 1, status: "ok", tee_coverage: 1, reason: null, venue_seen: true }, { round: 2, status: "common_only", tee_coverage: 0, reason: "no tee sheet", venue_seen: true }, { round: 3, status: "common_only", tee_coverage: 0, reason: "no tee sheet", venue_seen: true }];
  // genuinely excluded: the publisher says so and nothing moves
  const off = weatherStatus(mk({ applied_in_prices: false, label: "NOT APPLIED" }, false));
  assert.equal(off.short, "Weather not in prices");
  // applied, but only the field-wide shift (no tee times yet): must NOT say it is excluded
  const common = weatherStatus(mk({ applied_in_prices: true, label: "APPLIED", field, rounds }, false));
  assert.equal(common.short, "Weather in prices: whole field");
  assert.match(common.long, /round 1 −0\.40/);
  assert.match(common.long, /rounds 2 and 3 because tee times are not out yet/);
  // applied with player-level effects: zero rain is "no effect", not "not in the prices"
  const full = weatherStatus(mk({ applied_in_prices: true, label: "APPLIED", field, rounds }, true));
  assert.equal(full.short, "Weather included");
  assert.match(full.long, /through wind/);
  assert.match(full.long, /Rain has no effect for anyone/);
  assert.doesNotMatch(full.long, /Not in the prices/);
  // live runs only count rounds still to play
  assert.deepEqual(fieldWeatherRows(mk({ applied_in_prices: true, field }, false, "live", 2)).map((f) => f.round), [3]);
  assert.deepEqual(roundsWithoutTeeTimes(mk({ applied_in_prices: true, field, rounds }, false, "live", 2)), [3]);
  assert.equal(weatherStatus(mk({ applied_in_prices: true, label: "x" }, false)).short, "Weather has no effect yet");
  const lines = weatherEffectLines(mk({ applied_in_prices: true, field }, true));
  assert.equal(lines.find((l) => l.key === "wx_wind").spread, 0.05);
  assert.ok(lines.find((l) => l.key === "wx_wind").helped && lines.find((l) => l.key === "wx_wind").hurt);
  assert.equal(lines.find((l) => l.key === "wx_rain").spread, 0);
  assert.deepEqual(venueClassLines({ venue_class: "desert", wind_slope: 0.199, m_v: 0.9, wind_prior_source: "own history" }).length, 2);
  assert.match(venueClassLines({ venue_class: "desert", wind_slope: 0.199, m_v: 0.9 })[1], /matters less here/);
});

test("spread label reads the gap relative to this week's contenders; older documents fall back to the raw gap", async () => {
  const R = await import("../app/explain-rules.ts");
  const map = R.shapeMap(week);
  const markets = R.availableMarkets(week);
  const p = week.players.find((q) => R.playerShape(q, map, markets)?.higher.length);
  assert.ok(p, "fixture has a player with a bet-level difference");
  const old = R.playerShape(p, map, markets);
  assert.equal(old.spread_gap_rel, undefined, "no center: old behavior, no new field");
  const g = old.spread_gap;
  const flat = R.playerShape(p, map, markets, g);            // rel 0
  const steady = R.playerShape(p, map, markets, g + 0.3);    // rel -0.3
  const volatile = R.playerShape(p, map, markets, g - 0.3);  // rel +0.3
  for (const k of ["skill_gap", "spread_gap", "spread_ours", "spread_market", "pattern", "base_gap"]) assert.deepEqual(flat[k], old[k], `${k} unchanged`);
  assert.ok(Math.abs(flat.spread_gap_rel) < 1e-12 && Math.abs(steady.spread_gap_rel + 0.3) < 1e-12 && Math.abs(volatile.spread_gap_rel - 0.3) < 1e-12);
  assert.ok(!/steadier|boom-or-bust/.test(flat.sentence));
  assert.match(steady.sentence, /steadier.*[-−]0\.30 against this week's contenders/);
  assert.match(volatile.sentence, /boom-or-bust.*\+0\.30 against this week's contenders/);
  assert.equal(R.spreadGapOf(steady), steady.spread_gap_rel);
  assert.equal(R.spreadGapOf(old), old.spread_gap);          // older document: fall back
  // differ view: page-side documents are not centered; the exported flag is honoured only when every row carries spread_gap_rel
  assert.equal(R.differView(week).differ.spread_gap_centered, false);
  const exp = { ...week, differ: { rule: "r", ids: [p.id], groups: [], spread_gap_median: g, spread_gap_centered: true }, players: week.players.map((q) => (q.id === p.id ? { ...q, shape: steady } : q)) };
  assert.equal(R.differView(exp).differ.spread_gap_centered, true);
  const expOld = { ...exp, players: week.players.map((q) => (q.id === p.id ? { ...q, shape: old } : q)) };
  assert.equal(R.differView(expOld).differ.spread_gap_centered, false);
  // notes sentence: small, relative to this week's contenders, no "market generally implies" lean claim
  const n = R.spreadLeanNote(-0.0849, true);
  assert.match(n, /small/); assert.match(n, /relative to this week's contenders/); assert.ok(!/tends to favor/.test(n));
  assert.ok(!/relative to this week/.test(R.spreadLeanNote(0.02, false)));
  assert.equal(R.spreadLeanNote(null, true), null);
});

test("Data Golf gaps: chances relative to theirs, skill in strokes, contenders plus real top-20 chances; old documents have none", () => {
  assert.equal(hasDg(week), false);
  assert.deepEqual(dgGaps(week, "skill"), { higher: [], lower: [], scale: 0.3 });
  const raw = await_week;
  const players = raw.players.map((p, i) => ({ ...p, dg: { win: (p.probs?.win?.model ?? 0.01) * (i === 0 ? 0.5 : 1), top_20: 0.3, skill: 0.1 * i, skill_ours: 0.1 * i + (i === 1 ? 0.4 : i === 2 ? -0.25 : 0), skill_gap: i === 1 ? 0.4 : i === 2 ? -0.25 : 0 } }));
  const d = parseExplain({ ...raw, players });
  assert.equal(hasDg(d), true);
  const s = dgGaps(d, "skill", 4);
  assert.equal(s.higher[0].id, players[1].id);
  assert.equal(s.lower[0].gap, -0.25);
  assert.equal(biggestDgSkillGap(d).id, players[1].id);
  const w = dgGaps(d, "win", 4);
  if (players[0].probs?.win?.model) assert.ok(Math.abs(w.higher[0].gap - 1) < 1e-9);       // ours twice theirs = +100%
  assert.ok(w.higher.every((r) => r.gap >= 0.05) && w.lower.every((r) => r.gap <= -0.05));
  assert.equal(parseExplain({ ...raw, players: [{ ...raw.players[0], dg: { skill: "x" } }] }).players[0].dg, null);
});

/* ------------------------------------------------------------------ Why we differ (explain version 11) */
const { renderView } = await import("./helpers/render-harness.mjs");
const D = await import("../app/diverge-rules.ts");
const v11raw = await read("explain_week_v11.json");
const v11 = parseExplain(v11raw);
const rai = v11.players.find((p) => p.id === 18554);
const drawerHtml = (doc, player) => renderView("DivergeView.tsx", "DivergeSection", { props: { doc, player, onSelect() {} } });
const skillRows = (doc) => { const g = dgGaps(doc, "skill", 100); return g.higher.concat(g.lower); };

test("v11 documents parse the diverge, form and divergence blocks; the edge sentence is kept as published", () => {
  assert.ok(v11 && rai);
  assert.equal(rai.diverge.compare.verdict, "book_outlier");
  assert.equal(rai.diverge.compare.outlier, "book");
  assert.equal(rai.diverge.compare.markets.win.book, 0.0516);
  assert.equal(rai.diverge.dg_check.flag, true);
  assert.match(rai.diverge.dg_check.sentence, /0\.61 below Data Golf/);
  assert.equal(rai.diverge.edge_split.reference, "peers");
  assert.equal(rai.diverge.peers.ids.length, 4);
  assert.ok(rai.diverge.sentences.length >= 1 && rai.diverge.sentences.length <= 4);
  assert.equal(rai.form.windows.L8.mean, 1.8);
  assert.deepEqual(rai.form.seasons.map((s) => s.year), [2024, 2025, 2026]);
  assert.equal(rai.why_edge, rai.diverge.edge_split.sentence);   // not reworded by the old why-edge pattern
  assert.equal(v11.divergence.course_data.blank, true);
  assert.equal(v11.divergence.course_data.unavailable, "fit and history");
  assert.equal(rai.diverge.dg_check.skill_number_gap, -0.606);
  assert.match(v11.divergence.course_data.note, /Course fit and course history are blank/);
  assert.match(v11.divergence.form_source.scale, /adjusted strokes gained/);
  assert.equal(v11.divergence.dg_odds_fit.flagged.includes(18554), true);
  // a flagged player carries the sentence, an unflagged one does not
  assert.equal(v11.players.find((p) => p.diverge?.dg_check && !p.diverge.dg_check.flag).diverge.dg_check.sentence, null);
});

test("the course row of the legend keeps its relabelled text and the waterfall marks it as having no data", () => {
  const lg = v11.components_legend.course;
  assert.equal(lg.unavailable, "fit and history");
  assert.match(lg.label, /no course fit or history this week/i);
  assert.notEqual(lg.label, "Course fit and history");
  const wf = waterfall(rai, v11.components_legend);
  assert.equal(wf.rows.find((r) => r.key === "course").unavailable, "fit and history");
  assert.equal(wf.rows.find((r) => r.key === "kalman").unavailable, undefined);
  const html = renderView("ExplainBits.tsx", "Waterfall", { props: { ...wf, scale: 1 } });
  assert.match(html, /no data this week/);
});

test("documents at version 10 and below, live runs and withdrawn players carry no blocks and stay unchanged", () => {
  assert.equal(week.divergence, null);
  assert.ok(week.players.every((p) => p.diverge === null && p.form === null));
  assert.ok(live.players.every((p) => p.diverge === null && p.form === null));
  // the old why-edge wording is still produced for them
  const old = parseExplain({ ...await_week, players: [{ ...await_week.players[0], why_edge: "We are above the market by 0.26 strokes/round: mostly cross-tour +0.36, kalman trend -0.19." }] });
  assert.match(old.players[0].why_edge, /^We rate them higher than the market/);
  assert.equal(week.components_legend.course?.unavailable, undefined);
  // a withdrawn player has no diverge block and renders nothing in the section
  assert.equal(drawerHtml(v11, { ...rai, withdrawn: true, diverge: null, form: null }), "");
});

test("malformed blocks become null and never throw", () => {
  const bad = (diverge, form, divergence) => parseExplain({ ...v11raw, divergence, players: [{ ...v11raw.players[0], diverge, form }] });
  for (const junk of ["x", 5, [], { compare: 3, dg_check: "x", edge_split: [], peers: { ids: "x" }, sentences: 4 }]) {
    const d = bad(junk, junk, junk);
    assert.equal(d.players[0].diverge, null);
    assert.equal(d.players[0].form, null);
    assert.equal(d.divergence, null);
  }
  const partial = bad({ compare: { markets: { win: { ours: "a", dg: 0.1, book: 0.2 }, top_5: 7 }, verdict: "nonsense", outlier: "nobody" }, dg_check: { flag: "yes" }, edge_split: { edge: "big" }, peers: { ids: [1, "x", 2], names: ["A", 3], parts: [{ key: 1 }, { key: "a", label: "A", value: 1 }] }, sentences: ["ok", 3, ""] },
    { windows: { L8: { mean: "x" }, L24: { mean: 0.5, n: 24 } }, seasons: [{ year: "y" }, { year: 2025, mean: 1 }] }, { course_data: { note: 4 }, dg_odds_fit: { flagged: [1, "a"] } });
  const p = partial.players[0];
  assert.deepEqual(p.diverge.compare.markets.win, { ours: null, dg: 0.1, book: 0.2 });
  assert.equal(p.diverge.compare.verdict, "insufficient");
  assert.equal(p.diverge.compare.outlier, null);
  assert.equal(p.diverge.dg_check.flag, false);
  assert.equal(p.diverge.edge_split, null);
  assert.deepEqual(p.diverge.peers.ids, [1, 2]);
  assert.equal(p.diverge.peers.parts.length, 1);
  assert.deepEqual(p.diverge.sentences, ["ok"]);
  assert.deepEqual(Object.keys(p.form.windows), ["L24"]);
  assert.deepEqual(p.form.seasons.map((s) => s.year), [2025]);
  assert.equal(partial.divergence.course_data.note, null);
  assert.deepEqual(partial.divergence.dg_odds_fit.flagged, [1]);
  assert.doesNotThrow(() => drawerHtml(partial, p));
  assert.deepEqual(D.chanceRows(null), []);
  assert.equal(D.splitView(null), null);
  assert.deepEqual(D.peerPartRows(null), []);
});

test("chip: the biggest Data Golf skill gap skips players whose stated skill disagrees with their odds and ranks on the odds-implied gap", () => {
  // Aaron Rai is flagged: stated gap -0.61, against the odds -0.22. The old chip would have named him.
  const rows = skillRows(v11);
  const r = rows.find((x) => x.id === 18554);
  assert.equal(r.flag, true);
  assert.equal(r.gap, -0.124);                 // measured against the odds, shown with a warning
  assert.equal(r.stated, 2.114);
  const top = biggestDgSkillGap(v11);
  assert.ok(top && top.id !== 18554 && !top.flag);
  assert.ok(rows.filter((x) => !x.flag).every((x) => Math.abs(x.gap) <= Math.abs(top.gap) + 1e-12));
  // all flagged -> no chip
  const allFlagged = parseExplain({ ...v11raw, players: v11raw.players.map((p) => ({ ...p, diverge: p.diverge && { ...p.diverge, dg_check: p.diverge.dg_check && { ...p.diverge.dg_check, flag: true } } })) });
  assert.equal(biggestDgSkillGap(allFlagged), null);
  // documents without the odds check fall back to the stated gap
  const plain = parseExplain({ ...v11raw, players: v11raw.players.map((p) => ({ ...p, diverge: null })) });
  assert.equal(skillRows(plain).find((x) => x.id === 18554)?.gap, rai.dg.skill_gap);
  assert.ok(skillRows(plain).every((x) => x.flag === false && x.stated === x.dg));
});

test("chance table: rows need two sources, the odd-one-out column follows the verdict", () => {
  const rows = D.chanceRows(rai.diverge.compare);
  assert.deepEqual(rows.map((r) => r.market), ["win", "top_5", "top_10", "top_20", "make_cut"]);
  assert.equal(D.oddColumn(rai.diverge.compare), "book");
  const sparse = { markets: { win: { ours: 0.02, dg: null, book: null }, top_5: { ours: 0.1, dg: null, book: 0.2 }, top_10: { ours: null, dg: null, book: null } }, mean_logit_gap: {}, verdict: "no_dg", outlier: null };
  assert.deepEqual(D.chanceRows(sparse).map((r) => r.market), ["top_5"]);
  assert.equal(D.oddColumn(sparse), null);
  assert.equal(D.oddColumn({ ...rai.diverge.compare, verdict: "all_differ", outlier: "ours" }), null);   // nobody is named when all three differ
  for (const k of ["ours", "dg", "book"]) assert.equal(D.oddColumn({ ...rai.diverge.compare, verdict: `${k}_outlier`, outlier: k }), k);
});

test("the drawer section renders the chance table, odds warning, gap split, peers and form for Aaron Rai", () => {
  const html = drawerHtml(v11, rai);
  assert.match(html, /Why we differ/);
  assert.match(html, /On chances we are close to Data Golf/);
  assert.match(html, /<th class="is-odd">Sportsbooks/);                // the odd column is marked in the header...
  assert.equal((html.match(/<td data-label="Sportsbooks" class="is-odd">/g) ?? []).length, 5);   // ...and in each of the five rows
  assert.doesNotMatch(html, /<th class="is-odd">(Ours|Data Golf)/);
  assert.match(html, /5\.2%/);
  assert.match(html, /ex-callout is-warn/);                           // flagged: warning with the published sentence
  assert.match(html, /0\.61 below Data Golf/);
  assert.match(html, /players priced like them/);
  assert.match(html, /closest-priced players/);                       // two numbers side by side
  assert.match(html, /The sportsbooks rate them/);
  assert.doesNotMatch(html, /not explained by our model|no source in our model/);
  assert.match(html, /<button[^>]*class="ex-linkbtn"[^>]*>[^<]*MacIntyre/);
  assert.match(html, /Priced-like-them median/);
  assert.match(html, /aria-current="true"/);                          // this player's form row is highlighted
  assert.match(html, /adjusted strokes gained a round vs an average PGA Tour field/);
  assert.match(html, /Sportsbook implies/);
  assert.match(html, /1\.96/);                                        // the skill the sportsbook implies
  assert.match(html, /Course fit and course history are blank/);
  assert.match(html, /table-scroll/);
  // an unflagged player shows no warning, and no odd-one-out claim when the verdict names nobody
  const calm = v11.players.find((p) => p.diverge?.dg_check && !p.diverge.dg_check.flag && p.diverge.compare && !["book_outlier", "ours_outlier", "dg_outlier"].includes(p.diverge.compare.verdict));
  if (calm) { const h = drawerHtml(v11, calm); assert.doesNotMatch(h, /is-warn/); assert.doesNotMatch(h, /class="is-odd"/); }
});

test("form panel is hidden when nobody shown has form; the gap split reads its parts", () => {
  const noForm = parseExplain({ ...v11raw, players: v11raw.players.map((p) => ({ ...p, form: null })) });
  const html = drawerHtml(noForm, noForm.players.find((p) => p.id === 18554));
  assert.doesNotMatch(html, /Recent form/);
  assert.match(html, /Why we differ/);
  const s = D.splitView(rai.diverge.edge_split);
  assert.deepEqual([s.ours, s.book, s.particular], [-0.261, 0.122, -0.383]);
  const es = (o) => ({ kind: "normal", edge: 0.2, reference: "peers", reference_edge: 0.05, particular: 0.15, ours_vs_peers: 0.1, book_vs_peers: 0.2, sentence: "s", ...o });
  assert.equal(D.splitView(es({ kind: "in_line" })), null);          // nothing but the sentence for in-line and long-shot players
  assert.equal(D.splitView(es({ kind: "long_shot" })), null);
  assert.equal(D.splitView(es({ kind: "mixed" })), null);
  assert.equal(D.splitView(es({ book_vs_peers: null })), null);
  assert.equal(D.splitView(es({})).book, 0.2);
  const inLine = v11.players.find((p) => p.diverge?.edge_split?.kind === "in_line");
  assert.ok(inLine && !/closest-priced/.test(drawerHtml(v11, inLine)));
  // tolerant parser: unknown kind reads as normal, missing numbers are null, a bare sentence is kept
  const loose = parseExplain({ ...v11raw, players: [{ ...v11raw.players[0], diverge: { edge_split: { kind: "weird", sentence: "only text" }, peers: null } }] }).players[0].diverge.edge_split;
  assert.deepEqual([loose.kind, loose.edge, loose.ours_vs_peers, loose.reference], ["normal", null, null, "none"]);
  const t = D.formTable(v11, rai, rai.diverge.peers.ids, () => null);
  assert.equal(t.rows[0].self, true);
  assert.equal(t.rows.length, 5);
  assert.equal(D.formTable({ players: [{ ...rai, form: null }] }, { ...rai, form: null }, [], () => null), null);
});

test("flagged skill rows carry both gaps; a partial course gap reads as missing for some players", () => {
  const r = skillRows(v11).find((x) => x.id === 18554);
  assert.equal(r.numberGap, -0.606);
  assert.equal(r.gap, -0.124);
  const html = renderView("ExplainBits.tsx", "Waterfall", { props: { ...waterfall(rai, { ...v11.components_legend, course: { ...v11.components_legend.course, unavailable: "some players" } }), scale: 1 } });
  assert.match(html, /missing for some/);
  assert.equal(parseExplain({ ...v11raw, divergence: { ...v11raw.divergence, course_data: { ...v11raw.divergence.course_data, unavailable: 4 } } }).divergence.course_data.unavailable, null);
});

test("a mixed player (skill gap and bet edge disagree in sign) renders only the sentence", () => {
  const mixed = v11.players.find((p) => p.diverge?.edge_split?.kind === "mixed");
  assert.ok(mixed);
  assert.equal(mixed.diverge.edge_split.ours_vs_peers, null);
  const html = drawerHtml(v11, mixed);
  assert.match(html, /in shape rather than level/);
  assert.doesNotMatch(html, /closest-priced players|The sportsbooks rate them/);
});
