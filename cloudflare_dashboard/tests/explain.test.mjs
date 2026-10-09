import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  american, availableTags, biasDimensions, biasForDimension, biasSentence, binFinish, defaultCompetitors, explainKeyFor, filterWhy, finishBins, leaderboard, movers, nameMatches, orderEvents,
  parseExplain, pct, signed, sortWhy, toParText, toggleCompetitor, topEdges, topK, waterfall,
} from "../app/explain-rules.ts";

const read = async (n) => JSON.parse(await readFile(new URL(`./fixtures/${n}`, import.meta.url), "utf8"));
const week = parseExplain(await read("explain_week.json"));
const live = parseExplain(await read("explain_live.json"));

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

test("competitor overlay: same-odds neighbours preselected, capped, toggled", () => {
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
