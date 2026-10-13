import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";
import { renderView } from "./helpers/render-harness.mjs";
import {
  briefKeyFor, buildWhatIf, componentWords, fieldWords, helpWords, historySeries, isNeutral, medianProbSlopes, movementWords, NEUTRAL_SLIDERS, orderPlayers, parseBrief, pointsWords,
  scaledProb, searchBrief, spreadWords, tone, waveSide, waveWords, windSensitivity,
} from "../app/weather-brief-rules.ts";

const fixture = JSON.parse(await readFile(new URL("./fixtures/weather_brief.json", import.meta.url), "utf8"));
const legacyFixture = JSON.parse(await readFile(new URL("./fixtures/weather.json", import.meta.url), "utf8"));
const brief = parseBrief(fixture);
const KEY = "golfprice/weather/pga_528_2026-10-22/brief_latest.json";
const LEGACY_KEY = "golfprice/weather/pga_528_2026-10-22/latest.json";
const index = { events: [{ event_uid: "pga:528:2026-10-22", name: "Test Open", tour: "pga", date_start: "2026-10-22", weather_key: LEGACY_KEY, weather_brief_key: KEY }] };

test("the brief parses; anything else is null so the page falls back to the older weather file", () => {
  assert.ok(brief);
  assert.equal(brief.model_version, "v2.7");
  assert.equal(brief.rounds.length, 4);
  assert.equal(brief.players.length, 48);
  assert.equal(parseBrief(null), null);
  assert.equal(parseBrief([]), null);
  assert.equal(parseBrief({ schema: "golfprice.weather.v1" }), null);
  const bare = parseBrief({ schema: "golfprice.weather_brief.v1", players: [1, { dg_id: 4 }], rounds: [{ round: 1 }, "x"], what_if: { scenarios: [null] } });
  assert.equal(bare.players.length, 1);
  assert.equal(bare.rounds[0].forecast.AM, null);
  assert.equal(bare.whatIf, null);
  assert.deepEqual(bare.headline, []);
  assert.equal(briefKeyFor({ weather_brief_key: KEY }), KEY);
  assert.equal(briefKeyFor({}), null);
});

test("wording never shows a raw sign: helps/hurts, easier/harder, morning/afternoon", () => {
  assert.equal(fieldWords(-0.78), "easier by 0.78 strokes");
  assert.equal(fieldWords(0.5), "harder by 0.50 strokes");
  assert.equal(fieldWords(0.001), "no real change");
  assert.equal(fieldWords(-1), "easier by 1.00 stroke");
  assert.equal(helpWords(-0.12), "helps by 0.12 strokes");
  assert.equal(helpWords(0.12), "hurts by 0.12 strokes");
  assert.equal(helpWords(null), "not available");
  assert.equal(waveWords(0.45), "Morning wave favoured by 0.45 strokes");
  assert.equal(waveWords(-0.45), "Afternoon wave favoured by 0.45 strokes");
  assert.equal(waveWords(null), "no tee sheet yet");
  assert.equal(waveSide(0.45), "am");
  assert.equal(waveSide(-0.3), "pm");
  assert.equal(waveSide(0.01), "even");
  assert.match(spreadWords(1.02), /wider.*long shots/);
  assert.match(spreadWords(0.98), /bunch up.*favourites/);
  assert.equal(pointsWords(0.0025), "up 0.25 pts");
  assert.equal(pointsWords(-0.031), "down 3.1 pts");
  assert.equal(tone(-0.2), "good");
  assert.equal(tone(0.2), "bad");
  for (const text of [fieldWords(-0.78), helpWords(-0.12), waveWords(-0.4), pointsWords(-0.02), componentWords({ wind: -0.03, gust: 0.02, rain: 0 })]) assert.doesNotMatch(text, /[−-]\d|\+\d/, text);
  assert.equal(componentWords({ wind: -0.03, gust: 0.02, rain: 0 }), "Wind helps 0.03 strokes; Gusts hurts 0.02 strokes");
  assert.equal(componentWords({ rain: 0 }), "Nothing material");
});

test("wind sensitivity compares the course with the tour average in words", () => {
  const s = windSensitivity(brief.course);
  assert.equal(s.ratio, 1.31);
  assert.match(s.label, /More wind-sensitive than average/);
  assert.ok(Math.abs(s.perFive - 1.075) < 1e-9);
  assert.match(windSensitivity({ ...brief.course, multiplier: null, slope: 0.08, tourSlope: 0.164 }).label, /Less wind-sensitive|Barely/);
  assert.match(windSensitivity({ ...brief.course, multiplier: null, slope: null, tourSlope: null }).label, /unknown/);
});

test("players order by most helped first (missing last), search by name or id", () => {
  const ordered = orderPlayers(brief.players, "helped");
  assert.ok(ordered[0].total <= ordered[1].total);
  assert.ok(ordered.at(-1).total >= ordered[0].total);
  const hurt = orderPlayers(brief.players, "hurt");
  assert.ok(hurt[0].total >= hurt[1].total);
  const withNull = orderPlayers([{ name: "A", skill_rank: 1, total: null }, { name: "B", skill_rank: 2, total: 0.2 }, { name: "C", skill_rank: 3, total: -0.2 }], "helped");
  assert.deepEqual(withNull.map((p) => p.name), ["C", "B", "A"]);
  assert.deepEqual(orderPlayers(withNull, "hurt").map((p) => p.name), ["B", "C", "A"]);
  const first = brief.players[0];
  assert.ok(searchBrief(brief.players, first.name.split(",")[0]).some((p) => p.dg_id === first.dg_id));
  assert.equal(searchBrief(brief.players, String(first.dg_id))[0].dg_id, first.dg_id);
  assert.equal(searchBrief(brief.players, "zzzzzz").length, 0);
});

test("what-if: neutral sliders reproduce the base; wind and temperature interpolate between published points", () => {
  const model = buildWhatIf(brief);
  assert.deepEqual(model.axes, { wind: true, pm: true, temp: true, wet: true });
  assert.deepEqual(model.ranges.wind, [-10, 10]);
  assert.deepEqual(model.ranges.pm, [0, 10]);
  assert.equal(isNeutral(NEUTRAL_SLIDERS), true);
  const base = model.evaluate(NEUTRAL_SLIDERS);
  assert.ok(Math.abs(base.rounds[0].field - brief.rounds[0].field) < 1e-9);
  assert.ok(Math.abs(base.totals.get(brief.players[0].dg_id) - brief.players[0].total) < 1e-9);

  const sc = brief.whatIf.scenarios;
  const w5 = sc.find((s) => s.id === "wind5"), w10 = sc.find((s) => s.id === "wind10");
  const at5 = model.evaluate({ ...NEUTRAL_SLIDERS, wind: 5 });
  assert.ok(Math.abs(at5.rounds[0].field - w5.rounds[0].field) < 1e-9, "a published point is exact");
  const at75 = model.evaluate({ ...NEUTRAL_SLIDERS, wind: 7.5 });
  assert.ok(Math.abs(at75.rounds[0].field - (w5.rounds[0].field + w10.rounds[0].field) / 2) < 1e-9, "halfway between 5 and 10 is the average");
  assert.ok(model.evaluate({ ...NEUTRAL_SLIDERS, wind: 10 }).rounds[1].field > base.rounds[1].field, "more wind makes everyone's scoring harder");
  assert.ok(Math.abs(model.evaluate({ ...NEUTRAL_SLIDERS, wind: 40 }).rounds[0].field - w10.rounds[0].field) < 1e-9, "held at the last published point");

  // PM-only wind widens the wave split in favour of the morning wave.
  assert.ok(model.evaluate({ ...NEUTRAL_SLIDERS, pm: 10 }).rounds[0].wave > base.rounds[0].wave);
  // Wet is a toggle.
  const wet = model.evaluate({ ...NEUTRAL_SLIDERS, wet: true });
  assert.ok(wet.rounds[0].field > base.rounds[0].field);
  // Effects add when sliders combine.
  const both = model.evaluate({ ...NEUTRAL_SLIDERS, wind: 5, temp: 10 });
  const tOnly = model.evaluate({ ...NEUTRAL_SLIDERS, temp: 10 });
  assert.ok(Math.abs(both.rounds[0].field - (at5.rounds[0].field + tOnly.rounds[0].field - base.rounds[0].field)) < 1e-9);
});

test("what-if: players outside a scenario's biggest movers are scaled from their base effect and marked approximate", () => {
  const model = buildWhatIf(brief);
  const out = model.evaluate({ ...NEUTRAL_SLIDERS, pm: 10 });
  const w = brief.whatIf.scenarios.find((s) => s.id === "pm10");
  const moverId = [...w.movers.keys()][0];
  assert.ok(Math.abs(out.totals.get(moverId) - w.movers.get(moverId)) < 1e-9, "a listed mover is exact");
  assert.ok(!out.approx.has(moverId));
  const other = brief.players.find((p) => !w.movers.has(p.dg_id));
  assert.ok(out.approx.has(other.dg_id));
  assert.ok(Number.isFinite(out.totals.get(other.dg_id)));
  assert.equal(buildWhatIf({ ...brief, whatIf: null }), null);
});

test("what-if: a scenario's player_totals give every player an exact number (nothing approximate)", () => {
  const raw = structuredClone(fixture);
  for (const s of raw.what_if.scenarios) {
    s.player_totals = Object.fromEntries(raw.players.map((p, i) => [String(p.dg_id), (p.total_strokes ?? 0) + (s.id === "base" ? 0 : 0.01 * (i + 1))]));
  }
  const b = parseBrief(raw);
  const out = buildWhatIf(b).evaluate({ ...NEUTRAL_SLIDERS, pm: 10 });
  assert.equal(out.approx.size, 0);
  const p = raw.players[1];
  const pm10 = raw.what_if.scenarios.find((s) => s.id === "pm10");
  assert.ok(Math.abs(out.totals.get(p.dg_id) - pm10.player_totals[String(p.dg_id)]) < 1e-9);
});

test("probability change scales with the player's total; tiny totals use the field's median slope", () => {
  const slopes = medianProbSlopes(brief.players);
  assert.ok(slopes.win < 0, "helping (negative total) raises win chance");
  const p = { total: -0.2, prob: { win: 0.004, top_5: null, top_10: 0.01, top_20: 0.02 } };
  assert.ok(Math.abs(scaledProb(p, "win", -0.4, slopes) - 0.008) < 1e-12);
  assert.equal(scaledProb(p, "win", null, slopes), null);
  assert.ok(Math.abs(scaledProb({ total: 0.001, prob: { win: 0 } }, "win", -0.5, slopes) - slopes.win * -0.5) < 1e-12);
});

test("history needs two captures per round and describes the movement in words", () => {
  const series = historySeries(brief);
  assert.equal(series.length, 2);
  assert.equal(series[0].field.length, 3);
  assert.equal(historySeries({ ...brief, history: brief.history.slice(0, 1) }).length, 0);
  assert.match(movementWords([-0.7, -0.8, -1.1], "field"), /Got easier by 0.40/);
  assert.match(movementWords([0.3, 0.8], "wave"), /morning wave's edge grew by 0.50/);
  assert.match(movementWords([0.3, 0.31], "wave"), /Barely moved/);
  assert.equal(movementWords([0.3], "wave"), "");
});

test("the page renders the brief top to bottom with plain words", () => {
  const html = renderView("WeatherEffectsView.tsx", "WeatherEffectsView", { data: { "golfprice/index.json": index, [KEY]: fixture }, pathname: "/weather-effects" });
  for (const text of ["Weather headline", "This course and weather", "More wind-sensitive than average", "1.3x the tour average", "Round by round", "Who it helps and hurts", "What if the forecast changes", "How it has moved this week", "What the words mean", "Field effect", "Wave split", "Tee times not out", "model v2.7", "Wind, all day", "Extra afternoon wind", "Wet weather", "Search player"]) assert.ok(html.includes(text), text);
  assert.ok(html.includes("by skill only"), "top-30 filter");
  assert.match(html, /Morning wave favoured by|Afternoon wave favoured by/);
  assert.match(html, /helps by|hurts by/);
  assert.match(html, /easier by|harder by/);
  assert.doesNotMatch(html, /\[object Object\]|undefined|NaN|challenger/);
  assert.doesNotMatch(html, /(helps|hurts|easier|harder) by [−-]/);
});

test("without a brief the page falls back to the older weather document", () => {
  const plain = { events: [{ ...index.events[0], weather_brief_key: undefined }] };
  const html = renderView("WeatherEffectsView.tsx", "WeatherEffectsView", { data: { "golfprice/index.json": plain, [LEGACY_KEY]: legacyFixture }, pathname: "/weather-effects" });
  assert.match(html, /Wave comparison and field effect/);
  assert.doesNotMatch(html, /Who it helps and hurts/);
  // A brief key whose file is missing also falls back instead of showing an error.
  const html2 = renderView("WeatherEffectsView.tsx", "WeatherEffectsView", { data: { "golfprice/index.json": index, [LEGACY_KEY]: legacyFixture }, pathname: "/weather-effects" });
  assert.match(html2, /Wave comparison and field effect/);
  const none = renderView("WeatherEffectsView.tsx", "WeatherEffectsView", { data: { "golfprice/index.json": plain }, pathname: "/weather-effects" });
  assert.match(none, /No weather published for/);
});

test("This week links prominently to the weather page", async () => {
  const src = await readFile(new URL("../app/ExplainViews.tsx", import.meta.url), "utf8");
  assert.match(src, /<WeatherLinkCard event=\{event\} \/>/);
  const html = renderView("WeatherBriefView.tsx", "WeatherLinkCard", { data: { [KEY]: fixture }, props: { event: index.events[0] } });
  assert.match(html, /href="\/weather-effects\?e=pga%3A528%3A2026-10-22"/);
  assert.match(html, /Windy Friday afternoon/);
});

/* ------------------------------------------------------------------ production-only wording */
test("no user-facing string mentions challenger, v2.2, champion or arms (data keys are allowed)", async () => {
  const dir = new URL("../app/", import.meta.url);
  const files = (await readdir(dir)).filter((f) => /\.tsx?$/.test(f));
  const bad = /challenger|v2\.2|champion\b|comparison arm|\barms?\b/i;
  const hits = [];
  for (const file of files) {
    const src = (await readFile(new URL(file, dir), "utf8")).replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "").replace(/\??\.(challenger|champion)/g, ".data");   // property reads are data keys, not wording
    const literals = [...src.matchAll(/"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)'|`((?:[^`\\]|\\.)*)`/g)].map((m) => m[1] ?? m[2] ?? m[3] ?? "");
    const jsxText = [...src.matchAll(/>([^<>{}=]*[A-Za-z][^<>{}=]*)</g)].map((m) => m[1]);
    for (const raw of [...literals, ...jsxText]) {
      let text = raw;
      for (let i = 0; i < 4; i++) text = text.replace(/\$\{[^{}]*\}/g, " ");   // code inside a template is not wording
      if (/^[a-z_]+$/i.test(text.trim())) continue;   // bare data keys such as "challenger"
      if (bad.test(text)) hits.push(`${file}: ${text.slice(0, 400)} -> ${bad.exec(text)[0]}`);
    }
  }
  assert.deepEqual(hits, []);
});

test("with sliders moved, the round strip shows the new numbers with the forecast faintly beside them, and players get approximate marks", () => {
  const model = buildWhatIf(brief);
  const sliders = { wind: 10, pm: 10, temp: -10, wet: true };
  const result = model.evaluate(sliders);
  const strip = renderView("WeatherBriefView.tsx", "RoundStrip", { data: {}, props: { rounds: brief.rounds, live: result.rounds, sliders } });
  assert.match(strip, /\(forecast: /);
  assert.match(strip, /harder by|easier by/);
  assert.match(strip, /wind/);
  const players = renderView("WeatherBriefView.tsx", "PlayersPanel", { data: {}, props: { brief, result } });
  assert.match(players, /class="ghost"/);
  assert.match(players, /≈/);
  assert.doesNotMatch(players, /NaN|undefined/);
  const base = renderView("WeatherBriefView.tsx", "RoundStrip", { data: {}, props: { rounds: brief.rounds, live: null, sliders: NEUTRAL_SLIDERS } });
  assert.doesNotMatch(base, /\(forecast: /);
});
