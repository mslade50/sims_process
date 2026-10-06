import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { filterPlayers, forecastAgeHours, hhmm, parseWeather, parseWindow, sortPlayers, toHour } from "../app/weather-rules.ts";

const fixture = JSON.parse(await readFile(new URL("./fixtures/weather.json", import.meta.url), "utf8"));

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

test("time and window parsing tolerate the formats the sim may write", () => {
  assert.equal(toHour("2026-10-08T07:30"), 7.5);
  assert.equal(toHour("2026-10-08 13:15:00"), 13.25);
  assert.equal(toHour("07:45"), 7.75);
  assert.equal(toHour("garbage"), null);
  assert.equal(toHour(null), null);
  assert.deepEqual(parseWindow("07:00-09:30"), [7, 9.5]);
  assert.deepEqual(parseWindow("07:00 to 09:30"), [7, 9.5]);
  assert.deepEqual(parseWindow(["12:00", "14:30"]), [12, 14.5]);
  assert.deepEqual(parseWindow({ start: "7:00", end: "9:00" }), [7, 9]);
  assert.equal(parseWindow("early"), null);
  assert.equal(hhmm(7.5), "07:30");
  assert.equal(hhmm(null), "-");
});

test("the fixture document parses with its rounds, waves, field and per-player rows", () => {
  const doc = parseWeather(fixture);
  assert.ok(doc);
  assert.equal(doc.synthetic, true);
  assert.deepEqual(doc.rounds, [1, 2, 3, 4]);
  assert.equal(doc.forecast.hourly.length, 48);
  assert.equal(doc.waves.length, 4);
  assert.deepEqual(doc.waves[0].window, [7, 9.5]);
  assert.equal(doc.field.find((f) => f.round === 1).cut_line_shift, null);
  assert.equal(doc.field.find((f) => f.round === 2).cut_line_shift, 0.8);
  assert.equal(doc.players.length, 60);
  const p = doc.players[0];
  assert.ok(p.rounds[1] && p.rounds[2]);
  assert.ok(Math.abs(p.total12 - (p.rounds[1].mean_off_rel + p.rounds[2].mean_off_rel)) < 1e-9);
  assert.equal(doc.venue.wind_slope_source, "own history (synthetic)");
});

test("anything that is not the document is null; missing optional fields are tolerated", () => {
  assert.equal(parseWeather(null), null);
  assert.equal(parseWeather([]), null);
  assert.equal(parseWeather({ schema: "other" }), null);
  const bare = parseWeather({ schema: "golfprice.weather.v1", event_uid: "e", forecast: { hourly: [null, { round: 1 }, { round: 1, time_local: "2026-10-08T08:00", wind: 5 }] }, players: [1, { dg_id: 3 }, { dg_id: 4, round: 1, components: { wind: "x", gust: 0.1 } }], waves: [{ wave: "am" }] });
  assert.ok(bare);
  assert.equal(bare.forecast.hourly.length, 1);
  assert.equal(bare.forecast.hourly[0].wind_p10, null);
  assert.equal(bare.venue.class, "");
  assert.equal(bare.waves.length, 0);
  assert.equal(bare.players.length, 1);
  assert.deepEqual(bare.players[0].rounds[1].components, { gust: 0.1 });
  assert.equal(bare.players[0].total12, null);
  assert.deepEqual(bare.notes, []);
});

test("sorting puts missing values last in both directions; search matches names and ids", () => {
  const doc = parseWeather(fixture);
  const extra = { dg_id: 999, name: "Zed, Nobody", rounds: {}, total12: null, sdMax: null };
  const rows = [...doc.players, extra];
  const asc = sortPlayers(rows, "total12", "asc");
  const desc = sortPlayers(rows, "total12", "desc");
  assert.equal(asc.at(-1).dg_id, 999);
  assert.equal(desc.at(-1).dg_id, 999);
  assert.ok(asc[0].total12 <= asc[1].total12);
  assert.ok(desc[0].total12 >= desc[1].total12);
  assert.equal(sortPlayers(rows, "name", "asc")[0].name.localeCompare(sortPlayers(rows, "name", "asc")[1].name) <= 0, true);
  const first = doc.players[0];
  const [last, given] = first.name.split(", ");
  assert.ok(filterPlayers(doc.players, `${given} ${last}`).some((p) => p.dg_id === first.dg_id));
  assert.ok(filterPlayers(doc.players, first.name).some((p) => p.dg_id === first.dg_id));
  assert.ok(filterPlayers(doc.players, String(first.dg_id)).some((p) => p.dg_id === first.dg_id));
  assert.equal(filterPlayers(doc.players, "qqqqzzzz").length, 0);
  assert.equal(filterPlayers(doc.players, "  ").length, doc.players.length);
});

test("forecast age", () => {
  assert.equal(forecastAgeHours("2026-10-06T00:00:00Z", Date.parse("2026-10-06T06:00:00Z")), 6);
  assert.equal(forecastAgeHours("", 0), null);
});

test("the Weather effects page renders its empty state when nothing is published", async () => {
  const worker = await loadWorker();
  const response = await worker.fetch(new Request("https://golf.example/weather-effects", { headers: { accept: "text/html" } }), { ASSETS: { fetch: async () => new Response("nf", { status: 404 }) } }, { waitUntil() {}, passThroughOnException() {} });
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Weather effects/);
  assert.doesNotMatch(html, /Application error|Internal Server Error/);
});

test("the Worker serves golfprice/weather/<event>/latest.json read-only with no-cache", async () => {
  const worker = await loadWorker();
  const key = "golfprice/weather/pga_554_2026-10-01/latest.json";
  const bucket = { get: async (k) => (k === key ? { httpEtag: '"w1"', body: JSON.stringify(fixture), writeHttpMetadata() {} } : null) };
  const env = { ASSETS: { fetch: async () => new Response("nf", { status: 404 }) }, DASHBOARD_DATA: bucket };
  const ctx = { waitUntil() {}, passThroughOnException() {} };
  const ok = await worker.fetch(new Request(`https://golf.example/api/${key}`), env, ctx);
  assert.equal(ok.status, 200);
  assert.equal(ok.headers.get("cache-control"), "no-cache");
  assert.equal((await ok.json()).schema, "golfprice.weather.v1");
  const missing = await worker.fetch(new Request("https://golf.example/api/golfprice/weather/none/latest.json"), env, ctx);
  assert.equal(missing.status, 404);
  const write = await worker.fetch(new Request(`https://golf.example/api/${key}`, { method: "PUT", body: "{}" }), env, ctx);
  assert.equal(write.status, 405);
});
