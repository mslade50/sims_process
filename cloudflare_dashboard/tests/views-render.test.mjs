// Render smoke: every dashboard view mounts without throwing while loading, with no published data, and (where a fixture exists) with data.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { renderView } from "./helpers/render-harness.mjs";

const VIEWS = [
  ["players", "PlayerProfilesView.tsx", "PlayersView"],
  ["weekly-players", "WeeklyPlayersView.tsx", "WeeklyPlayersView"],
  ["run", "RunView.tsx", "RunView"],
  ["inputs", "InputsView.tsx", "InputsView"],
  ["research", "ResearchView.tsx", "ResearchView"],
  ["distributions", "DistributionView.tsx", "DistributionView"],
  ["sg-distributions", "views.tsx", "SgDistributionsView"],
  ["round-scores", "ScoringView.tsx", "ScoringView"],
  ["history", "views.tsx", "HistoryView"],
  ["performance", "views.tsx", "PerformanceView"],
  ["diagnostics", "views.tsx", "DiagnosticsView"],
  ["weather", "views.tsx", "WeatherView"],
  ["weather-effects", "WeatherEffectsView.tsx", "WeatherEffectsView"],
  ["this-week", "ExplainViews.tsx", "ThisWeekView"],
  ["why-priced", "ExplainViews.tsx", "WhyPricedView"],
];

test("the smoke list covers every view key the dashboard registers", () => {
  const app = readFileSync(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  const registered = [...app.matchAll(/^\s+"?([a-z-]+)"?: [A-Za-z]+View,$/gm)].map((m) => m[1]).sort();
  assert.deepEqual(VIEWS.map((v) => v[0]).sort(), registered);
});

for (const [key, file, exportName] of VIEWS) {
  test(`${key}: renders while loading`, () => {
    const html = renderView(file, exportName, { loading: true, pathname: `/${key}` });
    assert.ok(html.length > 0);
  });
  test(`${key}: renders with no published data`, () => {
    const html = renderView(file, exportName, { data: {}, pathname: `/${key}` });
    assert.ok(html.length > 0);
    assert.doesNotMatch(html, /\[object Object\]|undefined|NaN/);
  });
}

test("the built Worker serves every view route with 200 HTML", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const worker = (await import(workerUrl.href)).default;
  const env = { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } };
  for (const [key] of VIEWS) {
    const response = await worker.fetch(new Request(`https://golf.example/${key}`, { headers: { accept: "text/html" } }), env, { waitUntil() {}, passThroughOnException() {} });
    assert.equal(response.status, 200, key);
    assert.match(await response.text(), /Golf Model/, key);
  }
});
