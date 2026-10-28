// Render smoke: every dashboard view mounts without throwing while loading, with no published data, and (where a fixture exists) with data.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loadApp, renderView } from "./helpers/render-harness.mjs";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { NAVIGATION, OWNER_ONLY_VIEWS, OWNER_ONLY_API_PREFIXES, resolveRoute } from "../app/shell-rules.ts";

test("Shot data and Ops remain registered lazy owner routes in the real dashboard", () => {
  const app = readFileSync(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  const registered = [...app.matchAll(/^\s+"?([a-z-]+)"?: [A-Za-z]+View,$/gm)].map((m) => m[1]);
  const { DashboardApp } = loadApp("DashboardApp.tsx");
  for (const [key, name] of [["shot-data", "ShotDataView"], ["ops", "OpsView"]]) {
    assert.deepEqual(resolveRoute(key, registered), { kind: "view", key });
    assert.ok(OWNER_ONLY_VIEWS.includes(key), key);
    assert.ok(NAVIGATION.flatMap((g) => g.items).some((i) => i.key === key && i.href === `/${key}`), key);
    assert.ok(app.includes(`const ${name} = lazy(() => import("./${name}")`), `${key}: lazy import`);
    assert.ok(app.includes(`"${key}": () => import("./${name}")`), `${key}: client prefetch registration`);
    // The real shell waits for whoami during SSR. A missing entry renders NotFound immediately.
    const html = renderToStaticMarkup(React.createElement(DashboardApp, { initialView: key }));
    assert.doesNotMatch(html, /That page does not exist|There is no page at/);
  }
  assert.ok(OWNER_ONLY_API_PREFIXES.includes("/api/ops"));
  const missing = renderToStaticMarkup(React.createElement(DashboardApp, { initialView: "unregistered-test-route" }));
  assert.match(missing, /That page does not exist/, "negative control: absent registry entry");
});

const VIEWS = [
  ["shot-data", "ShotDataView.tsx", "ShotDataView"],
  ["players", "PlayerProfilesView.tsx", "PlayersView"],
  ["weekly-players", "WeeklyPlayersView.tsx", "WeeklyPlayersView"],
  ["ops", "OpsView.tsx", "OpsView"],
  ["run", "RunView.tsx", "RunView"],
  ["inputs", "InputsView.tsx", "InputsView"],
  ["research", "ResearchView.tsx", "ResearchView"],
  ["distributions", "DistributionView.tsx", "DistributionView"],
  ["sg-distributions", "views.tsx", "SgDistributionsView"],
  ["round-scores", "ScoringView.tsx", "ScoringView"],
  ["history", "views.tsx", "HistoryView"],
  ["performance", "views.tsx", "PerformanceView"],
  ["diagnostics", "views.tsx", "DiagnosticsView"],
  ["weather-effects", "WeatherEffectsView.tsx", "WeatherEffectsView"],
  ["weather-edits", "WeatherEditsView.tsx", "WeatherEditsView"],
  ["this-week", "ExplainViews.tsx", "ThisWeekView"],
  ["why-priced", "ExplainViews.tsx", "WhyPricedView"],
  ["bet-breakdowns", "BetBreakdownsView.tsx", "BetBreakdownsView"],
  ["bet-card", "BetCardView.tsx", "BetCardView"],
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

test("the built Worker serves existing routes and protects the new operations route", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const worker = (await import(workerUrl.href)).default;
  const env = { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } };
  for (const [key] of VIEWS) {
    const response = await worker.fetch(new Request(`https://golf.example/${key}`, { headers: { accept: "text/html" } }), env, { waitUntil() {}, passThroughOnException() {} });
    if (key === "ops") {
      assert.equal(response.status, 302, "the private operations page fails closed without owner configuration");
      assert.equal(response.headers.get("location"), "https://golf.example/this-week");
    } else {
      assert.equal(response.status, 200, key);
      assert.match(await response.text(), /Golf Model/, key);
    }
  }
});
