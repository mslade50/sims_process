// App shell (site audit wave 1): landing, redirects, not-found, nav groups, header oldest-as-of, data fallback decision, search, Run page.
import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";
import { DEFAULT_VIEW, NAVIGATION, NAV_ITEMS, REDIRECTS, decideFallback, navEntryFor, playerHref, resolveRoute, searchPlayers, suggestedGroup, unknownEventParam } from "../app/shell-rules.ts";
import { heldBackText, summarizeWeek } from "../app/ui-rules.ts";
import { finite, pct, shortDate, signedPct } from "../app/lib.ts";
import { JOB_TYPES } from "../app/jobs-rules.ts";

const app = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
const registered = [...app.matchAll(/^\s+"?([a-z-]+)"?: [A-Za-z]+View,$/gm)].map((m) => m[1]);

async function render(path) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  const worker = (await import(workerUrl.href)).default;
  const env = { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } };
  return worker.fetch(new Request(`https://golf.example${path}`, { headers: { accept: "text/html" } }), env, { waitUntil() {}, passThroughOnException() {} });
}

/* ------------------------------------------------------------------ routes */
test("the root and an empty segment land on This week", async () => {
  assert.equal(DEFAULT_VIEW, "this-week");
  assert.deepEqual(resolveRoute("", registered), { kind: "view", key: "this-week" });
  assert.deepEqual(resolveRoute(undefined, registered), { kind: "view", key: "this-week" });
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /initialView="this-week"/);
  const html = await (await render("/")).text();
  assert.match(html, /class="brand" href="\/this-week"/);
});

test("old routes redirect or show a retired notice; nothing is deleted", async () => {
  assert.deepEqual(resolveRoute("weather", registered), { kind: "redirect", key: "weather", to: "/weather-effects" });
  assert.deepEqual(resolveRoute("sg-distributions", registered), { kind: "redirect", key: "sg-distributions", to: "/players" });
  assert.deepEqual(resolveRoute("research", registered), { kind: "retired", key: "research" });
  for (const key of ["weather", "sg-distributions", "research", "history", "diagnostics"]) assert.ok(registered.includes(key), `${key} stays registered`);
  assert.deepEqual(Object.keys(REDIRECTS).sort(), ["sg-distributions", "weather"]);
  const html = await (await render("/research")).text();
  assert.match(html, /Betting backtests are no longer on this site/);
});

test("an unknown route is a visible not-found notice with the nav, not a redirect", async () => {
  assert.deepEqual(resolveRoute("nope", registered), { kind: "notfound", requested: "nope" });
  const response = await render("/nope");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /That page does not exist/);
  assert.match(html, /There is no page at \/nope/);
  assert.match(html, /aria-label="Dashboard navigation"/);
});

test("a redirect route renders a moved notice with a link (the browser then replaces the location)", async () => {
  const html = await (await render("/weather")).text();
  assert.match(html, /Go to the new page/);
  assert.match(html, /href="\/weather-effects"/);
});

/* ------------------------------------------------------------------ navigation */
test("navigation groups: Week, Model, Operate, Review, then a collapsed Archive", () => {
  assert.deepEqual(NAVIGATION.map((g) => g.label), ["Week", "Model", "Operate", "Review", "Archive"]);
  assert.deepEqual(NAVIGATION[0].items.map((i) => i.label), ["This week", "Players", "Why priced", "Distributions", "Scoring and weather"]);
  assert.deepEqual(NAVIGATION.at(-1).items.map((i) => i.label), ["History", "Diagnostics"]);
  assert.equal(NAVIGATION.at(-1).collapsed, true);
  assert.equal(NAVIGATION.filter((g) => g.collapsed).length, 1);
  assert.equal(NAV_ITEMS.find((i) => i.key === "performance").label, "Scorecard and P&L");
  assert.equal(NAV_ITEMS.find((i) => i.key === "run").label, "Run");
  for (const hidden of ["weather", "sg-distributions", "research"]) assert.ok(!NAV_ITEMS.some((i) => i.key === hidden || i.toggle?.some((t) => t.key === hidden)), `${hidden} is not in the nav`);
});

test("one Players entry toggles Field and Player; the active entry resolves for both routes", () => {
  const players = NAV_ITEMS.find((i) => i.label === "Players");
  assert.deepEqual(players.toggle.map((t) => t.href), ["/weekly-players", "/players"]);
  assert.equal(players.href, "/weekly-players");
  assert.equal(navEntryFor("players").item, players);
  assert.equal(navEntryFor("weekly-players").item, players);
  assert.equal(navEntryFor("weather-effects").item.label, "Scoring and weather");
  assert.equal(navEntryFor("nope"), null);
  for (const item of NAV_ITEMS) for (const key of [item.key, ...(item.toggle ?? []).map((t) => t.key)]) assert.ok(registered.includes(key), key);
});

test("the rendered nav shows the groups and keeps the Archive group closed", async () => {
  const html = await (await render("/this-week")).text();
  for (const label of ["Week", "Model", "Operate", "Review", "Archive", "Scoring and weather", "Scorecard and P&amp;L"]) assert.match(html, new RegExp(`>${label}<`), label);
  assert.match(html, /<details class="nav-group nav-archive"/);
  assert.doesNotMatch(html, /<details class="nav-group nav-archive" open/);
});

/* ------------------------------------------------------------------ header */
const HOUR = 3_600_000;
const NOW = Date.parse("2026-10-09T12:00:00Z");
const iso = (hoursAgo) => new Date(NOW - hoursAgo * HOUR).toISOString();
const index = {
  events: [
    { name: "Old week", date_start: "2026-10-01", runs: [{ as_of: iso(200) }] },
    { name: "Baycurrent Classic", course: "Baycurrent", date_start: "2026-10-08", event_uid: "pga:1", runs: [{ as_of: iso(40) }, { as_of: iso(22) }] },
    { name: "Open de Espana", course: "Madrid", date_start: "2026-10-08", event_uid: "euro:2", runs: [{ as_of: iso(32) }] },
  ],
};

test("header freshness is the OLDEST event's newest run, not the newest run of any event", () => {
  const week = summarizeWeek(index, NOW);
  assert.equal(week.title, "Baycurrent Classic · Open de Espana");
  assert.equal(week.sub, "Baycurrent · Madrid");
  assert.equal(week.oldestAsOf, iso(32), "Espana (32 h) is older than Baycurrent's newest (22 h); the old code reported 22 h");
  assert.deepEqual(week.events.map((e) => Math.round((e.ageMs ?? 0) / HOUR)), [22, 32]);
  assert.match(week.tooltip, /Baycurrent Classic: last run Oct 8, 10:00 AM ET \(22 h ago\)/);
  assert.match(week.tooltip, /Open de Espana: last run Oct 8, 12:00 AM ET \(32 h ago\)/);
  assert.ok(!week.tooltip.includes("Old week"), "only the newest date_start week");
});

test("a single event, an event without a run, and no events", () => {
  const single = summarizeWeek({ events: [{ name: "Solo", date_start: "2026-10-08", runs: [{ as_of: iso(3) }] }] }, NOW);
  assert.equal(single.oldestAsOf, iso(3));
  const noRun = summarizeWeek({ events: [{ name: "A", date_start: "2026-10-08", runs: [{ as_of: iso(5) }] }, { name: "B", date_start: "2026-10-08", runs: [] }] }, NOW);
  assert.equal(noRun.oldestAsOf, iso(5));
  assert.match(noRun.tooltip, /B: no run yet/);
  assert.equal(summarizeWeek({ events: [] }, NOW), null);
  assert.equal(summarizeWeek(null, NOW), null);
});

test("published_at and held_back render only when index.json carries them", () => {
  const plain = summarizeWeek(index, NOW);
  assert.equal(plain.publishedLine, null);
  assert.equal(plain.heldBackNote, null);
  const withFields = summarizeWeek({
    events: [
      { name: "A", date_start: "2026-10-08", runs: [{ as_of: iso(30) }], published_at: iso(6), held_back: { reason: "Round 2 pricing failed check c7; showing the previous validated run" } },
      { name: "B", date_start: "2026-10-08", runs: [{ as_of: iso(10) }], published_at: iso(2), held_back: null },
    ],
  }, NOW);
  assert.equal(withFields.publishedLine, "Published 6 h ago", "oldest publish");
  assert.match(withFields.heldBackNote, /^A newer run was not published: A: Round 2 pricing failed check c7/);
  assert.equal(heldBackText(["x", "y"]), "x; y");
  assert.equal(heldBackText(false), null);
  assert.equal(heldBackText({}), "a newer run was held back");
  assert.equal(summarizeWeek({ events: [{ name: "A", date_start: "2026-10-08", runs: [{ as_of: iso(1) }], published_at: "garbage" }] }, NOW).publishedLine, null);
});

test("the header source has no static Published dot or duplicate date, and labels legacy pages", () => {
  assert.doesNotMatch(app, /live-indicator|Latest golfprice run|displayDate/);
  assert.match(app, /Legacy data/);
  assert.match(app, /<HeaderSearch /);
});

test("?e= that names no published event is flagged; absent or valid is not", () => {
  const events = [{ event_uid: "pga:1" }, { event_uid: "euro:2" }];
  assert.equal(unknownEventParam("nope", events), true);
  assert.equal(unknownEventParam("euro:2", events), false);
  assert.equal(unknownEventParam(null, events), false);
  assert.equal(unknownEventParam("nope", undefined), false, "index not loaded yet");
  assert.equal(unknownEventParam("nope", []), false);
});

/* ------------------------------------------------------------------ data.ts fallback */
test("packaged fallback is used only when there is no Worker (network failure or a non-JSON body)", () => {
  assert.deepEqual(decideFallback({ kind: "network" }), { action: "fallback" });
  assert.deepEqual(decideFallback({ kind: "response", ok: false, status: 404, json: undefined }), { action: "fallback" }, "HTML 404 page = no Worker route");
  assert.deepEqual(decideFallback({ kind: "response", ok: true, status: 200, json: undefined }), { action: "fallback" }, "200 with an HTML body");
  assert.deepEqual(decideFallback({ kind: "response", ok: true, status: 200, json: { events: [] } }), { action: "use" });
});

test("a structured Worker error is surfaced, never masked by the packaged copy", () => {
  assert.deepEqual(decideFallback({ kind: "response", ok: false, status: 404, json: { ok: false, error: "not published yet" } }), { action: "surface", message: "This data could not be loaded (code 404): not published yet" });
  assert.deepEqual(decideFallback({ kind: "response", ok: false, status: 503, json: { ok: false, error: "dashboard bucket is not bound" } }), { action: "surface", message: "This data could not be loaded (code 503): dashboard bucket is not bound" });
  assert.equal(decideFallback({ kind: "response", ok: false, status: 500, json: { nothing: true } }).action, "surface", "non-ok JSON without an error string is still real");
  assert.equal(decideFallback({ kind: "response", ok: false, status: 400, json: { error: "Invalid data path" } }).action, "surface");
});

test("data.ts uses the decision and the production build drops the packaged snapshot", async () => {
  const data = await readFile(new URL("../app/data.ts", import.meta.url), "utf8");
  assert.match(data, /decideFallback/);
  assert.doesNotMatch(data, /\.catch\(\(\) => readJson/, "no blanket catch-and-retry on the static copy");
  const vite = await readFile(new URL("../vite.config.ts", import.meta.url), "utf8");
  assert.match(vite, /excludePackagedGolfprice/);
  assert.match(vite, /RESEARCH_LOCAL_PREVIEW/);
  const gitignore = await readFile(new URL("../.gitignore", import.meta.url), "utf8");
  assert.match(gitignore, /^\/public\/golfprice\/$/m);
  await assert.rejects(access(new URL("../dist/client/golfprice", import.meta.url)), "the production build output has no packaged golfprice copy");
});

/* ------------------------------------------------------------------ player search */
const players = [
  { name: "Hideki Matsuyama", dg_id: 13562, tours: ["pga"], status: "complete", profile_key: "a", country: "Japan" },
  { name: "Blair Hamilton", dg_id: 17639, tours: ["euro"], status: "partial", profile_key: "b" },
  { name: "Ludvig Åberg", dg_id: 23950, tours: ["pga", "euro"], status: "complete", profile_key: "c" },
  { name: "Old Timer", dg_id: null, profile_id: "historical-abc123", tours: [], status: "partial", profile_key: "d" },
  { name: "Matt Fitzpatrick", dg_id: 1, tours: ["pga"], status: "complete", profile_key: "e" },
];

test("search needs two characters, folds accents, ranks name-prefix first, and links to /players?player=<id>", () => {
  assert.deepEqual(searchPlayers(players, "h"), []);
  assert.deepEqual(searchPlayers(players, "aberg").map((p) => p.name), ["Ludvig Åberg"]);
  assert.deepEqual(searchPlayers(players, "ma").map((p) => p.name), ["Matt Fitzpatrick", "Hideki Matsuyama"], "prefix of first name, then of last name");
  assert.equal(playerHref(players[1]), "/players?player=17639");
  assert.equal(playerHref(players[3]), "/players?player=historical-abc123");
  assert.equal(searchPlayers(players, "zzzz").length, 0);
  assert.equal(searchPlayers(Array.from({ length: 30 }, (_, i) => ({ name: `Sam ${i}`, dg_id: i + 1, status: "complete", profile_key: `k${i}` })), "sam").length, 8);
});

test("the header search loads the catalog lazily and navigates with a full page load", async () => {
  const search = await readFile(new URL("../app/HeaderSearch.tsx", import.meta.url), "utf8");
  assert.match(search, /armed \? PLAYER_CATALOG_KEY : null/, "no request until the box is focused");
  assert.match(search, /window\.location\.assign\(playerHref/);
  const rules = await readFile(new URL("../app/shell-rules.ts", import.meta.url), "utf8");
  assert.match(rules, /golfprice\/player_profiles\/dossier-review\.json/);
  const html = await (await render("/this-week")).text();
  assert.match(html, /aria-label="Search golfers"/);
});

/* ------------------------------------------------------------------ Run page and wording */
test("the Run page suggests today's weekday group in New York time", () => {
  assert.equal(suggestedGroup(Date.parse("2026-10-12T14:00:00Z")), "Monday");
  assert.equal(suggestedGroup(Date.parse("2026-10-13T14:00:00Z")), "Tuesday");
  assert.equal(suggestedGroup(Date.parse("2026-10-14T14:00:00Z")), "Wednesday");
  assert.equal(suggestedGroup(Date.parse("2026-10-15T14:00:00Z")), "Thursday");
  assert.equal(suggestedGroup(Date.parse("2026-10-17T14:00:00Z")), "During event");
  assert.equal(suggestedGroup(Date.parse("2026-10-13T02:00:00Z")), "Monday", "Tuesday 02:00 UTC is still Monday evening in New York");
});

test("Run page source: Suggested now, collapsed rest, and 'Machine: unavailable'", async () => {
  const run = await readFile(new URL("../app/RunView.tsx", import.meta.url), "utf8");
  assert.match(run, /Suggested now/);
  assert.match(run, /run-collapsed/);
  assert.match(run, /Machine: unavailable/);
  assert.doesNotMatch(run, /Unknown until the page can load/);
});

test("job descriptions no longer talk about arms, comparison or the champion", () => {
  for (const spec of JOB_TYPES) assert.doesNotMatch(`${spec.does} ${spec.when}`, /\barms?\b|comparison|challenger|champion/i, spec.type);
});

/* ------------------------------------------------------------------ shared helpers (F1) */
test("lib helpers: finite, unsigned pct, signed pct, short date", () => {
  assert.equal(finite(0), true);
  assert.equal(finite(NaN), false);
  assert.equal(finite("1"), false);
  assert.equal(pct(0.1234), "12.3%");
  assert.equal(pct(0.0004), "0.04%");
  assert.equal(pct(0), "0.0%");
  assert.equal(pct(null), "—");
  assert.equal(signedPct(0.012), "+1.2%");
  assert.equal(signedPct(-0.004), "-0.4%");
  assert.equal(shortDate("not a date"), "—");
  assert.match(shortDate("2026-10-08T12:00:00Z"), /^Oct 8$/);
});
