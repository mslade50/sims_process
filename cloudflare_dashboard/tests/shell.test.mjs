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
  for (const key of ["sg-distributions", "research", "history", "diagnostics"]) assert.ok(registered.includes(key), `${key} stays registered`);
  assert.ok(!registered.includes("weather"), "the retired static weather view is gone; /weather only redirects");
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
  assert.deepEqual(NAVIGATION[0].items.map((i) => i.label), ["This week", "Weather", "Players", "Why priced", "Distributions", "Scoring"]);
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
  assert.equal(navEntryFor("weather-effects").item.label, "Weather");
  assert.equal(navEntryFor("nope"), null);
  for (const item of NAV_ITEMS) for (const key of [item.key, ...(item.toggle ?? []).map((t) => t.key)]) assert.ok(registered.includes(key), key);
});

test("the rendered nav shows the groups and keeps the Archive group closed", async () => {
  const html = await (await render("/this-week")).text();
  for (const label of ["Week", "Model", "Review", "Archive", "Weather", "Scoring", "Scorecard and P&amp;L"]) assert.match(html, new RegExp(`>${label}<`), label);
  // Owner-only links (Run, Weather edits) are added in the browser once /api/whoami says owner, so the server render never shows them.
  for (const label of ["Operate", "Run", "Weather edits"]) assert.doesNotMatch(html, new RegExp(`>${label}<`), label);
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

test("the redesigned shell shows the selected event's tour, name and course, and truncates instead of wrapping", async () => {
  const parts = await readFile(new URL("../app/ShellParts.tsx", import.meta.url), "utf8");
  const css = await readFile(new URL("../app/shell.css", import.meta.url), "utf8");
  assert.match(app, /<EventSwitcher events=\{events\} selected=\{selectedEvent\}/);
  assert.match(parts, /evsw-tour[\s\S]*selected\.name[\s\S]*selected\.course/);
  assert.match(css, /\.evsw-name strong \{[^}]*text-overflow: ellipsis/);
  assert.match(css, /\.evsw > button \{[^}]*max-width: min\(46vw/);
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

test("the header search reads the slim index (never the 16 MB catalog unprompted) and navigates with a full page load", async () => {
  const search = await readFile(new URL("../app/HeaderSearch.tsx", import.meta.url), "utf8");
  assert.match(search, /usePlayerDirectory\(\{ load: armed \|\| prefetch, allowFull: armed \}\)/, "the slim index may be warmed when idle; the full catalog only after the box is used");
  assert.doesNotMatch(search, /PLAYER_CATALOG_KEY/, "the header search never names the 16 MB catalog");
  assert.match(search, /window\.location\.assign\(playerHref/);
  const rules = await readFile(new URL("../app/shell-rules.ts", import.meta.url), "utf8");
  assert.match(rules, /golfprice\/player_profiles\/index\.json/);
  assert.match(rules, /golfprice\/player_profiles\/search-index\.json/);
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

test("Run page source: Suggested now, collapsed rest, and a plain runner-unavailable message", async () => {
  const run = await readFile(new URL("../app/RunView.tsx", import.meta.url), "utf8");
  assert.match(run, /Suggested now/);
  assert.match(run, /run-collapsed/);
  assert.match(run, /desktop runner status could not be loaded/i);
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

test("context bar: one event identity, stage from the newest run, default event matches the pages", async () => {
  const { eventHref, primaryTabKeys, selectedShellEvent, shellEvents, stageOf } = await import("../app/shell-rules.ts");
  const now = Date.parse("2026-10-08T20:00:00Z");
  const index = { events: [
    { event_uid: "pga:1", name: "Baycurrent", tour: "pga", date_start: "2026-10-08", date_end: "2026-10-11", explain_key: "k", runs: [{ as_of: "2026-10-08T12:21:00Z", kind: "live", after_round: 1 }, { as_of: "2026-10-07T12:00:00Z", kind: "week" }] },
    { event_uid: "euro:2", name: "Open de Espana", tour: "euro", date_start: "2026-10-08", runs: [{ as_of: "2026-10-07T09:00:00Z", kind: "week" }] },
    { event_uid: "pga:0", name: "Old", tour: "pga", date_start: "2026-10-01", runs: [] },
  ] };
  const events = shellEvents(index, now);
  assert.deepEqual(events.map((e) => e.uid), ["pga:1", "euro:2"], "only the newest week, explain-key event first");
  assert.equal(events[0].stage, "Prices after R1");
  assert.equal(events[0].status.stageNow, "R1 in play", "the calendar, not the newest run, says where the event is");
  assert.equal(events[0].tourLabel, "PGA Tour");
  assert.equal(events[1].stage, "Pre-tournament prices");
  assert.equal(selectedShellEvent(events, "euro:2")?.uid, "euro:2");
  assert.equal(selectedShellEvent(events, "nope")?.uid, "pga:1");
  assert.equal(stageOf({ date_end: "2026-10-04", runs: [] }, now).stage, "Final");
  assert.equal(eventHref("/this-week", "?e=a&player=7&x=1", "b"), "/this-week?e=b&x=1");
  assert.deepEqual(primaryTabKeys(true), ["this-week", "weekly-players", "bet-card", "weather-effects"]);
  assert.ok(!primaryTabKeys(false).includes("bet-card"));
});

test("freshness is usability: stage now, which round the prices reflect and whether they lag; age alone never colours", async () => {
  const { eventStatus, groupOf } = await import("../app/shell-rules.ts");
  const ev = (runs, extra = {}) => ({ tour: "pga", date_start: "2026-10-08", date_end: "2026-10-11", runs, ...extra });
  const live1 = [{ as_of: "2026-10-08T23:30:00Z", kind: "live", after_round: 1 }];
  // Friday 13:00 ET (R2 in play), run is 13 h old and reflects R1: stage and price round are separate, and the prices lag in play.
  const fri = Date.parse("2026-10-09T17:00:00Z");
  const a = eventStatus(ev(live1), fri);
  assert.equal(a.stageNow, "R2 in play");
  assert.equal(a.pricesLabel, "Prices after R1");
  assert.equal(a.state, "behind");
  assert.match(a.consequence, /R2 is in play/);
  // The same 13 h-old run early on Friday (before play) is fine: age alone is not a warning.
  const early = eventStatus(ev(live1), Date.parse("2026-10-09T09:00:00Z"));
  assert.equal(early.stageNow, "R1 complete");
  assert.equal(early.state, "ok");
  // Prices that miss a finished round lag the stage whatever their age.
  const lag = eventStatus(ev([{ as_of: "2026-10-08T12:00:00Z", kind: "week" }]), Date.parse("2026-10-09T02:00:00Z"));
  assert.equal(lag.stageNow, "R1 complete");
  assert.equal(lag.pricesLabel, "Pre-tournament prices");
  assert.equal(lag.state, "behind");
  // Before the event, a day-old run is simply current; no run is "none"; an ended event is "final".
  assert.equal(eventStatus(ev([{ as_of: "2026-10-06T12:00:00Z", kind: "week" }]), Date.parse("2026-10-07T12:00:00Z")).state, "ok");
  assert.equal(eventStatus(ev([]), fri).state, "none");
  assert.equal(eventStatus(ev(live1), Date.parse("2026-10-20T12:00:00Z")).state, "final");
  // DP World Tour hours differ from US hours.
  assert.equal(eventStatus(ev(live1, { tour: "euro" }), Date.parse("2026-10-09T08:00:00Z")).stageNow, "R2 in play");
  // One eyebrow rule: the nav group names the page.
  assert.equal(groupOf("players"), "Week");
  assert.equal(groupOf("bet-card"), "Review");
  assert.equal(groupOf("run"), "Operate");
  assert.equal(groupOf("history"), "Archive");
  assert.equal(groupOf("nope"), null);
});

/* ------------------------------------------------------------------ round 3: archive status, early fetch, chunk recovery */
import vm from "node:vm";
import { archivedStatusText } from "../app/freshness-rules.ts";
import { earlyFetchScript, explainWarmKey, resolveEventUid, mayReloadForChunk, isChunkLoadError } from "../app/shell-rules.ts";

test("retired archives show a neutral Archived state, never an age", () => {
  assert.equal(archivedStatusText(undefined), "Archived");
  assert.equal(archivedStatusText("2026-10-27T12:00:00Z"), "Archived · last data Oct 27, 2026");
  assert.doesNotMatch(archivedStatusText("2020-01-01"), /ago/);
  assert.match(app, /<ArchivedStatus/);
  assert.doesNotMatch(app, /label="Data"/, "the legacy bar no longer uses the freshness badge");
});

function runEarly({ path, search = "", remembered = null, index }) {
  const calls = [];
  const fetch = (url, init) => {
    calls.push({ url, cache: init.cache });
    const body = url === "/api/golfprice/index.json" ? index : {};
    return Promise.resolve({ ok: true, clone() { return this; }, json: async () => body });
  };
  const win = {};
  const ctx = { window: win, fetch, location: { pathname: path, search }, localStorage: { getItem: () => remembered }, URLSearchParams, String };
  vm.runInNewContext(earlyFetchScript(), ctx);
  return new Promise((resolve) => setTimeout(() => resolve({ calls, early: win.__earlyFetch }), 20));
}
const EV = (uid, tour, start, extra = {}) => ({ event_uid: uid, tour, date_start: start, explain_key: `golfprice/explain/${uid.replace(/:/g, "_")}/latest.json`, explain_run: "pre_x", ...extra });

test("the early-fetch script requests the index, then exactly the explain document this page will resolve", async () => {
  const index = { events: [EV("euro:1:2026-11-05", "euro", "2026-11-05"), EV("pga:2:2026-11-05", "pga", "2026-11-05"), EV("pga:0:2026-10-29", "pga", "2026-10-29")] };
  const cases = [
    { path: "/this-week" }, { path: "/" }, { path: "/this-week", search: "?e=pga%3A2%3A2026-11-05" }, { path: "/this-week", search: "?e=pga%3A0%3A2026-10-29" },
    { path: "/this-week", search: "?e=nope" }, { path: "/this-week", remembered: "pga:2:2026-11-05" }, { path: "/this-week", remembered: "pga:0:2026-10-29" },
  ];
  for (const c of cases) {
    const { calls } = await runEarly({ ...c, index });
    const params = new URLSearchParams(c.search ?? "");
    const uid = resolveEventUid(index, params.get("e"), c.remembered, 0);
    const want = explainWarmKey(index.events, uid);
    assert.equal(calls[0].url, "/api/golfprice/index.json");
    assert.equal(calls[0].cache, "no-cache");
    assert.deepEqual(calls.slice(1).map((x) => x.url), want ? [`/api/${want}`] : [], JSON.stringify(c));
  }
  const live = { events: [EV("pga:2:2026-11-05", "pga", "2026-11-05", { explain_run: "pre_x", distribution_runs: [{ kind: "live" }] })] };
  assert.equal((await runEarly({ path: "/this-week", index: live })).calls.length, 1, "a live run newer than the explain run leaves the choice to the page");
  assert.equal((await runEarly({ path: "/players", index })).calls.length, 1, "other pages only start the index");
  assert.equal((await runEarly({ path: "/weekly-players", index })).calls.length, 1, "other pages only start the index");
});

test("the early-fetch script warms the one-golfer slice and Why priced's document", async () => {
  const index = { events: [EV("pga:2:2026-11-05", "pga", "2026-11-05")] };
  const one = await runEarly({ path: "/players", search: "?player=10091", index });
  assert.deepEqual(one.calls.map((x) => x.url), ["/api/golfprice/index.json", "/api/golfprice/player_profiles/golfer/search-index.json?id=10091"]);
  assert.equal(one.calls[1].cache, "no-cache");
  assert.equal((await runEarly({ path: "/players", search: "?player=10091&generation=" + "a".repeat(20), index })).calls.length, 1, "a pinned generation uses another URL: left to the page");
  assert.equal((await runEarly({ path: "/players", search: "?player=../x", index })).calls.length, 1);
  const pre = await runEarly({ path: "/why-priced", index });
  assert.deepEqual(pre.calls.slice(1).map((x) => x.url), ["/api/golfprice/explain/pga_2_2026-11-05/latest.json"]);
  const withLive = { events: [EV("pga:2:2026-11-05", "pga", "2026-11-05", { explain_run: "live_R1_x", distribution_runs: [{ kind: "week", key: "golfprice/explain/pga_2/week_x.json", as_of: "2026-11-05T01:00:00Z" }, { kind: "live", key: "golfprice/explain/pga_2/live_R1_x.json" }] })] };
  assert.deepEqual((await runEarly({ path: "/why-priced", index: withLive })).calls.slice(1).map((x) => x.url), ["/api/golfprice/explain/pga_2/week_x.json"], "with a live run Why priced opens on the pre-tournament document");
  assert.equal((await runEarly({ path: "/why-priced", search: "?v=live", index: withLive })).calls.length, 1);
});

test("a deploy under an open tab reloads once a minute at most", () => {
  assert.equal(mayReloadForChunk(null, 1_000_000), true);
  assert.equal(mayReloadForChunk(String(1_000_000 - 5_000), 1_000_000), false);
  assert.equal(mayReloadForChunk(String(1_000_000 - 61_000), 1_000_000), true);
  assert.equal(isChunkLoadError(new TypeError("Failed to fetch dynamically imported module: /_next/static/chunks/x.js")), true);
  assert.equal(isChunkLoadError(new Error("boom")), false);
});
