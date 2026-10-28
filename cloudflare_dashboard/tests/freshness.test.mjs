import assert from "node:assert/strict";
import test from "node:test";
import { briefCurrency, calendarDate, dataThrough, daysSince, historyThrough, plainDate, scorecardLag, seasonField, shotFreshness, shotFreshnessLine, weatherRunKey } from "../app/freshness-rules.ts";
import { parseCourseHistory } from "../app/course-history-rules.ts";
import { inWindow } from "../worker/cron-rules.ts";
import { FRESHNESS_STATUS_KEY, SOURCES, carryForward, fieldFromHead, inboxRecord, judge, parseFreshness, windowOpenForMs } from "../worker/freshness-rules.ts";
import { runFreshness } from "../worker/freshness.ts";

const iso = (s) => Date.parse(s);

test("calendar dates: bare dates never shift, timestamps read as Eastern dates, junk is null", () => {
  assert.equal(calendarDate("2026-10-22"), "2026-10-22");
  assert.equal(calendarDate("2026-10-08 (strictly earlier events)"), "2026-10-08");
  assert.equal(calendarDate("2026-10-23T02:30:00Z"), "2026-10-22"); // 22:30 EDT the day before
  assert.equal(calendarDate("2026-10-07 21:51:30 UTC"), "2026-10-07");
  for (const bad of [null, undefined, 5, "", "soon"]) assert.equal(calendarDate(bad), null);
  assert.equal(plainDate("2026-10-22"), "Oct 22, 2026");
  assert.equal(dataThrough("2026-10-22"), "Data through Oct 22, 2026");
  assert.equal(dataThrough(undefined), null);
  assert.equal(daysSince("2026-10-20", iso("2026-10-28T15:00:00Z")), 8);
});

test("shot data freshness reads the published fields and tolerates their absence", () => {
  assert.deepEqual(shotFreshness({}), { through: null, generation: null });
  assert.deepEqual(shotFreshness({ deep_coverage: { shot_data: { newest_shot_event_date: "2026-10-18", export_generation: "abcdef0123456789ff" } } }), { through: "2026-10-18", generation: "abcdef0123456789ff" });
  assert.equal(shotFreshnessLine({ through: null, generation: null }), null);
  assert.equal(shotFreshnessLine({ through: "2026-10-18", generation: "abcdef0123456789ff" }), "Shot data through Oct 18, 2026 (export abcdef012345)");
  assert.equal(shotFreshnessLine({ through: null, generation: "g1" }), "Shot data export g1");
});

test("course history 'through' is the published source_through_date, else the newest round it holds", () => {
  assert.equal(historyThrough("2026-10-19", ["2026-10-25"]), "2026-10-19");
  assert.equal(historyThrough(undefined, ["2024-06-16", null, "2025-06-15"]), "2025-06-15");
  assert.equal(historyThrough(undefined, []), null);
  const doc = { schema: "golfprice.course_history.v1", status: "available", source_through_date: "2026-10-18", editions: [{ event_uid: "pga:1:2025-10-01", year: 2025, rounds: [{ round: 1, date: "2025-10-02" }] }], by_round: [] };
  assert.equal(parseCourseHistory(doc).through, "2026-10-18");
  assert.equal(parseCourseHistory({ ...doc, source_through_date: undefined }).through, "2025-10-02");
});

test("weather: earlier runs read their own immutable copy; the brief banner fires only when behind", () => {
  assert.equal(weatherRunKey("golfprice/weather/pga_528_2026-10-22/latest.json", "live", "R1_20261022T120000Z"), "golfprice/weather/pga_528_2026-10-22/live_R1_20261022T120000Z.json");
  for (const bad of [["x", "live", "../a"], ["x", "live", "a/b"], ["latest.json", "live", "r"], [null, "live", "r"], ["a/b.json", null, "r"]]) assert.equal(weatherRunKey(...bad), null);
  assert.equal(briefCurrency({}).state, "unknown");
  assert.equal(briefCurrency({ weather_run: "week_20261021T100000Z", weather_brief_run: "week_20261021T100000Z" }).state, "current");
  const behind = briefCurrency({ weather_run: "live_20261022T180000Z", weather_brief_run: "week_20261021T100000Z" });
  assert.equal(behind.state, "behind");
  assert.match(behind.message, /newer run/);
  assert.equal(briefCurrency({ weather_run: "live_20261022T180000Z" }).state, "behind");
  assert.equal(briefCurrency({ weather_run: "a", weather_brief_run: "a" }).state, "current");
});

test("scorecard lag compares the newest settled event with the newest completed one, not generated_at", () => {
  const index = [
    { event_uid: "pga:2:2026-10-22", name: "B Open", tour: "pga", date_start: "2026-10-22", date_end: "2026-10-25" },
    { event_uid: "pga:1:2026-10-15", name: "A Classic", tour: "pga", date_start: "2026-10-15", date_end: "2026-10-18" },
    { event_uid: "dpwt:9:2026-10-27", name: "Future", tour: "dpwt", date_start: "2026-10-27", date_end: "2026-10-30" },
  ];
  const settled = [{ event_uid: "pga:1:2026-10-15", event_start: "2026-10-15" }];
  assert.equal(scorecardLag(settled, index, iso("2026-10-28T12:00:00Z")).state, "behind");
  assert.match(scorecardLag(settled, index, iso("2026-10-28T12:00:00Z")).message, /B Open finished Oct 25, 2026/);
  assert.equal(scorecardLag(settled, index, iso("2026-10-26T12:00:00Z")).state, "current", "inside the settle grace");
  assert.equal(scorecardLag([...settled, { event_uid: "pga:2:2026-10-22", event_start: "2026-10-22" }], index, iso("2026-11-03T12:00:00Z")).state, "current");
  assert.equal(scorecardLag(settled, null, iso("2026-10-28T12:00:00Z")).state, "unknown");
  assert.equal(scorecardLag(settled, [], iso("2026-10-28T12:00:00Z")).state, "unknown");
});

test("season-neutral glossary keys win; season-suffixed keys are the fallback; no season is hard-coded", () => {
  assert.equal(seasonField({ ridge_lambda: 3, ridge_lambda_2026: 9 }, "ridge_lambda", 2026), 3);
  assert.equal(seasonField({ ridge_lambda_2026: 9 }, "ridge_lambda", 2026), 9);
  assert.equal(seasonField({ ridge_lambda_2027: 7, ridge_lambda_2026: 9 }, "ridge_lambda", 2027), 7);
  assert.equal(seasonField({ ridge_lambda_2026: 9, ridge_lambda_2027: 7 }, "ridge_lambda", null), 7, "newest suffixed key when the season is unknown");
  assert.equal(seasonField({}, "ridge_lambda", 2026), undefined);
  assert.equal(seasonField(null, "x"), undefined);
});

/* ------------------------------------------------------------------ worker rules */
test("odds window: opens Sunday evening; the first hours of a window are not judged", () => {
  const sunEvening = iso("2026-10-25T23:30:00Z"); // Sun 19:30 EDT, window opened 18:00
  assert.equal(inWindow("odds_reprice", sunEvening), true);
  assert.equal(inWindow("odds_reprice", iso("2026-10-24T15:00:00Z")), false); // Saturday
  assert.equal(windowOpenForMs(inWindow, "odds_reprice", iso("2026-10-24T15:00:00Z")), 0);
  const open = windowOpenForMs(inWindow, "odds_reprice", sunEvening);
  assert.ok(open >= 90 * 60_000 && open <= 105 * 60_000, `open for ${open}`);
  const spec = SOURCES.find((s) => s.id === "odds_signals");
  const read = { kind: "ok", uploadedMs: iso("2026-10-18T12:00:00Z"), head: '{"generated_at":"2026-10-18T12:00:00Z"}' };
  assert.equal(judge(spec, read, { now: sunEvening, inWindow }).state, "ok");
  const monLate = iso("2026-10-26T22:00:00Z"); // Monday afternoon: window continuous for over 12 h
  assert.equal(judge(spec, read, { now: monLate, inWindow }).state, "stale");
  const sat = iso("2026-10-24T15:00:00Z");
  assert.equal(judge(spec, read, { now: sat, inWindow }).state, "ok", "outside the window the weekly limit applies (6 days old is fine)");
  assert.equal(judge(spec, { kind: "absent" }, { now: sat, inWindow }).state, "skipped");
  assert.equal(judge(spec, { kind: "absent" }, { now: monLate, inWindow }).state, "missing");
});

test("weekly sources: 8 days; field first, upload time as fallback; read failure is never 'missing'", () => {
  const spec = SOURCES.find((s) => s.id === "legacy_manifest");
  const now = iso("2026-10-28T12:00:00Z");
  const ctx = { now, inWindow };
  assert.equal(judge(spec, { kind: "ok", uploadedMs: now, head: '{"generated_at": "2026-10-19T00:00:00Z"}' }, ctx).state, "stale", "an old own timestamp beats a fresh upload");
  assert.equal(judge(spec, { kind: "ok", uploadedMs: now, head: '{"generated_at": "2026-10-25T00:00:00Z"}' }, ctx).via, "generated_at");
  const fallback = judge(spec, { kind: "ok", uploadedMs: iso("2026-10-20T00:00:00Z"), head: "{}" }, ctx);
  assert.equal(fallback.via, "upload time");
  assert.equal(fallback.state, "stale");
  assert.equal(judge(spec, { kind: "error" }, ctx).state, "unreadable");
  assert.equal(judge(spec, { kind: "absent" }, ctx).state, "missing");
  assert.equal(judge(spec, { kind: "ok", uploadedMs: null, head: "" }, ctx).state, "unreadable");
  assert.deepEqual(fieldFromHead('{"a":1,"built_at":"2026-10-21T22:00:00Z","meta":{}}', ["generated_at", "built_at"]), { field: "built_at", value: "2026-10-21T22:00:00Z" });
});

test("transitions: file once when stale, once a day while it stays stale, recovered once; unreadable ticks change nothing", () => {
  const base = { id: "x", label: "X", key: "k", schedule: "weekly", observed_at: null, age_hours: 300, limit_hours: 192, via: "u", detail: "d" };
  const t0 = iso("2026-10-28T12:00:00Z");
  let r = carryForward(t0, [{ ...base, state: "stale" }], null);
  assert.deepEqual(r.transitions.map((t) => t.kind), ["stale"]);
  r.statuses[0].inboxed_at = new Date(t0).toISOString();
  r = carryForward(t0 + 900_000, [{ ...base, state: "stale" }], r.statuses);
  assert.equal(r.transitions.length, 0, "no duplicate on the next tick");
  assert.equal(r.statuses[0].stale_since, new Date(t0).toISOString());
  r = carryForward(t0 + 1_800_000, [{ ...base, state: "unreadable" }], r.statuses);
  assert.equal(r.transitions.length, 0);
  assert.equal(r.statuses[0].inboxed_at, new Date(t0).toISOString(), "an unreadable tick keeps the open episode");
  r = carryForward(t0 + 25 * 3_600_000, [{ ...base, state: "stale" }], r.statuses);
  assert.deepEqual(r.transitions.map((t) => t.kind), ["refile"]);
  r.statuses[0].inboxed_at = new Date(t0 + 25 * 3_600_000).toISOString();
  r = carryForward(t0 + 26 * 3_600_000, [{ ...base, state: "ok" }], r.statuses);
  assert.deepEqual(r.transitions.map((t) => t.kind), ["recovered"]);
  r = carryForward(t0 + 27 * 3_600_000, [{ ...base, state: "ok" }], r.statuses);
  assert.equal(r.transitions.length, 0);
});

test("inbox records match what the repair watcher reads and never name a phone path", () => {
  const status = { id: "bet_card", label: "owner bet card", key: "owner/bet_card.json", schedule: "odds_window", state: "stale", observed_at: null, age_hours: 20, limit_hours: 12, via: "generated_at", detail: "Last updated 20.0 h ago", stale_since: "2026-10-28T10:00:00.000Z", inboxed_at: null };
  const stale = inboxRecord({ id: "bet_card", kind: "stale", status }, "2026-10-28T12:00:00.000Z");
  assert.equal(stale.key, "ops/agent_inbox/dashboard-freshness/bet_card.json");
  assert.equal(stale.body.severity, "warning");
  assert.equal(stale.body.source, "dashboard-freshness");
  assert.equal(stale.body.dedupe_key, "dashboard-freshness:bet_card");
  assert.ok(typeof stale.body.summary === "string" && typeof stale.body.detail === "string" && typeof stale.body.ts === "string");
  const back = inboxRecord({ id: "bet_card", kind: "recovered", status: { ...status, state: "ok" } }, "2026-10-28T13:00:00.000Z");
  assert.equal(back.body.severity, "recovered");
  assert.equal(back.body.dedupe_key, stale.body.dedupe_key);
  assert.notEqual(back.key, stale.key);
  assert.doesNotMatch(JSON.stringify([stale, back]), /telegram/i);
});

/* ------------------------------------------------------------------ the check end to end (fake R2) */
function fakeBucket(initial = {}, { failKeys = [] } = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, { text: typeof v === "string" ? v : JSON.stringify(v), uploaded: new Date(iso("2026-10-28T11:00:00Z")) }]));
  const puts = [];
  return {
    store, puts,
    async get(key, options) {
      if (failKeys.includes(key)) throw new Error("R2 down");
      const o = store.get(key);
      if (!o) return null;
      const text = options?.range ? o.text.slice(options.range.offset, options.range.offset + options.range.length) : o.text;
      return { text: async () => text, uploaded: o.uploaded };
    },
    async put(key, body) { puts.push(key); store.set(key, { text: String(body), uploaded: new Date() }); return { key }; },
  };
}

const NOW = iso("2026-10-28T12:00:00Z"); // Wednesday 08:00 EDT: odds window open since Monday 00:00
const fresh = (at) => ({ generated_at: at });

function healthyData() {
  return {
    "golfprice/manifest.json": { schema: "x" },
    "golfprice/index.json": { schema: "x", published_at: "2026-10-27T12:00:00Z", events: [{ event_uid: "pga:2:2026-10-22", name: "B Open", tour: "pga", date_start: "2026-10-22", date_end: "2026-10-25", history_key: "golfprice/history/pga_2_2026-10-22.json" }] },
    "data/manifest.json": fresh("2026-10-26T12:00:00Z"),
    "golfprice/odds_signals/latest.json": fresh("2026-10-28T11:30:00Z"),
    "owner/bet_card.json": fresh("2026-10-28T11:30:00Z"),
    "owner/bet_breakdowns.json": { schema: "x", built_at: "2026-10-26T22:00:00Z", cols: {} },
    "golfprice/scorecard/latest.json": { generated_at: "2026-10-27T12:00:00Z", events: [{ event_uid: "pga:2:2026-10-22", event_start: "2026-10-22" }] },
    "golfprice/history/pga_2_2026-10-22.json": { source_through_date: "2026-10-18", editions: [] },
  };
}

test("check is off unless DATA_FRESHNESS_ENABLED is exactly 1 and writes nothing when off", async () => {
  const bucket = fakeBucket(healthyData());
  for (const flag of [undefined, "0", "true"]) {
    const out = await runFreshness({ DASHBOARD_DATA: bucket, DATA_FRESHNESS_ENABLED: flag }, NOW);
    assert.equal(out.enabled, false);
  }
  assert.equal(bucket.puts.length, 0);
});

test("healthy data: status object written, everything current, no inbox writes", async () => {
  const bucket = fakeBucket(healthyData());
  const inbox = fakeBucket();
  const out = await runFreshness({ DASHBOARD_DATA: bucket, AGENT_INBOX: inbox, DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(out.doc.ok, true, JSON.stringify(out.doc.sources.filter((s) => s.state !== "ok")));
  assert.deepEqual(bucket.puts, [FRESHNESS_STATUS_KEY]);
  assert.equal(inbox.puts.length, 0);
  const stored = parseFreshness(JSON.parse(bucket.store.get(FRESHNESS_STATUS_KEY).text));
  assert.equal(stored.stale_count, 0);
  assert.ok(stored.sources.some((s) => s.id.startsWith("history:")), "history documents are taken from the published index");
  assert.ok(stored.sources.some((s) => s.id === "scorecard_lag" && s.state === "ok"));
});

test("a stale bet card and a scorecard that lags files one inbox record each, then stays quiet, then recovers", async () => {
  const data = healthyData();
  data["owner/bet_card.json"] = fresh("2026-10-26T04:00:00Z");
  data["golfprice/scorecard/latest.json"] = { generated_at: "2026-10-28T08:00:00Z", events: [{ event_uid: "pga:1:2026-10-15", event_start: "2026-10-15" }] };
  const bucket = fakeBucket(data);
  const inbox = fakeBucket();
  const env = { DASHBOARD_DATA: bucket, AGENT_INBOX: inbox, DATA_FRESHNESS_ENABLED: "1" };
  let out = await runFreshness(env, NOW);
  assert.equal(out.doc.ok, false);
  assert.deepEqual(inbox.puts.sort(), ["ops/agent_inbox/dashboard-freshness/bet_card.json", "ops/agent_inbox/dashboard-freshness/scorecard_lag.json"]);
  out = await runFreshness(env, NOW + 900_000);
  assert.equal(inbox.puts.length, 2, "no duplicate record on the next tick");
  assert.equal(out.doc.stale_count, 2);
  data["owner/bet_card.json"] = fresh("2026-10-28T12:10:00Z");
  bucket.store.set("owner/bet_card.json", { text: JSON.stringify(data["owner/bet_card.json"]), uploaded: new Date(NOW) });
  out = await runFreshness(env, NOW + 1_800_000);
  assert.equal(out.doc.stale_count, 1);
  assert.equal(inbox.puts.length, 3);
  assert.match(inbox.puts[2], /bet_card\.recovered-/);
});

test("without the inbox binding the finding stays in the status object; R2 failures raise no finding", async () => {
  const data = healthyData();
  data["data/manifest.json"] = fresh("2026-10-01T00:00:00Z");
  const out = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(out.doc.inbox.bound, false);
  assert.equal(out.doc.sources.find((s) => s.id === "legacy_manifest").state, "stale");
  assert.match(out.notes.join(" "), /not bound/);
  const broken = await runFreshness({ DASHBOARD_DATA: fakeBucket(healthyData(), { failKeys: ["owner/bet_card.json"] }), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(broken.doc.sources.find((s) => s.id === "bet_card").state, "unreadable");
  assert.equal(broken.doc.ok, true);
});

/* ------------------------------------------------------------------ the built Worker */
async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

test("scheduled(): the freshness check runs alongside the other cron work and never touches Telegram", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket(healthyData());
  const waits = [];
  const realFetch = globalThis.fetch;
  let fetched = 0;
  globalThis.fetch = async () => { fetched += 1; return new Response("{}"); };
  try {
    await worker.scheduled({ scheduledTime: NOW, cron: "*/15 * * * *" }, { DASHBOARD_DATA: bucket, DATA_FRESHNESS_ENABLED: "1" }, { waitUntil: (p) => waits.push(p) });
    await Promise.all(waits);
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.ok(bucket.store.has(FRESHNESS_STATUS_KEY));
  assert.equal(fetched, 0);
});
