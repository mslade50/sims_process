import assert from "node:assert/strict";
import test from "node:test";
import { briefCurrency, calendarDate, dataThrough, daysSince, historyThrough, plainDate, scorecardLag, seasonField, shotFreshness, shotFreshnessLine, weatherRunKey } from "../app/freshness-rules.ts";
import { parseCourseHistory } from "../app/course-history-rules.ts";
import { inWindow } from "../worker/cron-rules.ts";
import { FRESHNESS_STATUS_KEY, SOURCES, carryForward, eventsActive, fieldFromHead, inboxRecord, judge, oddsCheckEvidence, parseFreshness, runnerSilent, windowOpenForMs } from "../worker/freshness-rules.ts";
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
  const monLate = iso("2026-10-26T22:00:00Z"); // Monday afternoon: window continuous for over 12 h
  const sat = iso("2026-10-24T15:00:00Z");
  const okRun = (now) => ({ lastSuccessMs: now - 20 * 60_000, jobsSeen: 5 });
  assert.equal(judge(spec, read, { now: sunEvening, inWindow, oddsCheck: okRun(sunEvening) }).state, "ok");
  assert.equal(judge(spec, read, { now: sat, inWindow }).state, "skipped", "window closed: not judged");
  assert.equal(judge(spec, { kind: "absent" }, { now: sat, inWindow }).state, "skipped");
  assert.equal(judge(spec, { kind: "absent" }, { now: monLate, inWindow, oddsCheck: okRun(monLate) }).state, "missing");
  // quiet market: the published file is 8 days old but checks keep succeeding, so it is healthy
  const quiet = judge(spec, read, { now: monLate, inWindow, oddsCheck: okRun(monLate) });
  assert.equal(quiet.state, "ok");
  assert.match(quiet.detail, /quiet market publishes nothing new/);
  // the checker stopped: no clean run for 5 h while the window is open
  const dead = judge(spec, read, { now: monLate, inWindow, oddsCheck: { lastSuccessMs: monLate - 5 * 3_600_000, jobsSeen: 8 } });
  assert.equal(dead.state, "stale");
  assert.equal(dead.via, "odds_reprice job");
  assert.equal(judge(spec, read, { now: monLate, inWindow, oddsCheck: { lastSuccessMs: null, jobsSeen: 0 } }).state, "stale", "no odds job at all while the window is open");
  assert.equal(judge(spec, read, { now: monLate, inWindow }).state, "unreadable", "queue unreadable: no judgment");
  assert.equal(judge(spec, read, { now: sunEvening, inWindow, oddsCheck: { lastSuccessMs: null, jobsSeen: 0 } }).state, "ok", "first hours of a window are not judged");
});

test("odds check evidence: newest clean odds_reprice run; failures, other types and bad exit codes do not count", () => {
  const queue = { jobs: [{ id: "a", type: "odds_reprice" }, { id: "b", type: "odds_reprice" }, { id: "c", type: "odds_reprice" }, { id: "d", type: "watch" }, { id: "e", type: "odds_reprice" }] };
  const statuses = new Map([
    ["a", { state: "done", finished_at: "2026-10-28T10:00:00Z", exit_code: 0 }],
    ["b", { state: "failed", finished_at: "2026-10-28T11:30:00Z", exit_code: 1 }],
    ["c", { state: "done", finished_at: "2026-10-28T11:00:00Z" }],
    ["d", { state: "done", finished_at: "2026-10-28T11:55:00Z", exit_code: 0 }],
    ["e", { state: "done", finished_at: "2026-10-28T11:45:00Z", exit_code: 2 }],
  ]);
  assert.deepEqual(oddsCheckEvidence(queue, statuses), { lastSuccessMs: iso("2026-10-28T11:00:00Z"), jobsSeen: 4 });
  assert.deepEqual(oddsCheckEvidence({ jobs: [] }, new Map()), { lastSuccessMs: null, jobsSeen: 0 });
  assert.equal(oddsCheckEvidence({ nope: 1 }, new Map()), null);
  assert.equal(oddsCheckEvidence(null, new Map()), null);
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
    async list({ prefix }) { return { objects: [...store.keys()].filter((k) => k.startsWith(prefix)).map((key) => ({ key })) }; },
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
    "jobs/queue.json": { schema: "x", jobs: [{ id: "odds-1", type: "odds_reprice", requested_at: "2026-10-28T11:30:00Z", requested_by: "cron" }] },
    "jobs/status/odds-1.json": { id: "odds-1", state: "done", exit_code: 0, finished_at: "2026-10-28T11:31:00Z" },
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
  const out = await runFreshness({ DASHBOARD_DATA: bucket, AGENT_INBOX: inbox, DATA_FRESHNESS_INBOX: "1", DATA_FRESHNESS_ENABLED: "1" }, NOW);
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
  const env = { DASHBOARD_DATA: bucket, AGENT_INBOX: inbox, DATA_FRESHNESS_INBOX: "1", DATA_FRESHNESS_ENABLED: "1" };
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
  assert.match(inbox.puts[2], /bet_card\.recovered\.json$/);
});

test("without the inbox binding the finding stays in the status object; R2 failures raise no finding", async () => {
  const data = healthyData();
  data["data/manifest.json"] = fresh("2026-10-01T00:00:00Z");
  const out = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(out.doc.inbox.bound, false);
  assert.equal(out.doc.sources.find((s) => s.id === "legacy_manifest").state, "stale");
  const broken = await runFreshness({ DASHBOARD_DATA: fakeBucket(healthyData(), { failKeys: ["owner/bet_card.json"] }), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(broken.doc.sources.find((s) => s.id === "bet_card").state, "unreadable");
  assert.equal(broken.doc.ok, true);
});

test("inbox filing is off unless DATA_FRESHNESS_INBOX is 1, even with the binding present (no escalation path by default)", async () => {
  const data = healthyData();
  data["owner/bet_card.json"] = fresh("2026-10-26T04:00:00Z");
  const inbox = fakeBucket();
  const out = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), AGENT_INBOX: inbox, DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(out.doc.sources.find((s) => s.id === "bet_card").state, "stale");
  assert.equal(inbox.puts.length, 0);
  assert.equal(out.doc.inbox.bound, false);
});

test("bet_breakdowns has no scheduled producer and is not judged", () => {
  assert.ok(!SOURCES.some((s) => s.id === "bet_breakdowns" || s.key === "owner/bet_breakdowns.json"));
});

test("off weeks and breaks are scheduled inactivity, however long; an announced event resumes judging", async () => {
  const data = healthyData();
  data["golfprice/index.json"] = { schema: "x", published_at: "2026-10-27T12:00:00Z", events: [{ event_uid: "pga:2:2026-10-10", tour: "pga", date_start: "2026-10-10", date_end: "2026-10-13" }] };
  data["owner/bet_card.json"] = fresh("2026-10-01T04:00:00Z");
  data["golfprice/odds_signals/latest.json"] = fresh("2026-10-01T04:00:00Z");
  data["data/manifest.json"] = fresh("2026-09-01T00:00:00Z");
  const out = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.ok(out.doc.sources.every((s) => s.state === "skipped"));
  assert.equal(out.doc.not_judged, true);
  assert.equal(out.doc.ok, false); // all skipped is never reported as healthy
  // December break: the last event ended in mid-November and none is published ahead (over 21 days): still inactive, not stale
  const december = iso("2026-12-20T15:00:00Z");
  data["golfprice/index.json"].events[0] = { event_uid: "pga:9:2026-11-12", tour: "pga", date_start: "2026-11-12", date_end: "2026-11-15" };
  const winter = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, december);
  assert.equal(winter.doc.stale_count, 0);
  assert.equal(winter.doc.not_judged, true);
  assert.match(winter.doc.notes.join(" "), /Scheduled inactivity/);
  // the first event of the new season is published ahead: judging resumes, and the old weekly sources now count
  data["golfprice/index.json"].events.push({ event_uid: "pga:1:2026-12-24", tour: "pga", date_start: "2026-12-24", date_end: "2026-12-27" });
  const resumed = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, december);
  assert.equal(resumed.doc.not_judged, false);
  assert.ok(resumed.doc.stale_count > 0);
  assert.equal(eventsActive([], NOW), null);
  assert.equal(eventsActive(null, NOW), null);
  assert.equal(eventsActive([{ date_start: "2026-10-30" }], NOW), true);
  assert.equal(eventsActive([{ date_start: "2026-10-01" }], NOW), false, "27 days back and nothing ahead is a break, not a lapse");
  assert.equal(eventsActive([{ date_start: "2026-10-10" }], NOW), false);
  assert.equal(eventsActive([{ date_start: "2026-09-01" }], NOW), false, "a 57-day gap is inactivity, not a lapsed pipeline");
});

test("end to end: quiet market with a successful checker is healthy; a failed checker is the finding", async () => {
  const data = healthyData();
  data["golfprice/odds_signals/latest.json"] = fresh("2026-10-26T01:00:00Z"); // 59 h old, nothing new published
  const quiet = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(quiet.doc.sources.find((x) => x.id === "odds_signals").state, "ok");
  assert.equal(quiet.doc.ok, true);
  data["jobs/status/odds-1.json"] = { id: "odds-1", state: "failed", exit_code: 1, finished_at: "2026-10-28T11:31:00Z" };
  const dead = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(dead.doc.sources.find((x) => x.id === "odds_signals").state, "stale");
  const unread = await runFreshness({ DASHBOARD_DATA: fakeBucket(healthyData(), { failKeys: ["jobs/queue.json"] }), DATA_FRESHNESS_ENABLED: "1" }, NOW);
  assert.equal(unread.doc.sources.find((x) => x.id === "odds_signals").state, "unreadable");
});

test("orphaned episodes: a rolled-over history id is closed with a recovered record; an unjudged scorecard_lag stays open", () => {
  const t0 = iso("2026-10-28T12:00:00Z");
  const open = (id) => ({ id, label: id, key: "k", schedule: "weekly", state: "stale", observed_at: null, age_hours: 1, limit_hours: 1, via: null, detail: "d", stale_since: new Date(t0).toISOString(), inboxed_at: new Date(t0).toISOString() });
  const r = carryForward(t0 + 900_000, [], [open("history:old"), open("scorecard_lag"), open("removed_source")], ["scorecard_lag"]);
  assert.deepEqual(r.transitions.map((t) => `${t.kind}:${t.id}`).sort(), ["recovered:history:old", "recovered:removed_source"]);
  assert.deepEqual(r.statuses.map((x) => x.id), ["scorecard_lag"]);
  assert.equal(r.statuses[0].stale_since, new Date(t0).toISOString());
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

test("a dead publisher in mid-season is flagged by about D+12 (silent runners); a break with live runners stays not_judged", async () => {
  const data = healthyData();
  data["golfprice/index.json"].events = [{ event_uid: "pga:1:2026-10-22", tour: "pga", date_start: "2026-10-22", date_end: "2026-10-25" }, { event_uid: "pga:2:2026-10-29", tour: "pga", date_start: "2026-10-29", date_end: "2026-11-01" }];
  data["data/manifest.json"] = fresh("2026-11-02T12:00:00Z");
  data["golfprice/odds_signals/latest.json"] = fresh("2026-11-02T01:00:00Z");
  data["owner/bet_card.json"] = fresh("2026-11-02T01:00:00Z");
  const d12 = iso("2026-11-13T15:00:00Z"); // 12 days after the last event start; the old 21-day rule would still be quiet
  data["jobs/heartbeat/desktop.json"] = { at: "2026-11-03T09:00:00Z" };
  const dead = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, d12);
  assert.equal(dead.doc.not_judged, false);
  assert.ok(dead.doc.stale_count > 0, JSON.stringify(dead.doc.sources.map((s) => [s.id, s.state])));
  assert.match(dead.doc.notes.join(" "), /heartbeat is old|heartbeat/);
  data["jobs/heartbeat/desktop.json"] = { at: "2026-11-13T14:50:00Z" };
  const quiet = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, d12);
  assert.equal(quiet.doc.not_judged, true);
  assert.equal(quiet.doc.stale_count, 0);
  delete data["jobs/heartbeat/desktop.json"];
  const none = await runFreshness({ DASHBOARD_DATA: fakeBucket(data), DATA_FRESHNESS_ENABLED: "1" }, d12);
  assert.equal(none.doc.not_judged, true, "no heartbeat object is no evidence");
  assert.equal(runnerSilent([{ at: "2026-11-13T00:00:00Z" }, { at: "2026-11-13T14:00:00Z" }], d12), false, "newest heartbeat counts");
  assert.equal(runnerSilent([], d12), false);
});
