import assert from "node:assert/strict";
import test from "node:test";
import { P1_STALE_MS, dueAlerts, emptyAlertState, evaluateP1, parseAlertState, pruneState, seedState, shouldAlert } from "../worker/alerts-rules.ts";

const NOW = Date.parse("2026-10-09T15:00:00Z");
const MIN = 60_000;
const iso = (ms) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");
const beat = (minutesAgo) => ({ at: iso(NOW - minutesAgo * MIN) });
const queuedJob = (minutesAgo, over = {}) => ({ id: "j-20261009T140000-aaaaaa", type: "watch", requested_at: iso(NOW - minutesAgo * MIN), state: "queued", ...over });

test("shouldAlert ports telegram_policy: 24 h since the last send and no reservation in the last hour", () => {
  assert.equal(shouldAlert(undefined, NOW), true);
  assert.equal(shouldAlert({ sent_at: NOW - 23 * 3_600_000 }, NOW), false);
  assert.equal(shouldAlert({ sent_at: NOW - 25 * 3_600_000 }, NOW), true);
  assert.equal(shouldAlert({ sent_at: NOW - 25 * 3_600_000, reserved_at: NOW - 30 * MIN }, NOW), false, "a fresh reservation blocks");
  assert.equal(shouldAlert({ sent_at: NOW - 25 * 3_600_000, reserved_at: NOW - 61 * MIN }, NOW), true);
});

test("P1: primary silent over 30 min AND (backup silent OR a job queued over 30 min)", () => {
  assert.equal(evaluateP1(beat(5), beat(500), [], NOW).down, false, "fresh primary");
  assert.equal(evaluateP1(beat(31), beat(5), [], NOW).down, false, "backup alive and nothing stuck");
  assert.equal(evaluateP1(beat(31), beat(40), [], NOW).down, true, "both silent");
  assert.equal(evaluateP1(beat(31), beat(5), [queuedJob(45)], NOW).down, true, "stuck queued job");
  assert.equal(evaluateP1(beat(31), beat(5), [queuedJob(10)], NOW).down, false, "recently queued job is not stuck");
  assert.equal(evaluateP1(null, null, [], NOW).down, true, "no heartbeats at all");
  assert.equal(evaluateP1(beat(30), beat(500), [], NOW).down, false, "exactly at the 30 min threshold");
  assert.equal(P1_STALE_MS, 30 * MIN);
});

test("dueAlerts: P2 one alert per failed job id; P3 only in the 07:30 ET tick; recovery note only after an outage", () => {
  const failed = [queuedJob(60, { id: "j-1", state: "failed", type: "tuesday" }), queuedJob(60, { id: "j-2", state: "done" }), queuedJob(60, { id: "j-3", state: "failed", type: "wednesday" })];
  const base = { jobs24h: failed, recentJobs: failed, primary: beat(1), backup: beat(1), state: emptyAlertState(), now: NOW, etHHMM: "11:00", etDate: "2026-10-09" };
  assert.deepEqual(dueAlerts(base).map((a) => a.key), ["p2-failed-j-1", "p2-failed-j-3"]);
  assert.deepEqual(dueAlerts({ ...base, etHHMM: "07:30" }).map((a) => a.key), ["p2-failed-j-1", "p2-failed-j-3", "p3-digest-2026-10-09"]);
  assert.deepEqual(dueAlerts({ ...base, recentJobs: [], state: { ...emptyAlertState(), p1_outage: { since: NOW - 3_600_000 } } }).map((a) => a.key), ["p1-recovered"]);
  assert.deepEqual(dueAlerts({ ...base, recentJobs: [], primary: beat(90), backup: beat(90) }).map((a) => a.key), ["p1-primary-down"]);
});

test("alert state parses defensively and prunes keys older than 7 days", () => {
  assert.deepEqual(parseAlertState("junk").keys, {});
  const state = parseAlertState({ keys: { old: { sent_at: NOW - 8 * 86_400_000 }, fresh: { sent_at: NOW - 3_600_000 }, bad: 5 }, p1_outage: { since: NOW } });
  assert.deepEqual(Object.keys(pruneState(state, NOW).keys), ["fresh"]);
  assert.equal(state.p1_outage.since, NOW);
});

/* ---------------------------------------------------------------- scheduled tick through the built Worker */
async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

function fakeBucket(initial = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, { body: typeof v === "string" ? v : JSON.stringify(v), v: 1 }]));
  return {
    store,
    puts: [],
    get: async (key) => {
      const o = store.get(key);
      return o ? { key, etag: `e${o.v}`, text: async () => o.body } : null;
    },
    async put(key, body, options = {}) {
      this.puts.push(key);
      const existing = store.get(key);
      const cond = options.onlyIf;
      if (cond?.etagMatches && (!existing || `e${existing.v}` !== cond.etagMatches)) return null;
      if (cond?.etagDoesNotMatch === "*" && existing) return null;
      store.set(key, { body: String(body), v: (existing?.v ?? 0) + 1 });
      return { key, etag: `e${store.get(key).v}` };
    },
    list: async () => ({ objects: [], truncated: false }),
  };
}

const realFetch = globalThis.fetch;
let telegram = [];
let telegramStatus = 200;
test.before(() => {
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input.url;
    if (href.startsWith("https://api.telegram.org/")) {
      telegram.push({ href, body: JSON.parse(init.body) });
      return new Response("{}", { status: telegramStatus });
    }
    return realFetch(input, init);
  };
});
test.after(() => {
  globalThis.fetch = realFetch;
});
test.beforeEach(() => {
  telegram = [];
  telegramStatus = 200;
});

async function tick(worker, env, scheduledTime) {
  const pending = [];
  await worker.scheduled({ scheduledTime, cron: "*/15 * * * *" }, env, { waitUntil: (p) => pending.push(p), passThroughOnException() {} });
  await Promise.all(pending);
}

const SECRETS = { TELEGRAM_BOT_TOKEN: "test-token-not-real", TELEGRAM_CHAT_ID: "123" };
const heartbeat = (machine, at) => ({ machine, at: iso(at), state: "idle" });
const outageBucket = (extra = {}) => fakeBucket({ "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [] }, ...extra });
const WED = Date.parse("2026-10-07T16:07:00Z"); // 12:07 ET: no fixed slot, only the watch window

test("E5 dark: with no Telegram secrets a dead primary is evaluated and logged, nothing is sent and no state object is written", async () => {
  const worker = await loadWorker();
  const bucket = outageBucket(); // no heartbeats at all -> P1 conditions true
  await tick(worker, { DASHBOARD_DATA: bucket }, WED);
  assert.equal(telegram.length, 0, "no fetch to Telegram");
  assert.equal(bucket.store.has("jobs/alerts/state.json"), false);
  assert.equal(bucket.puts.some((k) => k.startsWith("jobs/alerts/")), false);
  // Only one of the two secrets set is still dark.
  await tick(worker, { DASHBOARD_DATA: bucket, TELEGRAM_BOT_TOKEN: "x" }, WED + 15 * MIN);
  await tick(worker, { DASHBOARD_DATA: bucket, TELEGRAM_CHAT_ID: "1" }, WED + 30 * MIN);
  assert.equal(telegram.length, 0);
  assert.equal(bucket.store.has("jobs/alerts/state.json"), false);
});

test("E5 dark: a failed job and the 07:30 digest are also silent without secrets", async () => {
  const worker = await loadWorker();
  const failed = { id: "j-20261007T150000-aaaaaa", type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: iso(WED - 60 * MIN), verified: true };
  const bucket = outageBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [failed] },
    [`jobs/status/${failed.id}.json`]: { id: failed.id, state: "failed" },
    "jobs/heartbeat/desktop-2ki41v6.json": heartbeat("desktop-2ki41v6", WED - MIN),
  });
  await tick(worker, { DASHBOARD_DATA: bucket }, Date.parse("2026-10-09T11:30:00Z")); // 07:30 EDT
  assert.equal(telegram.length, 0);
  assert.equal(bucket.store.has("jobs/alerts/state.json"), false);
});

test("E5 live path (secrets present): P1 pages once per outage, honors the cooldown, sends one recovery note, then pages a new outage", async () => {
  const worker = await loadWorker();
  const bucket = outageBucket();
  const env = { DASHBOARD_DATA: bucket, ...SECRETS };
  await tick(worker, env, WED);
  assert.equal(telegram.length, 1);
  assert.match(telegram[0].body.text, /Golf runner ALERT/);
  assert.equal(telegram[0].body.chat_id, "123");
  assert.ok(telegram[0].href.includes("/bottest-token-not-real/sendMessage"));
  let state = JSON.parse(bucket.store.get("jobs/alerts/state.json").body);
  assert.ok(state.keys["p1-primary-down"].sent_at);
  assert.ok(state.p1_outage);
  await tick(worker, env, WED + 15 * MIN);
  await tick(worker, env, WED + 30 * MIN);
  assert.equal(telegram.length, 1, "cooldown: still one page for the same outage");
  // Primary comes back.
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify(heartbeat("desktop-2ki41v6", WED + 44 * MIN)), v: 1 });
  await tick(worker, env, WED + 45 * MIN);
  assert.equal(telegram.length, 2);
  assert.match(telegram[1].body.text, /recovered/);
  state = JSON.parse(bucket.store.get("jobs/alerts/state.json").body);
  assert.equal(state.p1_outage, null);
  assert.equal(state.keys["p1-primary-down"], undefined);
  await tick(worker, env, WED + 60 * MIN);
  assert.equal(telegram.length, 2, "no second recovery note");
  // A second outage later the same day is a new page (the key was cleared on recovery).
  await tick(worker, env, WED + 3 * 3_600_000);
  assert.equal(telegram.length, 3);
  assert.match(telegram[2].body.text, /ALERT/);
});

test("E5 live path: one page per failed job id; reserve-then-send means a failed send is not retried for an hour", async () => {
  const worker = await loadWorker();
  const failed = { id: "j-20261007T150000-aaaaaa", type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: iso(WED - 60 * MIN), verified: true };
  const bucket = outageBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [failed] },
    [`jobs/status/${failed.id}.json`]: { id: failed.id, state: "failed" },
    "jobs/heartbeat/desktop-2ki41v6.json": heartbeat("desktop-2ki41v6", WED - MIN),
    "jobs/alerts/state.json": { schema: "golfprice.alert_state.v1", keys: {}, p1_outage: null, seeded: { at: WED - 3_600_000, failed: [] } },
  });
  const env = { DASHBOARD_DATA: bucket, ...SECRETS };
  telegramStatus = 500;
  await tick(worker, env, WED);
  assert.equal(telegram.length, 1, "attempted once");
  let state = JSON.parse(bucket.store.get("jobs/alerts/state.json").body);
  assert.ok(state.keys[`p2-failed-${failed.id}`].reserved_at);
  assert.equal(state.keys[`p2-failed-${failed.id}`].sent_at, undefined, "nothing recorded as sent");
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify(heartbeat("desktop-2ki41v6", WED + 14 * MIN)), v: 1 });
  telegramStatus = 200;
  await tick(worker, env, WED + 15 * MIN);
  assert.equal(telegram.length, 1, "reservation holds for an hour");
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify(heartbeat("desktop-2ki41v6", WED + 74 * MIN)), v: 1 });
  await tick(worker, env, WED + 75 * MIN);
  assert.equal(telegram.length, 2);
  assert.match(telegram[1].body.text, /Golf job failed: tuesday/);
  state = JSON.parse(bucket.store.get("jobs/alerts/state.json").body);
  assert.ok(state.keys[`p2-failed-${failed.id}`].sent_at);
  await tick(worker, env, WED + 90 * MIN);
  assert.equal(telegram.length, 2, "one page per failed job id");
});

test("E5 live path: daily digest at 07:30 ET once; a lost reservation race sends nothing", async () => {
  const worker = await loadWorker();
  const t = Date.parse("2026-10-09T11:30:00Z");
  const bucket = outageBucket({
    "jobs/heartbeat/desktop-2ki41v6.json": heartbeat("desktop-2ki41v6", t - MIN),
    "jobs/heartbeat/mckinley_home.json": heartbeat("mckinley_home", t - MIN),
  });
  const env = { DASHBOARD_DATA: bucket, ...SECRETS };
  await tick(worker, env, t);
  assert.equal(telegram.length, 1);
  assert.match(telegram[0].body.text, /daily digest/);
  await tick(worker, env, t); // same slot again
  assert.equal(telegram.length, 1);
  // Race: another tick changes the state object between our read and our reservation write.
  const racy = outageBucket({ "jobs/heartbeat/desktop-2ki41v6.json": heartbeat("desktop-2ki41v6", t + 86_400_000 - MIN), "jobs/heartbeat/mckinley_home.json": heartbeat("mckinley_home", t + 86_400_000 - MIN) });
  const put = racy.put.bind(racy);
  racy.put = async (key, body, options) => {
    if (key === "jobs/alerts/state.json") racy.store.set(key, { body: JSON.stringify({ schema: "golfprice.alert_state.v1", keys: {} }), v: 50 });
    return put(key, body, options);
  };
  await tick(worker, { DASHBOARD_DATA: racy, ...SECRETS }, t + 86_400_000);
  assert.equal(telegram.length, 1, "lost reservation: no duplicate page");
});

test("first-run guard: seeding records already-failed jobs so they never page; a job that fails later does", async () => {
  const rules = seedState([queuedJob(60, { id: "j-old", state: "failed" }), queuedJob(60, { id: "j-ok", state: "done" })], NOW);
  assert.deepEqual(rules.seeded.failed, ["j-old"]);
  assert.deepEqual(parseAlertState(JSON.parse(JSON.stringify(rules))).seeded.failed, ["j-old"], "seed survives a round trip");
  const jobs = [queuedJob(60, { id: "j-old", state: "failed" }), queuedJob(30, { id: "j-new", state: "failed" })];
  const base = { jobs24h: jobs, recentJobs: jobs, primary: beat(1), backup: beat(1), state: rules, now: NOW, etHHMM: "11:00", etDate: "2026-10-09" };
  assert.deepEqual(dueAlerts(base).map((a) => a.key), ["p2-failed-j-new"]);

  const worker = await loadWorker();
  const failedAt = (id, ago) => ({ id, type: "tuesday", params: {}, requested_by: "o@example.com", requested_at: iso(WED - ago * MIN), verified: true });
  const a = failedAt("j-20261007T140000-aaaaaa", 120), b = failedAt("j-20261007T150000-bbbbbb", 60);
  const bucket = outageBucket({
    "jobs/queue.json": { schema: "golfprice.job_queue.v1", jobs: [a, b] },
    [`jobs/status/${a.id}.json`]: { id: a.id, state: "failed" },
    [`jobs/status/${b.id}.json`]: { id: b.id, state: "failed" },
    "jobs/heartbeat/desktop-2ki41v6.json": heartbeat("desktop-2ki41v6", WED - MIN),
  });
  const env = { DASHBOARD_DATA: bucket, ...SECRETS };
  await tick(worker, env, WED);
  assert.equal(telegram.length, 0, "the backlog of two old failures does not page on first run");
  const state = JSON.parse(bucket.store.get("jobs/alerts/state.json").body);
  assert.deepEqual(state.seeded.failed.sort(), [a.id, b.id].sort());
  // A third job fails afterwards: only it pages.
  const c = failedAt("j-20261007T160000-cccccc", 5);
  bucket.store.set("jobs/queue.json", { body: JSON.stringify({ schema: "golfprice.job_queue.v1", jobs: [a, b, c] }), v: 9 });
  bucket.store.set(`jobs/status/${c.id}.json`, { body: JSON.stringify({ id: c.id, state: "failed" }), v: 1 });
  bucket.store.set("jobs/heartbeat/desktop-2ki41v6.json", { body: JSON.stringify(heartbeat("desktop-2ki41v6", WED + 14 * MIN)), v: 1 });
  await tick(worker, env, WED + 15 * MIN);
  assert.equal(telegram.length, 1);
  assert.match(telegram[0].body.text, /Golf job failed: tuesday \(j-20261007T160000-cccccc\)/);
});
