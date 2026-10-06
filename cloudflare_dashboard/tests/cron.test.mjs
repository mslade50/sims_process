import assert from "node:assert/strict";
import test from "node:test";
import { SCHEDULE_CRONS, SLOTS, WINDOWS, dueJobs, easternClock, isMondaySettleTime } from "../worker/cron-rules.ts";
import { JOB_TYPES } from "../app/jobs-rules.ts";

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

/** Minimal in-memory R2 bucket with etag preconditions (enough for the jobs queue and status objects). */
function fakeBucket(initial = {}) {
  const store = new Map(Object.entries(initial).map(([k, v]) => [k, { text: JSON.stringify(v), etag: "e0" }]));
  let counter = 0;
  return {
    store,
    async get(key) {
      const o = store.get(key);
      return o ? { text: async () => o.text, etag: o.etag } : null;
    },
    async put(key, body, options = {}) {
      const cur = store.get(key);
      if (options.onlyIf?.etagMatches && (!cur || cur.etag !== options.onlyIf.etagMatches)) return null;
      if (options.onlyIf?.etagDoesNotMatch === "*" && cur) return null;
      counter += 1;
      store.set(key, { text: String(body), etag: `e${counter}` });
      return { key };
    },
    queue() {
      const o = store.get("jobs/queue.json");
      return o ? JSON.parse(o.text).jobs : [];
    },
  };
}

const MON_EDT = Date.parse("2026-10-05T12:30:00Z"); // Monday 08:30 EDT
const MON_EST = Date.parse("2026-11-09T13:30:00Z"); // Monday 08:30 EST (daylight saving ended 2026-11-01)

const types = (ms) => dueJobs(ms).map((j) => j.type);
const et = (iso) => Date.parse(iso); // explicit UTC instants below; the New York time is in the comment

test("one 30-minute trigger; Monday 08:30 New York is exact in summer and winter time", () => {
  assert.deepEqual([...SCHEDULE_CRONS], ["0,30 * * * *"]);
  assert.deepEqual(easternClock(MON_EDT), { weekday: "Mon", hour: 8, minute: 30 });
  assert.deepEqual(easternClock(MON_EST), { weekday: "Mon", hour: 8, minute: 30 });
  for (const [day, ok12, ok13] of [["2026-10-05", true, false], ["2026-11-09", false, true], ["2026-03-09", true, false], ["2026-03-02", false, true], ["2027-01-04", false, true], ["2026-07-06", true, false]]) {
    assert.equal(isMondaySettleTime(Date.parse(`${day}T12:30:00Z`)), ok12, `${day} 12:30Z`);
    assert.equal(isMondaySettleTime(Date.parse(`${day}T13:30:00Z`)), ok13, `${day} 13:30Z`);
  }
  assert.equal(isMondaySettleTime(Date.parse("2026-10-06T12:30:00Z")), false, "Tuesday never");
});

test("every scheduled type is in the job allow-list and every slot is on a trigger minute", () => {
  const allowed = new Set(JOB_TYPES.map((s) => s.type));
  for (const slot of SLOTS) {
    assert.ok(allowed.has(slot.type), slot.type);
    for (const t of slot.times) assert.match(t, /^\d\d:(00|30)$/, `${slot.type} ${t} must be a :00 or :30 time (the trigger fires at those minutes)`);
  }
  for (const win of WINDOWS) {
    assert.ok(allowed.has(win.type), win.type);
    for (const m of win.minutes) assert.ok(m === 0 || m === 30);
  }
});

test("the weekly table at New York times (EDT, October 2026)", () => {
  // Monday 2026-10-05
  assert.deepEqual(types(et("2026-10-05T12:30:00Z")), ["monday_settle"]); // 08:30
  assert.deepEqual(types(et("2026-10-05T15:30:00Z")), ["monday_settle"]); // 11:30 retry
  assert.deepEqual(types(et("2026-10-05T19:30:00Z")), ["monday_week"]); // 15:30
  assert.deepEqual(types(et("2026-10-05T20:00:00Z")), ["watch", "odds_reprice"]); // 16:00: window opens, watch first
  assert.deepEqual(types(et("2026-10-05T20:30:00Z")), ["odds_reprice"]); // 16:30: watch is hourly on Monday
  assert.deepEqual(types(et("2026-10-05T14:00:00Z")), []); // 10:00 Monday: nothing
  // Tuesday 12:00, Wednesday 18:00 and 22:30
  assert.deepEqual(types(et("2026-10-06T16:00:00Z")), ["tuesday", "watch", "odds_reprice"]); // pricing first, then watch, then odds
  assert.deepEqual(types(et("2026-10-07T22:00:00Z")), ["wednesday", "watch", "odds_reprice"]); // 18:00
  assert.deepEqual(types(et("2026-10-08T02:30:00Z")), ["wednesday", "odds_reprice"]); // Wed 22:30
  assert.deepEqual(types(et("2026-10-07T08:00:00Z")), ["watch", "odds_reprice"]); // Wed 04:00: overnight
  assert.deepEqual(types(et("2026-10-07T08:30:00Z")), ["odds_reprice"]); // Wed 04:30
  // Thursday 05:30 / 06:30 / 11:00 / 11:30
  assert.deepEqual(types(et("2026-10-08T09:30:00Z")), ["thursday", "watch", "odds_reprice"]);
  assert.deepEqual(types(et("2026-10-08T10:30:00Z")), ["thursday_close", "watch", "odds_reprice"]);
  assert.deepEqual(types(et("2026-10-08T15:00:00Z")), ["watch", "odds_reprice"]); // 11:00 inclusive
  assert.deepEqual(types(et("2026-10-08T15:30:00Z")), []); // 11:30: window closed
  // Rounds: Thu, Fri, Sat 14:30 / 20:30 / 22:30; Fri, Sat, Sun 01:00; Sun 14:30
  assert.deepEqual(types(et("2026-10-08T18:30:00Z")), ["after_round"]); // Thu 14:30
  assert.deepEqual(types(et("2026-10-09T00:30:00Z")), ["after_round"]); // Thu 20:30
  assert.deepEqual(types(et("2026-10-09T02:30:00Z")), ["after_round"]); // Thu 22:30
  assert.deepEqual(types(et("2026-10-09T05:00:00Z")), ["after_round"]); // Fri 01:00
  assert.deepEqual(types(et("2026-10-10T18:30:00Z")), ["after_round"]); // Sat 14:30
  assert.deepEqual(types(et("2026-10-11T18:30:00Z")), ["after_round"]); // Sun 14:30
  assert.deepEqual(types(et("2026-10-08T05:00:00Z")), ["watch", "odds_reprice"]); // Thu 01:00: no round slot on Thursday 01:00, the pre-tee window runs
});

test("winter time: the same New York clock times map to the UTC hour one later", () => {
  assert.deepEqual(types(et("2026-11-09T13:30:00Z")), ["monday_settle"]); // Mon 08:30 EST
  assert.deepEqual(types(et("2026-11-09T12:30:00Z")), []); // 07:30 EST: nothing
  assert.deepEqual(types(et("2026-11-10T17:00:00Z")), ["tuesday", "watch", "odds_reprice"]); // Tue 12:00 EST
  assert.deepEqual(types(et("2026-11-12T11:30:00Z")), ["thursday_close", "watch", "odds_reprice"]); // Thu 06:30 EST
});

test("daylight-saving change days keep New York wall-clock times (2026-11-01 fall back, 2027-03-14 spring forward)", () => {
  assert.deepEqual(easternClock(et("2026-11-01T05:30:00Z")), { weekday: "Sun", hour: 1, minute: 30 }); // 01:30 EDT (first occurrence)
  assert.deepEqual(easternClock(et("2026-11-01T06:30:00Z")), { weekday: "Sun", hour: 1, minute: 30 }); // 01:30 EST (second occurrence)
  assert.deepEqual(easternClock(et("2027-03-14T07:30:00Z")), { weekday: "Sun", hour: 3, minute: 30 }); // 02:30 does not exist: 07:30Z is 03:30 EDT
  assert.deepEqual(types(et("2026-11-02T13:30:00Z")), ["monday_settle"]); // Monday after the change: 08:30 EST
});

test("scheduled handler enqueues one monday_settle job as cron at 08:30 ET, and does nothing on the other DST trigger", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  await worker.scheduled({ scheduledTime: Date.parse("2026-10-05T13:30:00Z"), cron: "30 13 * * 1" }, { DASHBOARD_DATA: bucket }, ctx); // 09:30 EDT: skip
  await Promise.all(waits);
  assert.deepEqual(bucket.queue(), []);
  await worker.scheduled({ scheduledTime: MON_EDT, cron: "30 12 * * 1" }, { DASHBOARD_DATA: bucket }, ctx);
  await Promise.all(waits);
  const jobs = bucket.queue();
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].type, "monday_settle");
  assert.equal(jobs[0].requested_by, "cron");
  assert.equal(jobs[0].requested_at, "2026-10-05T12:30:00Z");
  assert.deepEqual(jobs[0].params, {});
  assert.match(jobs[0].id, /^j-20261005T123000-[0-9a-f]{6}$/);
});

test("winter time: the 13:30 UTC trigger enqueues and the 12:30 one skips", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  await worker.scheduled({ scheduledTime: Date.parse("2026-11-09T12:30:00Z"), cron: "30 12 * * 1" }, { DASHBOARD_DATA: bucket }, ctx);
  await worker.scheduled({ scheduledTime: MON_EST, cron: "30 13 * * 1" }, { DASHBOARD_DATA: bucket }, ctx);
  await Promise.all(waits);
  assert.deepEqual(bucket.queue().map((j) => j.requested_at), ["2026-11-09T13:30:00Z"]);
});

test("same-type rate limit: a still-queued or running settle blocks a second one; a finished one does not", async () => {
  const worker = await loadWorker();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  const earlier = { id: "j-20261005T120000-aaaaaa", type: "monday_settle", params: {}, requested_by: "o@example.com", requested_at: "2026-10-05T12:00:00Z", verified: true };
  const blocked = fakeBucket({ "jobs/queue.json": { jobs: [earlier] } });
  await worker.scheduled({ scheduledTime: MON_EDT }, { DASHBOARD_DATA: blocked }, ctx);
  await Promise.all(waits);
  assert.equal(blocked.queue().length, 1);
  const running = fakeBucket({ "jobs/queue.json": { jobs: [earlier] }, [`jobs/status/${earlier.id}.json`]: { id: earlier.id, state: "running" } });
  await worker.scheduled({ scheduledTime: MON_EDT }, { DASHBOARD_DATA: running }, ctx);
  await Promise.all(waits);
  assert.equal(running.queue().length, 1);
  const done = fakeBucket({ "jobs/queue.json": { jobs: [earlier] }, [`jobs/status/${earlier.id}.json`]: { id: earlier.id, state: "done" } });
  await worker.scheduled({ scheduledTime: MON_EDT }, { DASHBOARD_DATA: done }, ctx);
  await Promise.all(waits);
  assert.equal(done.queue().length, 2);
  assert.equal(done.queue()[1].requested_by, "cron");
});

test("a missing bucket or a broken queue never throws out of the scheduled handler", async () => {
  const worker = await loadWorker();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  await worker.scheduled({ scheduledTime: MON_EDT }, {}, ctx);
  await worker.scheduled({ scheduledTime: MON_EDT }, { DASHBOARD_DATA: fakeBucket({ "jobs/queue.json": { not: "a queue" } }) }, ctx);
  await Promise.all(waits);
});

test("HTTP can never create a cron job: POST /api/jobs rejects requested_by and anonymous callers", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const ctx = { waitUntil() {}, passThroughOnException() {} };
  const post = (body, headers = {}) => worker.fetch(new Request("https://golf.example/api/jobs", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) }), { DASHBOARD_DATA: bucket, ASSETS: { fetch: async () => new Response("no", { status: 404 }) } }, ctx);
  assert.equal((await post({ type: "monday_settle", requested_by: "cron", verified: true })).status, 401);
  assert.equal(bucket.queue().length, 0);
});

test("a Tuesday 12:00 tick enqueues pricing, then watch, then the odds check, as cron, each exactly once; a repeat tick queues nothing new", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  const tue = Date.parse("2026-10-06T16:00:00Z"); // Tuesday 12:00 EDT
  await worker.scheduled({ scheduledTime: tue, cron: "0,30 * * * *" }, { DASHBOARD_DATA: bucket }, ctx);
  await Promise.all(waits);
  assert.deepEqual(bucket.queue().map((j) => j.type), ["tuesday", "watch", "odds_reprice"]);
  assert.ok(bucket.queue().every((j) => j.requested_by === "cron" && j.verified === true && j.requested_at === "2026-10-06T16:00:00Z"));
  assert.equal(new Set(bucket.queue().map((j) => j.id)).size, 3);
  await worker.scheduled({ scheduledTime: tue + 30 * 60_000, cron: "0,30 * * * *" }, { DASHBOARD_DATA: bucket }, ctx); // 12:30: odds is still queued -> not duplicated
  await Promise.all(waits);
  assert.deepEqual(bucket.queue().map((j) => j.type), ["tuesday", "watch", "odds_reprice"]);
});

test("a finished odds check does not block the next one; a stuck queued one does until it is 6 h old", async () => {
  const worker = await loadWorker();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  const prev = { id: "j-20261006T153000-aaaaaa", type: "odds_reprice", params: {}, requested_by: "cron", requested_at: "2026-10-06T15:30:00Z", verified: true };
  const finished = fakeBucket({ "jobs/queue.json": { jobs: [prev] }, [`jobs/status/${prev.id}.json`]: { id: prev.id, state: "done" } });
  await worker.scheduled({ scheduledTime: Date.parse("2026-10-06T16:30:00Z") }, { DASHBOARD_DATA: finished }, ctx); // 12:30 Tuesday
  await Promise.all(waits);
  assert.deepEqual(finished.queue().map((j) => j.type), ["odds_reprice", "odds_reprice"]);
  const stuck = fakeBucket({ "jobs/queue.json": { jobs: [prev] } });
  await worker.scheduled({ scheduledTime: Date.parse("2026-10-06T16:30:00Z") }, { DASHBOARD_DATA: stuck }, ctx);
  await worker.scheduled({ scheduledTime: Date.parse("2026-10-06T21:30:00Z") }, { DASHBOARD_DATA: stuck }, ctx); // 17:30: 6 h after the stuck one -> allowed again
  await Promise.all(waits);
  assert.deepEqual(stuck.queue().map((j) => j.requested_at), ["2026-10-06T15:30:00Z", "2026-10-06T21:30:00Z"]);
});

test("a tick with nothing due never touches the bucket", async () => {
  const worker = await loadWorker();
  const bucket = fakeBucket();
  const waits = [];
  const ctx = { waitUntil: (p) => waits.push(p), passThroughOnException() {} };
  await worker.scheduled({ scheduledTime: Date.parse("2026-10-10T16:00:00Z") }, { DASHBOARD_DATA: bucket }, ctx); // Saturday 12:00
  await Promise.all(waits);
  assert.deepEqual(bucket.queue(), []);
  assert.equal(bucket.store.size, 0);
});
