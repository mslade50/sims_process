import assert from "node:assert/strict";
import test from "node:test";
import { MONDAY_SETTLE_CRONS, easternClock, isMondaySettleTime } from "../worker/cron-rules.ts";

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

test("both UTC triggers exist and exactly one of them is 08:30 New York in every season", () => {
  assert.deepEqual([...MONDAY_SETTLE_CRONS], ["30 12 * * 1", "30 13 * * 1"]);
  assert.deepEqual(easternClock(MON_EDT), { weekday: "Mon", hour: 8, minute: 30 });
  assert.deepEqual(easternClock(MON_EST), { weekday: "Mon", hour: 8, minute: 30 });
  for (const [day, ok12, ok13] of [["2026-10-05", true, false], ["2026-11-09", false, true], ["2026-03-09", true, false], ["2026-03-02", false, true], ["2027-01-04", false, true], ["2026-07-06", true, false]]) {
    assert.equal(isMondaySettleTime(Date.parse(`${day}T12:30:00Z`)), ok12, `${day} 12:30Z`);
    assert.equal(isMondaySettleTime(Date.parse(`${day}T13:30:00Z`)), ok13, `${day} 13:30Z`);
  }
  assert.equal(isMondaySettleTime(Date.parse("2026-10-06T12:30:00Z")), false, "Tuesday never");
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
