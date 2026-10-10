import assert from "node:assert/strict";
import test from "node:test";
import { classify, parseConfig, planRetention, handleRetention, retentionDue } from "../worker/retention-rules.ts";

const DAY = 86_400_000;
const NOW = Date.parse("2026-10-10T12:00:00Z");
const jid = (n) => `j-2026090${n}T120000-00000${n}`;
const obj = (key, ageDays, size = 100) => ({ key, size, uploaded: NOW - ageDays * DAY });

test("classify touches only status, logs and claims of well-formed job ids", () => {
  assert.equal(classify(`jobs/status/${jid(1)}.json`).kind, "job_status");
  assert.equal(classify(`jobs/logs/${jid(1)}.txt`).kind, "job_log");
  assert.equal(classify(`jobs/claims/${jid(1)}/desktop-2ki41v6.json`).kind, "job_claim");
  for (const k of ["jobs/queue.json", "jobs/heartbeat/a.json", "jobs/bundles/x/y.zip", "jobs/bundles/index/a.json", "jobs/retention/config.json", "golfprice/latest.json",
    "overrides/active.json", "jobs/status/../queue.json", "jobs/status/notanid.json", "jobs/status/x/j-20260901T120000-000001.json"]) assert.equal(classify(k), null, k);
});

test("plan: 7-day window, newest of each kind and queue-referenced ids survive", () => {
  const objects = [
    obj(`jobs/status/${jid(1)}.json`, 30), obj(`jobs/status/${jid(2)}.json`, 20), obj(`jobs/status/${jid(3)}.json`, 9), obj(`jobs/status/${jid(4)}.json`, 8.9),
    obj(`jobs/status/${jid(5)}.json`, 6), obj(`jobs/logs/${jid(1)}.txt`, 30), obj(`jobs/logs/${jid(2)}.txt`, 10), obj(`jobs/claims/${jid(1)}/a.json`, 30),
    obj("jobs/queue.json", 99), obj("jobs/heartbeat/a.json", 99), obj("golfprice/latest.json", 99),
  ];
  const plan = planRetention(objects, new Set([jid(2)]), NOW, 7);
  const keys = plan.delete.map((o) => o.key);
  assert.deepEqual(keys, [`jobs/logs/${jid(1)}.txt`, `jobs/status/${jid(1)}.json`, `jobs/status/${jid(3)}.json`, `jobs/status/${jid(4)}.json`]);
  assert.ok(!keys.includes(`jobs/status/${jid(2)}.json`), "queue-referenced id kept");
  assert.ok(!keys.includes(`jobs/status/${jid(5)}.json`), "inside the window kept");
  assert.ok(!keys.includes(`jobs/logs/${jid(2)}.txt`), "newest log kept even though old");
  assert.ok(!keys.includes(`jobs/claims/${jid(1)}/a.json`), "newest (only) claim kept");
  assert.ok(!keys.some((k) => !k.startsWith("jobs/status/") && !k.startsWith("jobs/logs/") && !k.startsWith("jobs/claims/")));
  assert.equal(plan.unclassified_ignored, 3);
  assert.equal(plan.summary.count, keys.length);
  assert.equal(plan.summary.bytes, keys.length * 100);
});

test("plan never goes under 7 days and the config parser fails closed", () => {
  const objects = [obj(`jobs/status/${jid(1)}.json`, 3), obj(`jobs/status/${jid(2)}.json`, 2)];
  assert.equal(planRetention(objects, new Set(), NOW, 1).summary.count, 0);
  assert.equal(parseConfig(null).mode, "off");
  assert.equal(parseConfig("{oops").mode, "off");
  assert.equal(parseConfig('{"mode":"delete-everything"}').mode, "off");
  assert.equal(parseConfig('{"mode":"apply","days":1}').days, 7);
  assert.equal(parseConfig('{"mode":"dry-run"}').mode, "dry-run");
});

function fakeBucket(files) {
  const store = new Map(Object.entries(files).map(([k, v]) => [k, { text: v.text ?? "{}", size: v.size ?? 100, uploaded: new Date(v.uploaded ?? NOW - 30 * DAY) }]));
  const deleted = [];
  return {
    store, deleted,
    async get(key) { const o = store.get(key); return o ? { text: async () => o.text } : null; },
    async put(key, body) { store.set(key, { text: String(body), size: String(body).length, uploaded: new Date(NOW) }); return {}; },
    async delete(keys) { for (const k of [].concat(keys)) { deleted.push(k); store.delete(k); } },
    async list({ prefix }) { return { objects: [...store].filter(([k]) => k.startsWith(prefix)).map(([key, o]) => ({ key, size: o.size, uploaded: o.uploaded })), truncated: false }; },
  };
}
const files = () => ({
  "jobs/queue.json": { text: JSON.stringify({ jobs: [{ id: jid(2) }] }) },
  "jobs/heartbeat/a.json": { text: JSON.stringify({ job_id: jid(3) }) },
  [`jobs/status/${jid(1)}.json`]: {}, [`jobs/status/${jid(2)}.json`]: {}, [`jobs/status/${jid(3)}.json`]: {}, [`jobs/status/${jid(4)}.json`]: { uploaded: NOW - 29 * DAY },
  [`jobs/status/${jid(5)}.json`]: { uploaded: NOW - 40 * DAY },
});

test("handler is OFF without a config object and deletes nothing in dry-run", async () => {
  const b = fakeBucket(files());
  assert.equal((await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true)).ran, false);
  b.store.set("jobs/retention/config.json", { text: '{"mode":"dry-run"}', size: 1, uploaded: new Date(NOW) });
  const out = await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true);
  assert.equal(out.ran, true);
  assert.equal(b.deleted.length, 0);
  const report = JSON.parse(b.store.get("jobs/retention/last_report.json").text);
  assert.equal(report.mode, "dry-run");
  assert.deepEqual(report.delete_keys.sort(), [`jobs/status/${jid(1)}.json`, `jobs/status/${jid(5)}.json`]);
});

test("apply deletes exactly the planned keys, records them first, keeps protected ones", async () => {
  const b = fakeBucket(files());
  b.store.set("jobs/retention/config.json", { text: '{"mode":"apply"}', size: 1, uploaded: new Date(NOW) });
  const out = await handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true);
  assert.equal(out.ran, true);
  assert.ok(!b.deleted.includes(`jobs/status/${jid(2)}.json`), "queue id");
  assert.ok(!b.deleted.includes(`jobs/status/${jid(3)}.json`), "heartbeat job id");
  assert.ok(!b.deleted.includes(`jobs/status/${jid(4)}.json`), "newest status");
  assert.deepEqual(b.deleted.sort(), [`jobs/status/${jid(1)}.json`, `jobs/status/${jid(5)}.json`]);
  assert.ok([...b.store.keys()].some((k) => k.startsWith("jobs/retention/applied/")), "applied record written");
  assert.ok(b.store.has("jobs/queue.json") && b.store.has("jobs/heartbeat/a.json"));
});

test("refuses to plan without a valid queue index", async () => {
  const b = fakeBucket({ ...files(), "jobs/queue.json": { text: "not json" } });
  b.store.set("jobs/retention/config.json", { text: '{"mode":"apply"}', size: 1, uploaded: new Date(NOW) });
  await assert.rejects(handleRetention({ scheduledTime: NOW }, { DASHBOARD_DATA: b }, true));
  assert.equal(b.deleted.length, 0);
});

test("daily slot is 03:15 New York time", () => {
  assert.equal(retentionDue(Date.parse("2026-10-10T07:15:00Z")), true); // EDT
  assert.equal(retentionDue(Date.parse("2026-11-10T08:15:00Z")), true); // EST
  assert.equal(retentionDue(Date.parse("2026-10-10T07:30:00Z")), false);
});
